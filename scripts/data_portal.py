"""CLI entrypoint for managing labeling platform data workflows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.assignment import AssignmentService  # noqa: E402
from src.labeling_app.core.models import Dataset  # noqa: E402
from src.labeling_app.db import client_scope  # noqa: E402
from src.labeling_app.settings import AppSettings, get_settings  # noqa: E402
from src.labeling_app.workflows import admin  # noqa: E402

app = typer.Typer(help="Manage Streamlit labeling platform data.", invoke_without_command=True)
console = Console()


def _parse_dataset(value: str) -> Dataset:
    try:
        return Dataset(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _require_settings() -> AppSettings:
    return get_settings()


def _interactive_dataset_prompt(allow_all: bool = False) -> str | None:
    suffix = "/all" if allow_all else ""
    default_value = "all" if allow_all else "aita"
    user_input = typer.prompt(f"Dataset [aita/scenario{suffix}]", default=default_value)
    normalized = user_input.strip().lower()
    if allow_all and normalized in {"all", ""}:
        return None
    dataset = _parse_dataset(normalized)
    return dataset.value


def _interactive_menu(ctx: typer.Context) -> None:
    console.print("[bold cyan]\nLLM Labeling Data Portal[/bold cyan]")
    options = {
        "1": "Ensure database schema",
        "2": "Sync seed data (dry run)",
        "3": "Sync seed data and apply",
        "4": "Export review records",
        "5": "Show dataset status",
        "0": "Exit",
    }

    while True:
        for key, label in options.items():
            console.print(f"  [green]{key}[/green] {label}")
        choice = typer.prompt("Select an option", default="0").strip()

        if choice == "0":
            console.print("Goodbye.")
            return
        if choice not in options:
            console.print("[yellow]Invalid selection. Please choose again.[/yellow]")
            continue

        try:
            if choice == "1":
                ctx.invoke(init_db)
            elif choice in {"2", "3"}:
                dataset_value = _interactive_dataset_prompt(allow_all=True)
                limit_input = typer.prompt(
                    "Limit number of records (press Enter for no limit)", default=""
                )
                limit = int(limit_input) if limit_input.strip() else None
                apply_changes = choice == "3" and typer.confirm(
                    "Apply changes to the database?", default=True
                )
                ctx.invoke(
                    push_dataset,
                    apply=apply_changes,
                    dataset=dataset_value,
                    run_file=None,
                    limit=limit,
                    record_start=None,
                    record_end=None,
                )
            elif choice == "4":
                dataset_value = _interactive_dataset_prompt(allow_all=True)
                ctx.invoke(pull_exports, target="reviews", dataset=dataset_value)
            elif choice == "5":
                dataset_value = _interactive_dataset_prompt(allow_all=False)
                ctx.invoke(status, dataset=dataset_value)
        except typer.BadParameter as exc:
            console.print(f"[red]Error:[/red] {exc}")
        except Exception as exc:  # pragma: no cover - defensive interactive fallback
            console.print(f"[red]Unexpected error:[/red] {exc}")


@app.callback()
def interactive_entry(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _interactive_menu(ctx)


@app.command("init-db")
def init_db() -> None:
    """Create database tables if they are missing."""
    settings = _require_settings()
    status = admin.ensure_database_schema(settings)

    table = Table(title="Database Schema Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Driver", status.driver)
    table.add_row("URL", status.url_display)
    table.add_row("Auth token", "present" if status.auth_token_present else "missing")
    table.add_row("Tables", ", ".join(status.tables) or "(none)")
    table.add_row("Missing tables", ", ".join(status.missing_tables) or "(none)")

    console.print(table)

    if status.is_complete:
        console.print("[green]Database schema initialized successfully.[/green]")
    else:
        console.print("[yellow]Schema ensured, but some expected tables are missing.[/yellow]")


@app.command("push")
def push_dataset(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Persist changes instead of running a dry-run diff.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Restrict sync to a single dataset (aita or scenario).",
    ),
    run_file: Path | None = typer.Option(
        None,
        "--run-file",
        help="Optional path to an outputs/runs/<run_id>/run.jsonl file for scenario ingestion.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optional cap on records ingested from --run-file.",
    ),
    record_start: int | None = typer.Option(
        None,
        "--record-start",
        help="Start offset for run ingestion (0-indexed).",
    ),
    record_end: int | None = typer.Option(
        None,
        "--record-end",
        help="End offset for run ingestion (non-inclusive).",
    ),
) -> None:
    """Sync all datasets from local seeds and scenario runs."""
    settings = _require_settings()

    range_tuple = None
    if record_start is not None or record_end is not None:
        range_tuple = (record_start or 0, record_end)

    datasets: list[Dataset] | None = None
    if dataset:
        datasets = [_parse_dataset(dataset)]

    results = admin.seed_datasets(
        datasets=datasets,
        apply=apply,
        run_file=run_file,
        limit=limit,
        record_range=range_tuple,
        settings=settings,
    )

    for result in results:
        _render_seed_result(result, apply)


def _render_seed_result(result: admin.SeedRunResult, applied: bool) -> None:
    diff = result.diff
    table = Table(title=f"{result.dataset.value.title()} seed diff")
    table.add_column("New")
    table.add_column("Changed")
    table.add_column("Deleted")
    table.add_column("Unchanged")
    table.add_column("Payloads")
    table.add_row(
        str(len(diff.new)),
        str(len(diff.changed)),
        str(len(diff.deleted)),
        str(diff.unchanged),
        str(result.payload_count),
    )
    console.print(table)

    if result.run_file:
        console.print(f"[cyan]Run file:[/cyan] {result.run_file}")
    if result.note:
        console.print(f"[yellow]{result.note}[/yellow]")

    if applied:
        if diff.has_changes():
            console.print(
                f"[green]Applied changes for {result.dataset.value} (total responses: {result.total_responses}).[/green]"
            )
        else:
            console.print(
                f"[blue]No changes detected for {result.dataset.value}; database already up-to-date.[/blue]"
            )
    else:
        console.print(
            f"[yellow]Dry run for {result.dataset.value}. Re-run with --apply to persist changes.[/yellow]"
        )


def _render_review_summary(summary: admin.ReviewExportSummary) -> None:
    export = summary.export
    console.print(
        f"[green]{summary.dataset.value.title()} reviews export: {export.record_count} records → {export.path}[/green]"
    )
    console.print(f"[cyan]Review table rows:[/cyan] {summary.review_only_count}")

    sample = summary.sample
    if sample:
        console.print("[magenta]Sample review:[/magenta]")
        keys_to_show = [
            ("llm_response_id", sample.get("llm_response_id")),
            ("reviewer_code", sample.get("reviewer_code")),
            ("score", sample.get("score")),
            ("notes", (sample.get("notes") or "")[:80] or None),
            ("review_created_at", sample.get("review_created_at")),
        ]
        for label, value in keys_to_show:
            console.print(f"  [dim]{label}:[/dim] {value}")


@app.command("pull")
def pull_exports(
    target: str = typer.Option(
        "reviews",
        "--target",
        "-t",
        help="Export target: reviews or evals.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Restrict export to a single dataset (aita or scenario).",
    ),
) -> None:
    """Export data for downstream analysis."""
    settings = _require_settings()
    datasets: list[Dataset] | None = None
    if dataset:
        datasets = [_parse_dataset(dataset)]

    if target == "reviews":
        summaries = admin.export_reviews_for_datasets(datasets=datasets, settings=settings)
        for summary in summaries:
            _render_review_summary(summary)
        return

    raise typer.BadParameter(f"Unsupported export target: {target}")


@app.command("status")
def status(
    dataset: str = typer.Option(
        "aita",
        "--dataset",
        "-d",
        help="Dataset to inspect.",
    ),
) -> None:
    """Show dataset statistics (total responses, coverage distribution)."""
    settings = _require_settings()
    dataset_enum = _parse_dataset(dataset)

    with client_scope(settings) as client:
        assignment = AssignmentService(client)
        progress = assignment.get_progress(dataset_enum, reviewer_code="__aggregate__")

        table = Table(title=f"{dataset_enum.value.title()} dataset status")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total responses", str(progress.total_responses))
        table.add_row(
            "Coverage distribution",
            ", ".join(
                f"{reviews} response{'s' if reviews != 1 else ''} with {count} review{'s' if count != 1 else ''}"
                for count, reviews in sorted(progress.coverage_by_count.items())
            )
            or "No entries",
        )

        console.print(table)


@app.command("delete-llm-reviews")
def delete_llm_reviews(
    dataset: str = typer.Option("aita"),
    confirm: bool = typer.Option(False, "--yes"),
) -> None:
    """Delete all LLM-generated reviews from the database."""
    settings = _require_settings()
    dataset_enum = _parse_dataset(dataset)

    with client_scope(settings) as client:
        # Count LLM reviews
        count_query = """
        SELECT COUNT(*) as count
        FROM reviews r
        JOIN llm_responses resp ON r.llm_response_id = resp.id
        WHERE r.reviewer_code LIKE 'llm:%' AND resp.dataset = ?
        """
        count = client.execute(count_query, [dataset_enum.value]).first_value()

        if count == 0:
            console.print("No LLM reviews found.")
            return

        console.print(f"Found {count} LLM reviews for dataset '{dataset}'")

        if not confirm:
            response = typer.confirm(f"Are you sure you want to delete {count} LLM reviews?")
            if not response:
                console.print("Cancelled.")
                return

        # Delete LLM reviews
        delete_query = """
        DELETE FROM reviews
        WHERE id IN (
            SELECT r.id
            FROM reviews r
            JOIN llm_responses resp ON r.llm_response_id = resp.id
            WHERE r.reviewer_code LIKE 'llm:%' AND resp.dataset = ?
        )
        """
        client.execute(delete_query, [dataset_enum.value])

        console.print(f"✅ Deleted {count} LLM reviews from {dataset} dataset")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
