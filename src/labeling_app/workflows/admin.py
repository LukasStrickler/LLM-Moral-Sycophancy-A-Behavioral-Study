"""Administrative workflows consolidating maintenance scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from ..core.models import Dataset
from ..db import client_scope, ensure_schema, queries
from ..settings import AppSettings, get_settings
from . import exporting, seeding

_EXPECTED_TABLES: tuple[str, ...] = ("llm_responses", "reviews")
_DEFAULT_DATASETS: tuple[Dataset, ...] = (Dataset.AITA, Dataset.SCENARIO)


@dataclass(frozen=True)
class SchemaStatus:
    """Summary of a database schema ensure operation."""

    url_display: str
    driver: str
    tables: list[str]
    expected_tables: Sequence[str]
    auth_token_present: bool

    @property
    def missing_tables(self) -> list[str]:
        return [table for table in self.expected_tables if table not in self.tables]

    @property
    def is_complete(self) -> bool:  # pragma: no cover - simple proxy property
        return not self.missing_tables


@dataclass(frozen=True)
class SeedRunResult:
    """Outcome of a seeding sync for a specific dataset."""

    dataset: Dataset
    payload_count: int
    diff: seeding.SeedDiff
    applied: bool
    total_responses: int
    source: str
    run_file: Path | None
    note: str | None


@dataclass(frozen=True)
class ReviewExportSummary:
    """Metadata about a review export suitable for CLI presentation."""

    dataset: Dataset
    export: exporting.ExportResult
    review_only_count: int

    @property
    def sample(self) -> dict | None:
        return self.export.sample


def ensure_database_schema(settings: AppSettings | None = None) -> SchemaStatus:
    """Create required tables (if missing) and report the resulting catalog."""

    settings = settings or get_settings()
    database_url = settings.require_database_url()
    url_display = _mask_url(database_url)

    with client_scope(settings) as client:
        ensure_schema(client)
        table_rows = client.execute("SELECT name FROM sqlite_master WHERE type='table'").rows
        tables = sorted(str(row[0]) for row in table_rows if row and row[0])
        return SchemaStatus(
            url_display=url_display,
            driver=client.driver,
            tables=tables,
            expected_tables=list(_EXPECTED_TABLES),
            auth_token_present=bool(settings.auth_token),
        )


def seed_datasets(
    datasets: Sequence[Dataset] | None = None,
    *,
    apply: bool = False,
    run_file: Path | None = None,
    limit: int | None = None,
    record_range: tuple[int, int | None] | None = None,
    settings: AppSettings | None = None,
) -> list[SeedRunResult]:
    """Sync one or more datasets from local seeds or scenario runs."""

    settings = settings or get_settings()
    selected = list(datasets or _DEFAULT_DATASETS)

    results: list[SeedRunResult] = []
    with client_scope(settings) as client:
        for dataset in selected:
            payloads = seeding.load_seed_payloads(dataset, settings=settings, limit=limit)
            source = "seed"
            used_run_file: Path | None = None
            note: str | None = None
            skipped_messages: list[str] = []

            if dataset is Dataset.SCENARIO:
                candidate_run_file = run_file or settings.find_latest_run_file()
                if candidate_run_file and candidate_run_file.exists():
                    used_run_file = candidate_run_file
                    payloads, skipped_messages = seeding.payloads_from_run_file(
                        candidate_run_file,
                        dataset=dataset,
                        limit=limit,
                        record_range=record_range,
                    )
                    source = "run"
            if skipped_messages:
                note = (
                    (note + " ") if note else ""
                ) + f"Skipped {len(skipped_messages)} run records (e.g., {skipped_messages[0]})."
            elif dataset is Dataset.SCENARIO:
                if run_file and not (used_run_file and source == "run"):
                    note = f"Run file not found: {run_file}"
                elif not run_file:
                    note = "No scenario run file detected; using seed payloads only."

            diff = seeding.sync_dataset(client, dataset, payloads, apply_changes=apply)
            total_responses = queries.count_responses(client, dataset)
            results.append(
                SeedRunResult(
                    dataset=dataset,
                    payload_count=len(payloads),
                    diff=diff,
                    applied=apply and diff.has_changes(),
                    total_responses=total_responses,
                    source=source,
                    run_file=used_run_file,
                    note=note,
                )
            )

    return results


def export_reviews_for_datasets(
    datasets: Sequence[Dataset] | None = None,
    *,
    settings: AppSettings | None = None,
) -> list[ReviewExportSummary]:
    """Export review data for the requested datasets and capture summary metadata."""

    settings = settings or get_settings()
    selected = list(datasets or _DEFAULT_DATASETS)

    summaries: list[ReviewExportSummary] = []
    with client_scope(settings) as client:
        for dataset in selected:
            export_result = exporting.export_reviews(client, dataset, settings=settings)
            reviews_only = queries.select_reviews_for_export(client, dataset)
            summaries.append(
                ReviewExportSummary(
                    dataset=dataset,
                    export=export_result,
                    review_only_count=len(reviews_only),
                )
            )

    return summaries


def _mask_url(url: str) -> str:
    """Mask credentials in database URLs while preserving file paths."""
    if "@" not in url:
        return url

    parsed = urlparse(url)

    # SQLite URLs have no netloc (or empty netloc), so '@' is part of the path
    if parsed.scheme in ("sqlite", "file") or not parsed.netloc:
        return url

    # For URLs with credentials (user:pass@host), mask the userinfo
    if "@" in parsed.netloc:
        _, host_part = parsed.netloc.rsplit("@", 1)
        masked_netloc = f"***@{host_part}"
        return urlunparse(
            (
                parsed.scheme,
                masked_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    return url


__all__ = [
    "SchemaStatus",
    "SeedRunResult",
    "ReviewExportSummary",
    "ensure_database_schema",
    "seed_datasets",
    "export_reviews_for_datasets",
]
