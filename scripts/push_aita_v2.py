#!/usr/bin/env python3
"""Push AITA seed v2 file to database.

This script loads the aita_seed_v2.jsonl file and pushes it to the database
using the same seeding infrastructure.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.models import Dataset  # noqa: E402
from src.labeling_app.db import client_scope, queries as db_queries  # noqa: E402
from src.labeling_app.settings import get_settings  # noqa: E402
from src.labeling_app.workflows import admin, seeding  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()

SEED_V2_FILE = PROJECT_ROOT / "data" / "humanLabel" / "seeds" / "aita_seed_v2.jsonl"


def load_v2_seed_payloads() -> list[seeding.LLMResponsePayload]:
    """Load payloads from aita_seed_v2.jsonl file."""
    if not SEED_V2_FILE.exists():
        console.print(f"[red]Error: Seed file not found: {SEED_V2_FILE}[/red]")
        sys.exit(1)
    
    payloads = []
    for record in seeding._load_jsonl(SEED_V2_FILE):
        payloads.append(seeding._payload_from_seed_record(record, Dataset.AITA))
    
    console.print(f"[green]Loaded {len(payloads)} records from {SEED_V2_FILE.name}[/green]")
    return payloads


def main() -> None:
    """Push AITA v2 seed to database."""
    console.print("[bold cyan]Push AITA Seed v2 to Database[/bold cyan]\n")
    
    # Load payloads from v2 file
    payloads = load_v2_seed_payloads()
    
    if not payloads:
        console.print("[yellow]No records to push[/yellow]")
        sys.exit(0)
    
    # Show preview
    console.print(f"\n[cyan]Preview of first record:[/cyan]")
    first = payloads[0]
    console.print(f"  Identifier: {first.identifier}")
    console.print(f"  Model: {first.model_id}")
    console.print(f"  Run ID: {first.run_id}")
    console.print(f"  Title: {first.prompt_title[:60]}...")
    console.print()
    
    # Ask for confirmation
    response = input("Push these records to database? [y/N]: ").strip().lower()
    if response != "y":
        console.print("[yellow]Push cancelled.[/yellow]")
        sys.exit(0)
    
    console.print()
    console.print("[yellow]Computing diff (dry run)...[/yellow]\n")
    
    # Compute diff
    settings = get_settings()
    with client_scope(settings) as client:
        diff = seeding.sync_dataset(
            client=client,
            dataset=Dataset.AITA,
            payloads=payloads,
            apply_changes=False,
        )
    
    # Show diff
    console.print(f"[cyan]Diff Summary:[/cyan]")
    console.print(f"  [green]New records:[/green] {len(diff.new)}")
    console.print(f"  [yellow]Changed records:[/yellow] {len(diff.changed)}")
    if len(diff.deleted) > 0:
        console.print(f"  [red]Records to delete:[/red] {len(diff.deleted)}")
        console.print(f"  [yellow]Warning: These records will be permanently deleted from the database[/yellow]")
    console.print(f"  [dim]Unchanged:[/dim] {diff.unchanged}")
    console.print()
    
    # Only proceed if there are changes
    if len(diff.new) == 0 and len(diff.changed) == 0 and len(diff.deleted) == 0:
        console.print("[yellow]No changes to apply. Database is already up-to-date.[/yellow]")
        sys.exit(0)
    
    # Ask for final confirmation
    console.print(f"[cyan]Will add {len(diff.new)} new records, update {len(diff.changed)} existing records.[/cyan]")
    if len(diff.deleted) > 0:
        console.print(f"[red]Will delete {len(diff.deleted)} records that are not in v2 file.[/red]")
    console.print()
    
    response = input("Apply changes to database? [y/N]: ").strip().lower()
    if response != "y":
        console.print("[yellow]Push cancelled.[/yellow]")
        sys.exit(0)
    
    console.print()
    console.print("[yellow]Applying changes to database...[/yellow]\n")
    
    # Apply all changes including deletions
    with client_scope(settings) as client:
        # Insert new records
        for payload in diff.new:
            db_queries.insert_response_row(
                client,
                dataset=payload.dataset,
                identifier=payload.identifier,
                prompt_title=payload.prompt_title,
                prompt_body=payload.prompt_body,
                model_response_text=payload.model_response_text,
                model_id=payload.model_id,
                run_id=payload.run_id,
                metadata_json=seeding.serialize_metadata(payload.metadata),
            )
        
        # Update changed records
        for existing, payload in diff.changed:
            db_queries.update_response_row(
                client,
                existing["id"],
                prompt_title=payload.prompt_title,
                prompt_body=payload.prompt_body,
                model_response_text=payload.model_response_text,
                model_id=payload.model_id,
                metadata_json=seeding.serialize_metadata(payload.metadata),
            )
        
        # Delete records not in v2 file
        for row in diff.deleted:
            db_queries.delete_response_row(client, row["id"])
    
    console.print(f"\n[green]Successfully pushed {len(diff.new)} new records to database![/green]")
    if len(diff.changed) > 0:
        console.print(f"[green]Updated {len(diff.changed)} existing records![/green]")
    if len(diff.deleted) > 0:
        console.print(f"[green]Deleted {len(diff.deleted)} records not in v2 file![/green]")


if __name__ == "__main__":
    main()

