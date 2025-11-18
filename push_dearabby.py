#!/usr/bin/env python3
"""Quick script to push Dear Abby sampled data to database."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.models import Dataset  # noqa: E402
from src.labeling_app.settings import get_settings  # noqa: E402
from src.labeling_app.workflows import admin  # noqa: E402
from scripts.data_portal import _render_seed_result, console  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    run_file = PROJECT_ROOT / "outputs/runs/run_f2d3e4b22e01564f/sampled_run.jsonl"
    
    if not run_file.exists():
        console.print(f"[red]Error:[/red] File not found: {run_file}")
        sys.exit(1)
    
    console.print(f"[cyan]Pushing Dear Abby data from:[/cyan] {run_file}")
    console.print("[yellow]Applying changes to database...[/yellow]\n")
    
    # Use admin.seed_datasets which handles run files correctly
    datasets = [Dataset.DEARABBY]
    results = admin.seed_datasets(
        settings=settings,
        datasets=datasets,
        run_file=run_file,
        limit=None,
        record_range=None,
        apply=True,
    )
    
    if results:
        for result in results:
            _render_seed_result(result, applied=True)
        console.print("\n[green]Done![/green] The data is now available in the labeling platform.")
    else:
        console.print("[red]No results returned.[/red]")
        sys.exit(1)

