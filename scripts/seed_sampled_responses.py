#!/usr/bin/env python3
"""Seed sampled responses from JSONL file into database.

This script loads sampled responses from outputs/scoring/sampled_responses.jsonl
and inserts them into the database using the seeding infrastructure.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.models import Dataset
from src.labeling_app.db import client_scope
from src.labeling_app.settings import get_settings
from src.labeling_app.workflows import seeding


INPUT_FILE = PROJECT_ROOT / "outputs" / "scoring" / "sampled_responses.jsonl"


def load_payloads_from_file(jsonl_path: Path) -> list[seeding.LLMResponsePayload]:
    """Load payloads from custom JSONL file."""
    if not jsonl_path.exists():
        print(f"Error: Input file not found: {jsonl_path}")
        sys.exit(1)

    payloads = []
    for record in seeding._load_jsonl(jsonl_path):
        payloads.append(seeding._payload_from_seed_record(record, Dataset.AITA))

    print(f"Loaded {len(payloads)} records from {jsonl_path.name}")
    return payloads


def main() -> None:
    """Main execution."""
    print("=" * 60)
    print("Seed Sampled Responses")
    print("=" * 60)
    print()

    # Load payloads from JSONL file
    print(f"Loading responses from: {INPUT_FILE}")
    payloads = load_payloads_from_file(INPUT_FILE)
    print()

    # Sync with database
    settings = get_settings()
    with client_scope(settings) as client:
        print("Syncing with database...")
        diff = seeding.sync_dataset(client, Dataset.AITA, payloads, apply_changes=True)

        print()
        print("=" * 60)
        print("Seeding Results")
        print("=" * 60)
        print(f"  - New records: {len(diff.new)}")
        print(f"  - Changed records: {len(diff.changed)}")
        print(f"  - Deleted records: {len(diff.deleted)}")
        print(f"  - Unchanged records: {diff.unchanged}")
        print()

        if diff.has_changes():
            print("✅ Changes applied to database")
        else:
            print("ℹ️  No changes needed (all records already in database)")
        print("=" * 60)


if __name__ == "__main__":
    main()
