#!/usr/bin/env python3
"""Validate current AITA reviews against backup files."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_CURRENT = Path("data/humanLabel/reviews/aita_reviews.jsonl")
DEFAULT_BACKUPS = (
    Path("data/humanLabel/reviews/aita_reviews.jsonl.backup_20251215_170856"),
    Path("data/humanLabel/reviews/aita_reviews.jsonl.pre_merge_backup"),
)
DEFAULT_IGNORE_FIELDS = ("llm_response_id", "review_updated_at")


@dataclass
class FileStats:
    path: Path
    total_lines: int = 0
    parsed_records: int = 0
    invalid_lines: int = 0
    full_hashes: set[str] = field(default_factory=set)
    full_dupes: int = 0
    key_hashes: set[str] = field(default_factory=set)
    key_dupes: int = 0
    records_by_full: dict[str, dict] = field(default_factory=dict)
    records_by_key: dict[str, dict] = field(default_factory=dict)


def _hash_record(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strip_fields(record: dict, ignore_fields: Iterable[str]) -> dict:
    return {key: value for key, value in record.items() if key not in ignore_fields}


def load_stats(path: Path, ignore_fields: Iterable[str]) -> FileStats:
    stats = FileStats(path=path)
    if not path.exists():
        print(f"Missing file: {path}")
        return stats

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stats.total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.invalid_lines += 1
                continue

            stats.parsed_records += 1

            full_hash = _hash_record(record)
            if full_hash in stats.full_hashes:
                stats.full_dupes += 1
            else:
                stats.full_hashes.add(full_hash)
                stats.records_by_full[full_hash] = record

            key_record = _strip_fields(record, ignore_fields)
            key_hash = _hash_record(key_record)
            if key_hash in stats.key_hashes:
                stats.key_dupes += 1
            else:
                stats.key_hashes.add(key_hash)
                stats.records_by_key[key_hash] = record

    return stats


def print_stats(stats: FileStats, label: str) -> None:
    print(f"{label}: {stats.path}")
    print(f"  Total lines:       {stats.total_lines}")
    print(f"  Parsed records:    {stats.parsed_records}")
    print(f"  Invalid lines:     {stats.invalid_lines}")
    print(f"  Unique full:       {len(stats.full_hashes)} (dupes {stats.full_dupes})")
    print(f"  Unique key:        {len(stats.key_hashes)} (dupes {stats.key_dupes})")
    print()


def write_missing(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} missing records to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT,
        help="Path to the current aita_reviews.jsonl",
    )
    parser.add_argument(
        "--backup",
        action="append",
        type=Path,
        dest="backups",
        help="Backup files to compare (can be repeated)",
    )
    parser.add_argument(
        "--ignore-field",
        action="append",
        dest="ignore_fields",
        help="Field names to ignore when computing key hashes (can be repeated)",
    )
    parser.add_argument(
        "--write-missing-full",
        type=Path,
        help="Write records missing by full hash to this JSONL file",
    )
    parser.add_argument(
        "--write-missing-key",
        type=Path,
        help="Write records missing by key hash to this JSONL file",
    )
    args = parser.parse_args()

    backups = args.backups or list(DEFAULT_BACKUPS)
    ignore_fields = args.ignore_fields or list(DEFAULT_IGNORE_FIELDS)

    print("Using ignore fields for key comparison:")
    for field in ignore_fields:
        print(f"  - {field}")
    print()

    current_stats = load_stats(args.current, ignore_fields)
    print_stats(current_stats, "CURRENT")

    missing_full_records: list[dict] = []
    missing_key_records: list[dict] = []

    for backup in backups:
        backup_stats = load_stats(backup, ignore_fields)
        print_stats(backup_stats, "BACKUP")

        missing_full = backup_stats.full_hashes - current_stats.full_hashes
        missing_key = backup_stats.key_hashes - current_stats.key_hashes

        print(f"Missing full hashes from {backup.name}: {len(missing_full)}")
        print(f"Missing key hashes  from {backup.name}: {len(missing_key)}")
        print()

        missing_full_records.extend(
            backup_stats.records_by_full[hash_key]
            for hash_key in missing_full
            if hash_key in backup_stats.records_by_full
        )
        missing_key_records.extend(
            backup_stats.records_by_key[hash_key]
            for hash_key in missing_key
            if hash_key in backup_stats.records_by_key
        )

    if args.write_missing_full:
        write_missing(args.write_missing_full, missing_full_records)
    if args.write_missing_key:
        write_missing(args.write_missing_key, missing_key_records)


if __name__ == "__main__":
    main()
