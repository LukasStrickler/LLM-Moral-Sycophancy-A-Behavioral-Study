"""Data seeding workflows for the labeling platform."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...benchmark.core.logging import setup_logger
from ..core.models import Dataset
from ..db import DatabaseClient
from ..db import queries as db_queries
from ..db.libsql import parse_metadata, serialize_metadata
from ..settings import AppSettings, get_settings

logger = setup_logger("seeding")


def _validate_identifier(identifier: str, context: str = "") -> tuple[str, str]:
    """Validate identifier format and extract prompt_id, model_id.

    Raises ValueError if identifier is malformed.
    """
    if ":" not in identifier:
        raise ValueError(
            f"Invalid identifier format {context}: '{identifier}'. "
            f"Expected format: 'prompt_id:model_id'"
        )

    parts = identifier.split(":", 1)
    prompt_id, model_id_part = parts[0], parts[1]

    if not prompt_id or not model_id_part:
        raise ValueError(
            f"Invalid identifier format {context}: '{identifier}'. "
            f"Both prompt_id and model_id must be non-empty."
        )

    return prompt_id, model_id_part


class SeedError(RuntimeError):
    """Raised when a seed file or run payload is malformed."""


@dataclass(slots=True)
class LLMResponsePayload:
    """Represents the canonical data required to populate an `LLMResponse` row."""

    dataset: Dataset
    identifier: str
    prompt_title: str
    prompt_body: str
    model_response_text: str
    model_id: str
    run_id: str
    metadata: dict

    def key(self) -> tuple[str, str, str]:
        return self.dataset.value, self.identifier, self.run_id

    def to_model_kwargs(self) -> dict:
        return {
            "dataset": self.dataset,
            "identifier": self.identifier,
            "prompt_title": self.prompt_title,
            "prompt_body": self.prompt_body,
            "model_response_text": self.model_response_text,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "metadata_json": self.metadata,
        }


@dataclass
class SeedDiff:
    """Diff summary between existing database state and incoming payloads."""

    new: list[LLMResponsePayload]
    changed: list[tuple[dict, LLMResponsePayload]]
    deleted: list[dict]
    unchanged: int

    def has_changes(self) -> bool:
        return bool(self.new or self.changed or self.deleted)


def _load_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise SeedError(f"Seed file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SeedError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def _payload_from_seed_record(record: dict, dataset: Dataset) -> LLMResponsePayload:
    required_fields = [
        "identifier",
        "prompt_title",
        "prompt_body",
        "model_response_text",
        "model_id",
        "run_id",
    ]
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise SeedError(f"Seed record missing fields {missing}: {record}")

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise SeedError(f"`metadata` must be an object for seed record: {record}")

    return LLMResponsePayload(
        dataset=dataset,
        identifier=str(record["identifier"]),
        prompt_title=str(record["prompt_title"]),
        prompt_body=str(record["prompt_body"]),
        model_response_text=str(record["model_response_text"]),
        model_id=str(record["model_id"]),
        run_id=str(record["run_id"]),
        metadata=metadata,
    )


def load_seed_payloads(
    dataset: Dataset,
    settings: AppSettings | None = None,
    limit: int | None = None,
) -> list[LLMResponsePayload]:
    """Load seed payloads for the requested dataset from JSONL."""
    settings = settings or get_settings()
    if dataset is Dataset.AITA:
        seed_path = settings.paths.seeds / "aita_seed.jsonl"
    elif dataset is Dataset.SCENARIO:
        seed_path = settings.paths.seeds / "scenario_seed.jsonl"
    else:
        raise SeedError(f"Unsupported dataset: {dataset}")

    payloads = [_payload_from_seed_record(record, dataset) for record in _load_jsonl(seed_path)]

    if limit is not None:
        if dataset is Dataset.AITA:
            # For AITA: simple limit (first N records)
            payloads = payloads[:limit]
        elif dataset is Dataset.SCENARIO:
            # For SCENARIO: even distribution across models and prompt IDs
            payloads = _apply_even_distribution_limit(payloads, limit)

    return payloads


def _apply_even_distribution_limit(
    payloads: list[LLMResponsePayload], limit: int
) -> list[LLMResponsePayload]:
    """Apply even distribution limit for SCENARIO dataset across models and prompt IDs."""
    if len(payloads) <= limit:
        return payloads

    # Sort payloads deterministically first (by identifier, then by model_id for consistency)
    sorted_payloads = sorted(payloads, key=lambda p: (p.identifier, p.model_id))

    # Group by prompt_id and model_id
    groups: dict[tuple[str, str], list[LLMResponsePayload]] = {}
    for payload in sorted_payloads:
        # Extract prompt_id and model_id from identifier (format: "prompt_id:model_id")
        try:
            prompt_id, _ = _validate_identifier(
                payload.identifier, f"in payload for model {payload.model_id}"
            )
            model_id = payload.model_id
            key = (prompt_id, model_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(payload)
        except ValueError as e:
            # Log and re-raise to fail fast
            logging.error(f"Malformed identifier in distribution: {e}")
            raise

    # Get all unique prompt IDs and model IDs (sorted deterministically)

    # Only consider combinations that actually have records
    available_combinations = sorted(groups.keys())
    total_combinations = len(available_combinations)
    if total_combinations == 0:
        return sorted_payloads[:limit]

    # Calculate base records per combination
    base_records_per_combination = limit // total_combinations
    extra_records = limit % total_combinations

    selected_payloads = []

    # Distribute evenly across available combinations (deterministic order)
    for i, key in enumerate(available_combinations):
        # Calculate how many records to take from this combination
        records_to_take = base_records_per_combination
        if i < extra_records:
            records_to_take += 1

        # Take up to the calculated amount from this group
        take_count = min(records_to_take, len(groups[key]))
        selected_payloads.extend(groups[key][:take_count])

        # Stop if we've reached the limit
        if len(selected_payloads) >= limit:
            break

    return selected_payloads


def diff_payloads(
    client: DatabaseClient,
    dataset: Dataset,
    payloads: Sequence[LLMResponsePayload],
) -> SeedDiff:
    """Compare payloads with current DB state and return a diff."""
    existing_rows = db_queries.select_responses_for_dataset(client, dataset)
    existing_map = {
        (row["dataset"], row["identifier"], row["run_id"]): row for row in existing_rows
    }
    incoming_map = {payload.key(): payload for payload in payloads}

    new_keys = set(incoming_map.keys()) - set(existing_map.keys())
    deleted_keys = set(existing_map.keys()) - set(incoming_map.keys())
    candidate_keys = set(existing_map.keys()) & set(incoming_map.keys())

    new_payloads = [incoming_map[key] for key in sorted(new_keys)]
    deleted_rows = [existing_map[key] for key in sorted(deleted_keys)]

    changed: list[tuple[dict, LLMResponsePayload]] = []
    unchanged_count = 0

    for key in sorted(candidate_keys):
        existing = existing_map[key]
        incoming = incoming_map[key]
        if _is_payload_changed(existing, incoming):
            changed.append((existing, incoming))
        else:
            unchanged_count += 1

    return SeedDiff(
        new=new_payloads, changed=changed, deleted=deleted_rows, unchanged=unchanged_count
    )


def _is_payload_changed(existing: dict[str, Any], payload: LLMResponsePayload) -> bool:
    return any(
        [
            existing["prompt_title"] != payload.prompt_title,
            existing["prompt_body"] != payload.prompt_body,
            existing["model_response_text"] != payload.model_response_text,
            existing["model_id"] != payload.model_id,
            parse_metadata(existing.get("metadata_json")) != (payload.metadata or {}),
        ]
    )


def apply_diff(client: DatabaseClient, diff: SeedDiff) -> None:
    """Apply a previously computed diff to the database."""
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
            metadata_json=serialize_metadata(payload.metadata),
        )

    for existing, payload in diff.changed:
        db_queries.update_response_row(
            client,
            existing["id"],
            prompt_title=payload.prompt_title,
            prompt_body=payload.prompt_body,
            model_response_text=payload.model_response_text,
            model_id=payload.model_id,
            metadata_json=serialize_metadata(payload.metadata),
        )

    for row in diff.deleted:
        db_queries.delete_response_row(client, row["id"])


def sync_dataset(
    client: DatabaseClient,
    dataset: Dataset,
    payloads: Sequence[LLMResponsePayload],
    apply_changes: bool = False,
) -> SeedDiff:
    """Sync a dataset using provided payloads and optionally apply changes."""
    diff = diff_payloads(client, dataset, payloads)
    if apply_changes and diff.has_changes():
        apply_diff(client, diff)
    return diff


def payloads_from_run_file(
    run_path: Path,
    dataset: Dataset = Dataset.SCENARIO,
    limit: int | None = None,
    record_range: tuple[int, int | None] | None = None,
) -> tuple[list[LLMResponsePayload], list[str]]:
    """Create payloads from a run artifact, skipping malformed records."""
    if dataset is not Dataset.SCENARIO:
        raise SeedError("Run ingestion is currently only supported for the scenario dataset.")

    if not run_path.exists():
        raise SeedError(f"Run file does not exist: {run_path}")

    records = []
    with run_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SeedError(f"Invalid JSON on line {line_no} of {run_path}: {exc}") from exc
            records.append(record)

    if record_range:
        start, end = record_range
        records = records[start:end]

    run_identifier = run_path.parent.name
    payloads: list[LLMResponsePayload] = []
    skipped: list[str] = []
    for record in records:
        try:
            payloads.append(_payload_from_run_record(record, run_identifier, dataset))
        except SeedError as exc:
            prompt_id = record.get("prompt_id")
            skipped.append(f"prompt={prompt_id or '?'}: {exc}")
            continue

    # Apply limit after payloads are created and validated
    if limit is not None:
        payloads = _apply_even_distribution_limit(payloads, limit)

    return payloads, skipped


def _payload_from_run_record(
    record: dict,
    run_identifier: str,
    dataset: Dataset,
) -> LLMResponsePayload:
    prompt_id = record.get("prompt_id")
    perspective = record.get("perspective")
    model_id = record.get("model_id")
    response_text = record.get("response_text")
    messages = record.get("messages") or []
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    prompt_body = user_messages[-1].get("content") or record.get("prompt", "") if user_messages else record.get("prompt", "")

    if not all([prompt_id, perspective, model_id]):
        raise SeedError("missing prompt_id/perspective/model_id")
    if not response_text:
        raise SeedError("empty response text")

    identifier = f"{prompt_id}:{model_id}"
    metadata = {
        "factors": record.get("factors"),
        "provider": record.get("provider"),
        "perspective": perspective,
        "messages": messages,
        "prompt_id": prompt_id,
    }

    prompt_title = f"{dataset.value.capitalize()} prompt {prompt_id} ({perspective})"

    run_id = str(record.get("run_id") or run_identifier)

    return LLMResponsePayload(
        dataset=dataset,
        identifier=identifier,
        prompt_title=prompt_title,
        prompt_body=prompt_body,
        model_response_text=response_text,
        model_id=str(model_id),
        run_id=str(run_id),
        metadata=metadata,
    )


__all__ = [
    "LLMResponsePayload",
    "SeedDiff",
    "SeedError",
    "load_seed_payloads",
    "payloads_from_run_file",
    "diff_payloads",
    "sync_dataset",
    "apply_diff",
]
