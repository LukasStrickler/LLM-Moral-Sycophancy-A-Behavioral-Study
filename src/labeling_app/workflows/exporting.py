"""Export workflows for reviewer data and evaluation bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.models import Dataset, response_to_assignment_payload
from ..db import DatabaseClient
from ..db import queries as db_queries
from ..db.libsql import parse_metadata
from ..settings import AppSettings, get_settings
from ..utils import _isoformat


@dataclass
class ExportResult:
    """Simple value object describing an export operation."""

    path: Path
    record_count: int
    sample: dict[str, Any] | None = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_reviews(
    client: DatabaseClient,
    dataset: Dataset,
    settings: AppSettings | None = None,
    output_path: Path | None = None,
) -> ExportResult:
    """Export raw review records joined with their prompts."""
    settings = settings or get_settings()
    if output_path is None:
        output_path = settings.paths.reviews / f"{dataset.value}_reviews.jsonl"
    _ensure_parent(output_path)

    count = 0
    sample_record: dict[str, Any] | None = None
    with output_path.open("w", encoding="utf-8") as handle:
        for row in db_queries.select_reviews_with_responses(client, dataset):
            record = {
                "dataset": row["dataset"],
                "llm_response_id": row["llm_response_id"],
                "identifier": row["identifier"],
                "prompt_title": row["prompt_title"],
                "prompt_body": row["prompt_body"],
                "model_response_text": row["model_response_text"],
                "model_id": row["model_id"],
                "run_id": row["run_id"],
                "reviewer_code": row["reviewer_code"],
                "score": row["score"],
                "notes": row["notes"],
                "review_created_at": _isoformat(row["review_created_at"]),
                "review_updated_at": _isoformat(row["review_updated_at"]),
                "metadata": parse_metadata(row.get("metadata_json")),
            }
            handle.write(json.dumps(record))
            handle.write("\n")
            count += 1
            if sample_record is None:
                sample_record = record

    return ExportResult(path=output_path, record_count=count, sample=sample_record)


def export_evaluation_bundle(
    client: DatabaseClient,
    dataset: Dataset,
    settings: AppSettings | None = None,
    output_path: Path | None = None,
) -> ExportResult:
    """Export per-response bundles including all associated reviews."""
    settings = settings or get_settings()
    if output_path is None:
        output_path = settings.paths.evals / f"{dataset.value}_llm_eval.jsonl"
    _ensure_parent(output_path)

    responses = db_queries.select_responses_for_export(client, dataset)
    reviews_by_response: dict[int, list[dict]] = {}
    for review in db_queries.select_reviews_for_export(client, dataset):
        reviews_by_response.setdefault(review["llm_response_id"], []).append(review)

    records_written = 0
    sample_record: dict[str, Any] | None = None
    with output_path.open("w", encoding="utf-8") as handle:
        for response in responses:
            review_items = reviews_by_response.get(response["id"], [])
            record = response_to_assignment_payload(response, review_count=len(review_items))
            record["reviews"] = [
                {
                    "reviewer_code": review["reviewer_code"],
                    "score": review["score"],
                    "notes": review["notes"],
                    "created_at": _isoformat(review["created_at"]),
                    "updated_at": _isoformat(review["updated_at"]),
                }
                for review in review_items
            ]
            handle.write(json.dumps(record))
            handle.write("\n")
            records_written += 1
            if sample_record is None:
                sample_record = record

    return ExportResult(path=output_path, record_count=records_written, sample=sample_record)


__all__ = ["ExportResult", "export_reviews", "export_evaluation_bundle"]
