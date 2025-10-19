"""Lightweight domain models and helpers for the labeling platform."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, TypedDict

from ..utils import _isoformat


def parse_metadata(raw_value: Any) -> dict[str, Any]:
    """Parse metadata payloads loaded from the database into dictionaries."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, bytes | bytearray):
        try:
            return json.loads(raw_value.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except Exception:
            return {}
    return {}


class Dataset(str, Enum):
    """Supported datasets for the labeling workflow."""

    AITA = "aita"
    SCENARIO = "scenario"

    @classmethod
    def from_value(cls, value: str) -> Dataset:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported dataset value: {value}") from exc


class LLMResponseRecord(TypedDict, total=False):
    """Row representation of the llm_responses table."""

    id: int
    dataset: str
    prompt_title: str
    prompt_body: str
    model_response_text: str
    identifier: str
    model_id: str
    run_id: str
    metadata_json: str | None
    created_at: Any


class ReviewRecord(TypedDict, total=False):
    """Row representation of the reviews table."""

    id: int
    llm_response_id: int
    reviewer_code: str
    score: float
    notes: str | None
    created_at: Any
    updated_at: Any


def response_to_assignment_payload(
    response: dict[str, Any],
    review_count: int,
) -> dict[str, Any]:
    """Mirror the previous ORM helper using dictionary-based rows."""
    metadata = parse_metadata(response.get("metadata_json"))
    dataset_value = response.get("dataset")
    if isinstance(dataset_value, Dataset):
        dataset_str = dataset_value.value
    elif isinstance(dataset_value, str):
        dataset_str = dataset_value
    else:
        dataset_str = Dataset(dataset_value).value
    return {
        "id": response.get("id"),
        "dataset": dataset_str,
        "identifier": response.get("identifier"),
        "prompt_title": response.get("prompt_title"),
        "prompt_body": response.get("prompt_body"),
        "model_response_text": response.get("model_response_text"),
        "model_id": response.get("model_id"),
        "run_id": response.get("run_id"),
        "metadata": metadata,
        "created_at": _isoformat(response.get("created_at")),
        "review_count": review_count,
    }


__all__ = [
    "Dataset",
    "LLMResponseRecord",
    "ReviewRecord",
    "response_to_assignment_payload",
    "parse_metadata",
]
