"""Lightweight domain models and helpers for the labeling platform."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

from ..utils import _isoformat


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
    from ..db.libsql import parse_metadata  # Import locally to avoid circular dependency

    metadata = parse_metadata(response.get("metadata_json"))
    dataset_value = response.get("dataset")
    if dataset_value is None:
        dataset_str = None
    elif isinstance(dataset_value, Dataset):
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
]
