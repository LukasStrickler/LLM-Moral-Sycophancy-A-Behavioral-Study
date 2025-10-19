"""Assignment logic for balanced reviewer workloads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..db import DatabaseClient
from ..db import queries as db_queries
from .models import Dataset, response_to_assignment_payload


@dataclass(frozen=True)
class ProgressSnapshot:
    """Represents reviewer progress within a dataset."""

    dataset: Dataset
    total_responses: int
    reviewer_completed: int
    coverage_by_count: dict[int, int]

    @property
    def remaining(self) -> int:
        return max(self.total_responses - self.reviewer_completed, 0)


class AssignmentService:
    """High-level API for retrieving assignments and persisting reviews."""

    def __init__(self, client: DatabaseClient) -> None:
        self.client = client

    def _reviewed_ids(self, dataset: Dataset, reviewer_code: str) -> set[int]:
        return db_queries.get_reviewed_response_ids(self.client, dataset, reviewer_code)

    def get_progress(self, dataset: Dataset, reviewer_code: str) -> ProgressSnapshot:
        total = db_queries.count_responses(self.client, dataset)
        reviewer_completed = db_queries.count_reviewer_completed(
            self.client, dataset, reviewer_code
        )
        coverage_map = db_queries.coverage_distribution(self.client, dataset)

        return ProgressSnapshot(
            dataset=dataset,
            total_responses=total,
            reviewer_completed=reviewer_completed,
            coverage_by_count=coverage_map,
        )

    def next_assignment(
        self,
        dataset: Dataset,
        reviewer_code: str,
        exclude_ids: Iterable[int] | None = None,
    ) -> dict | None:
        """Return the next LLM response (as a dict) following coverage priorities."""
        excluded: set[int] = set(exclude_ids or [])
        total_responses = db_queries.count_responses(self.client, dataset)
        reviewed_ids = self._reviewed_ids(dataset, reviewer_code)
        allow_repeats = total_responses > 0 and len(reviewed_ids) >= total_responses

        candidate = db_queries.next_response_candidate(
            self.client,
            dataset,
            excluded_ids=excluded,
            reviewed_ids=set() if allow_repeats else reviewed_ids,
            reviewer_code=reviewer_code,
        )

        if not candidate:
            return None

        return response_to_assignment_payload(
            candidate,
            review_count=int(candidate.get("review_count", 0)),
        )

    def submit_review(
        self,
        response_id: int,
        reviewer_code: str,
        score: float,
        notes: str | None = None,
    ) -> dict:
        """Create a new review for the given response and return the stored row."""
        dataset = db_queries.get_response_dataset(self.client, response_id)
        reviewer_history = self._reviewed_ids(dataset, reviewer_code)
        total_responses = db_queries.count_responses(self.client, dataset)
        has_reviewed = response_id in reviewer_history
        allow_repeats = total_responses > 0 and len(reviewer_history) >= total_responses

        if has_reviewed and not allow_repeats:
            raise ValueError(
                "Duplicate review denied until the reviewer completes every prompt in the dataset."
            )

        db_queries.insert_review(
            self.client,
            response_id=response_id,
            reviewer_code=reviewer_code,
            score=score,
            notes=notes,
        )

        review_row = db_queries.fetch_review_record(self.client, response_id, reviewer_code)
        if review_row is None:
            # Fallback if fetch fails
            review_id = db_queries.find_review_id(self.client, response_id, reviewer_code)
            if review_id is None:
                raise ValueError(
                    f"Could not find review ID for response_id={response_id}, "
                    f"reviewer_code={reviewer_code}"
                )
            review_row = {
                "id": review_id,
                "llm_response_id": response_id,
                "reviewer_code": reviewer_code,
                "score": score,
                "notes": notes,
            }
        return review_row


__all__ = ["AssignmentService", "ProgressSnapshot"]
