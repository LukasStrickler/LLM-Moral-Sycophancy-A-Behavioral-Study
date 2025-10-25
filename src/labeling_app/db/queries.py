"""Centralized SQL helpers for the labeling application."""

from __future__ import annotations

from ..core.models import Dataset
from .libsql import DatabaseClient


# ---------------------------------------------------------------------------
# Assignment helpers
# ---------------------------------------------------------------------------
def get_reviewed_response_ids(
    client: DatabaseClient, dataset: Dataset, reviewer_code: str
) -> set[int]:
    result = client.execute(
        """
        SELECT reviews.llm_response_id
        FROM reviews
        JOIN llm_responses AS responses ON responses.id = reviews.llm_response_id
        WHERE reviews.reviewer_code = ? AND responses.dataset = ?
        """,
        [reviewer_code, dataset.value],
    )
    return {int(row[0]) for row in result.rows}


def get_response_dataset(client: DatabaseClient, response_id: int) -> Dataset:
    value = client.execute(
        "SELECT dataset FROM llm_responses WHERE id = ?",
        [response_id],
    ).first_value()
    if value is None:
        raise ValueError(f"LLM response {response_id} does not exist")
    return Dataset(str(value))


def count_responses(client: DatabaseClient, dataset: Dataset) -> int:
    return int(
        client.execute(
            "SELECT COUNT(*) FROM llm_responses WHERE dataset = ?",
            [dataset.value],
        ).first_value(0)
    )


def count_reviewer_completed(client: DatabaseClient, dataset: Dataset, reviewer_code: str) -> int:
    return int(
        client.execute(
            """
            SELECT COUNT(*)
            FROM reviews
            JOIN llm_responses AS responses ON responses.id = reviews.llm_response_id
            WHERE reviews.reviewer_code = ? AND responses.dataset = ?
            """,
            [reviewer_code, dataset.value],
        ).first_value(0)
    )


def coverage_distribution(client: DatabaseClient, dataset: Dataset) -> dict[int, int]:
    rows = client.execute(
        """
        SELECT
            review_count,
            COUNT(*) AS total
        FROM (
            SELECT
                responses.id,
                (
                    SELECT COUNT(*) FROM reviews WHERE reviews.llm_response_id = responses.id AND reviews.reviewer_code NOT LIKE 'llm:%'
                ) AS review_count
            FROM llm_responses AS responses
            WHERE responses.dataset = ?
        )
        GROUP BY review_count
        """,
        [dataset.value],
    ).rows
    return {int(row[0] or 0): int(row[1] or 0) for row in rows}


def next_response_candidate(
    client: DatabaseClient,
    dataset: Dataset,
    excluded_ids: set[int],
    reviewed_ids: set[int],
    reviewer_code: str,
) -> dict | None:
    clauses: list[str] = ["responses.dataset = ?"]
    params: list[object] = [dataset.value]

    if excluded_ids:
        excluded_list = sorted(excluded_ids)
        placeholders = ",".join("?" for _ in excluded_list)
        clauses.append(f"responses.id NOT IN ({placeholders})")
        params.extend(excluded_list)

    if reviewed_ids:
        reviewed_list = sorted(reviewed_ids)
        placeholders = ",".join("?" for _ in reviewed_list)
        clauses.append(f"responses.id NOT IN ({placeholders})")
        params.extend(reviewed_list)

    where_sql = " AND ".join(clauses)

    # First pass: prioritize items with <3 reviews (2→3, 1→2, 0→1)
    query_primary = f"""
        SELECT
            responses.id,
            responses.dataset,
            responses.identifier,
            responses.prompt_title,
            responses.prompt_body,
            responses.model_response_text,
            responses.model_id,
            responses.run_id,
            responses.metadata_json,
            responses.created_at,
            COALESCE(review_counts.review_count, 0) AS review_count
        FROM llm_responses AS responses
        LEFT JOIN (
            SELECT llm_response_id, COUNT(*) AS review_count
            FROM reviews
            WHERE reviewer_code NOT LIKE 'llm:%'
            GROUP BY llm_response_id
        ) AS review_counts ON review_counts.llm_response_id = responses.id
        WHERE {where_sql} AND COALESCE(review_counts.review_count, 0) < 3
        ORDER BY review_count DESC, responses.created_at ASC, responses.id ASC
        LIMIT 1
    """

    result = client.execute(query_primary, params)
    if result.rows:
        return result.to_dicts()[0]

    # Fallback pass: if no items with <3 reviews available, allow items with ≥3 reviews
    # Prioritize items with fewer reviews (3→4, 4→5) and deprioritize items reviewer has already seen
    # Use combined aggregation for better performance
    query_fallback = f"""
        SELECT
            responses.id,
            responses.dataset,
            responses.identifier,
            responses.prompt_title,
            responses.prompt_body,
            responses.model_response_text,
            responses.model_id,
            responses.run_id,
            responses.metadata_json,
            responses.created_at,
            COALESCE(counts.review_count, 0) AS review_count,
            COALESCE(counts.reviewer_review_count, 0) AS reviewer_review_count
        FROM llm_responses AS responses
        LEFT JOIN (
            SELECT 
                llm_response_id, 
                COUNT(*) AS review_count,
                SUM(CASE WHEN reviewer_code = ? THEN 1 ELSE 0 END) AS reviewer_review_count
            FROM reviews
            WHERE reviewer_code NOT LIKE 'llm:%'
            GROUP BY llm_response_id
        ) AS counts ON counts.llm_response_id = responses.id
        WHERE {where_sql}
        ORDER BY 
            reviewer_review_count ASC,     -- Items reviewer has seen fewer times first
            review_count ASC,              -- Items with fewer reviews first (3→4 before 4→5)
            responses.created_at ASC, 
            responses.id ASC
        LIMIT 1
    """

    # Add reviewer_code parameter for the reviewer-specific count
    fallback_params = params.copy()
    fallback_params.append(reviewer_code)

    result = client.execute(query_fallback, fallback_params)
    if not result.rows:
        return None
    return result.to_dicts()[0]


def find_review_id(client: DatabaseClient, response_id: int, reviewer_code: str) -> int | None:
    return client.execute(
        """
        SELECT id
        FROM reviews
        WHERE llm_response_id = ? AND reviewer_code = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [response_id, reviewer_code],
    ).first_value()


def insert_review(
    client: DatabaseClient,
    response_id: int,
    reviewer_code: str,
    score: float,
    notes: str | None,
) -> bool:
    """Insert review, returns True if inserted, False if already exists."""
    try:
        client.execute(
            """
            INSERT INTO reviews (llm_response_id, reviewer_code, score, notes)
            VALUES (?, ?, ?, ?)
            """,
            [response_id, reviewer_code, score, notes],
        )
        return True
    except Exception as e:
        # Check if it's a duplicate (UNIQUE constraint violation)
        if "UNIQUE constraint failed" in str(e):
            return False
        raise


def update_review(
    client: DatabaseClient,
    review_id: int,
    score: float,
    notes: str | None,
) -> None:
    client.execute(
        """
        UPDATE reviews
        SET score = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        [score, notes, review_id],
    )


def fetch_review_record(
    client: DatabaseClient, response_id: int, reviewer_code: str
) -> dict | None:
    rows = client.execute(
        """
        SELECT
            id,
            llm_response_id,
            reviewer_code,
            score,
            notes,
            created_at,
            updated_at
        FROM reviews
        WHERE llm_response_id = ? AND reviewer_code = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [response_id, reviewer_code],
    ).to_dicts()
    if not rows:
        return None
    return rows[0]


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------
def select_responses_for_dataset(client: DatabaseClient, dataset: Dataset) -> list[dict]:
    return client.execute(
        """
        SELECT
            id,
            dataset,
            identifier,
            prompt_title,
            prompt_body,
            model_response_text,
            model_id,
            run_id,
            metadata_json
        FROM llm_responses
        WHERE dataset = ?
        """,
        [dataset.value],
    ).to_dicts()


def insert_response_row(
    client: DatabaseClient,
    *,
    dataset: Dataset,
    identifier: str,
    prompt_title: str,
    prompt_body: str,
    model_response_text: str,
    model_id: str,
    run_id: str,
    metadata_json: str,
) -> None:
    client.execute(
        """
        INSERT INTO llm_responses (
            dataset,
            identifier,
            prompt_title,
            prompt_body,
            model_response_text,
            model_id,
            run_id,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            dataset.value,
            identifier,
            prompt_title,
            prompt_body,
            model_response_text,
            model_id,
            run_id,
            metadata_json,
        ],
    )


def update_response_row(
    client: DatabaseClient,
    response_id: int,
    *,
    prompt_title: str,
    prompt_body: str,
    model_response_text: str,
    model_id: str,
    metadata_json: str,
) -> None:
    client.execute(
        """
        UPDATE llm_responses
        SET
            prompt_title = ?,
            prompt_body = ?,
            model_response_text = ?,
            model_id = ?,
            metadata_json = ?
        WHERE id = ?
        """,
        [
            prompt_title,
            prompt_body,
            model_response_text,
            model_id,
            metadata_json,
            response_id,
        ],
    )


def delete_response_row(client: DatabaseClient, response_id: int) -> None:
    client.execute("DELETE FROM llm_responses WHERE id = ?", [response_id])


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def select_reviews_with_responses(client: DatabaseClient, dataset: Dataset) -> list[dict]:
    return client.execute(
        """
        SELECT
            responses.id AS llm_response_id,
            responses.dataset AS dataset,
            responses.identifier AS identifier,
            responses.prompt_title AS prompt_title,
            responses.prompt_body AS prompt_body,
            responses.model_response_text AS model_response_text,
            responses.model_id AS model_id,
            responses.run_id AS run_id,
            responses.metadata_json AS metadata_json,
            reviews.reviewer_code AS reviewer_code,
            reviews.score AS score,
            reviews.notes AS notes,
            reviews.created_at AS review_created_at,
            reviews.updated_at AS review_updated_at
        FROM reviews
        JOIN llm_responses AS responses ON responses.id = reviews.llm_response_id
        WHERE responses.dataset = ?
        ORDER BY reviews.created_at ASC
        """,
        [dataset.value],
    ).to_dicts()


def select_responses_for_export(client: DatabaseClient, dataset: Dataset) -> list[dict]:
    return client.execute(
        """
        SELECT
            id,
            dataset,
            identifier,
            prompt_title,
            prompt_body,
            model_response_text,
            model_id,
            run_id,
            metadata_json,
            created_at
        FROM llm_responses
        WHERE dataset = ?
        ORDER BY created_at ASC
        """,
        [dataset.value],
    ).to_dicts()


def select_reviews_for_export(client: DatabaseClient, dataset: Dataset) -> list[dict]:
    return client.execute(
        """
        SELECT
            reviews.llm_response_id AS llm_response_id,
            reviews.reviewer_code AS reviewer_code,
            reviews.score AS score,
            reviews.notes AS notes,
            reviews.created_at AS created_at,
            reviews.updated_at AS updated_at
        FROM reviews
        JOIN llm_responses AS responses ON responses.id = reviews.llm_response_id
        WHERE responses.dataset = ?
        ORDER BY reviews.created_at ASC
        """,
        [dataset.value],
    ).to_dicts()


def count_llm_reviews(client: DatabaseClient, dataset: Dataset, model_id: str) -> int:
    """Count reviews from a specific LLM model."""
    reviewer_code = f"llm:{model_id}"
    return int(
        client.execute(
            """
            SELECT COUNT(*)
            FROM reviews
            JOIN llm_responses AS responses ON responses.id = reviews.llm_response_id
            WHERE reviews.reviewer_code = ? AND responses.dataset = ?
            """,
            [reviewer_code, dataset.value],
        ).first_value(0)
    )


def get_unlabeled_responses(
    client: DatabaseClient, 
    dataset: Dataset, 
    reviewer_code: str,
    limit: int | None = None
) -> list[dict]:
    """Get responses not yet reviewed by specified reviewer."""
    query = """
        SELECT
            responses.id,
            responses.dataset,
            responses.identifier,
            responses.prompt_title,
            responses.prompt_body,
            responses.model_response_text,
            responses.model_id,
            responses.run_id,
            responses.metadata_json,
            responses.created_at
        FROM llm_responses AS responses
        LEFT JOIN reviews ON reviews.llm_response_id = responses.id AND reviews.reviewer_code = ?
        WHERE responses.dataset = ? AND reviews.id IS NULL
        ORDER BY responses.created_at ASC
    """
    
    params = [reviewer_code, dataset.value]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    
    return client.execute(query, params).to_dicts()


__all__ = [
    "count_llm_reviews",
    "count_responses",
    "count_reviewer_completed",
    "coverage_distribution",
    "delete_response_row",
    "fetch_review_record",
    "find_review_id",
    "get_response_dataset",
    "get_reviewed_response_ids",
    "get_unlabeled_responses",
    "insert_response_row",
    "insert_review",
    "next_response_candidate",
    "select_responses_for_dataset",
    "select_responses_for_export",
    "select_reviews_for_export",
    "select_reviews_with_responses",
    "update_response_row",
    "update_review",
]
