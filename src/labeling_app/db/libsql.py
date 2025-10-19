"""Shared utilities for working with libsql (or local SQLite) clients."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import libsql_client

from ..settings import AppSettings, get_settings


@dataclass(frozen=True)
class QueryResult:
    """Normalized representation of a query result."""

    columns: list[str]
    rows: list[Sequence[Any]]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return rows as dictionaries keyed by column name."""
        return [dict(zip(self.columns, row, strict=False)) for row in self.rows]

    def first_value(self, default: Any = None) -> Any:
        """Convenience accessor for the first scalar value."""
        if not self.rows or not self.rows[0]:
            return default
        return self.rows[0][0]


class DatabaseClient:
    """Thin wrapper providing a uniform interface over libsql and sqlite3."""

    def __init__(self, driver: str, connection: Any, auth_token: str | None = None) -> None:
        self._driver = driver
        self._connection = connection
        self._auth_token = auth_token

    @property
    def driver(self) -> str:
        """Return the driver backing this client (libsql or sqlite)."""
        return self._driver

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> QueryResult:
        """Execute a single SQL statement and return a normalized result."""
        params = list(parameters or [])
        if self._driver == "libsql":
            result = self._connection.execute(sql, params)
            columns = list(getattr(result, "columns", []) or _description_to_columns(result))
            rows = list(result.rows)
            return QueryResult(columns=columns, rows=rows)

        cursor = self._connection.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description] if cursor.description else []
        self._connection.commit()
        return QueryResult(columns=columns, rows=rows)

    def executemany(self, sql: str, param_sets: Iterable[Sequence[Any]]) -> None:
        """Execute an INSERT/UPDATE/DELETE statement against multiple parameter sets.

        Transaction Behavior:
        - libsql: Each execute() call is auto-committed individually
        - sqlite: cursor.executemany() is implicitly transactional (all-or-nothing)
        - For libsql, partial failures may leave some records committed
        """
        if self._driver == "libsql":
            for params in param_sets:
                self._connection.execute(sql, list(params))
            return

        cursor = self._connection.cursor()
        cursor.executemany(sql, [list(params) for params in param_sets])
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        if self._driver == "libsql":
            # The libsql client exposes a `close` method for HTTP connections.
            close = getattr(self._connection, "close", None)
            if callable(close):
                close()
            return

        self._connection.close()


def _description_to_columns(result: Any) -> list[str]:
    """Fallback column extraction for result sets exposing DB-API descriptions."""
    description = getattr(result, "description", None)
    if not description:
        return []
    return [column[0] for column in description]


def _normalize_url(raw_url: str) -> tuple[str, str]:
    """Return (driver, normalized_url_or_path) for the configured database."""
    trimmed = raw_url.strip()
    if trimmed.startswith("libsql://"):
        return "libsql", trimmed.replace("libsql://", "https://", 1)
    if trimmed.startswith("https://") or trimmed.startswith("http://"):
        return "libsql", trimmed
    if trimmed.startswith("sqlite://"):
        return "sqlite", _sqlite_path(trimmed)
    raise RuntimeError(
        "Unsupported database URL. Use libsql:// for Turso or sqlite:/// for local development."
    )


def _sqlite_path(raw_url: str) -> str:
    """Convert a sqlite:// URL into a filesystem path (or :memory:)."""
    if raw_url.startswith("sqlite:///"):
        path = raw_url[len("sqlite:///") :]
    else:
        path = raw_url[len("sqlite://") :]

    if not path:
        return ":memory:"

    # Preserve absolute paths (which begin with /) and decode URI escapes.
    normalized = unquote(path)
    if normalized.startswith("file:"):
        normalized = normalized[len("file:") :]
    return normalized


def create_client(settings: AppSettings | None = None) -> DatabaseClient:
    """Instantiate a database client based on application settings.

    Thread Safety:
    - Each call creates a new DatabaseClient instance with its own connection
    - SQLite connections use check_same_thread=False for cross-thread compatibility
    - Callers must not share DatabaseClient instances across threads
    - Use client_scope() context manager for automatic cleanup
    """
    settings = settings or get_settings()
    database_url = settings.require_database_url()
    driver, normalized = _normalize_url(database_url)

    if driver == "libsql":
        return DatabaseClient(
            driver="libsql",
            connection=libsql_client.create_client_sync(
                url=normalized, auth_token=settings.auth_token
            ),
            auth_token=settings.auth_token,
        )

    # sqlite fallback for local development/testing
    connection = sqlite3.connect(
        normalized,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,  # Safe because each DatabaseClient has its own connection
    )
    connection.execute("PRAGMA foreign_keys = ON;")
    return DatabaseClient(driver="sqlite", connection=connection)


@contextmanager
def client_scope(settings: AppSettings | None = None) -> Iterator[DatabaseClient]:
    """Context manager that yields a database client and ensures cleanup."""
    client = create_client(settings)
    try:
        yield client
    finally:
        client.close()


def ensure_schema(client: DatabaseClient) -> None:
    """Create the required tables, indexes, and triggers if they do not exist."""
    client.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL CHECK (dataset IN ('aita', 'scenario')),
            prompt_title TEXT NOT NULL,
            prompt_body TEXT NOT NULL,
            model_response_text TEXT NOT NULL,
            identifier TEXT NOT NULL,
            model_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            metadata_json TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dataset, identifier, run_id)
        )
        """
    )
    client.execute(
        "CREATE INDEX IF NOT EXISTS ix_llm_responses_dataset_identifier "
        "ON llm_responses(dataset, identifier)"
    )
    client.execute(
        "CREATE INDEX IF NOT EXISTS ix_llm_responses_dataset_run ON llm_responses(dataset, run_id)"
    )

    client.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            llm_response_id INTEGER NOT NULL,
            reviewer_code TEXT NOT NULL,
            score REAL NOT NULL CHECK (score >= -1.0 AND score <= 1.0),
            notes TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (llm_response_id) REFERENCES llm_responses(id) ON DELETE CASCADE
        )
        """
    )
    client.execute(
        "CREATE INDEX IF NOT EXISTS ix_reviews_reviewer_dataset ON reviews(reviewer_code)"
    )
    client.execute("DROP INDEX IF EXISTS ux_reviews_response_reviewer")
    client.execute(
        """
        CREATE TRIGGER IF NOT EXISTS update_reviews_updated_at
            AFTER UPDATE ON reviews
            FOR EACH ROW
            BEGIN
                UPDATE reviews SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """
    )


def serialize_metadata(metadata: dict[str, Any] | None) -> str:
    """Serialize metadata dictionaries to JSON text for storage."""
    if not metadata:
        return json.dumps({})
    return json.dumps(metadata)


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
        except json.JSONDecodeError:
            return {}
    return dict(raw_value)


__all__ = [
    "DatabaseClient",
    "QueryResult",
    "client_scope",
    "create_client",
    "ensure_schema",
    "parse_metadata",
    "serialize_metadata",
]
