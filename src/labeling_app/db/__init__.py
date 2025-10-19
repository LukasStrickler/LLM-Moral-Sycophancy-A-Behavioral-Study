"""Convenience re-export of database helpers and query utilities."""

from . import queries
from .libsql import (
    DatabaseClient,
    QueryResult,
    client_scope,
    create_client,
    ensure_schema,
    parse_metadata,
    serialize_metadata,
)

__all__ = [
    "DatabaseClient",
    "QueryResult",
    "client_scope",
    "create_client",
    "ensure_schema",
    "parse_metadata",
    "serialize_metadata",
    "queries",
]
