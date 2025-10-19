"""Shared utility functions for the labeling platform."""

from __future__ import annotations

from datetime import datetime


def _isoformat(value: object | None) -> str:
    """Convert a value to ISO format string, handling None and datetime objects."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = ["_isoformat"]
