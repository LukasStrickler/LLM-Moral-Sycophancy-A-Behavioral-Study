"""Utilities for working with benchmark model lists."""

from __future__ import annotations

import json
from pathlib import Path

PREFERRED_TEST_MODEL = "openai/gpt-oss-20b:free"


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def load_models_list(path: Path, fallback: str) -> list[str]:
    """Load an ordered list of model ids from JSON.

    Falls back to ``fallback`` if the file is missing or empty. Ensures the
    preferred test model appears first when present.
    """

    if not path.exists():
        return [fallback]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    models: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict) and "id" in item:
                models.append(str(item["id"]))

    if not models:
        models = [fallback]

    ordered = _unique_preserve_order(models)
    if PREFERRED_TEST_MODEL in ordered:
        return [PREFERRED_TEST_MODEL] + [
            model for model in ordered if model != PREFERRED_TEST_MODEL
        ]
    return ordered
