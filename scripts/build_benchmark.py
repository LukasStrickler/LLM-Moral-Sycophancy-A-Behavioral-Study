#!/usr/bin/env python3
"""CLI to materialize the prompt grid into a JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional

    def load_dotenv(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
        return None


# Ensure project root is on sys.path so `src/...` imports work
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent


def ensure_project_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


if TYPE_CHECKING:
    pass

ensure_project_on_path()


DEFAULT_GRID_PATH = Path("outputs/raw/grid.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the prompt grid and write it to JSONL. "
            "If --limit is omitted, every combination is written."
        )
    )
    parser.add_argument(
        "--include-neutral",
        dest="include_neutral",
        action="store_true",
        default=True,
        help="Include neutral perspective prompts (default: True)",
    )
    parser.add_argument(
        "--no-include-neutral",
        dest="include_neutral",
        action="store_false",
        help="Exclude neutral perspective prompts",
    )
    parser.add_argument("--limit", type=int, help="Number of prompts to include (omit for ALL)")
    parser.add_argument(
        "--out", type=str, help="Output JSONL path (defaults to outputs/raw/grid.jsonl)"
    )
    return parser.parse_args()


def validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if limit <= 0:
        raise ValueError("--limit must be positive when provided")
    return limit


def select_prompts(include_neutral: bool, limit: int | None) -> Iterable:
    from src.benchmark.prompts.generator import generate_factor_grid

    factors = generate_factor_grid(include_neutral=include_neutral)
    total = len(factors)
    counts = Counter(f.perspective for f in factors)
    print(f"Grid generated: {total} factor combinations (include_neutral={include_neutral})")
    print(
        "Breakdown -> landlord: {landlord}, tenant: {tenant}, neutral: {neutral}".format(
            landlord=counts.get("landlord", 0),
            tenant=counts.get("tenant", 0),
            neutral=counts.get("neutral", 0),
        )
    )

    if limit is None or limit >= total:
        return factors
    return factors[:limit]


def write_grid(chats, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, prompt in enumerate(chats):
            record = {
                "prompt_id": f"{idx:03d}",  # Row number as prompt_id (e.g., "000", "001", "002")
                "factors": prompt.factors.to_payload(),
                "messages": [{"role": msg.role, "content": msg.content} for msg in prompt.messages],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chats)} chats to {out_path}")


def print_examples(chats, limit: int = 5) -> None:
    print("Sample user messages:")
    for idx, prompt in enumerate(chats[:limit], start=1):
        user_message = next((msg.content for msg in prompt.messages if msg.role == "user"), "")
        print(f"  {idx:02d}. {user_message}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    limit = validate_limit(args.limit)

    from src.benchmark.prompts.chat import build_chat_from_factors

    prompts = [build_chat_from_factors(f) for f in select_prompts(args.include_neutral, limit)]

    out_path = Path(args.out) if args.out else DEFAULT_GRID_PATH
    write_grid(prompts, out_path)
    # print_examples(prompts)


if __name__ == "__main__":
    main()
