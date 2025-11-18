#!/usr/bin/env python3
"""CLI to build Dear Abby prompt grid from sampled questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: object, **_kwargs: object) -> None:
        return None


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

DEFAULT_GRID_PATH = Path("outputs/raw/dearabby_grid.jsonl")
DEFAULT_SAMPLED_PATH = Path("data/humanLabel/raw/dearabby_sampled_questions.jsonl")


def ensure_project_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Dear Abby prompt grid from sampled questions"
    )
    parser.add_argument(
        "--sampled",
        type=str,
        default=str(DEFAULT_SAMPLED_PATH),
        help="Path to sampled questions JSONL",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output grid JSONL path (defaults to outputs/raw/dearabby_grid.jsonl)",
    )
    return parser.parse_args()


def load_sampled_questions(path: Path) -> list[dict]:
    """Load sampled questions from JSONL."""
    if not path.exists():
        print(f"ERROR: Sampled questions file not found: {path}")
        print("Please run the transformation notebook first to create sampled questions.")
        sys.exit(1)
    
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    return questions


def build_grid(questions: list[dict], out_path: Path) -> None:
    """Build grid.jsonl from sampled questions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    width = len(str(len(questions)))  # Dynamic width for prompt_id padding
    
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, question in enumerate(questions, start=1):
            question_id = question.get("question_id", f"{idx:03d}")
            # IMPORTANT: Only use question_text (from question_only column), NOT title
            # The title is stored in metadata only, never sent to the LLM
            question_text = question.get("question_text", "")
            
            # Create messages (same format as rent scenario)
            # Only question_text is sent to LLM, title is excluded
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"I need advice. {question_text}"},
            ]
            
            # Store metadata in factors dict (even though not using Factors class)
            # Title is stored here for reference but NOT sent to LLM
            factors = {
                "question_id": question_id,
                "year": question.get("year"),
                "date": question.get("date"),
                "title": question.get("title"),  # Metadata only, not in messages
                "letterId": question.get("letterId"),
            }
            
            record = {
                "prompt_id": f"{idx:0{width}d}",
                "factors": factors,  # Metadata stored here
                "messages": messages,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(questions)} prompts to {out_path}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    ensure_project_on_path()
    
    sampled_path = Path(args.sampled)
    out_path = Path(args.out) if args.out else DEFAULT_GRID_PATH
    
    questions = load_sampled_questions(sampled_path)
    print(f"Loaded {len(questions)} sampled questions")
    
    build_grid(questions, out_path)
    print(f"Grid saved to {out_path}")


if __name__ == "__main__":
    main()

