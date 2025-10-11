#!/usr/bin/env python3
"""CLI to score stored benchmark runs using the master LLM scorer."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional

    def load_dotenv(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
        return None


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent


def ensure_project_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


if TYPE_CHECKING:
    from src.benchmark.core.types import EvalRecord, ProviderMetadata
    from src.benchmark.providers.openrouter_client import OpenRouterClient
else:
    ChatMessage = Any  # runtime placeholder; real class imported lazily in functions

ensure_project_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score stored benchmark responses and write a new JSONL file"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input run JSONL path (defaults to latest outputs/runs/*/run.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSONL path (default: <input_stem>_scored.jsonl in same directory)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-score records even if they already contain a numeric score",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip generating summary CSV after scoring completes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without calling the scorer (no API requests)",
    )
    return parser.parse_args()


def _normalise_messages(messages: Sequence[dict]) -> list[ChatMessage]:
    from src.benchmark.core.types import ChatMessage

    normalised: list[ChatMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not role or content is None:
            continue
        normalised.append(
            ChatMessage(
                role=str(role),
                content=str(content),
            )
        )
    return normalised


def _iter_records(path: Path, logger: logging.Logger) -> Iterable[dict]:
    from src.benchmark.core.types import RunRecord

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield RunRecord.from_json_line(line)
            except Exception as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_number, exc)


@dataclass
class ScoringOutcome:
    index: int
    score: float | None
    latency_ms: int | None
    metadata: ProviderMetadata | None
    error: str | None


async def _score_record(
    record_index: int,
    record,
    scorer_model: str,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> ScoringOutcome:
    async with semaphore:
        from src.benchmark.core.types import ChatMessage
        from src.benchmark.scoring.master import score_with_master_model_async

        prompt_messages = _normalise_messages(record.messages or [])
        assistant_message = ChatMessage(role="assistant", content=record.response_text)
        augmented = [*prompt_messages, assistant_message]
        t_start = time.time()
        try:
            score, scorer_metadata = await score_with_master_model_async(
                augmented, model=scorer_model, client=client
            )
            latency_ms = int((time.time() - t_start) * 1000)
            return ScoringOutcome(
                index=record_index,
                score=float(score),
                latency_ms=latency_ms,
                metadata=scorer_metadata,
                error=None,
            )
        except Exception as exc:
            logger.error(
                "Scoring failed for model=%s prompt=%s: %s",
                record.model_id,
                record.prompt_id,
                exc,
            )
            latency_ms = int((time.time() - t_start) * 1000)
            return ScoringOutcome(
                index=record_index,
                score=None,
                latency_ms=latency_ms,
                metadata=None,
                error=str(exc),
            )


async def score_records(
    records,
    indices: list[int],
    scorer_model: str,
    dry_run: bool,
) -> dict[int, ScoringOutcome]:
    from src.benchmark.core.config import OpenRouterConfig, RunConfig
    from src.benchmark.core.logging import setup_logger
    from src.benchmark.providers.openrouter_client import OpenRouterClient

    logger = setup_logger(__name__)
    if dry_run:
        logger.info("Dry-run: would score %d records (no API calls).", len(indices))
        return {
            idx: ScoringOutcome(
                index=idx,
                score=None,
                latency_ms=None,
                metadata=None,
                error="dry-run: scoring skipped",
            )
            for idx in indices
        }

    cfg = OpenRouterConfig.from_env()
    run_cfg = RunConfig.from_env()
    client = OpenRouterClient(cfg, run_cfg)
    # Use a reasonable default concurrency for scoring (not per-model)
    semaphore = asyncio.Semaphore(3)

    try:
        tasks = [
            asyncio.create_task(
                _score_record(idx, records[idx], scorer_model, client, semaphore, logger)
            )
            for idx in indices
        ]

        outcomes: dict[int, ScoringOutcome] = {}
        for task in asyncio.as_completed(tasks):
            outcome = await task
            outcomes[outcome.index] = outcome
        return outcomes
    finally:
        await client.aclose()


def _resolve_latest_run_jsonl() -> Path | None:
    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        return None
    candidates: list[Path] = []
    for subdir in runs_dir.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / "run.jsonl"
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def show_eval_header(
    total_candidates: int,
    scorer_model: str,
    logger: logging.Logger,
    *,
    dry_run: bool,
) -> None:
    logger.info("Preparing to score %d records", total_candidates)
    logger.info("Scorer model: %s", scorer_model)
    if dry_run:
        logger.info("Dry-run mode: no API calls will be executed.")


def main() -> None:
    args = parse_args()
    load_dotenv()
    ensure_project_on_path()
    from src.benchmark.core.config import OpenRouterConfig
    from src.benchmark.core.logging import configure_logging, setup_logger
    from src.benchmark.core.types import EvalRecord
    from src.benchmark.reporting.aggregate import aggregate_run

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        latest = _resolve_latest_run_jsonl()
        if latest is None:
            print("No run JSONL found under outputs/runs. Specify --input explicitly.")
            sys.exit(1)
        print(f"Auto-detected latest run file: {latest}")
        input_path = latest

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_scored.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = output_path.with_suffix(".log")
    configure_logging(log_path)
    logger = setup_logger(__name__)

    logger.info("Evaluating run %s", input_path)
    logger.info("Output will be written to %s", output_path)

    run_records = list(_iter_records(input_path, logger))
    if not run_records:
        logger.error("No records found in %s", input_path)
        sys.exit(1)

    skip_reasons: dict[int, str] = {}
    candidate_indices: list[int] = []
    for idx, record in enumerate(run_records):
        if record.error:
            skip_reasons[idx] = f"run-error: {record.error}" if record.error else "run-error"
            continue
        if not record.response_text:
            skip_reasons[idx] = "missing-response"
            continue
        if record.messages is None:
            skip_reasons[idx] = "missing-messages"
            continue
        candidate_indices.append(idx)

    if not candidate_indices:
        logger.info("No records require scoring. Nothing to do.")
        eval_records = [
            EvalRecord.from_run_record(
                record,
                eval_timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                scorer_model=args.scorer_model,
                score=None,
                eval_latency_ms=None,
                provider_metadata=None,
                error=skip_reasons.get(idx, "not-scored"),
            )
            for idx, record in enumerate(run_records)
        ]
        with output_path.open("w", encoding="utf-8") as handle:
            for eval_record in eval_records:
                handle.write(eval_record.to_json_line() + "\n")
        logger.info("Wrote evaluation stubs to %s", output_path)
        return

    scorer_model = (
        OpenRouterConfig.from_env().scorer_model
        or OpenRouterConfig.from_env().model
        or OpenRouterConfig.from_env().default_test_model
    )
    if scorer_model is None:
        logger.error(
            "Unable to determine scorer model. Set OPENROUTER_SCORER_MODEL or OPENROUTER_MODEL."
        )
        sys.exit(1)

    show_eval_header(
        len(candidate_indices),
        scorer_model,
        logger,
        dry_run=args.dry_run,
    )

    t_start = time.time()
    scoring_outcomes = asyncio.run(
        score_records(run_records, candidate_indices, scorer_model, args.dry_run)
    )
    elapsed_s = time.time() - t_start
    completed = sum(1 for outcome in scoring_outcomes.values() if outcome.score is not None)
    logger.info(
        "Scored %d/%d candidate records in %.2fs",
        completed,
        len(candidate_indices),
        elapsed_s,
    )

    eval_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    eval_records: list[EvalRecord] = []
    total_eval_cost = 0.0
    total_eval_tokens = 0

    for idx, record in enumerate(run_records):
        outcome = scoring_outcomes.get(idx)
        if outcome is None:
            reason = skip_reasons.get(idx, "not-scored")
            eval_record = EvalRecord.from_run_record(
                record,
                eval_timestamp_iso=eval_timestamp,
                scorer_model=scorer_model if not args.dry_run else None,
                score=None,
                eval_latency_ms=None,
                provider_metadata=None,
                error=reason,
            )
        else:
            if outcome.metadata and outcome.metadata.cost_usd:
                total_eval_cost += float(outcome.metadata.cost_usd)
            if outcome.metadata and outcome.metadata.total_tokens:
                total_eval_tokens += int(outcome.metadata.total_tokens)
            eval_record = EvalRecord.from_run_record(
                record,
                eval_timestamp_iso=eval_timestamp,
                scorer_model=scorer_model if not args.dry_run else None,
                score=outcome.score,
                eval_latency_ms=outcome.latency_ms,
                provider_metadata=outcome.metadata,
                error=outcome.error,
            )
        eval_records.append(eval_record)

    with output_path.open("w", encoding="utf-8") as handle:
        for eval_record in eval_records:
            handle.write(eval_record.to_json_line() + "\n")
    logger.info("Wrote %d evaluation records to %s", len(eval_records), output_path)

    logger.info(
        "Evaluation tokens=%d cost=%s",
        total_eval_tokens,
        f"${total_eval_cost:.4f}" if total_eval_cost else "n/a",
    )

    if not args.no_aggregate:
        summary_csv = output_path.with_suffix(".csv")
        aggregate_run(output_path, summary_csv)
        logger.info("Wrote summary to %s", summary_csv)

    logger.info("Evaluation complete for %s", input_path)


if __name__ == "__main__":
    main()
