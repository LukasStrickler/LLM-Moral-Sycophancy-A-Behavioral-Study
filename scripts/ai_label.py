#!/usr/bin/env python3
"""CLI for batch AI labeling of responses in the database."""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional

    def load_dotenv(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
        return None


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

# Ensure project is on path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.core.config import ProviderConfig, RunConfig
from src.benchmark.core.logging import configure_logging, make_log_extra, setup_logger
from src.benchmark.core.models import load_models_config
from src.benchmark.providers.litellm_provider import LiteLLMProvider, _has_api_key_for_model
from src.labeling_app.core.models import Dataset
from src.labeling_app.db.libsql import create_client
from src.labeling_app.db.queries import (
    get_llm_reviewer_counts,
    get_llm_unlabeled_counts,
    get_unlabeled_responses,
    insert_review,
)
from src.labeling_app.llm.llm_scorer import score_response_async

logger = setup_logger("labeler")
plan_logger = setup_logger("planer")


class LabelingStats(TypedDict):
    """Statistics for labeling run."""

    successful: int
    failed: int
    skipped: int
    cost: float


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run AI labeling on responses in the database")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="aita", 
        choices=["aita", "scenario"],
        help="Dataset to label (default: aita)",
    )
    parser.add_argument(
        "--models", 
        type=str, 
        default="data/models/llm_labeling_models.json", 
        help="Path to models JSON file",
    )
    parser.add_argument("--model", type=str, help="Run a single model ID (overrides models file)")
    parser.add_argument(
        "--limit", type=int, help="Maximum number of responses to label (default: all)"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually insert scores to database (default: dry-run)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Maximum concurrent requests (default: 5)"
    )
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3, 
        help="Maximum retry attempts per response (default: 3)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def ensure_api_key(cfg: ProviderConfig) -> None:
    """Ensure at least one provider API key is configured."""
    if cfg.has_any_api_key():
        return
    logger.error("No provider API key is set. Set at least one of:")
    logger.error("  GOOGLE_AI_API_KEY, GROQ_API_KEY, HUGGINGFACE_API_KEY, CEREBRAS_API_KEY,")
    logger.error("  MISTRAL_API_KEY, COHERE_API_KEY, or OPENROUTER_API_KEY")
    logger.error("Add it to your environment or .env file.")
    sys.exit(1)


def resolve_models(
    args: argparse.Namespace, cfg: ProviderConfig
) -> tuple[list[str], dict[str, dict]]:
    """Resolve which models to run."""
    try:
        if args.model:
            # Load model configs to get concurrency settings for the specified model
            _, model_configs = load_models_config(Path(args.models), cfg.default_test_model)
            return [args.model], model_configs
        return load_models_config(Path(args.models), cfg.default_test_model)
    except ValueError as e:
        logger.error(f"Failed to resolve models: {e}")
        logger.error(f"Please check your models configuration file: {args.models}")
        sys.exit(1)




class _RetryDecision(NamedTuple):
    """Represents how the caller should respond to a failed scoring request."""

    should_retry: bool
    wait_seconds: float
    label: str
    summary: str


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_BASE_WAIT_SECONDS = 1.5
_MAX_WAIT_SECONDS = 65.0


def _seconds_until_next_minute(now: float | None = None) -> float:
    """Return the seconds until the next minute boundary (minimum of one second)."""

    current = now or time.time()
    remainder = current % 60.0
    wait = 60.0 - remainder if remainder else 60.0
    return max(1.0, wait)


def _retry_hint_seconds(exc: Exception) -> float | None:
    """Extract an explicit retry-after hint from headers or error payload."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    candidates: list[str] = []

    if headers:
        try:
            for key in headers.keys():  # type: ignore[attr-defined]
                value = headers.get(key)
                if not value:
                    continue
                lower = key.lower()
                if lower in {
                    "retry-after", 
                    "x-ratelimit-reset", 
                    "x-ratelimit-reset-requests", 
                    "x-ratelimit-reset-tokens",
                    # Cerebras-specific headers
                    "x-ratelimit-reset-requests-day",
                    "x-ratelimit-reset-tokens-minute",
                }:
                    candidates.append(str(value))
        except Exception:
            pass

    for attr in ("retry_after", "retry_after_ms"):
        value = getattr(exc, attr, None)
        if value:
            candidates.append(str(value))

    message = str(exc)
    match = re.search(r"retry in\s+([\d.]+)s", message, re.IGNORECASE)
    if match:
        candidates.append(match.group(1))
    delay_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', message)
    if delay_match:
        candidates.append(delay_match.group(1))

    for raw in candidates:
        raw = raw.strip().lower()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            minutes_match = re.fullmatch(r"(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?", raw)
            if minutes_match:
                minutes = float(minutes_match.group("m") or 0)
                seconds = float(minutes_match.group("s") or 0)
                value = minutes * 60 + seconds
            else:
                try:
                    from email.utils import parsedate_to_datetime

                    dt = parsedate_to_datetime(raw)
                    if dt is None:
                        continue
                    value = max(0.0, dt.timestamp() - time.time())
                except Exception:
                    continue
        if value > 1e8:  # treat as timestamp
            if value > 1e12:
                value /= 1000.0
            value = max(0.0, value - time.time())
        if value >= 0:
            return value
    return None


def _status_code_from_exception(exc: Exception) -> int | None:
    """Best-effort retrieval of an HTTP status code from LiteLLM exceptions."""

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_code = getattr(response, "status_code", None)
    return int(response_code) if isinstance(response_code, int) else None


def _is_daily_quota_error(exc: Exception) -> bool:
    """Return True when the error clearly signals a daily quota exhaustion."""

    message = str(exc).lower()
    quota_tokens = ("requests per day", "per-day quota", "quotaid", "daily quota", "rpd")
    return any(token in message for token in quota_tokens)


def _classify_exception(exc: Exception) -> tuple[str | None, str]:
    """Determine the retry label for an exception and capture a short summary."""

    import litellm

    status_code = _status_code_from_exception(exc)
    summary = str(exc).split("\n", 1)[0] if exc else ""

    if _is_daily_quota_error(exc):
        return "quota", summary or "daily quota"

    if isinstance(exc, litellm.exceptions.RateLimitError) or status_code == 429:
        return "limited", summary or "rate limit"

    if isinstance(exc, litellm.exceptions.BadRequestError):
        return "limited", summary or "bad request"

    transient_types = (
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.Timeout,
    )
    if isinstance(exc, transient_types):
        return "retry", summary or type(exc).__name__

    if status_code is not None:
        try:
            if litellm._should_retry(status_code):
                return "retry", summary or f"http {status_code}"
        except Exception:
            if status_code in _RETRYABLE_STATUS_CODES:
                return "retry", summary or f"http {status_code}"

    lowered = summary.lower()
    for token in ("temporary", "try again", "timeout", "connection reset", "overloaded"):
        if token in lowered:
            return "retry", summary or token

    return None, summary or type(exc).__name__


def _build_retry_decision(exc: Exception, attempt: int) -> _RetryDecision:
    """Create a retry decision for the given exception and attempt index (1-indexed)."""

    label, summary = _classify_exception(exc)
    if label == "quota":
        return _RetryDecision(False, 0.0, label, summary)
    if label is None:
        return _RetryDecision(False, 0.0, "nonretry", summary)

    wait_seconds = _retry_hint_seconds(exc)
    if wait_seconds is None:
        if label == "limited":
            wait_seconds = max(5.0, _seconds_until_next_minute() + 1.0)
        else:
            wait_seconds = _BASE_WAIT_SECONDS * (2 ** (attempt - 1))
    wait_seconds = min(_MAX_WAIT_SECONDS, wait_seconds)
    wait_seconds = max(1.0, wait_seconds + random.uniform(0.25, 0.75))

    return _RetryDecision(True, wait_seconds, label, summary)



RATE_LIMIT_GUARD: dict[str, float] = {}
RATE_LIMIT_LOCK = asyncio.Lock()


async def _respect_rate_limit_window(provider_name: str | None) -> None:
    if not provider_name:
        return
    while True:
        async with RATE_LIMIT_LOCK:
            available = RATE_LIMIT_GUARD.get(provider_name, 0.0)
        delay = available - time.time()
        if delay <= 0:
            return
        await asyncio.sleep(delay)

async def score_with_retry(
    client: LiteLLMProvider,
    response: dict,
    model_id: str,
    get_progress: Callable[[], str],
    max_attempts: int,
    task_num: int,
) -> tuple[float | None, bool, float]:
    """Score a single response with bounded retries tuned for RPM rate limits."""
    provider_name = client.get_provider_for_model(model_id)

    for attempt in range(1, max_attempts + 1):
        try:
            await _respect_rate_limit_window(provider_name)
            score, metadata = await score_response_async(
                prompt_title=response["prompt_title"],
                prompt_body=response["prompt_body"],
                model_response_text=response["model_response_text"],
                scorer_model=model_id,
                client=client,
                grid_id=str(response["id"]),
                task_id=str(task_num),
                progress=get_progress(),
            )
            cost = metadata.cost_usd or 0.0
            return score, True, cost
        except Exception as exc:
            decision = _build_retry_decision(exc, attempt)
            progress = get_progress()
            status_code = _status_code_from_exception(exc)
            details = [f"attempt={attempt}/{max_attempts}"]
            if status_code is not None:
                details.append(f"status={status_code}")
            details.append(f"error={type(exc).__name__}")

            if attempt >= max_attempts or not decision.should_retry:
                logger.error(
                    "Scoring failed",
                    extra=make_log_extra(
                        model=model_id,
                        grid=str(response["id"]),
                        task=str(task_num),
                        progress=progress,
                        tag="error",
                        status="giveup" if decision.should_retry else decision.label,
                        details=tuple(details),
                    ),
                )
                logger.debug("Scoring exception details", exc_info=exc)
                return None, False, 0.0

            wait_seconds = decision.wait_seconds
            if wait_seconds:
                if provider_name:
                    async with RATE_LIMIT_LOCK:
                        current = RATE_LIMIT_GUARD.get(provider_name, 0.0)
                        target = max(time.time() + wait_seconds, current)
                        RATE_LIMIT_GUARD[provider_name] = target
                details.append(f"wait={wait_seconds:.0f}s")
            logger.warning(
                "Retrying scorer",
                extra=make_log_extra(
                    model=model_id,
                    grid=str(response["id"]),
                    task=str(task_num),
                    progress=progress,
                    tag="retry",
                    status=decision.label,
                    details=tuple(details),
                ),
            )
            if wait_seconds:
                await asyncio.sleep(wait_seconds)
    return None, False, 0.0


async def run_labeling_for_model(
    model_id: str,
    dataset: Dataset,
    limit: int | None,
    apply: bool,
    concurrency: int,
    max_retries: int,
    model_configs: dict[str, dict],
    provider_config: ProviderConfig,
    run_config: RunConfig,
    get_global_task: Callable[[], Awaitable[int]] | None = None,
) -> LabelingStats:
    """Run labeling for a single model."""
    
    # Setup LiteLLM provider with model configs
    client = LiteLLMProvider(provider_config, run_config, model_configs)
    
    # Get database client
    db_client = create_client()
    
    try:
        # Get unlabeled responses for this model
        reviewer_code = f"llm:{model_id}"
        unlabeled = get_unlabeled_responses(db_client, dataset, reviewer_code, limit)
        
        if not unlabeled:
            return {"successful": 0, "failed": 0, "skipped": 0, "cost": 0.0}
        
        if len(unlabeled) <= 10:
            logger.debug(f"Model {model_id} pending response IDs: {[r['id'] for r in unlabeled]}")
        
        # Process responses with concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        stats_lock = asyncio.Lock()
        total_responses = len(unlabeled)
        successful = 0
        failed = 0
        skipped = 0
        total_cost = 0.0
        
        async def process_response(response: dict, idx: int) -> None:
            nonlocal successful, failed, skipped, total_cost
            
            def get_current_progress() -> str:
                """Get current progress per model in X/Y format."""
                # Progress is successful count for this model / total responses for this model
                # Reading int is atomic in Python, safe to read without lock
                return f"{successful}/{total_responses}"
            
            async with semaphore:
                # Get global task number across all models
                if get_global_task:
                    task_num = await get_global_task()
                else:
                    task_num = idx + 1  # Fallback to local index
                
                if not apply:
                    # Dry run - just log what would be done
                    logger.info(
                        "[DRY RUN] Would score response",
                        extra=make_log_extra(
                            model=model_id,
                            grid=str(response["id"]),
                            task=str(task_num),
                            progress=get_current_progress(),
                            tag="info",
                            status="dryrun",
                            details=(f"response_id={response['id']}",),
                        ),
                    )
                    async with stats_lock:
                        skipped += 1
                    return
                
                score, success, cost = await score_with_retry(
                    client, response, model_id, get_current_progress, max_retries, task_num
                )
                
                if success and score is not None:
                    # Insert the review into database
                    try:
                        inserted = insert_review(
                            db_client,
                            response["id"],
                            reviewer_code,
                            score,
                            f"AI-labeled by {model_id}",
                        )
                        if inserted:
                            async with stats_lock:
                                successful += 1
                                total_cost += cost
                            # Get fresh progress count after update
                            current_progress = get_current_progress()
                            logger.info(
                                "Scored response",
                                extra=make_log_extra(
                                    model=model_id,
                                    grid=str(response["id"]),
                                    task=str(task_num),
                                    progress=current_progress,
                                    tag="info",
                                    status="scored",
                                    details=(f"score={score:.2f}", f"response_id={response['id']}"),
                                ),
                            )
                        else:
                            async with stats_lock:
                                skipped += 1
                            logger.debug(
                                f"Response {response['id']} already reviewed by {reviewer_code}"
                            )
                    except Exception as e:
                        current_progress = get_current_progress()
                        logger.error(
                            "Failed to insert review",
                            extra=make_log_extra(
                                model=model_id,
                                grid=str(response["id"]),
                                task=str(task_num),
                                progress=current_progress,
                                tag="error",
                                status="db-fail",
                                details=(f"error={str(e)}", f"response_id={response['id']}"),
                            ),
                        )
                        async with stats_lock:
                            failed += 1
                else:
                    async with stats_lock:
                        failed += 1
        
        # Process all responses concurrently
        tasks = [process_response(response, idx) for idx, response in enumerate(unlabeled)]
        await asyncio.gather(*tasks)
        
        # Return stats - logging will be done by the caller (process_model wrapper)
        return {"successful": successful, "failed": failed, "skipped": skipped, "cost": total_cost}
        
    finally:
        await client.aclose()
        db_client.close()


async def main() -> None:
    """Main entry point."""
    # Load environment
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Set defaults: all models, no limits, aita dataset, apply enabled
    args.dataset = args.dataset or "aita"
    args.apply = True  # Enable by default
    args.limit = None  # No limits
    args.model = None  # All models
    
    # Setup logging
    log_dir = Path("outputs/ai_labeling") / str(int(time.time()))
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "labeling.log")
    
    # Suppress noisy warnings unless verbose
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("aiohttp").setLevel(logging.ERROR)
    
    logger.info("")
    logger.info(f"🚀 Starting AI labeling run ({'APPLY' if args.apply else 'DRY RUN'})")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Concurrency: {args.concurrency}")
    if args.limit:
        logger.info(f"   Limit: {args.limit}")
    logger.info("")
    
    # Setup configuration
    provider_config = ProviderConfig.from_env()
    # Disable LiteLLM's automatic retries - we handle retries ourselves using retry-after headers
    # which is more effective than exponential backoff
    run_config = RunConfig(request_timeout_s=60, max_retries=0)
    
    # Ensure API key is configured
    ensure_api_key(provider_config)
    
    # Resolve models to run
    all_models, model_configs = resolve_models(args, provider_config)

    dataset = Dataset(args.dataset)

    # Prefetch reviewer stats in a single DB round-trip
    db_client = create_client()
    try:
        reviewer_counts = get_llm_reviewer_counts(db_client, dataset)
        reviewer_targets = [f"llm:{model_id}" for model_id in all_models]
        unlabeled_counts = get_llm_unlabeled_counts(db_client, dataset, reviewer_targets)
    finally:
        db_client.close()

    # Filter models based on API key availability
    models: list[str] = []
    skipped_models: list[str] = []
    for model_id in all_models:
        if _has_api_key_for_model(provider_config, model_id):
            models.append(model_id)
        else:
            skipped_models.append(model_id)

    plan_entries: list[dict[str, object]] = []
    models_to_process: list[str] = []
    if models:
        for idx, model_id in enumerate(models, 1):
            reviewer_code = f"llm:{model_id}"
            pending = unlabeled_counts.get(reviewer_code, 0)
            completed = reviewer_counts.get(reviewer_code, 0)
            todo = "OPEN" if pending > 0 else "DONE"
            plan_entries.append(
                {
                    "idx": idx,
                    "model": model_id,
                    "pending": pending,
                    "completed": completed,
                    "todo": todo,
                }
            )
            if todo == "OPEN":
                models_to_process.append(model_id)

    if skipped_models:
        plan_logger.warning(
            "",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="warn",
                status="no-key",
                details=(f"count={len(skipped_models)}", ", ".join(skipped_models)),
            ),
        )
    if plan_entries:
        plan_logger.info(
            "",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="plan",
                details=(
                    f"total={len(plan_entries)}",
                    f"open={len(models_to_process)}",
                    f"done={len(plan_entries) - len(models_to_process)}",
                ),
            ),
        )
        for entry in plan_entries:
            plan_logger.info(
                "",
                extra=make_log_extra(
                    model=str(entry["model"]),
                    grid=None,
                    task=None,
                    progress=None,
                    tag="info",
                    status=str(entry["todo"]),
                    details=(
                        f"idx={entry['idx']}",
                        f"todo={entry['todo']}",
                        f"pending={entry['pending']}",
                        f"completed={entry['completed']}",
                    ),
                ),
            )
    else:
        plan_logger.info(
            "",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="plan",
                details=("no-models",),
            ),
        )
    
    if not models_to_process:
        logger.info(
            "No pending responses to label",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="plan",
                details=("all-complete",),
            ),
        )
        return
    
    # Run labeling for all models in parallel
    total_stats = {"successful": 0, "failed": 0, "skipped": 0, "cost": 0.0}
    
    # Global task counter across all models
    global_task_counter = 0
    global_task_lock = asyncio.Lock()
    
    async def get_next_global_task() -> int:
        """Get next global task number."""
        nonlocal global_task_counter
        async with global_task_lock:
            global_task_counter += 1
            return global_task_counter
    
    async def process_model(model_id: str) -> LabelingStats:
        """Process a single model and return its stats."""
        stats = await run_labeling_for_model(
            model_id=model_id,
            dataset=dataset,
            limit=args.limit,
            apply=args.apply,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            model_configs=model_configs,
            provider_config=provider_config,
            run_config=run_config,
            get_global_task=get_next_global_task,
        )
        
        # Log per-model summary
        status = "✅" if stats['failed'] == 0 else "⚠️"
        logger.info(
            f"{status} Model completed: {stats['successful']} successful, {stats['failed']} failed, {stats['skipped']} skipped",
            extra=make_log_extra(
                model=model_id,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="done",
                details=(
                    f"successful={stats['successful']}",
                    f"failed={stats['failed']}",
                    f"skipped={stats['skipped']}",
                    f"cost=${stats['cost']:.4f}",
                ),
            ),
        )
        
        return stats
    
    # Process all models in parallel
    all_stats = await asyncio.gather(*[process_model(model_id) for model_id in models_to_process])
    
    # Accumulate stats from all models
    for stats in all_stats:
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Log final summary
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info(f"📈 FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"   ✅ Successful: {total_stats['successful']}")
    logger.info(f"   ❌ Failed:     {total_stats['failed']}")
    logger.info(f"   ⏭️  Skipped:    {total_stats['skipped']}")
    logger.info(f"   💰 Cost:        ${total_stats['cost']:.4f}")
    logger.info(f"   📁 Logs:        {log_dir}")
    logger.info("")
    if not args.apply:
        logger.info(f"🔍 This was a dry run - no scores were saved")
    else:
        if total_stats['failed'] == 0:
            logger.info(f"✅ All scores have been saved to the database")
        else:
            logger.info(f"⚠️  Some scores failed - check logs for details")


if __name__ == "__main__":
    asyncio.run(main())
