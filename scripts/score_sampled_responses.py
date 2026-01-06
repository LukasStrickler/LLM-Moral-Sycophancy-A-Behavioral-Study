#!/usr/bin/env python3
"""Score all sampled responses with 3 models in parallel.

This script scores all AITA responses in the database with 3 models:
- cohere/command-r-08-2024 (for positive category)
- openrouter/google/gemini-2.0-flash-lite-001 (for negative category)
- openrouter/tngtech/deepseek-r1t-chimera:free (for all/middle category)

All 3 models run in parallel, each with independent rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: object, **_kwargs: object) -> None:
        return None

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

# Ensure project is on path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(ROOT / "outputs" / "runs"))

from src.benchmark.core.config import ProviderConfig, RunConfig
from src.benchmark.core.logging import configure_logging, make_log_extra, setup_logger
from src.benchmark.core.models import load_models_config
from src.benchmark.core.retry import (
    RATE_LIMIT_GUARD,
    RATE_LIMIT_LOCK,
    RetryDecision,
    build_retry_decision,
    respect_rate_limit_window,
)
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

logger = setup_logger("scorer")
plan_logger = setup_logger("planer")


class LabelingStats(TypedDict):
    """Statistics for labeling run."""
    successful: int
    failed: int
    skipped: int
    cost: float


def _status_code_from_exception(exc: Exception) -> int | None:
    """Extract HTTP status code from exception if available."""
    if hasattr(exc, "status_code"):
        return int(exc.status_code)
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return int(exc.response.status_code)
    return None


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
            await respect_rate_limit_window(provider_name)
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
            decision = build_retry_decision(exc, attempt)
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
            logger.info(f"No unlabeled responses for {model_id}")
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
        task_counter = 0
        
        async def process_response(response: dict, idx: int) -> None:
            nonlocal successful, failed, skipped, total_cost, task_counter
            
            def get_current_progress() -> str:
                """Get current progress per model in X/Y format."""
                return f"{successful}/{total_responses}"
            
            async with semaphore:
                task_counter += 1
                task_num = task_counter
                
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
        
        # Return stats
        return {"successful": successful, "failed": failed, "skipped": skipped, "cost": total_cost}
        
    finally:
        await client.aclose()
        db_client.close()


async def main() -> None:
    """Main entry point."""
    # Load environment
    load_dotenv()
    
    # Setup logging
    log_dir = Path("outputs/ai_labeling") / str(int(time.time()))
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "scoring.log")
    
    logger.info("")
    logger.info("🚀 Starting parallel scoring run")
    logger.info(f"   Dataset: aita")
    logger.info(f"   Concurrency per model: 5")
    logger.info("")
    
    # Setup configuration
    provider_config = ProviderConfig.from_env()
    # Disable LiteLLM's automatic retries - we handle retries ourselves
    run_config = RunConfig(request_timeout_s=60, max_retries=0)
    
    # Models to score with
    models = [
        "cohere/command-r-08-2024",
        "openrouter/google/gemini-2.0-flash-lite-001",
        "openrouter/tngtech/deepseek-r1t-chimera:free",
    ]
    
    # Load model configs
    models_file = Path("data/models/llm_labeling_models.json")
    _, model_configs = load_models_config(models_file, provider_config.default_test_model)
    
    # Filter models based on API key availability
    available_models: list[str] = []
    skipped_models: list[str] = []
    for model_id in models:
        if _has_api_key_for_model(provider_config, model_id):
            available_models.append(model_id)
        else:
            skipped_models.append(model_id)
    
    if skipped_models:
        logger.warning(f"Skipping models without API keys: {', '.join(skipped_models)}")
    
    if not available_models:
        logger.error("No models available - check API keys")
        return
    
    dataset = Dataset.AITA
    
    # Prefetch reviewer stats
    db_client = create_client()
    try:
        reviewer_counts = get_llm_reviewer_counts(db_client, dataset)
        reviewer_targets = [f"llm:{model_id}" for model_id in available_models]
        unlabeled_counts = get_llm_unlabeled_counts(db_client, dataset, reviewer_targets)
    finally:
        db_client.close()
    
    # Show plan
    plan_entries: list[dict[str, object]] = []
    for idx, model_id in enumerate(available_models, 1):
        reviewer_code = f"llm:{model_id}"
        pending = unlabeled_counts.get(reviewer_code, 0)
        completed = reviewer_counts.get(reviewer_code, 0)
        todo = "OPEN" if pending > 0 else "DONE"
        plan_entries.append({
            "idx": idx,
            "model": model_id,
            "pending": pending,
            "completed": completed,
            "todo": todo,
        })
    
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
                    f"open={sum(1 for e in plan_entries if e['todo'] == 'OPEN')}",
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
                        f"pending={entry['pending']}",
                        f"completed={entry['completed']}",
                    ),
                ),
            )
    
    # Run all models in parallel
    total_stats = {"successful": 0, "failed": 0, "skipped": 0, "cost": 0.0}
    
    async def process_model(model_id: str) -> LabelingStats:
        """Process a single model and return its stats."""
        # Process all remaining unlabeled responses
        limit = None
        
        stats = await run_labeling_for_model(
            model_id=model_id,
            dataset=dataset,
            limit=limit,  # Process 250 responses
            apply=True,
            concurrency=5,
            max_retries=3,
            model_configs=model_configs,
            provider_config=provider_config,
            run_config=run_config,
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
    
    # Process all models in parallel using asyncio.gather
    logger.info("Starting parallel execution of all models...")
    all_stats = await asyncio.gather(*[process_model(model_id) for model_id in available_models])
    
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
    if total_stats['failed'] == 0:
        logger.info(f"✅ All scores have been saved to the database")
    else:
        logger.info(f"⚠️  Some scores failed - check logs for details")


if __name__ == "__main__":
    asyncio.run(main())





