#!/usr/bin/env python3
"""CLI for batch AI labeling of responses in the database."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import TypedDict

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

from src.benchmark.core.config import OpenRouterConfig, RunConfig
from src.benchmark.core.logging import configure_logging, setup_logger
from src.benchmark.core.models import load_models_config
from src.benchmark.core.rate_limit import compute_backoff
from src.benchmark.providers.openrouter_client import OpenRouterClient
from src.labeling_app.core.models import Dataset
from src.labeling_app.db.libsql import create_client
from src.labeling_app.db.queries import get_unlabeled_responses, insert_review
from src.labeling_app.llm.llm_scorer import score_response_async

logger = setup_logger("ai-label")


class LabelingStats(TypedDict):
    """Statistics for labeling run."""
    successful: int
    failed: int
    skipped: int
    cost: float


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AI labeling on responses in the database"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="aita", 
        choices=["aita", "scenario"],
        help="Dataset to label (default: aita)"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        default="data/models/llm_labeling_models.json", 
        help="Path to models JSON file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Run a single model ID (overrides models file)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        help="Maximum number of responses to label (default: all)"
    )
    parser.add_argument(
        "--apply", 
        action="store_true", 
        help="Actually insert scores to database (default: dry-run)"
    )
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=5, 
        help="Maximum concurrent requests (default: 5)"
    )
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3, 
        help="Maximum retry attempts per response (default: 3)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    return parser.parse_args()


def ensure_api_key(cfg: OpenRouterConfig) -> None:
    """Ensure OpenRouter API key is configured."""
    if cfg.api_key:
        return
    print("ERROR: OPENROUTER_API_KEY is not set. Add it to your environment or .env file.")
    sys.exit(1)


def resolve_models(args: argparse.Namespace, cfg: OpenRouterConfig) -> tuple[list[str], dict[str, dict]]:
    """Resolve which models to run."""
    try:
        if args.model:
            # Load model configs to get concurrency settings for the specified model
            _, model_configs = load_models_config(Path(args.models), cfg.default_test_model)
            return [args.model], model_configs
        return load_models_config(Path(args.models), cfg.default_test_model)
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"Please check your models configuration file: {args.models}")
        sys.exit(1)


async def score_with_retry(
    client: OpenRouterClient,
    response: dict,
    model_id: str,
    progress: str,
    max_attempts: int,
) -> tuple[float | None, bool, float]:
    """Score a single response with exponential backoff retry."""
    for attempt in range(max_attempts):
        try:
            score, metadata = await score_response_async(
                prompt_title=response['prompt_title'],
                prompt_body=response['prompt_body'],
                model_response_text=response['model_response_text'],
                scorer_model=model_id,
                client=client,
                grid_id=str(response['id']),
                task_id=f"{model_id}+{response['id']}",
                progress=progress,
            )
            cost = metadata.cost_usd or 0.0
            return score, True, cost
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = compute_backoff(attempt)
                logger.warning(f"Retry {attempt+1}/{max_attempts} after {wait:.1f}s: {e}")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Failed after {max_attempts} attempts: {e}")
                return None, False, 0.0


async def run_labeling_for_model(
    model_id: str,
    dataset: Dataset,
    limit: int | None,
    apply: bool,
    concurrency: int,
    max_retries: int,
    model_configs: dict[str, dict],
    openrouter_config: OpenRouterConfig,
    run_config: RunConfig,
) -> LabelingStats:
    """Run labeling for a single model."""
    logger.info(f"Starting labeling with model: {model_id}")
    
    # Setup OpenRouter client with model configs
    client = OpenRouterClient(openrouter_config, run_config, model_configs)
    
    # Get database client
    db_client = create_client()
    
    try:
        # Get unlabeled responses for this model
        reviewer_code = f"llm:{model_id}"
        unlabeled = get_unlabeled_responses(db_client, dataset, reviewer_code, limit)
        
        if not unlabeled:
            logger.info(f"No unlabeled responses found for {reviewer_code}")
            return {"successful": 0, "failed": 0, "skipped": 0, "cost": 0.0}
        
        logger.info(f"Found {len(unlabeled)} unlabeled responses for {reviewer_code}")
        logger.info(f"Response IDs to process: {[r['id'] for r in unlabeled[:10]]}..." if len(unlabeled) > 10 else f"Response IDs: {[r['id'] for r in unlabeled]}")
        
        # Process responses with concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        stats_lock = asyncio.Lock()
        successful = 0
        failed = 0
        skipped = 0
        total_cost = 0.0
        
        async def process_response(response: dict, idx: int) -> None:
            nonlocal successful, failed, skipped, total_cost
            async with semaphore:
                progress = f"{idx+1}/{len(unlabeled)}"
                
                if not apply:
                    # Dry run - just log what would be done
                    logger.info(f"[DRY RUN] Would score response {response['id']} with {model_id}")
                    async with stats_lock:
                        skipped += 1
                    return
                
                score, success, cost = await score_with_retry(
                    client, response, model_id, progress, max_retries
                )
                
                if success and score is not None:
                    # Insert the review into database
                    try:
                        inserted = insert_review(
                            db_client,
                            response['id'],
                            reviewer_code,
                            score,
                            f"AI-labeled by {model_id}"
                        )
                        if inserted:
                            async with stats_lock:
                                successful += 1
                                total_cost += cost
                            logger.info(f"Scored response {response['id']}: {score}")
                        else:
                            async with stats_lock:
                                skipped += 1
                            logger.info(f"Response {response['id']} already reviewed by {reviewer_code}")
                    except Exception as e:
                        logger.error(f"Failed to insert review for response {response['id']}: {e}")
                        async with stats_lock:
                            failed += 1
                else:
                    async with stats_lock:
                        failed += 1
        
        # Process all responses concurrently
        tasks = [
            process_response(response, idx) 
            for idx, response in enumerate(unlabeled)
        ]
        await asyncio.gather(*tasks)
        
        logger.info(f"Model {model_id} completed: {successful} successful, {failed} failed, {skipped} skipped")
        logger.info(f"Total cost for {model_id}: ${total_cost:.4f}")
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
    
    # Setup logging
    log_dir = Path("outputs/ai_labeling") / str(int(time.time()))
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "labeling.log")
    
    # Suppress noisy warnings unless verbose
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("aiohttp").setLevel(logging.ERROR)
    
    logger.info(f"Starting AI labeling run (apply={args.apply})")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Models file: {args.models}")
    logger.info(f"Concurrency: {args.concurrency}")
    if args.limit:
        logger.info(f"Limit: {args.limit}")
    
    # Setup configuration
    openrouter_config = OpenRouterConfig.from_env()
    run_config = RunConfig.from_env()
    
    # Ensure API key is configured
    ensure_api_key(openrouter_config)
    
    # Resolve models to run
    models, model_configs = resolve_models(args, openrouter_config)
    
    # Convert dataset string to enum
    dataset = Dataset(args.dataset)
    
    # Run labeling for each model
    total_stats = {"successful": 0, "failed": 0, "skipped": 0, "cost": 0.0}
    
    for model_id in models:
        logger.info(f"Processing model: {model_id}")
        
        stats = await run_labeling_for_model(
            model_id=model_id,
            dataset=dataset,
            limit=args.limit,
            apply=args.apply,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            model_configs=model_configs,
            openrouter_config=openrouter_config,
            run_config=run_config,
        )
        
        # Accumulate stats
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total successful: {total_stats['successful']}")
    print(f"Total failed: {total_stats['failed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total API cost: ${total_stats['cost']:.4f}")
    print(f"Logs saved to: {log_dir}")
    
    if args.apply:
        print(f"✅ Scores have been saved to the database")
    else:
        print(f"🔍 This was a dry run - no scores were saved")


if __name__ == "__main__":
    asyncio.run(main())
