"""Main runner orchestration for paid benchmark."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logging import make_log_extra, setup_logger
from ..core.retry import build_retry_decision, classify_exception

import csv

from .budget import BudgetTracker
from .client import OpenRouterClient
from .config import PaidRunnerConfig
from .output import CSVWriter
from .state import RunState, StateManager

logger = setup_logger("paid-runner")


@dataclass
class ModelResult:
    """Result from a single model call."""

    model_id: str
    success: bool
    response_text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float = 0.0  # Always a float (0.0 for free models)
    latency_ms: int | None = None
    error: str | None = None
    retry_count: int = 0
    timestamp: str = ""


def _check_budget_before_request(
    budget_tracker: BudgetTracker,
    estimated_cost: float | None,
    model_id: str,
    prompt_id: str,
) -> tuple[bool, str | None]:
    """Check if we can afford a request before making it.

    Args:
        budget_tracker: Budget tracker
        estimated_cost: Estimated cost (if known, None otherwise)
        model_id: Model ID for logging
        prompt_id: Prompt ID for logging

    Returns:
        Tuple of (can_proceed, error_message)
    """
    # Strict check: budget already exceeded
    if budget_tracker.exceeded:
        error = "Budget limit already exceeded"
        logger.warning(
            "",
            extra=make_log_extra(
                model=model_id,
                grid=prompt_id,
                tag="budget",
                status="exceeded",
                details=(error,),
            ),
        )
        return False, error

    # If we have an estimated cost, check if we can afford it
    if estimated_cost is not None and estimated_cost > 0:
        if not budget_tracker.can_afford(estimated_cost):
            error = f"Estimated cost ${estimated_cost:.4f} would exceed budget"
            logger.warning(
                "",
                extra=make_log_extra(
                    model=model_id,
                    grid=prompt_id,
                    tag="budget",
                    status="would-exceed",
                    details=(error,),
                ),
            )
            return False, error

    return True, None


def _validate_cost_after_response(
    budget_tracker: BudgetTracker,
    cost: float,
    model_id: str,
    prompt_id: str,
) -> tuple[bool, str | None]:
    """Validate cost after receiving response and add to budget if valid.

    Args:
        budget_tracker: Budget tracker
        cost: Actual cost from response (0.0 for free models)
        model_id: Model ID for logging
        prompt_id: Prompt ID for logging

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Free models have cost 0.0, skip budget check
    if cost == 0.0:
        return True, None

    # Strict check: can we afford this cost?
    if not budget_tracker.can_afford(cost):
        error = f"Cost ${cost:.4f} would exceed budget limit ${budget_tracker.limit:.2f}"
        logger.warning(
            "",
            extra=make_log_extra(
                model=model_id,
                grid=prompt_id,
                tag="budget",
                status="would-exceed",
                details=(error,),
            ),
        )
        return False, error

    # Add cost to tracker (will raise if would exceed)
    try:
        budget_tracker.add_cost(cost)
    except ValueError as e:
        # Budget exceeded when adding cost
        return False, str(e)

    return True, None


async def run_model_with_retry(
    client: OpenRouterClient,
    model_id: str,
    messages: list[dict[str, str]],
    max_retries: int,
    budget_tracker: BudgetTracker,
    prompt_id: str,
) -> ModelResult:
    """Run a single model with independent retry logic.
    
    Each model retries independently - if one model fails, only that model retries.
    Other models continue normally and don't wait.

    Args:
        client: OpenRouter client
        model_id: Model ID to use
        messages: Chat messages
        max_retries: Maximum number of retries
        budget_tracker: Budget tracker
        prompt_id: Prompt identifier for logging

    Returns:
        ModelResult with success status and data
    """
    result = ModelResult(model_id=model_id, success=False)
    attempt = 0
    last_exc: Exception | None = None

    while attempt < max_retries:
        attempt += 1
        result.retry_count = attempt - 1

        try:
            # STRICT: Check budget before making request
            # We don't have estimated cost, so just check if already exceeded
            can_proceed, budget_error = _check_budget_before_request(
                budget_tracker, None, model_id, prompt_id
            )
            if not can_proceed:
                result.error = budget_error
                break

            # Make API call
            t0 = time.time()
            response = await client.chat_completion(model=model_id, messages=messages)
            latency_ms = int((time.time() - t0) * 1000)

            # Extract response data
            result.response_text = client.extract_text(response)
            input_tokens, output_tokens, total_tokens = client.extract_tokens(response)
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.total_tokens = total_tokens
            result.cost_usd = client.extract_cost(response)
            result.latency_ms = latency_ms
            result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Validate that we actually got a response with content
            if not result.response_text or len(result.response_text.strip()) == 0:
                result.error = "Empty response: API returned no content"
                logger.warning(
                    "",
                    extra=make_log_extra(
                        model=model_id,
                        grid=prompt_id,
                        tag="error",
                        status="empty-response",
                        details=("API returned empty response content",),
                    ),
                )
                # Retry if we haven't exhausted retries
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Brief wait before retry
                    continue
                break

            # STRICT: Validate cost after response and add to budget
            is_valid, cost_error = _validate_cost_after_response(
                budget_tracker, result.cost_usd, model_id, prompt_id
            )
            if not is_valid:
                result.error = cost_error
                break

            # Success!
            result.success = True
            logger.info(
                "",
                extra=make_log_extra(
                    model=model_id,
                    grid=prompt_id,
                    tag="success",
                    status="completed",
                    details=(
                        f"cost=${result.cost_usd:.4f}",
                        f"tokens={result.total_tokens}" if result.total_tokens else "",
                    ),
                ),
            )
            break

        except Exception as exc:
            last_exc = exc
            error_summary = str(exc).split("\n", 1)[0]
            
            # Check if this is a credit error - don't retry these
            is_credit_error = (
                "credit" in error_summary.lower() or
                "insufficient" in error_summary.lower() or
                "balance" in error_summary.lower()
            )
            
            if is_credit_error:
                # Credit errors are fatal - don't retry
                logger.error(
                    "",
                    extra=make_log_extra(
                        model=model_id,
                        grid=prompt_id,
                        tag="error",
                        status="credit-error",
                        details=(f"Fatal credit error: {error_summary}",),
                    ),
                )
                result.error = f"Credit error: {error_summary}"
                break
            
            decision = build_retry_decision(exc, attempt)

            # Log the error
            logger.warning(
                "",
                extra=make_log_extra(
                    model=model_id,
                    grid=prompt_id,
                    tag="retry" if decision.should_retry else "error",
                    status=decision.label or "error",
                    details=(
                        f"attempt={attempt}/{max_retries}",
                        f"error={error_summary[:50]}",
                    ),
                ),
            )

            # Check if we should give up
            if not decision.should_retry or attempt >= max_retries:
                result.error = error_summary
                break

            # Wait before retry (only this model waits, others continue)
            if decision.wait_seconds > 0:
                await asyncio.sleep(decision.wait_seconds)

            continue

    if not result.success and last_exc:
        result.error = result.error or str(last_exc)

    return result


async def process_prompt_synchronized(
    client: OpenRouterClient,
    prompt: dict[str, Any],
    models: list[str],
    config: PaidRunnerConfig,
    budget_tracker: BudgetTracker,
    csv_writer: CSVWriter,
    state: RunState,
    state_manager: StateManager,
) -> dict[str, ModelResult]:
    """Process a single prompt through all models with independent retries.
    
    Each model retries independently - if one fails, only that model retries.
    Results are written to CSV and state is saved immediately after each model completes.

    Args:
        client: OpenRouter client
        prompt: Prompt dictionary with 'identifier' and 'prompt_body'
        models: List of model IDs
        config: Runner configuration
        budget_tracker: Budget tracker
        csv_writer: CSV writer for immediate result writing
        state: Run state for immediate updates
        state_manager: State manager for immediate saving

    Returns:
        Dictionary mapping model_id -> ModelResult
    """
    prompt_id = prompt.get("identifier", "unknown")
    prompt_body = prompt.get("prompt_body", "")

    # Build messages for chat completion
    messages = [{"role": "user", "content": prompt_body}]

    logger.info(
        "",
        extra=make_log_extra(
            model="all",
            grid=prompt_id,
            tag="start",
            status="processing",
            details=(f"models={len(models)}",),
        ),
    )

    # Create a callback to write results immediately after each model completes
    async def run_model_and_write_immediately(model_id: str) -> ModelResult:
        """Run model and write result immediately. Each model retries independently."""
        result = await run_model_with_retry(
            client=client,
            model_id=model_id,
            messages=messages,
            max_retries=config.max_retries,
            budget_tracker=budget_tracker,
            prompt_id=prompt_id,
        )
        
        # Only write successful results to CSV
        # Double-check that we actually have response content before writing
        if result.success and result.response_text and len(result.response_text.strip()) > 0:
            csv_writer.write_result(
                prompt_identifier=prompt_id,
                prompt_body=prompt_body,
                model_id=model_id,
                response_text=result.response_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
                timestamp=result.timestamp,
                error=None,  # No error for successful results
                retry_count=result.retry_count,
            )
            # Update state immediately (just track cost)
            state.add_model_cost(model_id, result.cost_usd)
            # Save state immediately after each successful model
            state_manager.save_state(state)
        elif result.success and (not result.response_text or len(result.response_text.strip()) == 0):
            # Log warning if we marked as successful but have no content
            logger.warning(
                "",
                extra=make_log_extra(
                    model=model_id,
                    grid=prompt_id,
                    tag="error",
                    status="empty-response",
                    details=("Skipping write: marked successful but response is empty",),
                ),
            )
        
        return result

    # Create tasks for all models
    tasks = [
        run_model_and_write_immediately(model_id)
        for model_id in models
    ]

    # Run all models concurrently
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert to dictionary and handle exceptions
    results: dict[str, ModelResult] = {}
    for model_id, result in zip(models, results_list):
        if isinstance(result, Exception):
            error_result = ModelResult(
                model_id=model_id,
                success=False,
                error=str(result),
            )
            results[model_id] = error_result
            # Write error result immediately
            csv_writer.write_result(
                prompt_identifier=prompt_id,
                prompt_body=prompt_body,
                model_id=model_id,
                error=str(result),
            )
        else:
            results[model_id] = result

    # Check if all succeeded
    all_succeeded = all(r.success for r in results.values())
    if all_succeeded:
        logger.info(
            "",
            extra=make_log_extra(
                model="all",
                grid=prompt_id,
                tag="finish",
                status="completed",
                details=(f"all_models_succeeded",),
            ),
        )
    else:
        failed_models = [mid for mid, r in results.items() if not r.success]
        logger.warning(
            "",
            extra=make_log_extra(
                model="all",
                grid=prompt_id,
                tag="finish",
                status="partial",
                details=(f"failed_models={','.join(failed_models)}",),
            ),
        )

    return results


def _write_prompt_results(
    csv_writer: CSVWriter,
    prompt_id: str,
    prompt_body: str,
    results: dict[str, ModelResult],
) -> None:
    """Write all model results for a prompt to CSV.

    Args:
        csv_writer: CSV writer instance
        prompt_id: Prompt identifier
        prompt_body: Prompt body text
        results: Dictionary of model_id -> ModelResult
    """
    for model_id, result in results.items():
        csv_writer.write_result(
            prompt_identifier=prompt_id,
            prompt_body=prompt_body,
            model_id=model_id,
            response_text=result.response_text if result.success else "",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            timestamp=result.timestamp,
            error=result.error,
            retry_count=result.retry_count,
        )


def _load_completed_from_csv(csv_path: Path) -> set[tuple[str, str]]:
    """Load completed (prompt_id, model_id) pairs from CSV.
    
    Args:
        csv_path: Path to CSV results file
        
    Returns:
        Set of (prompt_id, model_id) tuples that have been completed
    """
    completed = set()
    
    if not csv_path.exists():
        return completed
    
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_id = row.get("prompt_identifier", "").strip()
                model_id = row.get("model_id", "").strip()
                error = row.get("error", "").strip()
                response_text = row.get("response_text", "").strip()
                
                # Only count as completed if:
                # 1. No error
                # 2. Response text exists and is non-empty
                if prompt_id and model_id and not error and response_text and len(response_text) > 0:
                    completed.add((prompt_id, model_id))
    except Exception:
        # If CSV is corrupted or can't be read, return empty set
        # This allows recovery by treating everything as incomplete
        pass
    
    return completed


def _is_prompt_complete(csv_path: Path, prompt_id: str, models: list[str]) -> bool:
    """Check if all models have completed for a prompt by checking CSV.

    Args:
        csv_path: Path to CSV results file
        prompt_id: Prompt identifier
        models: List of model IDs that must complete

    Returns:
        True if all models completed (no errors and has response text), False otherwise
    """
    if not csv_path.exists():
        return False
    
    completed_models = set()
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("prompt_identifier", "").strip() == prompt_id:
                    model_id = row.get("model_id", "").strip()
                    error = row.get("error", "").strip()
                    response_text = row.get("response_text", "").strip()
                    # Only count as completed if no error and has response text
                    if model_id and not error and response_text and len(response_text) > 0:
                        completed_models.add(model_id)
    except Exception:
        return False
    
    return all(model_id in completed_models for model_id in models)


def _count_completed_prompts(csv_path: Path, models: list[str]) -> int:
    """Count how many prompts are fully completed (all models done).

    Args:
        csv_path: Path to CSV results file
        models: List of all model IDs that must complete

    Returns:
        Number of prompts that have all models completed
    """
    if not csv_path.exists():
        return 0
    
    # Track which prompts have which models completed
    prompt_completions: dict[str, set[str]] = {}
    
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_id = row.get("prompt_identifier", "").strip()
                model_id = row.get("model_id", "").strip()
                error = row.get("error", "").strip()
                response_text = row.get("response_text", "").strip()
                
                # Only count if no error and has response text
                if prompt_id and model_id and not error and response_text and len(response_text) > 0:
                    if prompt_id not in prompt_completions:
                        prompt_completions[prompt_id] = set()
                    prompt_completions[prompt_id].add(model_id)
    except Exception:
        return 0
    
    # Count prompts where all models are completed
    count = 0
    for prompt_id, completed_models in prompt_completions.items():
        # Verify all models completed and have non-empty response text
        if all(model_id in completed_models for model_id in models):
            count += 1
    
    return count


def run_paid_benchmark(config: PaidRunnerConfig) -> dict[str, Any]:
    """Run paid benchmark with budget tracking and resumable state.

    Args:
        config: Runner configuration

    Returns:
        Summary dictionary with:
        - total_cost: Total cost spent
        - prompts_processed: Number of prompts processed
        - models_completed: Total number of model completions
        - budget_exceeded: Whether budget was exceeded
    """
    # Initialize state manager
    state_file = config.output_dir / "state.json"
    state_manager = StateManager(state_file)

    # Load or create state
    state = state_manager.load_state()
    if state is None:
        state = RunState.new()
        logger.info("Starting new run")
    else:
        logger.info(
            f"Resuming run from {state.last_updated}. "
            f"Total cost so far: ${state.total_cost:.4f}"
        )

    # Initialize budget tracker with current spending
    budget_tracker = BudgetTracker(limit=config.budget_limit, current=state.total_cost)

    # Initialize OpenRouter client
    client = OpenRouterClient(api_key=config.api_key, timeout=config.request_timeout)

    # Initialize CSV writer
    csv_path = config.output_dir / "results.csv"
    csv_append = state.total_cost > 0  # Append if resuming
    csv_writer = CSVWriter(csv_path, append=csv_append)
    csv_writer.open()

    try:
        # Get prompts to process with missing models (checks CSV for completion)
        csv_path = config.output_dir / "results.csv"
        prompt_data = list(state_manager.get_next_prompt(config.seed_file, csv_path, state, config.models))
        total_prompts = len(prompt_data)

        # Count total missing model runs
        total_missing_runs = sum(len(missing) for _, missing in prompt_data)

        # Count existing completed prompts (all models done) from CSV
        existing_completed = _count_completed_prompts(csv_path, config.models)
        target_completed = (
            existing_completed + config.prompt_limit
            if config.prompt_limit is not None
            else None
        )

        logger.info(
            f"Processing {total_prompts} prompts with {len(config.models)} models. "
            f"Total missing model runs: {total_missing_runs}. "
            f"Budget: ${config.budget_limit:.2f}, Remaining: ${budget_tracker.remaining:.2f}"
        )
        if config.prompt_limit is not None:
            logger.info(
                f"Prompt limit: {config.prompt_limit} new prompts. "
                f"Existing completed: {existing_completed}, "
                f"Target total: {target_completed}"
            )

        prompts_processed = 0
        models_completed = 0
        budget_exceeded = False

        # Create a single event loop to reuse for all prompts
        # This prevents "Event loop is closed" errors
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Process each prompt
        for prompt_idx, (prompt, missing_models) in enumerate(prompt_data, 1):
            prompt_id = prompt.get("identifier", f"prompt_{prompt_idx}")
            prompt_body = prompt.get("prompt_body", "")

            # STRICT: Check budget before starting prompt
            if budget_tracker.exceeded:
                logger.warning(
                    f"Budget limit reached. Stopping at prompt {prompt_idx}/{total_prompts}. "
                    f"Current: ${budget_tracker.current:.4f}, Limit: ${budget_tracker.limit:.2f}"
                )
                budget_exceeded = True
                break

            # Log which models we're running for this prompt
            if len(missing_models) < len(config.models):
                logger.info(
                    f"Processing prompt {prompt_idx}/{total_prompts}: {prompt_id} "
                    f"(running {len(missing_models)} missing models: {', '.join(missing_models)})"
                )
            else:
                logger.info(
                    f"Processing prompt {prompt_idx}/{total_prompts}: {prompt_id} "
                    f"(running all {len(missing_models)} models)"
                )

            # Process prompt through missing models only
            # Results are written immediately as each model completes
            try:
                # Use the existing event loop instead of creating a new one
                results = loop.run_until_complete(
                    process_prompt_synchronized(
                        client=client,
                        prompt=prompt,
                        models=missing_models,  # Only run missing models
                        config=config,
                        budget_tracker=budget_tracker,
                        csv_writer=csv_writer,
                        state=state,
                        state_manager=state_manager,
                    )
                )
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_id}: {e}", exc_info=True)
                # Write error results for missing models only
                for model_id in missing_models:
                    csv_writer.write_result(
                        prompt_identifier=prompt_id,
                        prompt_body=prompt_body,
                        model_id=model_id,
                        error=str(e),
                    )
                continue

            # Count completed models (state already updated in process_prompt_synchronized)
            completed_count = sum(1 for r in results.values() if r.success)
            models_completed += completed_count
            
            # Check if prompt is complete (all configured models done) by checking CSV
            is_complete = _is_prompt_complete(csv_path, prompt_id, config.models)

            # STRICT: Only mark prompt as processed if ALL models completed
            if is_complete:
                logger.info(
                    f"Prompt {prompt_id} completed (all {len(config.models)} models done)"
                )
                prompts_processed += 1

                # Check prompt limit: count total completed prompts now from CSV
                current_completed = _count_completed_prompts(csv_path, config.models)
                if config.prompt_limit is not None and current_completed >= target_completed:
                    logger.info(
                        f"Prompt limit reached: {current_completed} prompts completed "
                        f"(target: {target_completed}). Stopping."
                    )
                    break
            else:
                # Some models still missing - will be picked up in next run
                # Get missing models by checking CSV
                completed = _load_completed_from_csv(csv_path)
                remaining = [
                    model_id for model_id in config.models
                    if (prompt_id, model_id) not in completed
                ]
                if remaining:
                    logger.info(
                        f"Prompt {prompt_id} partially complete. "
                        f"Remaining models: {', '.join(remaining)}"
                    )

            # Update last processed prompt
            state.last_processed_prompt = prompt_id
            # State already saved immediately after each model completes, but save again here
            # to ensure last_processed_prompt is persisted
            state_manager.save_state(state)

            # STRICT: Check if budget exceeded after processing
            if budget_tracker.exceeded:
                logger.warning(
                    f"Budget limit reached after processing prompt {prompt_id}. "
                    f"Current: ${budget_tracker.current:.4f}, Limit: ${budget_tracker.limit:.2f}"
                )
                budget_exceeded = True
                break

        # Final summary - count completed prompts from CSV
        final_completed = _count_completed_prompts(csv_path, config.models)
        summary = {
            "total_cost": state.total_cost,
            "prompts_processed": prompts_processed,
            "models_completed": models_completed,
            "total_completed_prompts": final_completed,
            "budget_exceeded": budget_exceeded,
            "remaining_budget": budget_tracker.remaining,
        }

        logger.info(
            f"Run complete. Processed {prompts_processed} prompts, "
            f"{models_completed} model completions. "
            f"Total fully completed prompts: {final_completed}. "
            f"Total cost: ${state.total_cost:.4f}, "
            f"Remaining: ${budget_tracker.remaining:.2f}"
        )
        if config.prompt_limit is not None:
            logger.info(
                f"Prompt limit: {config.prompt_limit} new prompts requested. "
                f"Started with {existing_completed} completed, "
                f"now have {final_completed} completed "
                f"({final_completed - existing_completed} new)."
            )

        return summary

    finally:
        csv_writer.close()
        # Close HTTP client (use asyncio.run since we're in a sync function)
        try:
            # Check if there's a running event loop (shouldn't be in sync function)
            try:
                asyncio.get_running_loop()
                # If we get here, there's a running loop (unexpected in sync function)
                logger.warning("Event loop is running - client may not close properly")
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                # Wrap in try/except to handle case where event loop is already closed
                try:
                    asyncio.run(client.close())
                except RuntimeError:
                    # Event loop is closed, client will be cleaned up by garbage collection
                    pass
        except Exception as e:
            # Silently ignore cleanup errors
            logger.debug(f"Error closing HTTP client: {e}")
        # Final state save
        if state:
            state_manager.save_state(state)

