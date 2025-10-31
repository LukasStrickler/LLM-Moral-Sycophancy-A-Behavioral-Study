#!/usr/bin/env python3
"""CLI for batch AI labeling of responses in the database."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, TypedDict

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
    count_reviewer_completed,
    get_unlabeled_responses,
    insert_review,
)
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


async def score_with_retry(
    client: LiteLLMProvider,
    response: dict,
    model_id: str,
    get_progress: Callable[[], str],
    max_attempts: int,
    task_num: int,
) -> tuple[float | None, bool, float]:
    """Score a single response with retry-after header-based retry handling.
    
    Handles retries by extracting retry-after delays from HTTP response headers
    (much more effective than exponential backoff). LiteLLM's automatic retries
    are disabled (max_retries=0) so we have full control over retry timing.
    
    Args:
        client: LiteLLM provider instance (configured with max_retries=0)
        response: Response dictionary to score
        model_id: Model identifier
        get_progress: Callable that returns current progress string
        max_attempts: Maximum retry attempts
        task_num: Sequential task number (1, 2, 3...)
    """
    import litellm
    
    def parse_groq_time_format(time_str: str) -> float | None:
        """Parse Groq's time format (e.g., '2m59.56s', '7.66s') to seconds.
        
        Examples:
            '2m59.56s' -> 179.56 seconds
            '7.66s' -> 7.66 seconds
            '1m' -> 60 seconds
        """
        try:
            time_str = time_str.strip().lower()
            total_seconds = 0.0
            
            # Parse minutes (e.g., '2m' or '2m59.56s')
            if 'm' in time_str:
                minutes_match = re.search(r'(\d+(?:\.\d+)?)m', time_str)
                if minutes_match:
                    total_seconds += float(minutes_match.group(1)) * 60
            
            # Parse seconds (e.g., '7.66s' or '59.56s')
            if 's' in time_str:
                seconds_match = re.search(r'(\d+(?:\.\d+)?)s', time_str)
                if seconds_match:
                    total_seconds += float(seconds_match.group(1))
            
            return total_seconds if total_seconds > 0 else None
        except (ValueError, AttributeError, TypeError):
            return None
    
    for attempt in range(max_attempts):
        try:
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
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error - check multiple sources
            is_rate_limit = isinstance(e, litellm.exceptions.RateLimitError)
            
            # Also check for BadRequestError with 429 status code (Gemini uses this)
            if not is_rate_limit and isinstance(e, litellm.exceptions.BadRequestError):
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    is_rate_limit = True
            
            # Also check for APIError with 429 status code
            if not is_rate_limit and isinstance(e, litellm.exceptions.APIError):
                status_code = getattr(e, "status_code", None)
                if status_code == 429:
                    is_rate_limit = True
            
            # Also check APIConnectionError - sometimes rate limits come as connection errors
            # (e.g., Cohere trial key rate limits)
            if not is_rate_limit and isinstance(e, litellm.exceptions.APIConnectionError):
                # Check if the error message indicates a rate limit
                if ("limited to" in error_msg.lower() 
                    or "rate limit" in error_msg.lower()
                    or "api calls / minute" in error_msg.lower()
                    or "requests per minute" in error_msg.lower()):
                    is_rate_limit = True
            
            # Fallback to string-based detection if exception type doesn't match
            if not is_rate_limit:
                is_rate_limit = (
                    "RateLimitError" in error_msg
                    or "rate limit" in error_msg.lower()
                    or "429" in error_msg
                    or "RESOURCE_EXHAUSTED" in error_msg
                    or "limited to" in error_msg.lower()  # Cohere trial key messages
                    or "requests per minute" in error_msg.lower()
                    or "api calls / minute" in error_msg.lower()
                    or "quota exceeded" in error_msg.lower()
                    or "too many requests" in error_msg.lower()
                )
            
            # Extract retry-after delay from response headers if available
            default_wait_time = 2.0 * (attempt + 1)  # Default exponential backoff
            wait_time = default_wait_time
            found_retry_time = False  # Track if we successfully extracted retry time from response
            
            # Check if error message contains retry instructions first (e.g., Gemini's "Please retry in X.XXXs")
            # This takes precedence over daily quota detection - if API says to retry, we should retry
            has_retry_instruction = re.search(r'Please retry in ([\d.]+)s', error_msg, re.IGNORECASE)
            if has_retry_instruction:
                wait_time = float(has_retry_instruction.group(1)) + 2.0  # Add 2s buffer
                found_retry_time = True
            
            # Also check for retryDelay in JSON structure (Gemini's error details)
            if not found_retry_time:
                retry_delay_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', error_msg, re.IGNORECASE)
                if retry_delay_match:
                    wait_time = float(retry_delay_match.group(1)) + 2.0  # Add 2s buffer
                    found_retry_time = True
            
            # Detect if it's a daily quota limit (RPD) vs per-minute limit (RPM)
            # Only treat as non-retryable if there's no retry instruction in the error message
            is_daily_quota = False
            if is_rate_limit and not found_retry_time:
                # Check for daily quota indicators in Google's error response
                # quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                # quotaMetric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
                daily_indicators = [
                    r'"quotaId"\s*:\s*"[^"]*RequestsPerDay[^"]*"',  # quotaId contains "RequestsPerDay"
                    r'"quotaId"\s*:\s*"[^"]*PerDay[^"]*"',  # quotaId contains "PerDay"
                    r'limit:\s*\d+\s*RPD',  # Explicit RPD mention
                    r'Requests Per Day',  # Explicit mention
                ]
                for pattern in daily_indicators:
                    if re.search(pattern, error_msg, re.IGNORECASE):
                        is_daily_quota = True
                        break
            
            # For daily quotas without retry instructions, don't retry - quota resets at midnight UTC
            if is_daily_quota and not found_retry_time:
                logger.warning(
                    f"Daily quota (RPD) limit exceeded - skipping retries. "
                    f"Quota resets at midnight UTC.",
                    extra=make_log_extra(
                        model=model_id,
                        grid=str(response["id"]),
                        task=str(task_num),
                        progress=get_progress(),
                        tag="warning",
                        status="daily-quota-exceeded",
                        details=("RPD limit hit, no retries",),
                    ),
                )
                return None, False, 0.0
            
            if is_rate_limit:
                # Try to extract retry time from response headers (works for RateLimitError and APIError)
                # LiteLLM exceptions may have response in different attributes:
                # - RateLimitError.response (httpx.Response)
                # - APIError.response (httpx.Response)
                # - APIConnectionError might not have response directly
                
                # Try multiple ways to access the response
                # LiteLLM exceptions may wrap httpx exceptions or have response in different places
                http_response = None
                for attr_name in ["response", "http_response", "resp", "litellm_response", "_response"]:
                    http_response = getattr(e, attr_name, None)
                    if http_response:
                        break
                
                # If no direct response, check wrapped exceptions
                if not http_response:
                    for attr_name in ["exception", "original_exception", "cause", "__cause__"]:
                        underlying = getattr(e, attr_name, None)
                        if underlying:
                            http_response = getattr(underlying, "response", None)
                            if http_response:
                                break
                
                # Try to get response from exception's __dict__ if available
                if not http_response and hasattr(e, "__dict__"):
                    for key, value in e.__dict__.items():
                        if "response" in key.lower() and value and hasattr(value, "headers"):
                            http_response = value
                            break
                
                # Extract headers from response using httpx's built-in methods
                # httpx.Response.headers is a case-insensitive Headers object
                headers = None
                if http_response:
                    if hasattr(http_response, "headers"):
                        headers = http_response.headers
                    elif hasattr(http_response, "get"):
                        # Might be a dict-like object
                        headers = http_response.get("headers") or http_response.get("Headers")
                        if headers and not hasattr(headers, "get"):
                            # Convert dict to something we can use
                            try:
                                import httpx
                                headers = httpx.Headers(headers)
                            except:
                                pass
                
                # Extract retry-after from headers using httpx's case-insensitive access
                if headers:
                    # httpx.Headers.get() is case-insensitive, so we can use lowercase
                    # Check standard retry-after header first
                    retry_after = headers.get("retry-after")
                    if retry_after:
                        try:
                            wait_time = float(retry_after) + 2.0  # Add 2s buffer
                            found_retry_time = True
                        except (ValueError, TypeError):
                            pass
                    
                    # Try X-RateLimit-Reset headers if retry-after not found
                    # Groq uses x-ratelimit-reset-requests and x-ratelimit-reset-tokens with time format (e.g., "2m59.56s")
                    # Other providers may use timestamp format
                    if not found_retry_time:
                        reset_header = (
                            headers.get("x-ratelimit-reset-requests") 
                            or headers.get("x-ratelimit-reset-tokens")
                            or headers.get("x-ratelimit-reset")
                        )
                        if reset_header:
                            # Try Groq's time format first (e.g., "2m59.56s", "7.66s")
                            parsed_time = parse_groq_time_format(reset_header)
                            if parsed_time:
                                wait_time = parsed_time + 2.0  # Add 2s buffer
                                found_retry_time = True
                            else:
                                # Fallback: Try timestamp format (numeric)
                                try:
                                    reset_timestamp = float(reset_header)
                                    # Check if it's milliseconds (large number) or seconds
                                    if reset_timestamp > 1e10:
                                        reset_timestamp = reset_timestamp / 1000  # Convert ms to seconds
                                    current_time = time.time()
                                    wait_time = max(1.0, reset_timestamp - current_time + 2)
                                    found_retry_time = True
                                except (ValueError, TypeError):
                                    pass
                    
                    # Check all headers for rate limit related keys if still not found
                    # Some providers might use non-standard header names
                    if not found_retry_time and hasattr(headers, "keys"):
                        try:
                            for header_key in headers.keys():
                                header_lower = header_key.lower()
                                if ("retry" in header_lower or "rate" in header_lower or "limit" in header_lower):
                                    header_value = headers.get(header_key)
                                    if header_value:
                                        # Try to parse as time format or number
                                        parsed_time = parse_groq_time_format(str(header_value))
                                        if parsed_time:
                                            wait_time = parsed_time + 2.0
                                            found_retry_time = True
                                            break
                                        try:
                                            wait_seconds = float(header_value)
                                            if 0 < wait_seconds < 3600:  # Reasonable wait time (0-1 hour)
                                                wait_time = wait_seconds + 2.0
                                                found_retry_time = True
                                                break
                                        except (ValueError, TypeError):
                                            continue
                        except Exception:
                            pass
                
                # Parse error message for retryDelay if headers didn't provide it
                # Note: We already checked for "Please retry in" and "retryDelay" earlier,
                # so this is only a fallback if headers weren't found
                if not found_retry_time:
                    # Pattern 1: "Please retry in X.XXXs" from Gemini error message (most precise)
                    # This is a fallback - we already checked this earlier but it's here for completeness
                    retry_msg_match = re.search(r'Please retry in ([\d.]+)s', error_msg, re.IGNORECASE)
                    if retry_msg_match:
                        wait_time = float(retry_msg_match.group(1)) + 2.0  # Add 2s buffer
                        found_retry_time = True
                    else:
                        # Pattern 2: "retryDelay": "8s" from JSON (fallback if message not found)
                        retry_delay_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', error_msg, re.IGNORECASE)
                        if retry_delay_match:
                            wait_time = float(retry_delay_match.group(1)) + 2.0  # Add 2s buffer
                            found_retry_time = True
                
                # Pattern 3: Cohere trial key - "limited to X API calls / minute"
                # Check this separately (not nested in else) to catch per-minute limits
                # For per-minute limits, wait until next minute (60s)
                if not found_retry_time and is_rate_limit:
                    if ("per minute" in error_msg.lower() 
                        or "requests per minute" in error_msg.lower()
                        or "api calls / minute" in error_msg.lower()
                        or "api calls/minute" in error_msg.lower()
                        or "calls / minute" in error_msg.lower()
                        or "calls/minute" in error_msg.lower()):
                        wait_time = 60.0  # Wait until next minute window
                        found_retry_time = True
                
                # If still no retry time found, fall back to exponential backoff
                # Log debug info to help diagnose why headers weren't found
                if not found_retry_time:
                    # Debug: log exception structure and what headers we found
                    debug_info = []
                    debug_info.append(f"exception={type(e).__name__}")
                    debug_info.append(f"has_response={hasattr(e, 'response')}")
                    
                    if http_response:
                        debug_info.append(f"http_response_type={type(http_response).__name__}")
                        if hasattr(http_response, "headers"):
                            debug_info.append(f"has_headers=True")
                            # Try to show what headers are available
                            try:
                                if hasattr(http_response.headers, "keys"):
                                    header_keys = list(http_response.headers.keys())[:5]  # First 5 headers
                                    debug_info.append(f"header_keys={header_keys}")
                            except:
                                pass
                        else:
                            debug_info.append(f"has_headers=False")
                    else:
                        debug_info.append(f"http_response=None")
                    
                    logger.debug(
                        f"Rate limit - no retry-after header found. {' | '.join(debug_info)}",
                        extra=make_log_extra(
                            model=model_id,
                            grid=str(response["id"]),
                            task=str(task_num),
                            progress=get_progress(),
                            tag="debug",
                            status="rate-limit-no-header",
                            details=tuple(debug_info),
                        ),
                    )
                    # Exponential backoff for RPM limits: 30s, 60s, 60s, 120s
                    # Most providers are RPM-limited, so longer waits are needed
                    if attempt == 0:
                        wait_time = 30.0  # 30s for first retry
                    elif attempt == 1:
                        wait_time = 60.0  # 1m for second retry
                    elif attempt == 2:
                        wait_time = 60.0  # 1m for third retry
                    else:
                        wait_time = 120.0  # 2m for subsequent retries
            
            if attempt < max_attempts - 1:
                current_progress = get_progress()
                if is_rate_limit:
                    # Only log at warning level if:
                    # 1. We couldn't extract retry time from response (found_retry_time = False)
                    #    (uses exponential backoff: 30s, 60s, 60s, 120s)
                    # 2. OR wait time is very long (>120s) - indicates potential issue
                    # Otherwise use debug level for successful retry time extraction from headers
                    if not found_retry_time or wait_time > 120.0:
                        logger.warning(
                            f"Rate limit - retry {attempt+1}/{max_attempts} in {wait_time:.0f}s",
                            extra=make_log_extra(
                                model=model_id,
                                grid=str(response["id"]),
                                task=str(task_num),
                                progress=current_progress,
                                tag="warning",
                                status="rate-limit-retry",
                                details=(f"wait_time={wait_time:.1f}s",),
                            ),
                        )
                    else:
                        # Successfully extracted retry time - log at debug level
                        logger.debug(
                            f"Rate limit - retry {attempt+1}/{max_attempts} in {wait_time:.0f}s",
                            extra=make_log_extra(
                                model=model_id,
                                grid=str(response["id"]),
                                task=str(task_num),
                                progress=current_progress,
                                tag="debug",
                                status="rate-limit-retry",
                                details=(f"wait_time={wait_time:.1f}s",),
                            ),
                        )
                else:
                    # Non-rate-limit errors - log as warning
                    logger.warning(
                        f"Error - retry {attempt+1}/{max_attempts} in {wait_time:.0f}s",
                        extra=make_log_extra(
                            model=model_id,
                            grid=str(response["id"]),
                            task=str(task_num),
                            progress=current_progress,
                            tag="warning",
                            status="retry",
                            details=(f"error={error_msg[:80]}", f"wait_time={wait_time:.1f}s"),
                        ),
                    )
                await asyncio.sleep(wait_time)
            else:
                # Create concise error message for rate limits
                if is_rate_limit:
                    # Extract key info: quota limit and retry delay if available
                    quota_match = re.search(r'limit:\s*(\d+)', error_msg, re.IGNORECASE)
                    retry_match = re.search(r'Please retry in ([\d.]+)s', error_msg, re.IGNORECASE)
                    quota_info = f" (quota: {quota_match.group(1)})" if quota_match else ""
                    retry_info = f" - retry in {retry_match.group(1)}s" if retry_match else ""
                    concise_msg = f"Rate limit exceeded{quota_info}{retry_info}"
                else:
                    concise_msg = error_msg[:100]
                
                error_type = "Rate limit" if is_rate_limit else "Error"
                logger.error(
                    f"{error_type} after {max_attempts} attempts: {concise_msg}",
                    extra=make_log_extra(
                        model=model_id,
                        grid=str(response["id"]),
                        task=str(task_num),
                        progress=get_progress(),
                        tag="error",
                        status="final-failure",
                        details=(f"error={concise_msg}",),
                    ),
                )
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
    logger.info(
        "Starting labeling",
        extra=make_log_extra(
            model=model_id,
            grid=None,
            task=None,
            progress=None,
            tag="info",
            status="start",
            details=(f"model={model_id}",),
        ),
    )
    
    # Setup LiteLLM provider with model configs
    client = LiteLLMProvider(provider_config, run_config, model_configs)
    
    # Get database client
    db_client = create_client()
    
    try:
        # Get unlabeled responses for this model
        reviewer_code = f"llm:{model_id}"
        unlabeled = get_unlabeled_responses(db_client, dataset, reviewer_code, limit)
        
        if not unlabeled:
            logger.info(
                "No unlabeled responses found",
                extra=make_log_extra(
                    model=model_id,
                    grid=None,
                    task=None,
                    progress=None,
                    tag="info",
                    status="no-data",
                    details=(f"reviewer_code={reviewer_code}",),
                ),
            )
            return {"successful": 0, "failed": 0, "skipped": 0, "cost": 0.0}
        
        logger.info(
            f"Found {len(unlabeled)} unlabeled responses",
            extra=make_log_extra(
                model=model_id,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="found-responses",
                details=(f"count={len(unlabeled)}", f"reviewer_code={reviewer_code}"),
            ),
        )
        if len(unlabeled) <= 10:
            logger.debug(f"Response IDs: {[r['id'] for r in unlabeled]}")
        
        # Get initial review count from DB for this model
        initial_review_count = count_reviewer_completed(db_client, dataset, reviewer_code)
        
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
                            status="dry-run",
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
                                status="insert-failed",
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
        
        logger.info(
            f"Model completed: {successful} successful, {failed} failed, {skipped} skipped",
            extra=make_log_extra(
                model=model_id,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="model-completed",
                details=(
                    f"successful={successful}",
                    f"failed={failed}",
                    f"skipped={skipped}",
                    f"cost=${total_cost:.4f}",
                ),
            ),
        )
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
    
    # Filter models based on API key availability
    models = []
    skipped_models = []
    for model_id in all_models:
        if _has_api_key_for_model(provider_config, model_id):
            models.append(model_id)
        else:
            skipped_models.append(model_id)
    
    # Log which models will be processed
    logger.info("")
    logger.info(f"📋 Found {len(all_models)} model(s) in config")
    if skipped_models:
        logger.warning(f"⚠️  {len(skipped_models)} model(s) skipped - no API key:")
        for model_id in skipped_models:
            logger.warning(f"   - {model_id}")
    logger.info(f"✅ {len(models)} model(s) will be processed:")
    for idx, model_id in enumerate(models, 1):
        logger.info(f"   {idx:2d}. {model_id}")
    logger.info("")
    
    # Convert dataset string to enum
    dataset = Dataset(args.dataset)
    
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
        logger.info(
            "Processing model",
            extra=make_log_extra(
                model=model_id,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="processing",
                details=(f"model={model_id}",),
            ),
        )
        
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
                status="model-summary",
                details=(
                    f"successful={stats['successful']}",
                    f"failed={stats['failed']}",
                    f"skipped={stats['skipped']}",
                ),
            ),
        )
        
        return stats
    
    # Process all models in parallel
    all_stats = await asyncio.gather(*[process_model(model_id) for model_id in models])
    
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
