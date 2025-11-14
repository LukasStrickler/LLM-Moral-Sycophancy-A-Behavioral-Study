"""Shared retry logic for LiteLLM API calls."""

from __future__ import annotations

import asyncio
import random
import re
import time
from typing import NamedTuple

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_BASE_WAIT_SECONDS = 1.5
_MAX_WAIT_SECONDS = 65.0


class RetryDecision(NamedTuple):
    """Represents how the caller should respond to a failed API request."""

    should_retry: bool
    wait_seconds: float
    label: str
    summary: str


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
    # Match "retry in Xs" pattern
    match = re.search(r"retry in\s+([\d.]+)s", message, re.IGNORECASE)
    if match:
        candidates.append(match.group(1))
    # Match Google API format: "retryDelay": "31s" (with or without quotes, with or without s suffix)
    delay_match = re.search(r'"retryDelay"\s*:\s*"?(\d+(?:\.\d+)?)s?"', message, re.IGNORECASE)
    if delay_match:
        candidates.append(delay_match.group(1))
    # Also try to extract from response body if it's JSON
    try:
        response = getattr(exc, "response", None)
        if response:
            # Try to get response body/text
            body = getattr(response, "text", None) or getattr(response, "body", None)
            if body:
                import json
                try:
                    error_data = json.loads(body) if isinstance(body, str) else body
                    # Navigate through Google API error structure
                    if isinstance(error_data, dict):
                        error_obj = error_data.get("error", {})
                        details = error_obj.get("details", [])
                        for detail in details:
                            if isinstance(detail, dict) and detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                                retry_delay = detail.get("retryDelay")
                                if retry_delay:
                                    candidates.append(str(retry_delay))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
    except Exception:
        pass

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


def _is_trial_rate_limit_error(exc: Exception) -> bool:
    """Return True when the error signals a trial key rate limit (e.g., 10 calls/minute)."""
    message = str(exc)
    # Match the specific Cohere trial key error pattern
    return "You are using a Trial key, which is limited to " in message


def extract_concise_error_message(exc: Exception) -> str:
    """Extract a concise error message from an exception, avoiding huge JSON dumps."""
    import litellm
    
    # Handle trial key rate limit errors first
    if _is_trial_rate_limit_error(exc):
        return "Trial Key 10 RPM Limit"
    
    # Handle LiteLLM exceptions with structured error info
    if isinstance(exc, litellm.exceptions.RateLimitError):
        # Try to extract retry delay from message
        msg = str(exc)
        retry_match = re.search(r"retry in\s+([\d.]+)s", msg, re.IGNORECASE)
        if retry_match:
            return f"Rate limit (retry in {retry_match.group(1)}s)"
        return "Rate limit exceeded"
    
    if isinstance(exc, litellm.exceptions.NotFoundError):
        # Extract key info from NotFoundError
        msg = str(exc)
        # Look for common patterns like "No endpoints found" or model names
        if "No endpoints found" in msg:
            return "No endpoints found (check data policy)"
        if "model" in msg.lower():
            return "Model not found"
        return "Resource not found"
    
    if isinstance(exc, litellm.exceptions.BadRequestError):
        msg = str(exc)
        # For quota errors that come as BadRequestError
        if "quota" in msg.lower() or "429" in msg:
            retry_match = re.search(r"retry in\s+([\d.]+)s", msg, re.IGNORECASE)
            if retry_match:
                return f"Quota exceeded (retry in {retry_match.group(1)}s)"
            return "Quota exceeded"
        # Extract first meaningful line
        first_line = msg.split("\n", 1)[0]
        if len(first_line) > 100:
            return first_line[:97] + "..."
        return first_line or "Bad request"
    
    # For other exceptions, get the first line and truncate if too long
    msg = str(exc)
    first_line = msg.split("\n", 1)[0]
    
    # If it's a huge JSON dump, try to extract key info
    if "{" in first_line and len(first_line) > 200:
        # Try to extract error message from JSON-like structure
        error_match = re.search(r'"message"\s*:\s*"([^"]+)"', first_line, re.IGNORECASE)
        if error_match:
            error_msg = error_match.group(1)
            if len(error_msg) > 100:
                return error_msg[:97] + "..."
            return error_msg
        # Fall back to exception type and status code
        status_code = _status_code_from_exception(exc)
        if status_code:
            return f"{type(exc).__name__} (HTTP {status_code})"
        return type(exc).__name__
    
    # Truncate if still too long
    if len(first_line) > 100:
        return first_line[:97] + "..."
    
    return first_line or type(exc).__name__


def classify_exception(exc: Exception) -> tuple[str | None, str]:
    """Determine the retry label for an exception and capture a short summary."""
    import litellm

    status_code = _status_code_from_exception(exc)
    summary = extract_concise_error_message(exc)

    if _is_daily_quota_error(exc):
        return "quota", summary or "daily quota"

    # Trial rate limit errors should be retried with minute-based wait
    if _is_trial_rate_limit_error(exc):
        return "limited", summary or "trial rate limit"

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


def build_retry_decision(exc: Exception, attempt: int) -> RetryDecision:
    """Create a retry decision for the given exception and attempt index (1-indexed)."""
    label, summary = classify_exception(exc)
    if label == "quota":
        return RetryDecision(False, 0.0, label, summary)
    if label is None:
        return RetryDecision(False, 0.0, "nonretry", summary)

    wait_seconds = _retry_hint_seconds(exc)
    if wait_seconds is None:
        if label == "limited":
            # For trial rate limits, wait until next minute
            if _is_trial_rate_limit_error(exc):
                wait_seconds = _seconds_until_next_minute() + 1.0
            else:
                wait_seconds = max(5.0, _seconds_until_next_minute() + 1.0)
        else:
            wait_seconds = _BASE_WAIT_SECONDS * (2 ** (attempt - 1))
    wait_seconds = min(_MAX_WAIT_SECONDS, wait_seconds)
    wait_seconds = max(1.0, wait_seconds + random.uniform(0.25, 0.75))

    return RetryDecision(True, wait_seconds, label, summary)


# Rate limit guard for respecting rate limits across concurrent requests
RATE_LIMIT_GUARD: dict[str, float] = {}
RATE_LIMIT_LOCK = asyncio.Lock()


async def respect_rate_limit_window(provider_name: str | None) -> None:
    """Wait if necessary to respect rate limit windows for a provider."""
    if not provider_name:
        return
    while True:
        async with RATE_LIMIT_LOCK:
            available = RATE_LIMIT_GUARD.get(provider_name, 0.0)
        delay = available - time.time()
        if delay <= 0:
            return
        await asyncio.sleep(delay)

