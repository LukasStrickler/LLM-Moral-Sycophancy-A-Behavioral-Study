from __future__ import annotations

import asyncio
import logging
import time
from email.utils import parsedate_to_datetime

import httpx

from ..core.config import OpenRouterConfig, RunConfig
from ..core.logging import (
    MODEL_PREFIX_WIDTH,
    PROGRESS_PREFIX_WIDTH,
    TAG_PREFIX_WIDTH,
    make_log_extra,
    setup_logger,
)
from ..core.rate_limit import TokenBucket, compute_backoff
from ..core.types import ChatMessage, ModelResponse, ProviderMetadata, message_dict

MODEL_SHORT_WIDTH = MODEL_PREFIX_WIDTH
TAG_WIDTH = TAG_PREFIX_WIDTH
SEP = " | "


def _short_model_id(model_id: str) -> str:
    base = model_id.split("/", 1)[-1]
    return base.split(":", 1)[0]


TAG_LABEL_MAP = {
    "rate-limit": "limited",
    "provider-error": "error",
    "debug": "debug",
}


def _tag(label: str) -> str:
    display = TAG_LABEL_MAP.get(label, label)
    return f"[{display}]".ljust(TAG_WIDTH)


def _ratelimit_prefix(
    tag: str,
    model_id: str,
    *,
    grid: str | None = None,
    task: str | None = None,
    progress: str | None = None,
) -> str:
    parts: list[str] = [_tag(tag), f"{_short_model_id(model_id):<{MODEL_SHORT_WIDTH}}"]
    if grid:
        parts.append(f"g={grid}")
    if task:
        parts.append(f"t={task}")
    if progress is not None:
        parts.append(f"p={progress:<{PROGRESS_PREFIX_WIDTH}}".rstrip())
    return SEP.join(parts)


def _log_event(
    logger: logging.Logger,
    level: int,
    *,
    tag: str,
    status: str,
    model_id: str,
    grid: str | None,
    task: str | None,
    progress: str | None,
    details: tuple[str, ...],
) -> None:
    prefix = _ratelimit_prefix(tag, model_id, grid=grid, task=task, progress=progress)
    detail_text = "  ".join(part for part in details if part)
    logger.log(
        level,
        "",
        extra={
            "model_prefix": prefix,
            "status_label": status,
            "details": detail_text,
        },
    )


class OpenRouterClient:
    # Provider-level defaults for OpenRouter free models
    DEFAULT_RPS = 0.367  # 22 RPM, maximizes free tier limit of 20 RPM
    DEFAULT_BURST = 5

    def __init__(
        self,
        cfg: OpenRouterConfig,
        run_cfg: RunConfig | None = None,
        model_configs: dict[str, dict] | None = None,
    ):
        self.cfg = cfg
        self.run_cfg = run_cfg or RunConfig()
        self.model_configs = model_configs or {}
        short = "OR"
        self.logger = setup_logger(short)
        self._client: httpx.AsyncClient | None = None
        self._limiters: dict[str, TokenBucket] = {}

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self.run_cfg.request_timeout_s)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _rate_burst_for_model(self, model_id: str) -> tuple[float, int]:
        # Check for per-model configuration first
        model_config = self.model_configs.get(model_id, {})
        rate_limit = model_config.get("rate_limit", {})

        if rate_limit:
            rate = rate_limit.get("rps", self.DEFAULT_RPS)
            burst = rate_limit.get("burst", self.DEFAULT_BURST)
        else:
            # Fall back to provider defaults
            rate = self.DEFAULT_RPS
            burst = self.DEFAULT_BURST

        rate = max(rate, 0.05)  # ensure progress even if values are tiny
        burst = max(burst, 1)
        return rate, burst

    def _get_limiter(self, model_id: str) -> TokenBucket:
        limiter = self._limiters.get(model_id)
        if limiter is None:
            rate, burst = self._rate_burst_for_model(model_id)
            self.logger.debug(
                "Limiter created for %s: rate=%.2f rps, burst=%d", model_id, rate, burst
            )
            limiter = TokenBucket(rate=rate, capacity=burst)
            self._limiters[model_id] = limiter
        return limiter

    async def chat_async(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        *,
        grid_id: str | None = None,
        task_id: str | None = None,
        progress: str | None = None,
    ) -> ProviderMetadata:
        if not self.cfg.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        model_id = model or self.cfg.model or self.cfg.default_test_model
        if not model_id:
            raise ValueError("No model id provided and no default configured")
        limiter = self._get_limiter(model_id)
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model_id,
            "messages": [message_dict(m) for m in messages],
            "temperature": 0.2,
        }
        grid_ref = grid_id
        task_ref = task_id
        progress_ref = progress
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        client = await self._ensure_client()
        attempt = 0
        finish_reason: str | None = None
        request_id: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        provider_cost: float | None = None

        while True:
            await limiter.acquire()
            t0 = time.time()
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                if attempt >= self.run_cfg.max_retries:
                    raise
                delay = compute_backoff(attempt)
                self.logger.warning(
                    "",
                    extra=make_log_extra(
                        model=model_id,
                        tag="limited",
                        status="network-error",
                        details=(f"error={e}", f"retry_in={delay:0.2f}s"),
                    ),
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue

            dt_ms = int((time.time() - t0) * 1000)
            # Handle rate limit or server errors
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= self.run_cfg.max_retries:
                    text = resp.text
                    raise RuntimeError(f"HTTP {resp.status_code}: {text}")
                retry_after = resp.headers.get("Retry-After")
                delay = compute_backoff(attempt)
                if retry_after:
                    retry_after = retry_after.strip()
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            parsed = parsedate_to_datetime(retry_after)
                        except (TypeError, ValueError):
                            self.logger.debug(
                                "",
                                extra=make_log_extra(
                                    model=model_id,
                                    tag="debug",
                                    status="invalid-retry-after",
                                    details=(str(retry_after),),
                                ),
                            )
                        else:
                            delay = max(0.0, parsed.timestamp() - time.time())
                reset_header = resp.headers.get("X-RateLimit-Reset") or resp.headers.get(
                    "X-Ratelimit-Reset"
                )
                if reset_header:
                    try:
                        reset_val = float(reset_header)
                        if reset_val > 1e10:
                            reset_val /= 1000.0
                        reset_delay = max(0.0, reset_val - time.time())
                        delay = max(delay, reset_delay)
                    except ValueError:
                        self.logger.debug(
                            "",
                            extra=make_log_extra(
                                model=model_id,
                                tag="debug",
                                status="invalid-reset",
                                details=(str(reset_header),),
                            ),
                        )
                self.logger.warning(
                    "",
                    extra=make_log_extra(
                        model=model_id,
                        grid=grid_ref,
                        task=task_ref,
                        progress=progress_ref,
                        tag="limited",
                        status="limited",
                        details=(
                            f"attempt={attempt + 1:02d}",
                            f"sleep={delay:0.2f}s",
                            f"reset={reset_header}" if reset_header else "",
                        ),
                    ),
                )
                await asyncio.sleep(delay)
                self.logger.info(
                    "",
                    extra=make_log_extra(
                        model=model_id,
                        grid=grid_ref,
                        task=task_ref,
                        progress=progress_ref,
                        tag="limited",
                        status="retry",
                        details=(f"attempt={attempt + 1:02d}", "retrying"),
                    ),
                )
                attempt += 1
                continue

            resp.raise_for_status()
            obj = resp.json()
            if isinstance(obj, dict) and "error" in obj:
                err = obj.get("error", {})
                err_message = err.get("message", "Unknown error")
                err_code = err.get("code")
                retryable_payload_codes = {429, 500, 502, 503, 504}
                if err_code in retryable_payload_codes and attempt < self.run_cfg.max_retries:
                    delay = compute_backoff(attempt)
                    self.logger.warning(
                        "",
                        extra=make_log_extra(
                            model=model_id,
                            grid=grid_ref,
                            task=task_ref,
                            progress=progress_ref,
                            tag="error",
                            status="provider-error",
                            details=(
                                f"attempt={attempt + 1:02d}",
                                f"code={err_code}",
                                f"msg={err_message}",
                                f"sleep={delay:0.2f}s",
                            ),
                        ),
                    )
                    await asyncio.sleep(delay)
                    self.logger.info(
                        "",
                        extra=make_log_extra(
                            model=model_id,
                            grid=grid_ref,
                            task=task_ref,
                            progress=progress_ref,
                            tag="error",
                            status="provider-retry",
                            details=(f"attempt={attempt + 1:02d}", "retrying"),
                        ),
                    )
                    attempt += 1
                    continue
                raise RuntimeError(f"HTTP payload error (code={err_code}): {err_message}")
            try:
                choice = obj["choices"][0]
                text = choice["message"]["content"].strip()
                finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
            except Exception as e:
                raise RuntimeError(f"Unexpected response: {obj}") from e
            request_id = obj.get("id")
            usage = obj.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            provider_info = obj.get("provider")
            if isinstance(provider_info, dict):
                provider_cost = provider_info.get("cost", {}).get("total")
            elif isinstance(provider_info, str):
                cost_info = obj.get("usage", {}).get("total_cost")
                if isinstance(cost_info, int | float):
                    provider_cost = float(cost_info)
            model_response = ModelResponse(
                model_id=model_id,
                response_text=text,
                latency_ms=dt_ms,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=provider_cost,
                request_id=request_id,
                finish_reason=finish_reason,
                raw_response=obj,
            )

            provider_name = None
            provider = obj.get("provider")
            if isinstance(provider, dict):
                provider_name = provider.get("name")
            elif isinstance(provider, str):
                provider_name = provider

            return ProviderMetadata(
                provider_name=provider_name or "openrouter",
                model_response=model_response,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=provider_cost,
                raw_response=obj,
            )

    # Synchronous convenience wrapper
    def chat(self, messages: list[ChatMessage], model: str | None = None) -> ProviderMetadata:
        """Synchronously call :meth:`chat_async` using ``asyncio.run``.

        This helper must not be invoked from within an active event loop; callers in
        async contexts should await :meth:`chat_async` directly.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.chat_async(messages, model=model))
        raise RuntimeError(
            "chat() cannot be used inside a running event loop; call chat_async instead"
        )
