from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.config import ProviderConfig, RunConfig
from ..core.logging import (
    MODEL_PREFIX_WIDTH,
    PROGRESS_PREFIX_WIDTH,
    make_log_extra,
    setup_logger,
)
from ..core.retry import build_retry_decision, extract_concise_error_message
from ..core.types import Factors, ProviderMetadata, RunRecord
from ..prompts.chat import build_chat_from_factors
from ..prompts.generator import generate_factor_grid
from ..providers.litellm_provider import LiteLLMProvider


logger = setup_logger("run")


async def _append_record(
    out_path: Path,
    new_record: RunRecord,
    file_lock: asyncio.Lock,
) -> None:
    """Append new record to file.
    
    Always appends - no deduplication. Each run creates new records.
    Filtering happens before running to skip already successful records.
    """
    async with file_lock:
        # Append new record
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(new_record.to_json_line() + "\n")
            handle.flush()

MODEL_COL_WIDTH = 28
TOKEN_COL_WIDTH = 4
COST_COL_WIDTH = 8
PROVIDER_COL_WIDTH = 12
MODEL_SHORT_WIDTH = MODEL_PREFIX_WIDTH
TAG_WIDTH = 10  # Width for tag formatting
PREFIX_SEPARATOR = " | "


def _fmt(label: str, value: Any, width: int, align: str = "<") -> str:
    text = "-" if value in (None, "") else str(value)
    return f"{label}={text:{align}{width}}"


def _fmt_seconds(label: str, millis: int | None, width: int) -> str:
    if millis is None:
        return _fmt(label, "n/a", width, ">")
    seconds = millis / 1000
    return _fmt(label, f"{seconds:0.2f}", width, ">")


def _fmt_tokens(label: str, value: int | None) -> str:
    if value is None:
        return _fmt(label, None, TOKEN_COL_WIDTH, ">")
    text = f"{value:0{TOKEN_COL_WIDTH}d}"
    if len(text) > TOKEN_COL_WIDTH:
        text = str(value)
    return _fmt(label, text, max(len(text), TOKEN_COL_WIDTH), ">")


def _fmt_cost(label: str, value: float | None) -> str:
    text = None if value is None else f"{value:0.4f}"
    return _fmt(label, text, COST_COL_WIDTH, ">")


TAG_LABEL_MAP = {
    "start": "start",
    "finish": "finish",
    "error": "error",
    "rate-limit": "limited",
    "provider-error": "error",
}


def _tag(label: str) -> str:
    display = TAG_LABEL_MAP.get(label, label)
    content = f"[{display}]"
    return f"{content:<{TAG_WIDTH}}"


def _short_model_id(model_id: str) -> str:
    base = model_id.split("/", 1)[-1]
    before_colon = base.split(":", 1)[0]
    return before_colon


def _progress_str(current: int, total: int) -> str:
    width = max(3, len(str(total)))
    return f"{current:0{width}d}/{total:0{width}d}"


def _prefix(
    tag: str,
    model_id: str,
    grid_seq: int,
    task_id: int,
    progress: str | None,
) -> str:
    parts = [
        _tag(tag),
        f"{_short_model_id(model_id):<{MODEL_SHORT_WIDTH}}",
        f"g={grid_seq:03d}",
        f"t={task_id:03d}",
    ]
    if progress is not None:
        parts.append(f"p={progress:<{PROGRESS_PREFIX_WIDTH}}".rstrip())
    return PREFIX_SEPARATOR.join(parts)


class ProgressTracker:
    """Track total, started, and completed tasks safely across async workers."""

    def __init__(self, total: int):
        self.total = int(total)
        self._started = 0
        self._completed = 0
        self._lock = asyncio.Lock()

    async def mark_start(self) -> tuple[int, int]:
        async with self._lock:
            self._started += 1
            return self._started, self._completed

    async def mark_done(self) -> int:
        async with self._lock:
            self._completed += 1
            return self._completed


@dataclass
class TaskSpec:
    """Metadata for a single (model, prompt) evaluation."""

    model_id: str
    index: int


async def _worker(
    sem: asyncio.Semaphore,
    client: LiteLLMProvider | None,
    out_path: Path,
    file_lock: asyncio.Lock,
    factors: Sequence[Factors],
    task: TaskSpec,
    run_id: int,
    dry_run: bool,
    progress: ProgressTracker,
    task_id: int,
    prompt_id_map: dict[str, str] | None = None,
    max_retries: int = 5,
) -> None:
    async with sem:
        t_task0 = time.time()
        factor = factors[task.index]
        prompt = build_chat_from_factors(factor)
        # Use prompt_id from grid.jsonl (sequential number) if available, otherwise use hash
        from ..core.types import make_prompt_id
        factor_hash = make_prompt_id(factor)
        prompt_id = (prompt_id_map or {}).get(factor_hash, prompt.prompt_id)
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started, completed = await progress.mark_start()
        progress_str = _progress_str(completed, progress.total)
        logger.info(
            "",
            extra=make_log_extra(
                model=task.model_id,
                grid=f"{task.index:03d}",
                task=f"{task_id:03d}",
                progress=progress_str,
                tag="start",
                status="start",
            ),
        )

        provider_metadata: ProviderMetadata | None = None
        provider_label: str | None = None
        latency_ms: int | None = None
        answer_text: str = ""
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        cost_usd: float | None = None
        request_id: str | None = None
        finish_reason: str | None = None
        raw_response: dict[str, Any] | None = None

        attempt = 0
        last_exc: Exception | None = None

        while attempt < max_retries:
            attempt += 1
            try:
                if dry_run:
                    provider_label = provider_label or "dry-run"
                    answer_text = f"[dry-run] {task.model_id} response to prompt {prompt.prompt_id}"
                    latency_ms = 0
                    break  # Success for dry-run
                else:
                    if client is None:
                        raise RuntimeError("LiteLLM provider is unavailable (dry-run disabled)")
                    
                    # request start intentionally not logged to keep output compact
                    provider_metadata = await client.chat_async(
                        prompt.messages,
                        model=task.model_id,
                        grid_id=f"{task.index:03d}",
                        task_id=f"{task_id:03d}",
                        progress=progress_str,
                    )
                    provider_label = provider_metadata.provider_name or provider_label
                    model_resp = provider_metadata.model_response
                    latency_ms = model_resp.latency_ms
                    answer_text = model_resp.response_text
                    input_tokens = provider_metadata.input_tokens
                    output_tokens = provider_metadata.output_tokens
                    total_tokens = provider_metadata.total_tokens
                    cost_usd = provider_metadata.cost_usd
                    request_id = model_resp.request_id
                    finish_reason = model_resp.finish_reason
                    raw_response = provider_metadata.raw_response
                    logger.debug(
                        "",
                        extra=make_log_extra(
                            model=task.model_id,
                            grid=f"{task.index:03d}",
                            task=f"{task_id:03d}",
                            progress=progress_str,
                            tag="debug",
                            status="request-done",
                            details=(
                                f"prompt={getattr(prompt, 'prompt_id', None)}",
                                f"took={latency_ms}ms",
                            ),
                        ),
                    )
                    break  # Success - exit retry loop

            except Exception as exc:
                last_exc = exc
                decision = build_retry_decision(exc, attempt)
                
                # Check if this is a trial key error - these should always retry until next minute
                from ..core.retry import _is_trial_rate_limit_error
                is_trial_error = _is_trial_rate_limit_error(exc)
                
                # Log the error/retry
                status_code = getattr(exc, "status_code", None)
                details = [f"attempt={attempt}/{max_retries}"]
                if status_code is not None:
                    details.append(f"status={status_code}")
                details.append(f"error={type(exc).__name__}")

                # For trial key errors, always retry (don't fail after max_retries)
                # For other errors, check max_retries and should_retry
                should_give_up = not is_trial_error and (attempt >= max_retries or not decision.should_retry)
                
                if should_give_up:
                    # Give up - log error and create failure record
                    status_prefix = _prefix("error", task.model_id, task.index, task_id, progress_str)
                    logger.error(
                        "Prompt worker failure",
                        extra={
                            "model_prefix": status_prefix,
                            "status_label": "giveup" if decision.should_retry else decision.label,
                            "details": tuple(details),
                        },
                    )
                    logger.debug("Prompt worker exception details", exc_info=exc)
                    break  # Exit retry loop - will create failure record below
                else:
                    # Retry - wait and continue
                    wait_seconds = decision.wait_seconds
                    if wait_seconds:
                        details.append(f"wait={wait_seconds:.0f}s")
                    logger.warning(
                        "Retrying prompt",
                        extra=make_log_extra(
                            model=task.model_id,
                            grid=f"{task.index:03d}",
                            task=f"{task_id:03d}",
                            progress=progress_str,
                            tag="retry",
                            status=decision.label,
                            details=tuple(details),
                        ),
                    )
                    if wait_seconds:
                        await asyncio.sleep(wait_seconds)
                    continue  # Retry

        # Only save successful records (like AI labeling)
        total_ms = int((time.time() - t_task0) * 1000)
        if last_exc is None:
            # Success (either dry_run or successful API call) - create and save record
            record = RunRecord.success(
                run_id=run_id,
                timestamp_iso=timestamp_iso,
                model_id=task.model_id,
                provider=provider_label,
                prompt_id=prompt_id,  # Use prompt_id from grid.jsonl (sequential number)
                perspective=factor.perspective,
                factors=asdict(factor),
                messages=prompt.messages,
                response_text=answer_text,
                latency_ms=latency_ms if latency_ms is not None else total_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                request_id=request_id,
                finish_reason=finish_reason,
                raw_response=raw_response,
            )
            # Append successful record
            await _append_record(out_path, record, file_lock)
            logger.debug(
                "",
                extra=make_log_extra(
                    model=task.model_id,
                    grid=f"{task.index:03d}",
                    task=f"{task_id:03d}",
                    progress=None,
                    tag="debug",
                    status="write",
                    details=(f"path={out_path}", f"prompt_id={record.prompt_id}"),
                ),
            )

            wall_ms = total_ms
            done = await progress.mark_done()
            progress_value = _progress_str(done, progress.total)
            metrics = (
                _fmt_seconds("wall", wall_ms, 6),
                _fmt_tokens("in", record.input_tokens),
                _fmt_tokens("out", record.output_tokens),
                _fmt_tokens("tot", record.total_tokens),
                _fmt_cost("cost", record.cost_usd),
                _fmt("provider", record.provider, PROVIDER_COL_WIDTH),
            )
            logger.info(
                "",
                extra=make_log_extra(
                    model=task.model_id,
                    grid=f"{task.index:03d}",
                    task=f"{task_id:03d}",
                    progress=progress_value,
                    tag="finish",
                    status="finish",
                    details=metrics,
                ),
            )
        else:
            # Failure - log but don't save (will be retried on next run)
            # Error already logged above in retry logic
            done = await progress.mark_done()
            progress_value = _progress_str(done, progress.total)
            error_msg = extract_concise_error_message(last_exc) if last_exc else "Unknown error"
            error_metrics = (
                _fmt_seconds("wall", total_ms, 6),
                _fmt("reason", error_msg, 48),
            )
            logger.info(
                "",
                extra=make_log_extra(
                    model=task.model_id,
                    grid=f"{task.index:03d}",
                    task=f"{task_id:03d}",
                    progress=progress_value,
                    tag="error",
                    status="error",
                    details=error_metrics,
                ),
            )


async def run_local_benchmark_async(
    *,
    limit: int = 10,
    factors_list_override: Sequence[Factors] | None = None,
    assistant_models: list[str],
    model_configs: dict[str, dict] | None = None,
    out_path: Path,
    dry_run: bool = False,
    prompt_id_map: dict[str, str] | None = None,
) -> Path:
    """Run the benchmark asynchronously and persist results to ``out_path``."""

    cfg = ProviderConfig.from_env()
    run_cfg = RunConfig.from_env()
    max_retries = run_cfg.max_retries
    client: LiteLLMProvider | None = None
    if not dry_run:
        client = LiteLLMProvider(cfg, run_cfg, model_configs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't truncate file - append/update mode for grid hash-based runs

    factors = (
        list(factors_list_override)
        if factors_list_override is not None
        else generate_factor_grid()
    )
    selected = factors[:limit] if limit < len(factors) else factors

    # Get next sequential run_id for this grid hash
    from ..core.types import get_next_run_id
    run_id = get_next_run_id(out_path)

    # Filter out already successful records - only process missing or failed prompts
    from ..prompts.chat import build_chat_from_factors
    from ..core.types import make_prompt_id
    successful_keys: set[tuple[str, str]] = set()
    if out_path.exists():
        try:
            for existing_record in RunRecord.iter_jsonl(out_path):
                if existing_record.is_success() and existing_record.prompt_id:
                    successful_keys.add((existing_record.prompt_id, existing_record.model_id))
        except Exception as exc:
            logger.warning("Error reading existing records for filtering: %s", exc)

    # Filter selected factors to only include prompts that haven't been successfully completed
    # We need to track which (factor_idx, model_id) combinations need processing
    filtered_tasks: list[tuple[int, int, str]] = []  # (factor_idx, filtered_idx, model_id)
    filtered_factors_list: list[Factors] = []
    factor_to_filtered_idx: dict[int, int] = {}
    
    for idx, factor in enumerate(selected):
        # Get prompt_id from grid.jsonl (sequential number) if available
        factor_hash = make_prompt_id(factor)
        grid_prompt_id = (prompt_id_map or {}).get(factor_hash) if prompt_id_map else None
        for model_id in assistant_models:
            # Use grid prompt_id (sequential number) for filtering
            key = (grid_prompt_id or factor_hash, model_id)
            if key not in successful_keys:
                # This (factor, model) combination needs processing
                if idx not in factor_to_filtered_idx:
                    # First time seeing this factor, add it to filtered_factors
                    filtered_idx = len(filtered_factors_list)
                    filtered_factors_list.append(factor)
                    factor_to_filtered_idx[idx] = filtered_idx
                else:
                    filtered_idx = factor_to_filtered_idx[idx]
                filtered_tasks.append((idx, filtered_idx, model_id))

    if not filtered_tasks:
        logger.info("All prompts already successfully completed. Nothing to process.")
        return out_path

    filtered_factors = filtered_factors_list

    # Grid.jsonl is copied to run folder in scripts/run_benchmark.py
    # No need for grid.json snapshot - grid.jsonl is the source of truth

    # Create per-model semaphores to prevent cross-model blocking
    model_configs = model_configs or {}
    model_semaphores: dict[str, asyncio.Semaphore] = {}
    for model_id in assistant_models:
        model_config = model_configs.get(model_id, {})
        model_concurrency = model_config.get("concurrency")
        if model_concurrency is None:
            if not model_config:
                raise ValueError(
                    f"Model {model_id} not found in models.json. "
                    f"Available models: {list(model_configs.keys())}. "
                    f"Please add the model configuration with required 'concurrency' field."
                )
            else:
                raise ValueError(
                    f"Model {model_id} missing required 'concurrency' field in models.json. "
                    f"Current config: {model_config}. "
                    f"Please add 'concurrency' field (e.g., 'concurrency': 3)."
                )
        # Validate concurrency is a positive integer
        if not isinstance(model_concurrency, int) or model_concurrency <= 0:
            raise ValueError(
                f"Model {model_id} concurrency must be a positive integer, "
                f"got {model_concurrency!r}. "
                f"Please set 'concurrency' to a positive integer (e.g., 'concurrency': 3)."
            )
        model_semaphores[model_id] = asyncio.Semaphore(model_concurrency)

    file_lock = asyncio.Lock()
    # Only create tasks for filtered (factor, model) combinations
    total_tasks = len(filtered_tasks)
    logger.info(
        "",
        extra={
            "model_prefix": "[info] | setup",
            "status_label": "schedule",
            "details": (
                f"total={total_tasks} tasks (filtered from {len(assistant_models) * len(selected)} possible) "
                f"models={len(assistant_models)} prompts={len(filtered_factors)} "
                f"per-model-concurrency"
            ),
        },
    )
    progress = ProgressTracker(total_tasks)
    global_task_id = 1
    tasks: list[asyncio.Task[None]] = []
    for orig_idx, filtered_idx, model_id in filtered_tasks:
        task = asyncio.create_task(
            _worker(
                model_semaphores[model_id],
                client,
                out_path,
                file_lock,
                filtered_factors,
                TaskSpec(model_id=model_id, index=filtered_idx),
                run_id,
                dry_run,
                progress,
                global_task_id,
                prompt_id_map,
                max_retries,
            )
        )
        tasks.append(task)
        global_task_id += 1

    await asyncio.gather(*tasks)
    if client is not None:
        await client.aclose()
    return out_path
