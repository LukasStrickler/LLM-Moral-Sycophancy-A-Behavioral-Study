"""Async runner for custom grids (Dear Abby, etc.) that don't use Factors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
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
from ..core.types import ChatMessage, ProviderMetadata, RunRecord
from ..providers.litellm_provider import LiteLLMProvider

logger = setup_logger("run-custom")


async def _append_record(
    out_path: Path,
    new_record: RunRecord,
    file_lock: asyncio.Lock,
) -> None:
    """Append new record to file."""
    async with file_lock:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(new_record.to_json_line() + "\n")
            handle.flush()


def _progress_str(current: int, total: int) -> str:
    width = max(3, len(str(total)))
    return f"{current:0{width}d}/{total:0{width}d}"


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
    custom_prompts: Sequence[dict[str, Any]],
    task: TaskSpec,
    run_id: int,
    dry_run: bool,
    progress: ProgressTracker,
    task_id: int,
    prompt_id_map: dict[str, str] | None = None,
    max_retries: int = 5,
) -> None:
    """Worker for custom prompts (Dear Abby style)."""
    async with sem:
        t_task0 = time.time()
        custom_prompt = custom_prompts[task.index]
        # Convert message dicts to ChatMessage objects (or use if already ChatMessage)
        message_dicts = custom_prompt["messages"]
        messages = []
        for msg in message_dicts:
            if isinstance(msg, ChatMessage):
                messages.append(msg)
            elif isinstance(msg, dict) and "role" in msg and "content" in msg and msg.get("content"):
                messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
        
        if not messages:
            logger.error(f"No valid messages found for prompt {custom_prompt.get('prompt_id')}: {message_dicts}")
            return
        prompt_id = (prompt_id_map or {}).get(
            custom_prompt["prompt_id"], custom_prompt["prompt_id"]
        )
        factors_dict = custom_prompt.get("factors_dict", {})
        
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
                    answer_text = f"[dry-run] {task.model_id} response to prompt {prompt_id}"
                    latency_ms = 0
                    break
                else:
                    if client is None:
                        raise RuntimeError("LiteLLM provider is unavailable (dry-run disabled)")
                    
                    provider_metadata = await client.chat_async(
                        messages,
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
                    break

            except Exception as exc:
                last_exc = exc
                decision = build_retry_decision(exc, attempt)
                from ..core.retry import _is_trial_rate_limit_error
                is_trial_error = _is_trial_rate_limit_error(exc)
                
                status_code = getattr(exc, "status_code", None)
                details = [f"attempt={attempt}/{max_retries}"]
                if status_code is not None:
                    details.append(f"status={status_code}")
                details.append(f"error={type(exc).__name__}")

                should_give_up = not is_trial_error and (attempt >= max_retries or not decision.should_retry)
                
                if should_give_up:
                    logger.error("Prompt worker failure", exc_info=exc)
                    break
                else:
                    wait_seconds = decision.wait_seconds
                    if wait_seconds:
                        details.append(f"wait={wait_seconds:.0f}s")
                    logger.warning("Retrying prompt", extra={"details": tuple(details)})
                    if wait_seconds:
                        await asyncio.sleep(wait_seconds)
                    continue

        total_ms = int((time.time() - t_task0) * 1000)
        if last_exc is None:
            # Success
            record = RunRecord.success(
                run_id=run_id,
                timestamp_iso=timestamp_iso,
                model_id=task.model_id,
                provider=provider_label,
                prompt_id=prompt_id,
                perspective=None,  # Not applicable for custom prompts
                factors=factors_dict,  # Store metadata here
                messages=messages,
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
            await _append_record(out_path, record, file_lock)
            
            done = await progress.mark_done()
            progress_value = _progress_str(done, progress.total)
            logger.info(
                "",
                extra=make_log_extra(
                    model=task.model_id,
                    grid=f"{task.index:03d}",
                    task=f"{task_id:03d}",
                    progress=progress_value,
                    tag="finish",
                    status="finish",
                ),
            )
        else:
            # Failure
            done = await progress.mark_done()
            progress_value = _progress_str(done, progress.total)
            error_msg = extract_concise_error_message(last_exc) if last_exc else "Unknown error"
            logger.info(
                "",
                extra=make_log_extra(
                    model=task.model_id,
                    grid=f"{task.index:03d}",
                    task=f"{task_id:03d}",
                    progress=progress_value,
                    tag="error",
                    status="error",
                    details=(f"reason={error_msg}",),
                ),
            )


async def run_custom_grid_async(
    *,
    custom_prompts: Sequence[dict[str, Any]],
    assistant_models: list[str],
    model_configs: dict[str, dict] | None = None,
    out_path: Path,
    dry_run: bool = False,
    prompt_id_map: dict[str, str] | None = None,
) -> Path:
    """Run benchmark for custom grids (Dear Abby, etc.)."""
    cfg = ProviderConfig.from_env()
    run_cfg = RunConfig.from_env()
    max_retries = run_cfg.max_retries
    client: LiteLLMProvider | None = None
    if not dry_run:
        client = LiteLLMProvider(cfg, run_cfg, model_configs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    from ..core.types import get_next_run_id
    run_id = get_next_run_id(out_path)
    
    # Filter out already successful records
    from ..core.types import RunRecord
    successful_keys: set[tuple[str, str]] = set()
    if out_path.exists():
        try:
            for existing_record in RunRecord.iter_jsonl(out_path):
                if existing_record.is_success() and existing_record.prompt_id:
                    successful_keys.add((existing_record.prompt_id, existing_record.model_id))
        except Exception as exc:
            logger.warning("Error reading existing records for filtering: %s", exc)
    
    # Create tasks
    filtered_tasks: list[tuple[int, int, str]] = []
    filtered_prompts: list[dict[str, Any]] = []
    prompt_to_idx: dict[int, int] = {}
    
    for idx, prompt in enumerate(custom_prompts):
        prompt_id = prompt.get("prompt_id", "")
        for model_id in assistant_models:
            key = (prompt_id, model_id)
            if key not in successful_keys:
                if idx not in prompt_to_idx:
                    filtered_idx = len(filtered_prompts)
                    filtered_prompts.append(prompt)
                    prompt_to_idx[idx] = filtered_idx
                else:
                    filtered_idx = prompt_to_idx[idx]
                filtered_tasks.append((idx, filtered_idx, model_id))
    
    if not filtered_tasks:
        logger.info("All prompts already successfully completed. Nothing to process.")
        return out_path
    
    # Create per-model semaphores
    model_configs = model_configs or {}
    model_semaphores: dict[str, asyncio.Semaphore] = {}
    for model_id in assistant_models:
        model_config = model_configs.get(model_id, {})
        model_concurrency = model_config.get("concurrency", 3)
        if not isinstance(model_concurrency, int) or model_concurrency <= 0:
            raise ValueError(f"Model {model_id} concurrency must be a positive integer")
        model_semaphores[model_id] = asyncio.Semaphore(model_concurrency)
    
    file_lock = asyncio.Lock()
    total_tasks = len(filtered_tasks)
    logger.info(
        "",
        extra={
            "model_prefix": "[info] | setup",
            "status_label": "schedule",
            "details": (
                f"total={total_tasks} tasks "
                f"models={len(assistant_models)} prompts={len(filtered_prompts)}"
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
                filtered_prompts,
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

