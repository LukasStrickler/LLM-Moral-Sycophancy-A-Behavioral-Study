from __future__ import annotations

import asyncio
import json
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
from ..core.types import Factors, ProviderMetadata, RunRecord
from ..prompts.chat import build_chat_from_factors
from ..prompts.generator import generate_factor_grid
from ..providers.litellm_provider import LiteLLMProvider

logger = setup_logger("run")

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
) -> None:
    async with sem:
        t_task0 = time.time()
        factor = factors[task.index]
        prompt = build_chat_from_factors(factor)
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

        try:
            provider_metadata: ProviderMetadata | None = None
            provider_label: str | None = None
            latency_ms: int | None = None
            answer_text: str
            input_tokens: int | None = None
            output_tokens: int | None = None
            total_tokens: int | None = None
            cost_usd: float | None = None
            request_id: str | None = None
            finish_reason: str | None = None
            raw_response: dict[str, Any] | None = None

            if dry_run:
                provider_label = provider_label or "dry-run"
                answer_text = f"[dry-run] {task.model_id} response to prompt {prompt.prompt_id}"
                latency_ms = 0
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

            total_ms = int((time.time() - t_task0) * 1000)
            record = RunRecord.success(
                run_id=run_id,
                timestamp_iso=timestamp_iso,
                model_id=task.model_id,
                provider=provider_label,
                prompt_id=f"{task.index:03d}",  # Row number as prompt_id
                perspective=factor.perspective,
                factors=factor.to_payload(),
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
        except Exception as exc:
            status_prefix = _prefix("error", task.model_id, task.index, task_id, progress_str)
            logger.exception(
                "Prompt worker failure",
                extra={
                    "model_prefix": status_prefix,
                    "status_label": "exception",
                    "details": str(exc),
                },
            )
            total_ms = int((time.time() - t_task0) * 1000)
            record = RunRecord.failure(
                run_id=run_id,
                timestamp_iso=timestamp_iso,
                model_id=task.model_id,
                provider=provider_label or "unknown",
                prompt_id=f"{task.index:03d}",  # Row number as prompt_id
                perspective=getattr(factor, "perspective", None),
                factors=factor.to_payload() if hasattr(factor, "to_payload") else None,
                messages=getattr(prompt, "messages", None),
                error=str(exc),
            )

        line = record.to_json_line()
        async with file_lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        logger.debug(
            "",
            extra=make_log_extra(
                model=task.model_id,
                grid=f"{task.index:03d}",
                task=f"{task_id:03d}",
                progress=None,
                tag="debug",
                status="write",
                details=(f"path={out_path}", f"size={len(line) + 1}B"),
            ),
        )

        wall_ms = total_ms
        done = await progress.mark_done()
        progress_value = _progress_str(done, progress.total)
        if record.is_success():
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
            error_metrics = (
                _fmt_seconds("wall", wall_ms, 6),
                _fmt("reason", record.error, 48),
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
) -> Path:
    """Run the benchmark asynchronously and persist results to ``out_path``."""

    cfg = ProviderConfig.from_env()
    run_cfg = RunConfig.from_env()
    client: LiteLLMProvider | None = None
    if not dry_run:
        client = LiteLLMProvider(cfg, run_cfg, model_configs)
    run_id = int(time.time())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")  # truncate file for fresh run

    factors = (
        list(factors_list_override)
        if factors_list_override is not None
        else generate_factor_grid()
    )
    selected = factors[:limit] if limit < len(factors) else factors

    grid_snapshot = {
        "total": len(factors),
        "selected": len(selected),
        "factors": [factor.to_payload() for factor in selected],
        "assistant_models": assistant_models,
        "mode": "dry-run" if dry_run else "live",
    }
    grid_path = out_path.with_name(out_path.stem + "_grid.json")
    grid_path.write_text(json.dumps(grid_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

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
    total_tasks = len(assistant_models) * len(selected)
    logger.info(
        "",
        extra={
            "model_prefix": "[info] | setup",
            "status_label": "schedule",
            "details": (
                f"total={total_tasks} models={len(assistant_models)} prompts={len(selected)} "
                f"per-model-concurrency"
            ),
        },
    )
    progress = ProgressTracker(total_tasks)
    global_task_id = 1
    tasks: list[asyncio.Task[None]] = []
    for model_id in assistant_models:
        for idx in range(len(selected)):
            task = asyncio.create_task(
                _worker(
                    model_semaphores[model_id],
                    client,
                    out_path,
                    file_lock,
                    selected,
                    TaskSpec(model_id=model_id, index=idx),
                    run_id,
                    dry_run,
                    progress,
                    global_task_id,
                )
            )
            tasks.append(task)
            global_task_id += 1

    await asyncio.gather(*tasks)
    if client is not None:
        await client.aclose()
    return out_path
