from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path

_LOG_FILE_PATH: Path | None = None
_REGISTERED_LOGGERS: dict[str, logging.Logger] = {}
_LOCK = threading.Lock()


def _remove_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(log_file: Path | None) -> None:
    """Set a global log file path and retrofit existing loggers with it."""

    global _LOG_FILE_PATH
    with _LOCK:
        _LOG_FILE_PATH = log_file

        for logger in _REGISTERED_LOGGERS.values():
            _remove_file_handlers(logger)
            if log_file is not None:
                _add_file_handler(logger, log_file)


LEVEL_COL_WIDTH = 7
LOGGER_COL_WIDTH = 8

DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "__main__": "main",
    "runner": "run",
    "labeler": "Labeler",
    "planer": "Planer",
    "provider": "prov",
    "LLM-Labeler": "LLMLb",
    "labeling-config": "LblCfg",
    "provider:OO": "OpRo",
    "OO": "OpRo",
    "run": "run",
    "main": "main",
}

# Provider name normalization for display
PROVIDER_NAME_MAP: dict[str, str] = {
    "openrouter": "OR",
    "google_ai_studio": "GoogleAI",
    "gemini": "GoogleAI",
    "groq": "Groq",
    "huggingface": "HF",
    "hugging_face": "HF",
    "cerebras": "Cerebras",
    "mistral": "Mistral",
    "cohere": "Cohere",
}


def normalize_provider_name(provider: str | None) -> str:
    """Normalize provider name for consistent display."""
    if not provider:
        return ""
    provider_lower = provider.lower().replace("-", "_").replace(" ", "_")
    return PROVIDER_NAME_MAP.get(provider_lower, provider[:10].title())


MODEL_PREFIX_WIDTH = 30
PROGRESS_PREFIX_WIDTH = 7


def _format_opt(label: str, value: str | None, width: int) -> str:
    text = (value or "").strip()
    if not text:
        return "".ljust(width)
    combined = f"{label}{text}"
    return combined[:width].ljust(width)


PROGRESS_COLUMN_WIDTH = PROGRESS_PREFIX_WIDTH


TAG_COL_WIDTH = 6
STATUS_COL_WIDTH = 11
LOG_LEVEL_MAP = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "critical",
}


def _shorten_model(value: str | None) -> str:
    if not value:
        return ""
    base = value.split("/", 1)[-1]
    base = base.split(":", 1)[0]
    return base[:MODEL_PREFIX_WIDTH]


def make_log_extra(
    *,
    model: str | None = None,
    grid: str | None = None,
    task: str | None = None,
    progress: str | None = None,
    tag: str | None = None,
    status: str | None = None,
    details: str | Iterable[str] | None = None,
) -> dict[str, object]:
    """Build a structured logging payload for the custom formatter."""

    return {
        "model_id": _shorten_model(model),
        "grid_id": grid,
        "task_id": task,
        "progress_state": progress,
        "tag_label": tag,
        "status_label": status,
        "details": details,
    }


def _normalise_cell(value: str | None, *, width: int, pad: str = "0") -> str:
    if value in (None, ""):
        return "".ljust(width)
    text = str(value)
    if text.isdigit() and len(text) < width:
        return text.zfill(width)
    trimmed = text[:width]
    return trimmed.ljust(width)


def _coerce_progress(value: str | None) -> str:
    if not value:
        return "".ljust(PROGRESS_COLUMN_WIDTH)
    if "/" not in value:
        return value[:PROGRESS_COLUMN_WIDTH].ljust(PROGRESS_COLUMN_WIDTH)
    current, total = value.split("/", 1)
    total_width = max(len(total), 3)
    current = current.zfill(total_width)
    total = total.zfill(total_width)
    formatted = f"{current}/{total}"
    return formatted[:PROGRESS_COLUMN_WIDTH].ljust(PROGRESS_COLUMN_WIDTH)


def _coerce_details(details: str | Iterable[str] | None) -> str:
    if details is None:
        return ""
    if isinstance(details, str):
        return details.strip()
    joined = "  ".join(str(part).strip() for part in details if part)
    return joined.strip()


TAG_DISPLAY_MAP: dict[str, str] = {
    "warning": "warn",
    "warn": "warn",
    "retry": "retry",
    "limited": "limit",
    "limit": "limit",
    "quota": "quota",
    "error": "error",
    "info": "info",
    "debug": "debug",
}

STATUS_DISPLAY_MAP: dict[str, str] = {
    "found-responses": "found",
    "insert-failed": "db-fail",
    "model-completed": "done",
    "model-summary": "summary",
    "request-done": "req-done",
    "missing-api-key": "no-key",
    "provider-error": "prov-err",
    "daily-quota-exceeded": "quota",
    "rate-limit-retry": "retry",
    "dryrun": "dry-run",
    "giveup": "giveup",
    "progress": "progress",
    "start": "start",
    "scored": "scored",
    "none": "no-data",
    "limited": "limited",
    "retry": "retry",
    "quota": "quota",
    "nonretry": "no-retry",
    "plan": "plan",
    "OPEN": "open",
    "DONE": "done",
}


def _normalise_tag(level_name: str, tag_label: str | None) -> str:
    base = tag_label or LOG_LEVEL_MAP.get(level_name, level_name.lower())
    if not base:
        base = level_name.lower()
    display = TAG_DISPLAY_MAP.get(base.lower(), base.lower())
    return display[:TAG_COL_WIDTH].ljust(TAG_COL_WIDTH)


def _normalise_status(status_label: str | None) -> str:
    if not status_label:
        return "".ljust(STATUS_COL_WIDTH)
    text = STATUS_DISPLAY_MAP.get(status_label, status_label)
    if len(text) <= STATUS_COL_WIDTH:
        return text.ljust(STATUS_COL_WIDTH)
    compact = text.replace("-", "")
    if len(compact) <= STATUS_COL_WIDTH:
        return compact.ljust(STATUS_COL_WIDTH)
    return text[:STATUS_COL_WIDTH].ljust(STATUS_COL_WIDTH)


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        display_name = DISPLAY_NAME_OVERRIDES.get(record.name, record.name)

        model_raw = getattr(record, "model_id", None)
        grid_raw = getattr(record, "grid_id", None)
        task_raw = getattr(record, "task_id", None)
        progress_raw = getattr(record, "progress_state", None)

        grid_val = _normalise_cell(grid_raw, width=3).strip()
        task_val = _normalise_cell(task_raw, width=3).strip()
        progress_val = _coerce_progress(progress_raw).strip()

        model = _normalise_cell(model_raw, width=MODEL_PREFIX_WIDTH, pad=" ")
        grid = _format_opt("G=", grid_val, 5)
        task = _format_opt("T=", task_val, 5)
        progress = _format_opt("P=", progress_val, PROGRESS_COLUMN_WIDTH + 2)

        level_name = record.levelname.upper()
        origin = display_name[:LOGGER_COL_WIDTH].ljust(LOGGER_COL_WIDTH)
        action_label = getattr(record, "status_label", None)
        tag_label = getattr(record, "tag_label", None)

        tag_text = _normalise_tag(level_name, tag_label)
        status_text = _normalise_status(action_label)

        details_text = _coerce_details(getattr(record, "details", record.message))

        time_part = f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"
        left_part = (
            f"{time_part} | {progress}" f" | {model:<{MODEL_PREFIX_WIDTH}}" f" | {grid} | {task}"
        )

        middle_part = f"{tag_text} | {origin} | {status_text}"

        if details_text:
            right_part = f"[{details_text}]"
            return f"{left_part} : {middle_part} : {right_part}"
        return f"{left_part} : {middle_part}"


def _create_formatter() -> logging.Formatter:
    return _StructuredFormatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _add_stream_handler(logger: logging.Logger) -> None:
    if any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        return
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_create_formatter())
    logger.addHandler(stream_handler)


def _add_file_handler(logger: logging.Logger, log_file: Path) -> None:
    target_path = log_file.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(getattr(handler, "baseFilename", "")).resolve()
            if handler_path == target_path:
                return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setFormatter(_create_formatter())
    logger.addHandler(file_handler)


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a logger configured with stream and optional file handlers."""

    level_name_raw = os.getenv("LOG_LEVEL", "INFO")
    level_name = level_name_raw.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        logging.warning("Invalid LOG_LEVEL '%s'; defaulting to INFO.", level_name_raw)
        level = logging.INFO

    logger = logging.getLogger(name)

    with _LOCK:
        logger.setLevel(level)
        _add_stream_handler(logger)

        final_log_path = log_file or _LOG_FILE_PATH
        if final_log_path is not None:
            _add_file_handler(logger, final_log_path)

        _REGISTERED_LOGGERS.setdefault(name, logger)

    return logger


def configure_litellm_logging() -> None:
    """Configure LiteLLM's internal logging to respect our LOG_LEVEL."""
    try:
        import litellm
    except ImportError:
        return  # LiteLLM not installed yet

    level_name_raw = os.getenv("LOG_LEVEL", "INFO")
    level_name = level_name_raw.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    # Configure LiteLLM to use our logging level
    litellm.set_verbose = level <= logging.DEBUG
    # Suppress LiteLLM's internal loggers unless we're in DEBUG mode
    if level > logging.DEBUG:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
