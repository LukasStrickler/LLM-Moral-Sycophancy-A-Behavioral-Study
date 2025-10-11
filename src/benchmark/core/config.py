from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class OpenRouterConfig:
    api_key: str | None
    base_url: str
    model: str | None
    scorer_model: str | None
    # convenience default for testing if no model is set
    default_test_model: str = "openai/gpt-oss-20b:free"

    @staticmethod
    def from_env() -> OpenRouterConfig:
        return OpenRouterConfig(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_MODEL"),
            scorer_model=os.getenv("OPENROUTER_SCORER_MODEL"),
        )


@dataclass
class RunConfig:
    request_timeout_s: int = 60
    max_retries: int = 3

    @staticmethod
    def from_env() -> RunConfig:
        return RunConfig(
            request_timeout_s=_parse_int_env("BENCH_TIMEOUT_S", 60),
            max_retries=_parse_int_env("BENCH_MAX_RETRIES", 3),
        )
