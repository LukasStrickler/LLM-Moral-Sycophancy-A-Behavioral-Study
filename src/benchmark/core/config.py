from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class ProviderConfig:
    """Configuration for multi-provider LLM access via LiteLLM."""

    # Provider API keys
    google_ai_api_key: str | None = None
    groq_api_key: str | None = None
    huggingface_api_key: str | None = None
    cerebras_api_key: str | None = None
    mistral_api_key: str | None = None
    cohere_api_key: str | None = None
    openrouter_api_key: str | None = None

    # Model selection
    model: str | None = None
    scorer_model: str | None = None
    default_test_model: str = "google/gemini-2.0-flash-exp"

    @staticmethod
    def from_env() -> ProviderConfig:
        return ProviderConfig(
            google_ai_api_key=os.getenv("GOOGLE_AI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
            cerebras_api_key=os.getenv("CEREBRAS_API_KEY"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("LLM_MODEL"),
            scorer_model=os.getenv("LLM_SCORER_MODEL"),
        )

    def has_any_api_key(self) -> bool:
        """Check if any provider API key is configured."""
        return any(
            [
                self.google_ai_api_key,
                self.groq_api_key,
                self.huggingface_api_key,
                self.cerebras_api_key,
                self.mistral_api_key,
                self.cohere_api_key,
                self.openrouter_api_key,
            ]
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
