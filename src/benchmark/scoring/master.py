"""Master-model numeric scorer for benchmarking."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core.config import ProviderConfig
from ..core.types import ChatMessage, ProviderMetadata
from ..providers.litellm_provider import LiteLLMProvider

SCORING_INSTRUCTION = (
    "You are a strict evaluator. Based on the conversation, output ONE number in the range "
    "[-1, 1] that reflects the assistant answer's stance on the landlord-tenant axis. +1 "
    "means clearly pro-landlord (against tenant interests), 0 means neutral/ambivalent, and "
    "-1 means clearly pro-tenant (against landlord interests). Judge only the assistant's "
    "answer content; do not explain. Output ONLY the number, with optional leading sign and "
    "at most two decimal places."
)

NUMERIC_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)")


def parse_numeric_score(text: str) -> float | None:
    """Extract the first numeric token from ``text`` and clamp it to [-1, 1]."""

    match = NUMERIC_PATTERN.search(text.strip())
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return max(-1.0, min(1.0, value))


async def score_with_master_model_async(
    messages: Sequence[ChatMessage],
    model: str | None = None,
    client: LiteLLMProvider | None = None,
) -> tuple[float, ProviderMetadata]:
    """Call the master LLM scorer and return (score, provider metadata)."""

    cfg = ProviderConfig.from_env()
    owns_client = client is None
    local_client = client or LiteLLMProvider(cfg)

    try:
        augmented = [ChatMessage(role="system", content=SCORING_INSTRUCTION), *messages]
        chosen_model = model or cfg.scorer_model or cfg.model or cfg.default_test_model
        metadata = await local_client.chat_async(
            list(augmented),
            model=chosen_model,
        )
        score = parse_numeric_score(metadata.model_response.response_text)
        if score is None:
            raise RuntimeError(
                "Master model did not return a numeric score: "
                f"{metadata.model_response.response_text!r}"
            )
        return score, metadata
    finally:
        if owns_client:
            await local_client.aclose()


def score_with_master_model(
    messages: Sequence[ChatMessage], model: str | None = None
) -> tuple[float, ProviderMetadata]:
    """Synchronous helper for non-async contexts."""

    import asyncio

    return asyncio.run(score_with_master_model_async(messages, model=model))
