from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    role: MessageRole
    content: str


@dataclass
class Factors:
    perspective: str  # 'neutral' | 'landlord' | 'tenant'
    base_rent: int
    amount: int
    relationship_quality: str  # 'good' | 'poor'
    justification: str | None = None
    tenant_quality: str | None = None  # landlord view
    landlord_quality: str | None = None  # tenant view

    def to_payload(self) -> dict[str, Any]:
        d = asdict(self)
        # tool payload should exclude None
        return {k: v for k, v in d.items() if v is not None}


def make_prompt_id(factors: Factors) -> str:
    # Stable hash from canonical JSON of sorted keys
    s = json.dumps(factors.to_payload(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class PromptInstance:
    prompt_id: str
    factors: Factors
    messages: list[ChatMessage]


@dataclass
class ModelResponse:
    model_id: str
    response_text: str
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ProviderMetadata:
    provider_name: str | None
    model_response: ModelResponse
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    raw_response: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_response": asdict(self.model_response),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "raw_response": self.raw_response,
        }


def message_dict(m: ChatMessage) -> dict[str, Any]:
    return {"role": m.role, "content": m.content}


@dataclass
class RunRecord:
    """Canonical representation of a single benchmark result row."""

    run_id: int
    timestamp_iso: str
    model_id: str
    provider: str | None
    prompt_id: str | None
    perspective: str | None
    factors: dict[str, Any] | None
    messages: list[dict[str, Any]] | None
    response_text: str
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    request_id: str | None
    finish_reason: str | None
    raw_response: dict[str, Any] | None
    error: str | None

    @staticmethod
    def _normalise_messages(
        messages: Sequence[ChatMessage | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalised: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                normalised.append(message_dict(message))
            else:
                filtered = {
                    key: value
                    for key, value in message.items()
                    if key in {"role", "content"} and value is not None
                }
                normalised.append(filtered)
        return normalised

    @classmethod
    def success(
        cls,
        *,
        run_id: int,
        timestamp_iso: str,
        model_id: str,
        provider: str | None,
        prompt_id: str,
        perspective: str,
        factors: dict[str, Any],
        messages: Sequence[ChatMessage | dict[str, Any]],
        response_text: str,
        latency_ms: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        cost_usd: float | None,
        request_id: str | None,
        finish_reason: str | None,
        raw_response: dict[str, Any] | None,
    ) -> RunRecord:
        return cls(
            run_id=run_id,
            timestamp_iso=timestamp_iso,
            model_id=model_id,
            provider=provider,
            prompt_id=prompt_id,
            perspective=perspective,
            factors=factors,
            messages=cls._normalise_messages(messages),
            response_text=response_text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
            finish_reason=finish_reason,
            raw_response=raw_response,
            error=None,
        )

    @classmethod
    def failure(
        cls,
        *,
        run_id: int,
        timestamp_iso: str,
        model_id: str,
        provider: str,
        prompt_id: str | None,
        perspective: str | None,
        factors: dict[str, Any] | None,
        messages: Sequence[ChatMessage | dict[str, Any]] | None,
        error: str,
    ) -> RunRecord:
        payload_messages: list[dict[str, Any]] | None = None
        if messages is not None:
            payload_messages = cls._normalise_messages(messages)
        return cls(
            run_id=run_id,
            timestamp_iso=timestamp_iso,
            model_id=model_id,
            provider=provider,
            prompt_id=prompt_id,
            perspective=perspective,
            factors=factors,
            messages=payload_messages,
            response_text="",
            latency_ms=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=None,
            request_id=None,
            finish_reason=None,
            raw_response=None,
            error=error,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp_iso": self.timestamp_iso,
            "model_id": self.model_id,
            "provider": self.provider,
            "prompt_id": self.prompt_id,
            "perspective": self.perspective,
            "factors": self.factors,
            "messages": self.messages,
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "raw_response": self.raw_response,
            "error": self.error,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RunRecord:
        return cls(
            run_id=int(payload.get("run_id", 0)),
            timestamp_iso=str(payload.get("timestamp_iso", "")),
            model_id=str(payload.get("model_id", "unknown")),
            provider=payload.get("provider"),
            prompt_id=payload.get("prompt_id"),
            perspective=payload.get("perspective"),
            factors=payload.get("factors"),
            messages=payload.get("messages"),
            response_text=str(payload.get("response_text", "")),
            latency_ms=payload.get("latency_ms"),
            input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"),
            total_tokens=payload.get("total_tokens"),
            cost_usd=payload.get("cost_usd"),
            request_id=payload.get("request_id"),
            finish_reason=payload.get("finish_reason"),
            raw_response=payload.get("raw_response"),
            error=payload.get("error"),
        )

    @classmethod
    def from_json_line(cls, line: str) -> RunRecord:
        return cls.from_payload(json.loads(line))

    @classmethod
    def iter_jsonl(cls, path: Path) -> Iterator[RunRecord]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield cls.from_json_line(line)
                except json.JSONDecodeError:
                    logging.getLogger(__name__).warning(
                        "Skipping malformed JSON line %s in %s", line_number, path
                    )
                    continue

    def is_success(self) -> bool:
        return self.error in (None, "")


@dataclass
class EvalRecord:
    """Evaluation output capturing scores produced after a benchmark run."""

    run_id: int
    run_timestamp_iso: str | None
    eval_timestamp_iso: str
    model_id: str
    provider: str | None
    prompt_id: str | None
    perspective: str | None
    factors: dict[str, Any] | None
    messages: list[dict[str, Any]] | None
    response_text: str
    scorer_model: str | None
    score: float | None
    eval_latency_ms: int | None
    eval_provider: str | None
    eval_input_tokens: int | None
    eval_output_tokens: int | None
    eval_total_tokens: int | None
    eval_cost_usd: float | None
    eval_request_id: str | None
    eval_finish_reason: str | None
    eval_raw_response: dict[str, Any] | None
    error: str | None

    @classmethod
    def from_run_record(
        cls,
        run_record: RunRecord,
        *,
        eval_timestamp_iso: str,
        scorer_model: str | None,
        score: float | None,
        eval_latency_ms: int | None,
        provider_metadata: ProviderMetadata | None,
        error: str | None,
    ) -> EvalRecord:
        return cls(
            run_id=run_record.run_id,
            run_timestamp_iso=run_record.timestamp_iso,
            eval_timestamp_iso=eval_timestamp_iso,
            model_id=run_record.model_id,
            provider=run_record.provider,
            prompt_id=run_record.prompt_id,
            perspective=run_record.perspective,
            factors=run_record.factors,
            messages=run_record.messages,
            response_text=run_record.response_text,
            scorer_model=scorer_model,
            score=score,
            eval_latency_ms=eval_latency_ms,
            eval_provider=provider_metadata.provider_name if provider_metadata else None,
            eval_input_tokens=provider_metadata.input_tokens if provider_metadata else None,
            eval_output_tokens=provider_metadata.output_tokens if provider_metadata else None,
            eval_total_tokens=provider_metadata.total_tokens if provider_metadata else None,
            eval_cost_usd=provider_metadata.cost_usd if provider_metadata else None,
            eval_request_id=(
                provider_metadata.model_response.request_id if provider_metadata else None
            ),
            eval_finish_reason=(
                provider_metadata.model_response.finish_reason if provider_metadata else None
            ),
            eval_raw_response=provider_metadata.raw_response if provider_metadata else None,
            error=error,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_timestamp_iso": self.run_timestamp_iso,
            "eval_timestamp_iso": self.eval_timestamp_iso,
            "model_id": self.model_id,
            "provider": self.provider,
            "prompt_id": self.prompt_id,
            "perspective": self.perspective,
            "factors": self.factors,
            "messages": self.messages,
            "response_text": self.response_text,
            "scorer_model": self.scorer_model,
            "score": self.score,
            "eval_latency_ms": self.eval_latency_ms,
            "eval_provider": self.eval_provider,
            "eval_input_tokens": self.eval_input_tokens,
            "eval_output_tokens": self.eval_output_tokens,
            "eval_total_tokens": self.eval_total_tokens,
            "eval_cost_usd": self.eval_cost_usd,
            "eval_request_id": self.eval_request_id,
            "eval_finish_reason": self.eval_finish_reason,
            "eval_raw_response": self.eval_raw_response,
            "error": self.error,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EvalRecord:
        return cls(
            run_id=int(payload.get("run_id", 0)),
            run_timestamp_iso=payload.get("run_timestamp_iso"),
            eval_timestamp_iso=str(payload.get("eval_timestamp_iso", "")),
            model_id=str(payload.get("model_id", "unknown")),
            provider=payload.get("provider"),
            prompt_id=payload.get("prompt_id"),
            perspective=payload.get("perspective"),
            factors=payload.get("factors"),
            messages=payload.get("messages"),
            response_text=str(payload.get("response_text", "")),
            scorer_model=payload.get("scorer_model"),
            score=payload.get("score"),
            eval_latency_ms=payload.get("eval_latency_ms"),
            eval_provider=payload.get("eval_provider"),
            eval_input_tokens=payload.get("eval_input_tokens"),
            eval_output_tokens=payload.get("eval_output_tokens"),
            eval_total_tokens=payload.get("eval_total_tokens"),
            eval_cost_usd=payload.get("eval_cost_usd"),
            eval_request_id=payload.get("eval_request_id"),
            eval_finish_reason=payload.get("eval_finish_reason"),
            eval_raw_response=payload.get("eval_raw_response"),
            error=payload.get("error"),
        )

    @classmethod
    def from_json_line(cls, line: str) -> EvalRecord:
        return cls.from_payload(json.loads(line))

    @classmethod
    def iter_jsonl(cls, path: Path) -> Iterator[EvalRecord]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield cls.from_json_line(line)
                except json.JSONDecodeError:
                    logging.getLogger(__name__).warning(
                        "Skipping malformed JSON line %s in %s", line_number, path
                    )
                    continue
