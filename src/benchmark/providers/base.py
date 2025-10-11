from __future__ import annotations

from typing import Protocol

from ..core.types import ChatMessage


class ChatProvider(Protocol):
    def chat(self, messages: list[ChatMessage], model: str | None = None) -> tuple[str, int | None]:
        """
        Send chat messages via the provider and return a tuple ``(response_text, latency_ms)``.

        Implementations should:

        - raise ``ValueError`` when ``messages`` is empty, since most APIs require at least
          one user message.
        - surface network, authentication, rate-limit, and model-availability issues via
          provider-specific exceptions so callers can handle them (e.g., propagate
          ``httpx.HTTPError`` subclasses or custom errors).
        - document whether the operation is idempotent; providers are expected to avoid
          destructive side effects and may retry transient failures internally.
        - accept ``model`` values supported by the provider and raise a descriptive error for
          invalid or unavailable identifiers.

        ``latency_ms`` may be ``None`` when the provider cannot supply timing metrics.
        """
        ...
