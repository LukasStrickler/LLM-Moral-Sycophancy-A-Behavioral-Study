from __future__ import annotations

import re
import time
from typing import Any

import litellm
from litellm import ModelResponse as LiteLLMResponse

from ..core.config import ProviderConfig, RunConfig
from ..core.logging import (
    configure_litellm_logging,
    make_log_extra,
    normalize_provider_name,
    setup_logger,
)
from ..core.types import ChatMessage, ModelResponse, ProviderMetadata, message_dict

# Configure LiteLLM logging on import
configure_litellm_logging()

# Configure LiteLLM global settings (executed once at module import)
# These settings affect all LiteLLM calls and should be set at module level
# to avoid concurrency issues from per-instance configuration
litellm.drop_params = True  # Drop unsupported parameters
litellm.suppress_debug_info = True  # Reduce verbosity

MODEL_SHORT_WIDTH = 20
TAG_WIDTH = 9
SEP = " | "
# Default temperature for chat completions
DEFAULT_TEMPERATURE = 0.2


def _short_model_id(model_id: str) -> str:
    """Extract short model ID for display."""
    base = model_id.split("/", 1)[-1]
    return base.split(":", 1)[0]


TAG_LABEL_MAP = {
    "rate-limit": "limited",
    "provider-error": "error",
    "debug": "debug",
}

# Provider name mapping for display normalization
PROVIDER_EXTRACTION_MAP: dict[str, str] = {
    "openrouter": "openrouter",
    "gemini": "google_ai_studio",
    "groq": "groq",
    "huggingface": "huggingface",
    "cerebras": "cerebras",
    "mistral": "mistral",
    "cohere": "cohere",
}

# Groq model mappings
GROQ_MODEL_MAP: dict[str, str] = {
    "llama-3.3-70b": "groq/llama-3.3-70b-versatile",
    "llama-3-3-70b": "groq/llama-3.3-70b-versatile",
    "llama-3.1-8b": "groq/llama-3.1-8b-instant",
    "llama-3-1-8b": "groq/llama-3.1-8b-instant",
    "gpt-oss-20b": "groq/openai/gpt-oss-20b",
    "gpt-oss-120b": "groq/openai/gpt-oss-120b",
    "qwen-3-32b": "groq/qwen/qwen3-32b",
    "qwen3-32b": "groq/qwen/qwen3-32b",
}

# OpenRouter model prefixes (models that should route through OpenRouter)
OPENROUTER_MODEL_KEYWORDS = ["deepseek", "glm-4.5", "glm-4", "nemotron"]
OPENROUTER_LEGACY_PREFIXES = ["openai/", "meta-llama/"]


def _normalize_openrouter_prefix(model_id: str) -> str:
    """Ensure model_id has openrouter/ prefix if not already present."""
    return model_id if model_id.startswith("openrouter/") else f"openrouter/{model_id}"


def _check_model_variants(model_id_lower: str, variants: list[str]) -> bool:
    """Check if model_id contains any of the given variant strings."""
    return any(variant in model_id_lower for variant in variants)


def _normalize_model_variant(model_id_lower: str) -> str:
    """Normalize model ID variants to canonical form for lookup.

    Handles common variant spellings:
    - "llama-3-3-70b" -> "llama-3.3-70b"
    - "llama-3-1-70b" -> "llama-3.1-70b"
    - "llama-3-1-8b" -> "llama-3.1-8b"
    - "qwen3-32b" -> "qwen-3-32b"
    """
    # Normalize dash-separated version numbers (3-3 -> 3.3, 3-1 -> 3.1)
    import re

    # Replace patterns like "llama-3-3" or "llama-3-1" with "llama-3.3" or "llama-3.1"
    normalized = re.sub(r"llama-3-([38])-", r"llama-3.\1-", model_id_lower)
    # Replace "qwen3-" with "qwen-3-"
    normalized = re.sub(r"qwen3-", r"qwen-3-", normalized)
    return normalized


def _extract_choice_text(choice: Any) -> tuple[str, str | None]:
    """Extract text and finish_reason from a choice object.

    Handles both regular (has message attribute) and streaming choices.

    Returns:
        Tuple of (text, finish_reason)
    """
    if hasattr(choice, "message"):
        return (choice.message.content or "", choice.finish_reason)
    # Fallback for streaming choices or other formats
    text = getattr(choice, "content", "") or getattr(choice, "delta", {}).get("content", "") or ""
    return (text, getattr(choice, "finish_reason", None))


def _extract_usage_info(response: LiteLLMResponse) -> tuple[int | None, int | None, int | None]:
    """Extract token usage information from response.

    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_tokens)
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return (None, None, None)
    return (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )


def _extract_cost(response: LiteLLMResponse) -> float | None:
    """Extract cost information from response if available using LiteLLM's public API."""
    try:
        cost = litellm.completion_cost(response)
        if cost is not None:
            return float(cost)
    except Exception:
        # Fallback to None if cost extraction fails
        pass
    return None


def _serialize_raw_response(response: LiteLLMResponse) -> dict | None:
    """Convert response to dictionary for raw_response field."""
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "__dict__"):
            return response.__dict__
    except Exception:
        pass
    return None


def _tag(label: str) -> str:
    display = TAG_LABEL_MAP.get(label, label)
    return f"[{display}]".ljust(TAG_WIDTH)


def _map_model_id_to_litellm(model_id: str) -> str:
    """Map our internal model ID format to LiteLLM provider format.

    This function handles conversion from our standardized model IDs
    (e.g., "google/gemini-2.0-flash-exp") to LiteLLM's expected format
    (e.g., "gemini/gemini-2.0-flash-exp").

    Provider priority order (checked in sequence):
    1. Google AI Studio (gemini models) - highest priority
    2. Groq (llama, gpt-oss, qwen models)
    3. Hugging Face (phi, qwen-2.5 models)
    4. Cerebras (llama models)
    5. Mistral AI (mistral models)
    6. Cohere (command models)
    7. OpenRouter (deepseek, glm-4.5, nemotron, legacy openai/meta-llama formats)
    8. Default fallback to OpenRouter

    Examples:
        >>> _map_model_id_to_litellm("google/gemini-2.0-flash-exp")
        'gemini/gemini-2.0-flash-exp'
        >>> _map_model_id_to_litellm("groq/llama-3.3-70b")
        'groq/llama-3.3-70b-8192'
        >>> _map_model_id_to_litellm("openai/gpt-oss-20b:free")
        'openrouter/openai/gpt-oss-20b:free'

    Args:
        model_id: Our internal model ID format (e.g., "google/gemini-2.0-flash-exp")

    Returns:
        LiteLLM-compatible model ID string (e.g., "gemini/gemini-2.0-flash-exp")
    """
    model_id_lower = model_id.lower()

    # Google AI Studio models - check FIRST before generic google/ check
    if model_id.startswith("google/gemini"):
        model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
        return f"gemini/{model_name}"
    # Check if "gemini" appears as a path segment
    parts = model_id_lower.split("/")
    if "gemini" in parts:
        gemini_idx = parts.index("gemini")
        if gemini_idx == 0:
            return model_id
        return f"gemini/{'/'.join(parts[gemini_idx:])}"

    # Groq models - check if already in correct format first, then check mapping
    if "groq" in model_id_lower or ("llama-3" in model_id_lower and "groq" not in model_id_lower):
        # If already has groq/ prefix, pass through as-is (JSON has correct LiteLLM format)
        if model_id.startswith("groq/"):
            return model_id
        
        # Otherwise, try to map using GROQ_MODEL_MAP
        normalized_id = _normalize_model_variant(model_id_lower)
        for variant, litellm_id in GROQ_MODEL_MAP.items():
            if variant in normalized_id:
                return litellm_id

    # Hugging Face models
    if _check_model_variants(model_id_lower, ["phi", "qwen-2.5", "qwen2.5-72b"]):
        if "phi" in model_id_lower:
            return "huggingface/microsoft/Phi-3.5-mini-instruct"
        if "qwen-2.5-72b" in model_id_lower or "qwen2.5-72b" in model_id_lower:
            return "huggingface/Qwen/Qwen2.5-72B-Instruct"

    # Cerebras models
    if "cerebras" in model_id_lower and "llama-3.1-70b" in model_id_lower:
        return "cerebras/llama-3.1-70b-instruct"

    # Mistral and Cohere models - JSON already has correct LiteLLM format, pass through as-is
    if model_id.startswith("mistral/"):
        return model_id
    if model_id.startswith("cohere/"):
        return model_id

    # OpenRouter-specific models or default fallback - always normalize OpenRouter prefix
    return _normalize_openrouter_prefix(model_id)


def _extract_provider_from_model(model_id: str) -> str | None:
    """Extract provider name from LiteLLM model ID."""
    if "/" not in model_id:
        return None
    provider = model_id.split("/", 1)[0].lower()
    return PROVIDER_EXTRACTION_MAP.get(provider, provider)


def _has_api_key_for_model(provider_config: ProviderConfig, model_id: str) -> bool:
    """Check if an API key is available for the given model.

    Args:
        provider_config: Provider configuration with API keys
        model_id: Model ID in our internal format (e.g., "google/gemini-2.0-flash-exp")

    Returns:
        True if API key is available for this model, False otherwise
    """
    litellm_model = _map_model_id_to_litellm(model_id)
    provider = _extract_provider_from_model(litellm_model)
    if not provider:
        return False

    # Map provider names to ProviderConfig attributes
    provider_attr_map = {
        "google_ai_studio": "google_ai_api_key",
        "groq": "groq_api_key",
        "huggingface": "huggingface_api_key",
        "cerebras": "cerebras_api_key",
        "mistral": "mistral_api_key",
        "cohere": "cohere_api_key",
        "openrouter": "openrouter_api_key",
    }

    attr_name = provider_attr_map.get(provider)
    if attr_name:
        api_key = getattr(provider_config, attr_name, None)
        return api_key is not None and api_key.strip() != ""
    return False


class LiteLLMProvider:
    """LiteLLM-based provider supporting multiple LLM providers."""

    def __init__(
        self,
        cfg: ProviderConfig,
        run_cfg: RunConfig | None = None,
        model_configs: dict[str, dict] | None = None,
    ):
        self.cfg = cfg
        self.run_cfg = run_cfg or RunConfig()
        self.model_configs = model_configs or {}
        self.logger = setup_logger("provider")

    def _get_api_key_for_model(self, litellm_model: str) -> str | None:
        """Get the appropriate API key for a given LiteLLM model ID.

        Returns the API key from ProviderConfig that matches the provider
        for the given model, or None if no matching key is configured.

        Args:
            litellm_model: LiteLLM model ID (e.g., "gemini/gemini-2.0-flash-exp")

        Returns:
            API key string if available, None otherwise
        """
        provider = _extract_provider_from_model(litellm_model)
        if not provider:
            return None

        # Map provider names to ProviderConfig attributes
        provider_attr_map = {
            "google_ai_studio": "google_ai_api_key",
            "groq": "groq_api_key",
            "huggingface": "huggingface_api_key",
            "cerebras": "cerebras_api_key",
            "mistral": "mistral_api_key",
            "cohere": "cohere_api_key",
            "openrouter": "openrouter_api_key",
        }

        attr_name = provider_attr_map.get(provider)
        if attr_name:
            return getattr(self.cfg, attr_name, None)
        return None

    def _parse_litellm_response(
        self,
        response: LiteLLMResponse,
        model_id: str,
        latency_ms: int,
        provider_name: str | None,
    ) -> ProviderMetadata:
        """Parse LiteLLM response into ProviderMetadata.

        Extracts text, tokens, cost, and other metadata from a LiteLLM response.
        Handles both regular and streaming choice formats.

        Args:
            response: The LiteLLM ModelResponse object
            model_id: The original model ID used for the request
            latency_ms: Request latency in milliseconds
            provider_name: Provider name extracted from model ID

        Returns:
            ProviderMetadata with parsed response data

        Raises:
            RuntimeError: If response has no choices or invalid structure
        """
        # Validate choices exist
        if not response.choices or len(response.choices) == 0:
            raise RuntimeError(
                f"LiteLLM returned empty choices list for model {model_id}. "
                f"Response: {response}"
            )

        choice = response.choices[0]
        text, finish_reason = _extract_choice_text(choice)
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_info(response)
        cost_usd = _extract_cost(response)
        raw_response_dict = _serialize_raw_response(response)

        # Create ModelResponse
        model_response = ModelResponse(
            model_id=model_id,
            response_text=text.strip(),
            latency_ms=latency_ms,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            request_id=getattr(response, "id", None),
            finish_reason=finish_reason,
            raw_response=raw_response_dict,
        )

        # Normalize provider name for display
        display_provider = normalize_provider_name(provider_name)

        return ProviderMetadata(
            provider_name=display_provider or provider_name,
            model_response=model_response,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            raw_response=model_response.raw_response,
        )

    async def chat_async(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        *,
        grid_id: str | None = None,
        task_id: str | None = None,
        progress: str | None = None,
    ) -> ProviderMetadata:
        """Send chat messages via LiteLLM and return provider metadata.

        This method sends a chat completion request to LiteLLM, which automatically
        routes to the appropriate provider based on the model ID. The method handles
        model ID mapping, response parsing, error handling, and metadata extraction.

        Args:
            messages: List of chat messages to send. Must include at least one message
                with role 'user' or 'assistant'.
            model: Optional model ID to use. If not provided, uses the model from
                ProviderConfig.model or ProviderConfig.default_test_model.
            grid_id: Optional grid identifier for logging and tracking. Used in
                benchmark runs to identify which prompt configuration this request
                belongs to.
            task_id: Optional task identifier for logging and tracking. Used in
                benchmark runs to identify specific tasks within a grid.
            progress: Optional progress indicator string for logging. Shows current
                progress in benchmark runs (e.g., "12/50").

        Returns:
            ProviderMetadata containing:
            - provider_name: Normalized provider name (e.g., "GoogleAI", "Groq")
            - model_response: ModelResponse with text, tokens, cost, latency
            - input_tokens: Number of prompt tokens used
            - output_tokens: Number of completion tokens generated
            - total_tokens: Total tokens used
            - cost_usd: Estimated cost in USD (if available)
            - raw_response: Raw response dictionary from LiteLLM

        Raises:
            RuntimeError: If no provider API key is configured, or if LiteLLM
                completion fails, or if the response has no choices.
            ValueError: If no model ID is provided and no default is configured.

        Example:
            >>> provider = LiteLLMProvider(ProviderConfig.from_env())
            >>> messages = [ChatMessage(role="user", content="Hello")]
            >>> metadata = await provider.chat_async(messages, model="google/gemini-2.0-flash-exp")
            >>> print(metadata.model_response.response_text)
        """
        if not self.cfg.has_any_api_key():
            raise RuntimeError("No provider API key configured. Set at least one provider API key.")

        model_id = model or self.cfg.model or self.cfg.default_test_model
        if not model_id:
            raise ValueError("No model id provided and no default configured")

        # Map model ID to LiteLLM format
        litellm_model = _map_model_id_to_litellm(model_id)
        provider_name = _extract_provider_from_model(litellm_model)

        # Prepare messages for LiteLLM
        litellm_messages = [message_dict(m) for m in messages]

        # Get API key for this specific model/provider
        api_key = self._get_api_key_for_model(litellm_model)

        # Validate API key is available before making the request
        if not api_key:
            provider_display = normalize_provider_name(provider_name) or provider_name or "unknown"
            error_msg = (
                f"No API key configured for provider '{provider_display}' (model: {model_id}). "
                f"Please set the appropriate API key environment variable (e.g., GOOGLE_AI_API_KEY, "
                f"GROQ_API_KEY, etc.) or configure it in your .env file."
            )
            self.logger.warning(
                "",
                extra=make_log_extra(
                    model=model_id,
                    grid=grid_id,
                    task=task_id,
                    progress=progress,
                    tag="warning",
                    status="missing-api-key",
                    details=(f"provider={provider_display}",),
                ),
            )
            raise RuntimeError(error_msg)

        # Call LiteLLM
        t0 = time.time()
        try:
            # Pass API key directly to acompletion() to avoid mutating os.environ
            # This prevents concurrency issues when multiple provider instances exist
            completion_kwargs = {
                "model": litellm_model,
                "messages": litellm_messages,
                "temperature": DEFAULT_TEMPERATURE,
                "timeout": self.run_cfg.request_timeout_s,
                "num_retries": self.run_cfg.max_retries,
            }
            if api_key:
                completion_kwargs["api_key"] = api_key

            response: LiteLLMResponse = await litellm.acompletion(**completion_kwargs)
        except (
            litellm.exceptions.APIError,
            litellm.exceptions.RateLimitError,
            Exception,
        ) as e:
            # Catch specific LiteLLM exceptions plus generic Exception as fallback
            # for any unexpected errors (network issues, timeouts, etc.)
            # Preserve the original exception to maintain access to response headers
            # (RateLimitError has a 'response' attribute with httpx.Response containing headers)
            error_msg = str(e)
            
            # For rate limit errors that we can handle, suppress verbose logging
            # Check multiple exception types and error message patterns
            is_rate_limit = isinstance(e, litellm.exceptions.RateLimitError)
            can_handle_rate_limit = False
            
            # Also check for BadRequestError with 429 status code (Gemini uses this)
            if not is_rate_limit and isinstance(e, litellm.exceptions.BadRequestError):
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota exceeded" in error_msg.lower():
                    is_rate_limit = True
            
            # Also check APIConnectionError for rate limit messages (e.g., Cohere trial keys)
            if not is_rate_limit and isinstance(e, litellm.exceptions.APIConnectionError):
                if ("limited to" in error_msg.lower() 
                    or "rate limit" in error_msg.lower()
                    or "api calls / minute" in error_msg.lower()
                    or "requests per minute" in error_msg.lower()):
                    is_rate_limit = True
            
            if is_rate_limit:
                # Check if we can extract retry delay from headers or error message
                http_response = getattr(e, "response", None)
                if http_response and hasattr(http_response, "headers"):
                    headers = http_response.headers
                    can_handle_rate_limit = bool(
                        headers.get("retry-after")
                        or headers.get("x-ratelimit-reset")
                        or headers.get("x-ratelimit-reset-requests")
                    )
                
                # Also check if error message contains parseable retry delay
                if not can_handle_rate_limit:
                    can_handle_rate_limit = bool(
                        re.search(r'Please retry in ([\d.]+)s', error_msg, re.IGNORECASE)
                        or re.search(r'"retryDelay"\s*:\s*"(\d+)s"', error_msg, re.IGNORECASE)
                        or "per minute" in error_msg.lower()
                        or "api calls / minute" in error_msg.lower()
                        or "requests per minute" in error_msg.lower()
                    )
            
            if is_rate_limit and can_handle_rate_limit:
                # Don't log here - ai_label.py handles rate limit logging with retry details
                # This prevents duplicate logs
                pass
            elif is_rate_limit:
                # Rate limit but can't handle - still suppress verbose logging
                # ai_label.py will handle the final error logging
                pass
            else:
                # Log full error for unhandled errors or non-rate-limit errors
                self.logger.error(
                    "",
                    extra=make_log_extra(
                        model=model_id,
                        grid=grid_id,
                        task=task_id,
                        progress=progress,
                        tag="error",
                        status="provider-error",
                        details=(f"error={error_msg}",),
                    ),
                )
            # Re-raise the original exception to preserve response headers
            # Callers can access e.response.headers for retry-after information
            raise

        dt_ms = int((time.time() - t0) * 1000)

        # Parse response and return metadata
        return self._parse_litellm_response(response, model_id, dt_ms, provider_name)

    def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        *,
        grid_id: str | None = None,
        task_id: str | None = None,
        progress: str | None = None,
    ) -> ProviderMetadata:
        """Synchronously call :meth:`chat_async` using ``asyncio.run``.

        This helper must not be invoked from within an active event loop; callers in
        async contexts should await :meth:`chat_async` directly.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.chat_async(
                    messages, model=model, grid_id=grid_id, task_id=task_id, progress=progress
                )
            )
        raise RuntimeError(
            "chat() cannot be used inside a running event loop; call chat_async instead"
        )

    async def aclose(self) -> None:
        """Cleanup method for async context manager compatibility.
        
        LiteLLMProvider is stateless and doesn't require cleanup,
        but this method exists for consistency with async patterns.
        """
        pass
