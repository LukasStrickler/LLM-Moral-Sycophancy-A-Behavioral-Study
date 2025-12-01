"""OpenRouter client wrapper using direct REST API calls."""

from __future__ import annotations

from typing import Any, TypedDict

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx not installed. Install with: pip install httpx"
    )

from ..core.logging import setup_logger

logger = setup_logger("paid-client")


class ResponseUsage(TypedDict, total=False):
    """Usage information from OpenRouter response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class ResponseChoice(TypedDict, total=False):
    """Choice from OpenRouter response."""

    message: dict[str, str]
    finish_reason: str | None


class ResponseDict(TypedDict, total=False):
    """OpenRouter API response structure."""

    choices: list[ResponseChoice]
    usage: ResponseUsage
    cost: float
    id: str


class OpenRouterClient:
    """Wrapper for OpenRouter API client using direct REST API calls."""

    def __init__(self, api_key: str, timeout: int = 60) -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/LukasStrickler/LLM-Moral-Sycophancy-A-Behavioral-Study",  # Optional but recommended
                "X-Title": "LLM Moral Sycophancy Study",  # Optional but recommended
            },
        )

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ResponseDict:
        """Make async chat completion request using direct REST API.

        Args:
            model: Model ID (e.g., "openai/gpt-4o")
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ResponseDict with choices, usage, cost, and metadata

        Raises:
            Exception: If API call fails
        """
        # Build request payload with usage accounting enabled
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,  # Non-streaming by default
            "usage": {"include": True},  # Enable usage accounting to get cost info
            **kwargs,
        }
        
        # Make direct REST API call to OpenRouter
        url = f"{self.base_url}/chat/completions"
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse JSON response
            response_data = response.json()
            
            # Debug logging for Amazon models to diagnose empty responses
            if "amazon" in model.lower():
                logger.debug(f"Amazon model {model} raw response: {response_data}")
                if "choices" in response_data:
                    for i, choice in enumerate(response_data.get("choices", [])):
                        logger.debug(f"Choice {i}: {choice}")
                        if "message" in choice:
                            logger.debug(f"Message {i}: {choice['message']}")
            
            return self._parse_response(response_data, model)
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (including credit errors)
            error_message = f"HTTP {e.response.status_code}"
            
            # Try to extract error details from response
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_info = error_data["error"]
                    if isinstance(error_info, dict):
                        error_code = error_info.get("code", "")
                        error_msg = error_info.get("message", str(e))
                        
                        # Check for credit-related errors
                        if "credit" in error_code.lower() or "credit" in error_msg.lower():
                            error_message = f"Insufficient credits: {error_msg}"
                        elif "balance" in error_code.lower() or "balance" in error_msg.lower():
                            error_message = f"Insufficient balance: {error_msg}"
                        else:
                            error_message = f"{error_code}: {error_msg}"
                    else:
                        error_message = str(error_info)
                else:
                    error_message = str(e)
            except Exception:
                # If we can't parse the error response, use the status text
                error_message = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            
            # Create a custom exception with the error message
            raise RuntimeError(error_message) from e
        except httpx.HTTPError as e:
            # Handle other HTTP errors (network, timeout, etc.)
            raise RuntimeError(f"HTTP error: {str(e)}") from e
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    def _parse_response(self, response: dict[str, Any], model: str) -> ResponseDict:
        """Parse OpenRouter REST API response to ResponseDict.

        Args:
            response: Raw response dict from OpenRouter REST API
            model: Model ID to check if it's a free model

        Returns:
            ResponseDict with structured data
        """
        result: ResponseDict = {
            "choices": [],
            "usage": {},
        }

        # Extract choices
        if "choices" in response and isinstance(response["choices"], list):
            for choice in response["choices"]:
                message = choice.get("message", {})
                if not isinstance(message, dict):
                    message = {
                        "role": getattr(message, "role", "assistant"),
                        "content": getattr(message, "content", "") or "",
                    }
                
                result["choices"].append({
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": choice.get("finish_reason"),
                })

        # Extract usage information
        if "usage" in response and isinstance(response["usage"], dict):
            usage = response["usage"]
            result["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }

        # Extract cost (returns 0.0 for free models)
        cost = self._extract_cost_from_response(response, result, model)
        result["cost"] = cost
        if "usage" in result:
            result["usage"]["cost"] = cost

        # Store response ID if available
        if "id" in response:
            result["id"] = response["id"]

        return result

    def _extract_cost_from_response(
        self, response: dict[str, Any], parsed: ResponseDict, model: str
    ) -> float:
        """Extract cost from response dict.

        Args:
            response: Raw response dict from REST API
            parsed: Partially parsed response dict
            model: Model ID to check if it's a free model

        Returns:
            Cost in USD. Free models return 0.0.
        """
        # Check if model is free (ends with :free)
        if model.endswith(":free"):
            return 0.0

        # Check response.usage.cost (most common location)
        if "usage" in response and isinstance(response["usage"], dict):
            usage = response["usage"]
            if "cost" in usage and usage["cost"] is not None:
                return float(usage["cost"])

        # Check response.cost (fallback)
        if "cost" in response and response["cost"] is not None:
            return float(response["cost"])

        # Check parsed usage dict (final fallback)
        usage = parsed.get("usage", {})
        if isinstance(usage, dict) and "cost" in usage and usage["cost"] is not None:
            return float(usage["cost"])

        # No cost found - assume free model
        return 0.0

    def extract_cost(self, response: ResponseDict) -> float:
        """Extract cost from response.

        Args:
            response: Response dictionary from chat_completion

        Returns:
            Cost in USD (0.0 if not available)
        """
        if "cost" in response and response["cost"] is not None:
            return float(response["cost"])

        usage = response.get("usage", {})
        if isinstance(usage, dict) and "cost" in usage and usage["cost"] is not None:
            return float(usage["cost"])

        return 0.0

    def extract_tokens(self, response: ResponseDict) -> tuple[int | None, int | None, int | None]:
        """Extract token usage from response.

        Args:
            response: ResponseDict from chat_completion

        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens)
        """
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            return (None, None, None)

        return (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

    def extract_text(self, response: ResponseDict) -> str:
        """Extract response text from response.

        Args:
            response: Response dictionary from chat_completion

        Returns:
            Response text content
        """
        choices = response.get("choices", [])
        if not choices:
            return ""

        first_choice = choices[0]
        message = first_choice.get("message", {})
        return message.get("content", "")

