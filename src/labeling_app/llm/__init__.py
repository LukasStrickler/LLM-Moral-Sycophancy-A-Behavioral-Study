"""LLM-based automatic labeling for moral sycophancy evaluation."""

from .llm_scorer import score_response_async, score_response

__all__ = ["score_response_async", "score_response"]
