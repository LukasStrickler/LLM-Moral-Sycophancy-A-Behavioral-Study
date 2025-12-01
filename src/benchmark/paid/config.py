"""Configuration for paid runner."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

# Type aliases for clarity
ModelId = NewType("ModelId", str)
PromptId = NewType("PromptId", str)

# Default configuration constants
# For model selection rationale and grid composition, see SOTA_MODELS.md in this directory
#
# Sycophancy Benchmark Grid (16 models):
# Comprehensive coverage across providers, architectures, and geographic origins
# Includes historical models for generation-to-generation comparison
# Note: openai/gpt-5 and openai/gpt-4o removed - we already have data for them
DEFAULT_MODELS: list[str] = [
    # Anthropic (3 models)
    "anthropic/claude-opus-4.5",           # Latest flagship - strongest alignment baseline
    "anthropic/claude-sonnet-4.5",         # Best value - size comparison within family
    "anthropic/claude-sonnet-4",           # Previous generation Sonnet - historical comparison
    
    # Google (4 models)
    "google/gemini-3-pro-preview",         # Latest flagship - state-of-the-art multimodal
    "google/gemini-2.5-pro",               # Previous generation Pro - historical comparison
    "google/gemini-2.5-flash",             # Efficient workhorse - speed/cost balance
    "google/gemma-3n-e4b-it",        # On-device optimized - mobile/edge AI, MatFormer architecture
    
    # OpenAI (2 models)
    "openai/gpt-5.1",                      # Latest iteration - mainstream commercial AI
    "openai/gpt-oss-120b",                 # First open-weight - transparency and cost
    
    # Amazon (1 model)
    # "amazon/nova-premier-v1",              # Flagship multimodal - powers Rufus shopping assistant
    
    # AllenAI (1 model)
    "allenai/olmo-3-32b-think",           # 32B reasoning model - deep reasoning and complex logic
    
    # xAI (1 model)
    "x-ai/grok-4.1-fast:free",             # Latest model - different training philosophy
    
    # Chinese Models (3 models)
    "moonshotai/kimi-k2-thinking",         # Trillion-param reasoning - beats GPT-5
    "deepseek/deepseek-r1",                # Open-source reasoning - MIT licensed
    "qwen/qwen3-max",                      # Alibaba flagship - outperforms many Western models
    
    # European (1 model)
    "mistralai/mistral-medium-3.1",        # European perspective - multilingual focus
]

DEFAULT_BUDGET: float = 10.0  # Default $10 budget
DEFAULT_SEED_FILE = Path("data/humanLabel/seeds/aita_prompt_seed.jsonl")
DEFAULT_OUTPUT_BASE = Path("outputs/paid_runs")


@dataclass
class PaidRunnerConfig:
    """Configuration for paid runner using OpenRouter."""

    api_key: str
    models: list[str]
    budget_limit: float
    seed_file: Path
    output_dir: Path
    max_retries: int = 10
    request_timeout: int = 120
    prompt_limit: int | None = None

    @staticmethod
    def from_env(
        api_key: str | None = None,
        models: list[str] | None = None,
        budget_limit: float | None = None,
        seed_file: Path | None = None,
        output_dir: Path | None = None,
        max_retries: int = 10,
        request_timeout: int = 120,
        prompt_limit: int | None = None,
    ) -> PaidRunnerConfig:
        """Create config from environment variables and arguments."""
        # Check for PAID_OPENROUTER_API_KEY first, then fall back to OPENROUTER_API_KEY
        api_key = api_key or os.getenv("PAID_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key must be provided via --api-key, "
                "PAID_OPENROUTER_API_KEY env var, or OPENROUTER_API_KEY env var"
            )

        # Default models if not specified
        if not models:
            models = DEFAULT_MODELS.copy()

        if len(models) == 0:
            raise ValueError(
                "At least one model must be specified. "
                "Provide --models argument or use default models."
            )

        # Default budget if not specified
        if budget_limit is None:
            budget_limit = DEFAULT_BUDGET
        elif budget_limit <= 0:
            raise ValueError(
                f"Budget limit must be a positive number, got {budget_limit}. "
                f"Example: --budget 10.0 for $10"
            )

        if seed_file is None:
            seed_file = DEFAULT_SEED_FILE
        seed_file = Path(seed_file)
        if not seed_file.exists():
            raise FileNotFoundError(
                f"Seed file not found: {seed_file}. "
                f"Please provide a valid path with --seed-file or use default: {DEFAULT_SEED_FILE}"
            )

        # Compute hash of seed file for run folder
        seed_file_hash = _compute_file_hash(seed_file)

        if output_dir is None:
            # Use hash-based folder: outputs/paid_runs/run_{hash}/
            output_dir = DEFAULT_OUTPUT_BASE / f"run_{seed_file_hash}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate prompt limit if provided
        if prompt_limit is not None and prompt_limit <= 0:
            raise ValueError(
                f"Prompt limit must be a positive number, got {prompt_limit}. "
                f"Example: --prompt-limit 10"
            )

        return PaidRunnerConfig(
            api_key=api_key,
            models=models,
            budget_limit=budget_limit,
            seed_file=seed_file,
            output_dir=output_dir,
            max_retries=max_retries,
            request_timeout=request_timeout,
            prompt_limit=prompt_limit,
        )


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA1 hash of file contents for stable run folder naming.
    
    Args:
        file_path: Path to file to hash
        
    Returns:
        16-character hex hash
    """
    hasher = hashlib.sha1()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]

