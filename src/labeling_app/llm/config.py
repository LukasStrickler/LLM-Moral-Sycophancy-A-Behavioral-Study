"""Configuration for LLM-based labeling."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ...benchmark.core.config import OpenRouterConfig, RunConfig
from ...benchmark.core.models import load_models_config
from ...benchmark.scoring.master import SCORING_INSTRUCTION

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_FILE = PROJECT_ROOT / "data" / "models" / "llm_labeling_models.json"
PROMPT_FILE = PROJECT_ROOT / "data" / "prompts" / "sycophancy_scoring.json"

logger = logging.getLogger(__name__)


class LabelingConfig:
    """Configuration for LLM labeling operations."""
    
    def __init__(self, models_file: Path | None = None, prompt_file: Path | None = None):
        models_file = models_file or MODELS_FILE
        prompt_file = prompt_file or PROMPT_FILE
        
        # Load models from data/models/ using benchmark infrastructure
        self.models, self.model_configs = load_models_config(
            models_file, 
            "openai/gpt-oss-20b:free"
        )
        
        # Load prompt from data/prompts/
        self.scoring_prompt = self._load_prompt(prompt_file)
        
        # Use benchmark configs
        self.openrouter_config = OpenRouterConfig.from_env()
        self.run_config = RunConfig.from_env()
    
    def _load_prompt(self, prompt_file: Path) -> str:
        """Load the scoring prompt from JSON file."""
        if not prompt_file.exists():
            logger.warning(f"Prompt file not found: {prompt_file}, using fallback")
            return SCORING_INSTRUCTION
        
        try:
            with prompt_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("instruction", data.get("system", ""))
            if not prompt:
                logger.warning(f"Prompt file has no 'instruction' or 'system' field: {prompt_file}, using fallback")
                return SCORING_INSTRUCTION
            return prompt
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in prompt file: {prompt_file}, error: {e}")
            return SCORING_INSTRUCTION
    
    @property
    def default_model(self) -> str | None:
        """Get the default scorer model."""
        # Try LLM-specific env var first
        llm_model = os.getenv("LLM_LABELING_MODEL")
        if llm_model:
            return llm_model
        
        # Use first model from config
        if self.models:
            return self.models[0]
        
        # Fall back to OpenRouter scorer model
        if self.openrouter_config.scorer_model:
            return self.openrouter_config.scorer_model
        
        # Final fallback to default test model
        return self.openrouter_config.default_test_model
    
    @property
    def concurrency(self) -> int:
        """Get the concurrency limit for parallel scoring."""
        try:
            value = int(os.getenv("LLM_LABELING_CONCURRENCY", "5"))
            # Clamp to reasonable range
            if value < 1:
                logger.warning(f"Concurrency value {value} too low, using minimum of 1")
                return 1
            elif value > 50:
                logger.warning(f"Concurrency value {value} too high, using maximum of 50")
                return 50
            return value
        except ValueError:
            logger.warning(f"Invalid concurrency value, using default of 5")
            return 5


def get_labeling_config() -> LabelingConfig:
    """Get a configured LabelingConfig instance."""
    return LabelingConfig()
