"""Configuration for LLM-based labeling."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ...benchmark.core.config import ProviderConfig, RunConfig
from ...benchmark.core.logging import setup_logger
from ...benchmark.core.models import load_models_config
from ...benchmark.providers.litellm_provider import _has_api_key_for_model
from ...benchmark.scoring.master import SCORING_INSTRUCTION

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_FILE = PROJECT_ROOT / "data" / "models" / "llm_labeling_models.json"
PROMPT_FILE = PROJECT_ROOT / "data" / "prompts" / "sycophancy_scoring.json"

logger = setup_logger("labeling-config")

# Cache to avoid repeated warnings
_config_cache: LabelingConfig | None = None
_skipped_models_logged = False


class LabelingConfig:
    """Configuration for LLM labeling operations."""

    def __init__(self, models_file: Path | None = None, prompt_file: Path | None = None):
        models_file = models_file or MODELS_FILE
        prompt_file = prompt_file or PROMPT_FILE

        # Load models from data/models/ using benchmark infrastructure
        all_models, all_model_configs = load_models_config(
            models_file, "google/gemini-2.0-flash-exp"
        )

        # Load prompt from data/prompts/
        self.scoring_prompt = self._load_prompt(prompt_file)

        # Use benchmark configs
        self.provider_config = ProviderConfig.from_env()
        self.run_config = RunConfig.from_env()

        # Filter out models without API keys and log warnings (only once)
        self.models = []
        self.model_configs = {}
        skipped_models = []
        
        global _skipped_models_logged

        for model_id in all_models:
            if _has_api_key_for_model(self.provider_config, model_id):
                self.models.append(model_id)
                if model_id in all_model_configs:
                    self.model_configs[model_id] = all_model_configs[model_id]
            else:
                skipped_models.append(model_id)

        # Log warnings only once, not on every instantiation
        if skipped_models and not _skipped_models_logged:
            logger.warning(f"⚠️  {len(skipped_models)} model(s) skipped - no API key")
            logger.info(f"📋 {len(self.models)} models available")
            _skipped_models_logged = True

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
                logger.warning(
                    f"Prompt file has no 'instruction' or 'system' field: {prompt_file}, using fallback"
                )
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

        # Fall back to provider scorer model
        if self.provider_config.scorer_model:
            return self.provider_config.scorer_model

        # Final fallback to default test model
        return self.provider_config.default_test_model

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
            logger.warning("Invalid concurrency value, using default of 5")
            return 5


def get_labeling_config() -> LabelingConfig:
    """Get a configured LabelingConfig instance (cached)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = LabelingConfig()
    return _config_cache
