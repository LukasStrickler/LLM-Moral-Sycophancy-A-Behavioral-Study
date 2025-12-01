"""State management for resumable runs."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ModelId, PromptId


@dataclass
class RunState:
    """State of a paid runner execution.
    
    Simplified state that only tracks:
    - Total cost (aggregate)
    - Cost per model (for tracking/modeling)
    - Last processed prompt (for resuming)
    
    Completion status is determined from CSV, not stored in state.
    """

    started_at: str
    last_updated: str
    total_cost: float
    model_costs: dict[str, float] = field(default_factory=dict)  # model_id -> total cost for that model
    last_processed_prompt: str | None = None  # PromptId, but use str for compatibility

    @classmethod
    def new(cls) -> RunState:
        """Create a new run state."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return cls(
            started_at=now,
            last_updated=now,
            total_cost=0.0,
            model_costs={},
        )

    def update(self) -> None:
        """Update last_updated timestamp."""
        self.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def add_model_cost(self, model_id: str, cost: float) -> None:
        """Add cost for a model completion.

        Args:
            model_id: Model ID
            cost: Cost in USD
        """
        if model_id not in self.model_costs:
            self.model_costs[model_id] = 0.0
        self.model_costs[model_id] += cost
        self.total_cost += cost
        self.update()

    def to_dict(self) -> dict:
        """Convert state to dictionary for JSON serialization."""
        return {
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_cost": self.total_cost,
            "model_costs": self.model_costs,
            "last_processed_prompt": self.last_processed_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunState:
        """Create state from dictionary."""
        return cls(
            started_at=data["started_at"],
            last_updated=data["last_updated"],
            total_cost=data["total_cost"],
            model_costs=data.get("model_costs", {}),
            last_processed_prompt=data.get("last_processed_prompt"),
        )


class StateManager:
    """Manage run state persistence."""

    def __init__(self, state_file: Path) -> None:
        """Initialize state manager.

        Args:
            state_file: Path to state JSON file
        """
        self.state_file = state_file

    def load_state(self) -> RunState | None:
        """Load existing state from file.

        Returns:
            RunState if file exists, None otherwise
        """
        if not self.state_file.exists():
            return None

        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return RunState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If state file is corrupted, log warning and return None to start fresh
            # This allows recovery from corrupted state files
            import logging
            logger = logging.getLogger("paid-runner.state")
            logger.warning(
                f"State file {self.state_file} is corrupted or invalid: {e}. "
                f"Starting fresh run. If you need to recover, check the file manually."
            )
            return None

    def save_state(self, state: RunState) -> None:
        """Save state to file.

        Args:
            state: RunState to save
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_prompts(seed_file: Path) -> Iterator[dict[str, Any]]:
        """Load prompts from seed file.

        Args:
            seed_file: Path to JSONL seed file

        Yields:
            Prompt dictionaries with 'identifier' and 'prompt_body' keys

        Raises:
            FileNotFoundError: If seed file doesn't exist
        """
        if not seed_file.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_file}")

        return _load_prompts_from_file(seed_file)

    def get_next_prompt(
        self, seed_file: Path, csv_path: Path, state: RunState | None, models: list[str]
    ) -> Iterator[tuple[dict, list[str]]]:
        """Get next unprocessed prompt with list of missing models.
        
        Missing models are determined by checking the CSV file, not state.

        Args:
            seed_file: Path to seed file
            csv_path: Path to CSV results file
            state: Current run state (None for new run)
            models: List of model IDs

        Yields:
            Tuples of (prompt_dict, missing_models_list)
        """
        # Load completed model-prompt pairs from CSV
        completed = _load_completed_from_csv(csv_path)
        
        for prompt in _load_prompts_from_file(seed_file):
            prompt_id = prompt.get("identifier")
            if not prompt_id:
                continue

            # Get missing models for this prompt by checking CSV
            missing_models = [
                model_id for model_id in models
                if (prompt_id, model_id) not in completed
            ]
            
            # Skip if all models completed
            if not missing_models:
                continue

            yield prompt, missing_models


def _load_completed_from_csv(csv_path: Path) -> set[tuple[str, str]]:
    """Load completed (prompt_id, model_id) pairs from CSV.
    
    Args:
        csv_path: Path to CSV results file
        
    Returns:
        Set of (prompt_id, model_id) tuples that have been completed
    """
    completed = set()
    
    if not csv_path.exists():
        return completed
    
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_id = row.get("prompt_identifier", "").strip()
                model_id = row.get("model_id", "").strip()
                error = row.get("error", "").strip()
                
                # Only count as completed if there's no error (or error is empty)
                if prompt_id and model_id and not error:
                    completed.add((prompt_id, model_id))
    except Exception:
        # If CSV is corrupted or can't be read, return empty set
        # This allows recovery by treating everything as incomplete
        pass
    
    return completed


def _load_prompts_from_file(seed_file: Path) -> Iterator[dict[str, Any]]:
    """Load prompts from JSONL file.

    Args:
        seed_file: Path to JSONL seed file

    Yields:
        Prompt dictionaries with 'identifier' and 'prompt_body' keys
    """
    with seed_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # Log but continue - skip invalid lines
                import logging
                logger = logging.getLogger("paid-runner.state")
                logger.warning(
                    f"Skipping invalid JSON on line {line_num} of {seed_file}: {e}"
                )
                continue

