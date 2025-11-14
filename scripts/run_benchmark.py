#!/usr/bin/env python3
"""CLI for running the benchmark against OpenRouter models."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional

    def load_dotenv(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
        return None


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

DEFAULT_GRID_PATH = Path("outputs/raw/grid.jsonl")


def ensure_project_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark against one or more OpenRouter models"
    )
    parser.add_argument("--limit", type=int, default=9999, help="Number of prompts per model")
    parser.add_argument(
        "--models",
        type=str,
        default="data/models/benchmark_models.json",
        help="Ordered model list JSON",
    )
    parser.add_argument("--model", type=str, help="Run a single model id")
    parser.add_argument("--out", type=str, help="Custom run output path or JSONL file")
    parser.add_argument(
        "--grid", type=str, default=str(DEFAULT_GRID_PATH), help="Path to prebuilt grid JSONL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip provider calls and emit stub responses",
    )
    return parser.parse_args()


def ensure_api_key(cfg) -> None:
    if cfg.has_any_api_key():
        return
    print("ERROR: No provider API key is set. Set at least one of:")
    print("  GOOGLE_AI_API_KEY, GROQ_API_KEY, HUGGINGFACE_API_KEY, CEREBRAS_API_KEY,")
    print("  MISTRAL_API_KEY, COHERE_API_KEY, or OPENROUTER_API_KEY")
    print("Add it to your environment or .env file.")
    sys.exit(1)


def resolve_models(args: argparse.Namespace, cfg) -> tuple[list[str], dict[str, dict]]:
    ensure_project_on_path()
    from src.benchmark.core.models import load_models_config

    try:
        if args.model:
            # Load model configs to get concurrency settings for the specified model
            _, model_configs = load_models_config(Path(args.models), cfg.default_test_model)
            return [args.model], model_configs
        return load_models_config(Path(args.models), cfg.default_test_model)
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"Please check your models configuration file: {args.models}")
        sys.exit(1)


def resolve_run_paths(out_arg: str | None, grid_hash: str) -> tuple[Path, Path]:
    """Resolve run paths using grid hash instead of timestamp."""
    if out_arg:
        user_path = Path(out_arg)
        if user_path.suffix == ".jsonl":
            user_path.parent.mkdir(parents=True, exist_ok=True)
            return user_path.parent, user_path
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path, user_path / "run.jsonl"

    # Folder structure: outputs/runs/run_{grid_hash}/run.jsonl
    runs_dir = Path("outputs/runs")
    run_folder = runs_dir / f"run_{grid_hash}"
    run_folder.mkdir(parents=True, exist_ok=True)
    run_jsonl = run_folder / "run.jsonl"
    return run_folder, run_jsonl


def parse_factors(entry: dict):
    ensure_project_on_path()
    from src.benchmark.core.types import Factors

    payload = entry.get("factors") or {}
    required = {"perspective", "base_rent", "amount"}
    if not required.issubset(payload):
        return None
    try:
        # Handle amount which can be None (XX) or an int
        amount = payload["amount"]
        if amount is not None:
            amount = int(amount)
        
        return Factors(
            perspective=str(payload["perspective"]),
            base_rent=int(payload["base_rent"]),
            amount=amount,  # Can be None or int
            opposite_quality=payload.get("opposite_quality"),  # Can be None, "good", or "poor"
            justification=payload.get("justification"),
        )
    except (TypeError, ValueError):
        return None


def load_grid(path: Path, limit: int, logger) -> tuple[list, dict]:
    """Load grid and return (factors, prompt_id_map).
    
    prompt_id_map maps factors to prompt_id from grid.jsonl.
    """
    if not path.exists():
        logger.error(
            (
                "Grid file not found: %s. Run 'python scripts/build_benchmark.py' first."
            ),
            path,
        )
        sys.exit(1)

    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    factors = []
    prompt_id_map: dict = {}  # Maps factors to prompt_id from grid.jsonl
    for entry in records:
        factor = parse_factors(entry)
        if factor:
            factors.append(factor)
            # Use prompt_id from grid.jsonl (sequential number) as source of truth
            prompt_id = entry.get("prompt_id")
            if prompt_id:
                # Create a key from factors for mapping
                from src.benchmark.core.types import make_prompt_id
                factor_key = make_prompt_id(factor)  # Hash of factors
                prompt_id_map[factor_key] = prompt_id

    if not factors:
        logger.error("Grid file %s does not contain any valid prompts.", path)
        sys.exit(1)

    if limit is None:
        return factors, prompt_id_map
    if limit <= 0:
        return [], prompt_id_map
    return factors[:limit] if limit < len(factors) else factors, prompt_id_map


def show_run_header(
    prompt_count: int,
    models: list[str],
    grid_path: Path,
    logger,
    *,
    dry_run: bool,
) -> None:
    logger.info("Running benchmark for %d prompts per model (grid=%s)", prompt_count, grid_path)
    logger.info("Assistant models (ordered):")
    for idx, model_id in enumerate(models, start=1):
        logger.info("  %02d. %s", idx, model_id)
    if dry_run:
        logger.info("Dry-run mode: using synthetic responses; provider calls skipped.")


def main() -> None:
    args = parse_args()
    load_dotenv()
    ensure_project_on_path()

    from src.benchmark.core.config import ProviderConfig
    from src.benchmark.core.logging import configure_logging, setup_logger
    from src.benchmark.run.runner_async import run_local_benchmark_async

    cfg = ProviderConfig.from_env()
    if not args.dry_run:
        ensure_api_key(cfg)

    all_models, model_configs = resolve_models(args, cfg)
    grid_path = Path(args.grid) if args.grid else DEFAULT_GRID_PATH
    
    # Filter models based on API key availability (like AI labeling)
    from src.benchmark.providers.litellm_provider import _has_api_key_for_model
    
    models: list[str] = []
    skipped_models: list[str] = []
    for model_id in all_models:
        if args.dry_run or _has_api_key_for_model(cfg, model_id):
            models.append(model_id)
        else:
            skipped_models.append(model_id)

    # Set up logger first (needed for load_grid)
    logger = setup_logger("main")
    
    # Load all factors from grid to compute hash (before applying limit)
    all_factors, all_prompt_id_map = load_grid(grid_path, None, logger)
    from src.benchmark.core.types import compute_grid_hash

    grid_hash = compute_grid_hash(all_factors)
    
    # Now load selected factors with limit
    selected_factors, prompt_id_map = load_grid(grid_path, args.limit, logger)

    run_dir, run_jsonl = resolve_run_paths(args.out, grid_hash)
    configure_logging(run_dir / "run.log")
    
    # Copy grid.jsonl to run folder as source of truth for prompts
    import shutil
    grid_jsonl_dest = run_dir / "grid.jsonl"
    if grid_path.exists() and not grid_jsonl_dest.exists():
        shutil.copy2(grid_path, grid_jsonl_dest)
        logger.info("Copied grid.jsonl to run folder: %s", grid_jsonl_dest)

    # Show overview log (like AI labeling) - count completed tasks per model
    from src.benchmark.core.types import RunRecord
    from src.benchmark.core.logging import make_log_extra
    
    # Log skipped models (no API key)
    if skipped_models:
        logger.warning(
            "",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="warn",
                status="no-key",
                details=(f"count={len(skipped_models)}", ", ".join(skipped_models)),
            ),
        )
    
    plan_entries: list[dict[str, object]] = []
    models_to_process: list[str] = []
    total_prompts = len(selected_factors)
    
    # Initialize completed_counts before the if block
    completed_counts: dict[str, int] = {}
    if run_jsonl.exists():
        # Count completed tasks per model
        try:
            for record in RunRecord.iter_jsonl(run_jsonl):
                if record.is_success() and record.model_id and record.prompt_id:
                    completed_counts[record.model_id] = completed_counts.get(record.model_id, 0) + 1
        except Exception as exc:
            logger.warning("Error reading existing records for overview: %s", exc)
    
    for idx, model_id in enumerate(models, 1):
        completed = completed_counts.get(model_id, 0)
        pending = total_prompts - completed
        todo = "OPEN" if pending > 0 else "DONE"
        plan_entries.append(
            {
                "idx": idx,
                "model": model_id,
                "pending": pending,
                "completed": completed,
                "todo": todo,
            }
        )
        if todo == "OPEN":
            models_to_process.append(model_id)
    
    # Show overview
    if plan_entries:
        logger.info(
            "",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="plan",
                details=(
                    f"total={len(plan_entries)}",
                    f"open={len(models_to_process)}",
                    f"done={len(plan_entries) - len(models_to_process)}",
                ),
            ),
        )
        for entry in plan_entries:
            logger.info(
                "",
                extra=make_log_extra(
                    model=str(entry["model"]),
                    grid=None,
                    task=None,
                    progress=None,
                    tag="info",
                    status=str(entry["todo"]),
                    details=(
                        f"idx={entry['idx']}",
                        f"todo={entry['todo']}",
                        f"pending={entry['pending']}",
                        f"completed={entry['completed']}",
                    ),
                ),
            )
    
    if not models_to_process:
        logger.info(
            "All prompts already successfully completed for all models",
            extra=make_log_extra(
                model=None,
                grid=None,
                task=None,
                progress=None,
                tag="info",
                status="plan",
                details=("all-complete",),
            ),
        )
        return

    show_run_header(
        len(selected_factors),
        models,
        grid_path,
        logger,
        dry_run=args.dry_run,
    )

    t_start = time.time()
    asyncio.run(
        run_local_benchmark_async(
            limit=len(selected_factors),
            factors_list_override=selected_factors,
            assistant_models=models,
            model_configs=model_configs,
            out_path=run_jsonl,
            dry_run=args.dry_run,
            prompt_id_map=prompt_id_map,
        )
    )
    elapsed_s = time.time() - t_start
    wall_s = max(elapsed_s, 1e-6)
    logger.info("Benchmark generation finished in %.2fs", wall_s)
    logger.info("Run records saved to %s", run_jsonl)


if __name__ == "__main__":
    main()
