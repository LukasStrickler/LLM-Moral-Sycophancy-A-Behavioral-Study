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


def resolve_run_paths(out_arg: str | None, timestamp: int) -> tuple[Path, Path]:
    if out_arg:
        user_path = Path(out_arg)
        if user_path.suffix == ".jsonl":
            user_path.parent.mkdir(parents=True, exist_ok=True)
            return user_path.parent, user_path
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path, user_path / "run.jsonl"

    run_dir = Path(f"outputs/runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "run.jsonl"


def parse_factors(entry: dict):
    ensure_project_on_path()
    from src.benchmark.core.types import Factors

    payload = entry.get("factors") or {}
    required = {"perspective", "base_rent", "amount", "relationship_quality"}
    if not required.issubset(payload):
        return None
    try:
        return Factors(
            perspective=str(payload["perspective"]),
            base_rent=int(payload["base_rent"]),
            amount=int(payload["amount"]),
            relationship_quality=str(payload["relationship_quality"]),
            justification=payload.get("justification"),
            tenant_quality=payload.get("tenant_quality"),
            landlord_quality=payload.get("landlord_quality"),
        )
    except (TypeError, ValueError):
        return None


def load_grid(path: Path, limit: int, logger) -> list:
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
    for entry in records:
        factor = parse_factors(entry)
        if factor:
            factors.append(factor)

    if not factors:
        logger.error("Grid file %s does not contain any valid prompts.", path)
        sys.exit(1)

    if limit is None:
        return factors
    if limit <= 0:
        return []
    return factors[:limit] if limit < len(factors) else factors


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

    run_timestamp = int(time.time())
    run_dir, run_jsonl = resolve_run_paths(args.out, run_timestamp)
    configure_logging(run_dir / "run.log")
    logger = setup_logger("main")
    models, model_configs = resolve_models(args, cfg)
    grid_path = Path(args.grid) if args.grid else DEFAULT_GRID_PATH

    selected_factors = load_grid(grid_path, args.limit, logger)

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
        )
    )
    elapsed_s = time.time() - t_start
    wall_s = max(elapsed_s, 1e-6)
    logger.info("Benchmark generation finished in %.2fs", wall_s)
    logger.info("Run records saved to %s", run_jsonl)


if __name__ == "__main__":
    main()
