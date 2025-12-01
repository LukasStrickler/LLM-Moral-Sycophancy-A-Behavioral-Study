#!/usr/bin/env python3
"""Entry point for paid benchmark runner using OpenRouter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional
    def load_dotenv(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
        return None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

from src.benchmark.paid.config import PaidRunnerConfig
from src.benchmark.paid.runner import run_paid_benchmark


def _parse_models(models_str: str | None) -> list[str] | None:
    """Parse comma-separated model list.

    Args:
        models_str: Comma-separated string of model IDs, or None

    Returns:
        List of model IDs, or None if not provided

    Raises:
        SystemExit: If models string is empty after parsing
    """
    if not models_str:
        return None

    models = [m.strip() for m in models_str.split(",") if m.strip()]
    if len(models) == 0:
        print(
            "Error: At least one model must be specified. "
            "Provide comma-separated list like: 'openai/gpt-4o,anthropic/claude-3.5-sonnet'",
            file=sys.stderr,
        )
        sys.exit(1)

    return models


def _print_config_summary(config: PaidRunnerConfig, args: argparse.Namespace) -> None:
    """Print configuration summary.

    Args:
        config: Runner configuration
        args: Parsed command-line arguments
    """
    if not args.models:
        print(f"Using default models ({len(config.models)}): {', '.join(config.models)}")
    if not args.budget:
        print(f"Using default budget: ${config.budget_limit:.2f}")
    if config.prompt_limit is not None:
        print(f"Prompt limit: {config.prompt_limit} new fully completed prompts")
    print(f"Seed file: {config.seed_file}")
    print(f"Output directory: {config.output_dir}")
    print()


def _print_run_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Print run summary.

    Args:
        summary: Summary dictionary from run_paid_benchmark
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"Prompts processed: {summary['prompts_processed']}")
    print(f"Models completed: {summary['models_completed']}")
    print(f"Total fully completed prompts: {summary['total_completed_prompts']}")
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Remaining budget: ${summary['remaining_budget']:.2f}")
    print(f"Budget exceeded: {summary['budget_exceeded']}")
    print(f"\nResults saved to: {output_dir / 'results.csv'}")
    print(f"State saved to: {output_dir / 'state.json'}")
    print("=" * 80)


def _check_resume_state(state_file: Path, resume: bool) -> bool:
    """Check and handle resume state.

    Args:
        state_file: Path to state file
        resume: Whether resume was requested

    Returns:
        True if should continue, False if should exit
    """
    if resume and not state_file.exists():
        print(
            f"Warning: --resume specified but no state file found at {state_file}",
            file=sys.stderr,
        )
        print("Starting new run instead.", file=sys.stderr)
        return True

    if not resume and state_file.exists():
        print(
            f"Warning: State file exists at {state_file}",
            file=sys.stderr,
        )
        print(
            "Use --resume to continue from existing state, "
            "or delete state file to start fresh.",
            file=sys.stderr,
        )
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            return False

    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run paid benchmark using OpenRouter API with budget tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (6 models, $10 budget)
  # Just set PAID_OPENROUTER_API_KEY in .env and run:
  poetry run python scripts/run_paid_benchmark.py

  # Run with custom models and budget
  poetry run python scripts/run_paid_benchmark.py \\
    --models openai/gpt-4o,anthropic/claude-3.5-sonnet,google/gemini-2.0-flash-exp,meta-llama/llama-3.1-70b-instruct,deepseek/deepseek-chat,mistralai/mixtral-8x7b-instruct \\
    --budget 20.0

  # Resume from existing state
  poetry run python scripts/run_paid_benchmark.py --resume --budget 20.0
        """,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set PAID_OPENROUTER_API_KEY or OPENROUTER_API_KEY env var)",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model IDs (default: 6 popular models). Example: 'openai/gpt-4o,anthropic/claude-3.5-sonnet,...'",
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget limit in USD (default: 10.0 for $10). Example: 10.0",
    )

    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Path to seed JSONL file (default: data/humanLabel/seeds/aita_prompt_seed.jsonl)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV and state files (default: outputs/paid_runs)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum retries per model (default: 10)",
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )

    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Limit number of new fully completed prompts (default: no limit). Example: --prompt-limit 10",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing state (automatically detected if state file exists)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Parse models
    models = _parse_models(args.models)

    # Create configuration
    try:
        config = PaidRunnerConfig.from_env(
            api_key=args.api_key,
            models=models,
            budget_limit=args.budget,
            seed_file=args.seed_file,
            output_dir=args.output_dir,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            prompt_limit=args.prompt_limit,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print(
            "\nTip: Make sure to set PAID_OPENROUTER_API_KEY in your .env file "
            "or provide --api-key argument.",
            file=sys.stderr,
        )
        return 1

    # Print configuration summary
    _print_config_summary(config, args)

    # Check resume state
    state_file = config.output_dir / "state.json"
    if not _check_resume_state(state_file, args.resume):
        return 1

    # Run benchmark
    try:
        summary = run_paid_benchmark(config)
        _print_run_summary(summary, config.output_dir)
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. State has been saved and can be resumed.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

