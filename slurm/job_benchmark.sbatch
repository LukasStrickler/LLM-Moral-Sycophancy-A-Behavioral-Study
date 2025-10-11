#!/bin/bash
#SBATCH --job-name=ms-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --parsable
# Optional/cluster-specific:
#SBATCH --partition=cpu            # adjust per cluster (e.g., general, cpu, short)
#SBATCH --account=your_account     # if required by the cluster

set -euo pipefail

# Project directory on the cluster (set via: sbatch --export=PROJECT_DIR=/path ...)
: "${PROJECT_DIR:?Set PROJECT_DIR to the repository root before submitting the job}"

module load python/3.11 || true

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

# Activate virtual environment located inside the project
if [ -d "$PROJECT_DIR/.venv" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
else
  echo "Virtual environment not found at $PROJECT_DIR/.venv" >&2
  exit 1
fi

cd "$PROJECT_DIR" || exit 1

# Load environment variables from .env if present (OPENROUTER_API_KEY, etc.)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Run benchmark
poetry run python scripts/run_benchmark.py \
  --include-neutral \
  --limit 150
