#!/bin/bash
#SBATCH --job-name=ms-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --parsable
# Optional/cluster-specific:
#SBATCH --partition=gpu            # adjust per cluster (e.g., gpu, a100, short)
#SBATCH --account=your_account     # if required by the cluster

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the repository root before submitting the job}"

module load python/3.11 || true
module load cuda/12.1 || true      # adjust per cluster

mkdir -p "$PROJECT_DIR/logs"

if [ -d "$PROJECT_DIR/.venv" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
else
  echo "Virtual environment not found at $PROJECT_DIR/.venv" >&2
  exit 1
fi

cd "$PROJECT_DIR" || exit 1

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Train RoBERTa regression scorer (pending implementation)
echo "train_roberta.py not yet implemented - update this job once the trainer lands" >&2
exit 1
