# HPC Cluster Usage (Slurm) - WIP

This folder contains job scripts for running benchmarks and training on Slurm clusters (e.g., University of Mannheim, KIT).

## Status: Work in Progress

The Slurm integration is currently under development and will be implemented in a future milestone.

## Planned Features

- **Benchmark Jobs**: CPU-based benchmark execution via `sbatch`
- **Eval Jobs**: CPU-based response scoring and evaluation
- **Training Jobs**: GPU-based RoBERTa model training
- **Cluster Support**: University of Mannheim and KIT cluster configurations
- **Job Monitoring**: Queue management and log tracking

## Implementation Plan

1. **Job Scripts**: Complete `job_benchmark.sbatch`, `job_eval.sbatch`, and `job_train_roberta.sbatch`
2. **Cluster Configuration**: Adapt partition/account settings for target clusters
3. **Environment Setup**: Automated Poetry environment activation
4. **Monitoring Tools**: Job status tracking and log management
5. **Documentation**: Complete usage guide and troubleshooting

## Current Files

- `job_benchmark.sbatch`: Placeholder for benchmark job script
- `job_eval.sbatch`: Placeholder for evaluation job script
- `job_train_roberta.sbatch`: Placeholder for training job script

## Future Usage

Once implemented, jobs will be submitted with:
```bash
# Run benchmark (CPU)
sbatch slurm/job_benchmark.sbatch

# Score responses (CPU)
sbatch slurm/job_eval.sbatch

# Train RoBERTa scorer (GPU)
sbatch slurm/job_train_roberta.sbatch
```

For now, use local execution as described in the main README.