# Command-Line Tools

This directory contains CLI tools for managing the LLM moral sycophancy study pipeline, from benchmark generation to human labeling data management.

## Overview

The scripts provide both interactive and non-interactive interfaces for:
- **Benchmark Generation**: Creating prompt grids and running LLM evaluations
- **Response Scoring**: Evaluating LLM responses with master models
- **Data Management**: Managing human labeling platform data and exports

## Tools

### Benchmark Tools

#### `build_benchmark.py`
Generates prompt grids for benchmarking without making API calls.

```bash
# Build a small test grid
poetry run python scripts/build_benchmark.py --include-neutral --limit 10

# Build full grid
poetry run python scripts/build_benchmark.py --include-neutral
```

**Output**: `outputs/raw/grid.jsonl` - Prompt configuration file

#### `run_benchmark.py`
Executes benchmark runs against LLM providers.

```bash
# Run with default models
poetry run python scripts/run_benchmark.py --limit 5 --include-neutral --models data/models.json

# Run single model
poetry run python scripts/run_benchmark.py --model openai/gpt-oss-20b:free --limit 5

# Dry run (no API calls)
poetry run python scripts/run_benchmark.py --dry-run --limit 5
```

**Output**: `outputs/runs/<run_id>/` directory with:
- `run.jsonl` - Model responses
- `run_grid.json` - Prompt configuration snapshot
- `run.log` - Execution logs

#### `eval_benchmark.py`
Scores benchmark results using master LLM or future ML models.

```bash
# Score specific run
poetry run python scripts/eval_benchmark.py --input outputs/runs/<run_id>/run.jsonl

# Score with aggregation
poetry run python scripts/eval_benchmark.py --input outputs/runs/<run_id>/run.jsonl --aggregate
```

**Output**: 
- `run_scored.jsonl` - Responses with numerical scores
- `summary.csv` - Aggregated statistics (if `--aggregate` used)

### Data Management Tool

#### `data_portal.py`
Comprehensive CLI for managing the human labeling platform data lifecycle.

**Interactive Mode** (default):
```bash
poetry run python scripts/data_portal.py
```

Presents a Rich-enhanced menu with options:
1. Ensure database schema
2. Sync seed data (dry run)
3. Sync seed data and apply
4. Export review records
5. Show dataset status
0. Exit

**Non-Interactive Commands**:

```bash
# Database management
poetry run python scripts/data_portal.py init-db

# Data synchronization
poetry run python scripts/data_portal.py push --dataset aita
poetry run python scripts/data_portal.py push --dataset scenario
poetry run python scripts/data_portal.py push --dataset all --apply

# Data export
poetry run python scripts/data_portal.py pull --target reviews --dataset scenario
poetry run python scripts/data_portal.py pull --target reviews --dataset all

# Status and monitoring
poetry run python scripts/data_portal.py status --dataset aita
poetry run python scripts/data_portal.py status --dataset scenario
```

**Advanced Options**:
```bash
# Sync with specific run file
poetry run python scripts/data_portal.py push --dataset scenario --run-file outputs/runs/123456/run.jsonl

# Limit records during sync
poetry run python scripts/data_portal.py push --dataset scenario --limit 100

# Record range for partial sync
poetry run python scripts/data_portal.py push --dataset scenario --record-start 0 --record-end 50
```

## Data Portal Workflows

### 1. Initial Setup
```bash
# 1. Initialize database schema
poetry run python scripts/data_portal.py init-db

# 2. Sync AITA seed data
poetry run python scripts/data_portal.py push --dataset aita --apply

# 3. Sync scenario data from latest run
poetry run python scripts/data_portal.py push --dataset scenario --apply
```

### 2. Regular Data Management
```bash
# Check dataset status
poetry run python scripts/data_portal.py status --dataset scenario

# Export collected reviews
poetry run python scripts/data_portal.py pull --target reviews --dataset scenario

# Sync new scenario runs
poetry run python scripts/data_portal.py push --dataset scenario --run-file outputs/runs/<new_run_id>/run.jsonl --apply
```

### 3. Data Export for Analysis
```bash
# Export all reviews for notebook analysis
poetry run python scripts/data_portal.py pull --target reviews --dataset all

# Check coverage before export
poetry run python scripts/data_portal.py status --dataset aita
poetry run python scripts/data_portal.py status --dataset scenario
```

## Integration with Labeling Platform

The data portal CLI integrates seamlessly with the Streamlit labeling app:

1. **Seeds** (`data/humanLabel/seeds/`) are synced to database via `push`
2. **Streamlit app** (`src/labeling_app/app.py`) reads from database
3. **Reviews** are exported via `pull` for analysis notebooks

### Data Flow
```
Seeds (JSONL) → Database (Turso) → Streamlit UI → Reviews (JSONL) → Notebooks
     ↑                                                                    ↓
     └── data_portal.py push ───────────────── data_portal.py pull ──────┘
```

## Configuration

### Environment Variables
The data portal respects these environment variables:

```bash
# Database connection
TURSO_DATABASE_URL=libsql://your-database.turso.io
TURSO_AUTH_TOKEN=your_auth_token

# Data paths
LABEL_DATA_ROOT=data/humanLabel
STREAMLIT_RUNS_ROOT=outputs/runs
```

### File Locations
- **Seeds**: `data/humanLabel/seeds/*.jsonl`
- **Reviews**: `data/humanLabel/reviews/*.jsonl`
- **Runs**: `outputs/runs/<run_id>/run.jsonl`

## Error Handling

The CLI provides comprehensive error handling:

- **Database errors**: Clear messages for connection issues
- **File errors**: Validation of input files and paths
- **Data validation**: Schema validation for JSONL files
- **Dry run mode**: Preview changes before applying

### Common Issues

1. **Missing database credentials**: Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`
2. **Schema not initialized**: Run `init-db` command
3. **File not found**: Verify paths to seeds and run files
4. **Permission errors**: Check file permissions for data directories

## Development

### Adding New Commands
The CLI uses Typer for command definition. Add new commands in `scripts/data_portal.py`:

```python
@app.command("new-command")
def new_command(param: str = typer.Option(...)) -> None:
    """Description of new command."""
    # Implementation
```

### Testing
```bash
# Test database operations
poetry run python scripts/data_portal.py init-db
poetry run python scripts/data_portal.py status --dataset aita

# Test data sync (dry run)
poetry run python scripts/data_portal.py push --dataset scenario
```

For detailed information about the labeling platform, see [src/labeling_app/README.md](../src/labeling_app/README.md).
