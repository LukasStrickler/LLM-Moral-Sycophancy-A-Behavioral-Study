"""CLI entrypoint for ModernBERT sycophancy scoring pipeline.

Commands:
    create-dataset: Combine CSV files and create train/val splits
    train: Train ModernBERT regression model (with optional Optuna search)
    score: Score responses using trained model and generate plots

Note: The train and score commands execute src/scoring/*.py as subprocesses
because those modules load data at import time. This CLI provides a unified
interface with proper argument handling.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(
    help="ModernBERT sycophancy scoring pipeline.",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(ctx: typer.Context) -> None:
    """ModernBERT sycophancy scoring pipeline."""
    if ctx.invoked_subcommand is None:
        console.print("[bold cyan]ModernBERT Sycophancy Scoring Pipeline[/bold cyan]\n")
        console.print("Available commands:")
        console.print(
            "  [green]create-dataset[/green]  Combine CSV files and create train/val splits"
        )
        console.print("  [green]train[/green]           Train ModernBERT regression model")
        console.print("  [green]score[/green]           Score responses and generate plots")
        console.print(
            "\nRun [cyan]poetry run python scripts/score_sycophancy.py COMMAND --help[/cyan] for details."
        )


@app.command("create-dataset")
def create_dataset_cmd(
    file1: Path = typer.Option(
        Path("combined_consensus.csv"),
        "--file1",
        "-f1",
        help="First input CSV file",
    ),
    file2: Path = typer.Option(
        Path("final_consensus_export_v2_response.csv"),
        "--file2",
        "-f2",
        help="Second input CSV file",
    ),
    file3: Path = typer.Option(
        Path("final_consensus_export_v2-1_FINALEDITS20-12.csv"),
        "--file3",
        "-f3",
        help="Third input CSV file",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Output directory for generated files",
    ),
    test_size: float = typer.Option(
        0.15,
        "--test-size",
        "-t",
        help="Fraction of data to use for validation (0.0-1.0)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Combine CSV files and create train/val splits.

    Takes multiple CSV files with human consensus scores, normalizes column names,
    combines them into a single dataset, and creates an 85/15 train/validation split.

    Output files:
        - Aggregate_file.csv: Combined dataset
        - training_data.csv: Training split
        - validation_data.csv: Validation split
    """
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: [cyan]poetry install --with ml[/cyan]")
        raise typer.Exit(1)

    console.print("[bold]Creating dataset...[/bold]\n")

    # Validate input files
    for f in [file1, file2, file3]:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)

    # Load and normalize datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    console.print(f"  File 1: {len(df1)} rows from {file1}")
    console.print(f"  File 2: {len(df2)} rows from {file2}")
    console.print(f"  File 3: {len(df3)} rows from {file3}")

    # Normalize columns
    cols_df1 = ["response_id", "prompt_body", "model_response_text", "chosen_consensus"]
    df1_clean = df1[cols_df1].copy()

    cols_df2_src = ["response_id", "prompt_body", "model_response_text", "final_consensus"]
    df2_clean = df2[cols_df2_src].copy()
    df2_clean = df2_clean.rename(columns={"final_consensus": "chosen_consensus"})

    cols_df3_src = ["response_id", "prompt_body", "model_response_text", "final_score"]
    df3_clean = df3[cols_df3_src].copy()
    df3_clean = df3_clean.rename(columns={"final_score": "chosen_consensus"})

    # Combine and clean
    df_combined = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)
    required_cols = ["response_id", "prompt_body", "model_response_text", "chosen_consensus"]
    df_final = df_combined.dropna(subset=required_cols).reset_index(drop=True)

    console.print(f"\n[green]Combined dataset: {len(df_final)} samples[/green]")

    if len(df_final) < 150:
        console.print(f"[red]Not enough data points ({len(df_final)}) for a robust split.[/red]")
        raise typer.Exit(1)

    # Save aggregate file
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "Aggregate_file.csv"
    df_final.to_csv(aggregate_path, index=False)
    console.print(f"  Saved: {aggregate_path}")

    # Create train/val split
    train_df, val_df = train_test_split(
        df_final,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    train_path = output_dir / "training_data.csv"
    val_path = output_dir / "validation_data.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    console.print(f"  Saved: {train_path} ({len(train_df)} samples)")
    console.print(f"  Saved: {val_path} ({len(val_df)} samples)")
    console.print("\n[green]✓ Dataset creation complete[/green]")


def _create_train_config_snippet(
    train_csv: Path,
    val_csv: Path,
    output_dir: Path,
    model_name: str,
    use_optuna: bool,
    n_trials: int,
    use_kfold: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> str:
    """Generate Python code snippet to override train.py globals."""
    return f'''
# --- CLI Configuration Override ---
TRAIN_CSV_PATH = "{train_csv.resolve()}"
VAL_CSV_PATH = "{val_csv.resolve()}"
MODEL_PATH = "{model_name}"
OUTPUT_DIR = "{output_dir.resolve()}"
seed = {seed}
RUN_SINGLE_MODEL = {not use_optuna}
use_kfold = {use_kfold}
n_trials_optuna = {n_trials}
use_upsampling_grid = False

SINGLE_MODEL_PARAMS = {{
    "learning_rate": {learning_rate},
    "per_device_train_batch_size": {batch_size},
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.12583141206455262,
    "num_train_epochs": {epochs},
    "adam_epsilon": 1.071087092582853e-08,
    "warmup_ratio": 0.18651889995585025,
    "lr_scheduler_type": "constant"
}}
# --- End CLI Configuration ---
'''


@app.command("train")
def train_cmd(
    train_csv: Path = typer.Option(
        Path("training_data.csv"),
        "--train-csv",
        help="Path to training CSV file",
    ),
    val_csv: Path = typer.Option(
        Path("validation_data.csv"),
        "--val-csv",
        help="Path to validation CSV file",
    ),
    output_dir: Path = typer.Option(
        Path("./modernbert_chosen_consensus_advanced"),
        "--output-dir",
        "-o",
        help="Output directory for model and logs",
    ),
    model_name: str = typer.Option(
        "answerdotai/ModernBERT-base",
        "--model",
        "-m",
        help="HuggingFace model name or path",
    ),
    use_optuna: bool = typer.Option(
        False,
        "--optuna",
        help="Run Optuna hyperparameter search",
    ),
    n_trials: int = typer.Option(
        180,
        "--n-trials",
        help="Number of Optuna trials (if --optuna)",
    ),
    use_kfold: bool = typer.Option(
        False,
        "--kfold",
        help="Use K-fold cross-validation",
    ),
    epochs: int = typer.Option(
        5,
        "--epochs",
        "-e",
        help="Number of training epochs (single run mode)",
    ),
    batch_size: int = typer.Option(
        3,
        "--batch-size",
        "-b",
        help="Training batch size",
    ),
    learning_rate: float = typer.Option(
        8.7e-5,
        "--lr",
        help="Learning rate",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show configuration without running training",
    ),
) -> None:
    """Train ModernBERT regression model for sycophancy scoring.

    Supports two modes:
        - Single run: Train with specified hyperparameters (default)
        - Optuna search: Hyperparameter optimization with optional K-fold CV

    The model predicts sycophancy scores on a [-1, 1] scale where:
        - -1.0: Challenges OP (non-sycophantic)
        -  0.0: Neutral
        - +1.0: Agrees with OP (sycophantic)

    Note: This command executes src/scoring/train.py as a subprocess because
    the training module loads data at import time.
    """
    # Validate input files
    for f, name in [(train_csv, "Training CSV"), (val_csv, "Validation CSV")]:
        if not f.exists():
            console.print(f"[red]{name} not found: {f}[/red]")
            console.print("Run [cyan]create-dataset[/cyan] first to generate training data.")
            raise typer.Exit(1)

    console.print("[bold]ModernBERT Sycophancy Scorer Training[/bold]\n")
    console.print(f"  Training data:   {train_csv.resolve()}")
    console.print(f"  Validation data: {val_csv.resolve()}")
    console.print(f"  Base model:      {model_name}")
    console.print(f"  Output:          {output_dir.resolve()}")
    console.print(f"  Mode:            {'Optuna search' if use_optuna else 'Single run'}")
    if use_optuna:
        console.print(f"  Trials:          {n_trials}")
        console.print(f"  K-fold:          {use_kfold}")
    else:
        console.print(f"  Epochs:          {epochs}")
        console.print(f"  Batch size:      {batch_size}")
        console.print(f"  Learning rate:   {learning_rate}")
    console.print()

    # Generate config snippet
    config_snippet = _create_train_config_snippet(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=output_dir,
        model_name=model_name,
        use_optuna=use_optuna,
        n_trials=n_trials,
        use_kfold=use_kfold,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    if dry_run:
        console.print("[yellow]Dry run - showing configuration:[/yellow]")
        console.print(config_snippet)
        console.print("\n[cyan]To run training, remove --dry-run flag[/cyan]")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write temporary config file
    config_file = output_dir / "_cli_config.py"
    config_file.write_text(config_snippet)

    console.print(f"[cyan]Config written to: {config_file}[/cyan]")
    console.print("[yellow]Starting training...[/yellow]\n")
    console.print("=" * 60)

    # Execute training via subprocess
    # We modify the train.py source temporarily by prepending config
    train_module_path = PROJECT_ROOT / "src" / "scoring" / "train.py"
    original_content = train_module_path.read_text()

    # Find the line after imports to insert config
    # Insert after the global config section (after line ~104)
    lines = original_content.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("UPSAMPLING_GRID = ["):
            # Find end of UPSAMPLING_GRID
            for j in range(i, len(lines)):
                if lines[j].strip() == "]":
                    insert_idx = j + 1
                    break
            break

    if insert_idx is None:
        # Fallback: insert after line 104
        insert_idx = 104

    # Create modified script content
    modified_lines = lines[:insert_idx] + [config_snippet] + lines[insert_idx:]
    modified_content = "\n".join(modified_lines)

    # Write to temp file
    temp_train_file = output_dir / "_train_configured.py"
    temp_train_file.write_text(modified_content)

    try:
        # Run the modified training script
        result = subprocess.run(
            [sys.executable, str(temp_train_file)],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )

        console.print("=" * 60)
        if result.returncode == 0:
            console.print("\n[green]✓ Training complete![/green]")
            console.print(f"  Model saved to: {output_dir.resolve()}")
        else:
            console.print(f"\n[red]✗ Training failed with exit code {result.returncode}[/red]")
            raise typer.Exit(result.returncode)
    finally:
        # Cleanup temp files
        if temp_train_file.exists():
            temp_train_file.unlink()
        if config_file.exists():
            config_file.unlink()


@app.command("score")
def score_cmd(
    input_csv: Path = typer.Option(
        Path("results.csv"),
        "--input",
        "-i",
        help="Input CSV with model responses (must have 'model_id', 'response_text' columns)",
    ),
    model_dir: Path = typer.Option(
        Path("modernbert_chosen_consensus_advanced"),
        "--model-dir",
        "-m",
        help="Directory containing trained model",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots and eval.txt (default: model_dir)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show configuration without running scoring",
    ),
) -> None:
    """Score responses using trained ModernBERT model.

    Reads responses from a CSV file, scores each with the trained model,
    and generates analysis plots and summary statistics.

    Required CSV columns:
        - model_id: Identifier for the LLM that generated the response
        - response_text: The response text to score

    Output:
        - eval.txt: Per-model summary statistics
        - plots/: Visualization directory with analysis charts

    Note: This command executes src/scoring/inference.py as a subprocess.
    """
    # Validate inputs
    if not input_csv.exists():
        console.print(f"[red]Input CSV not found: {input_csv}[/red]")
        raise typer.Exit(1)

    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        console.print("Train a model first with: [cyan]score_sycophancy.py train[/cyan]")
        raise typer.Exit(1)

    if output_dir is None:
        output_dir = model_dir

    console.print("[bold]ModernBERT Sycophancy Scorer Inference[/bold]\n")
    console.print(f"  Input:  {input_csv.resolve()}")
    console.print(f"  Model:  {model_dir.resolve()}")
    console.print(f"  Output: {output_dir.resolve()}")
    console.print()

    # Generate config override
    # Note: We must include df/total_rows because the skip logic removes them
    config_snippet = f'''
# --- CLI Configuration Override ---
CSV_PATH = "{input_csv.resolve()}"
MODEL_DIR = "{model_dir.resolve()}"
OUTPUT_PATH = Path("{output_dir.resolve()}") / "eval.txt"
PLOTS_DIR = Path("{output_dir.resolve()}") / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Count rows up front so tqdm progress is accurate
df = pd.read_csv(CSV_PATH)
total_rows = len(df)
# --- End CLI Configuration ---
'''

    if dry_run:
        console.print("[yellow]Dry run - showing configuration:[/yellow]")
        console.print(config_snippet)
        console.print("\n[cyan]To run scoring, remove --dry-run flag[/cyan]")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[yellow]Starting scoring...[/yellow]\n")
    console.print("=" * 60)

    # Read and modify inference script
    inference_module_path = PROJECT_ROOT / "src" / "scoring" / "inference.py"
    original_content = inference_module_path.read_text()

    # Find insertion point (after imports, before data loading)
    lines = original_content.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "CSV_PATH = " in line and not line.strip().startswith("#"):
            insert_idx = i
            break

    if insert_idx is None:
        insert_idx = 40  # Fallback

    # Create modified script - replace the config lines
    modified_lines = []
    skip_until_device = False
    for i, line in enumerate(lines):
        if i == insert_idx:
            modified_lines.append(config_snippet)
            skip_until_device = True
        elif skip_until_device:
            if line.strip().startswith("device = ") or line.strip().startswith("# Device"):
                skip_until_device = False
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    modified_content = "\n".join(modified_lines)

    # Write to temp file
    temp_inference_file = output_dir / "_inference_configured.py"
    temp_inference_file.write_text(modified_content)

    try:
        # Run the modified inference script
        result = subprocess.run(
            [sys.executable, str(temp_inference_file)],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )

        console.print("=" * 60)
        if result.returncode == 0:
            console.print("\n[green]✓ Scoring complete![/green]")
            console.print(f"  Results: {output_dir / 'eval.txt'}")
            console.print(f"  Plots:   {output_dir / 'plots'}")
        else:
            console.print(f"\n[red]✗ Scoring failed with exit code {result.returncode}[/red]")
            raise typer.Exit(result.returncode)
    finally:
        # Cleanup temp file
        if temp_inference_file.exists():
            temp_inference_file.unlink()


if __name__ == "__main__":
    app()
