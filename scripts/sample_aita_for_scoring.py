#!/usr/bin/env python3
"""Sample ~1000 AITA responses for scoring, excluding existing seed records.

This script:
1. Loads existing AITA seed files (aita_seed.jsonl and aita_seed_v2.jsonl)
   to identify already-sampled records
2. Loads compiled AITA outputs CSV
3. Filters out records that match existing identifiers
4. Performs stratified sampling across models and scenarios
5. Exports new sample to outputs/scoring/sampled_responses.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
EXISTING_SEED_FILE_1 = PROJECT_ROOT / "data" / "humanLabel" / "seeds" / "aita_seed.jsonl"
EXISTING_SEED_FILE_2 = PROJECT_ROOT / "data" / "humanLabel" / "seeds" / "aita_seed_v2.jsonl"
COMPILED_CSV = PROJECT_ROOT / "data" / "humanLabel" / "raw" / "aita_compiled_llm_outputs.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoring"
OUTPUT_FILE = OUTPUT_DIR / "sampled_responses.jsonl"
TARGET_SAMPLE_SIZE = 1000  # New records to sample
RANDOM_STATE = 42
MIN_MODEL_RESPONSES = 20  # Minimum responses required for a model to be included

# Constants for seed formatting
DEFAULT_SOURCE = "reddit-aita-v2"
DEFAULT_VERSION = "2025-01"
DEFAULT_TOPIC = "general"
SEED_RUN_ID = "scoring-run-v1"


def load_existing_seed_identifiers(seed_file: Path) -> set[str]:
    """Load identifiers from existing seed file to exclude from new sample."""
    if not seed_file.exists():
        print(f"Warning: Existing seed file not found: {seed_file}")
        return set()

    identifiers = set()
    with open(seed_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    identifiers.add(record.get("identifier", ""))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(identifiers)} existing identifiers from {seed_file.name}")
    return identifiers


def slugify_model(model_name: str) -> str:
    """Convert model name to identifier-safe slug."""
    safe = str(model_name).lower()
    for ch in " /\\:":
        safe = safe.replace(ch, "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch == "-")
    return safe.strip("-") or "model"


def build_identifier(row: pd.Series) -> str:
    """Build identifier from row data (matches existing format)."""
    scenario_id = int(row["scenario_number"])
    model_slug = slugify_model(row.get("model_name", "model"))
    return f"aita_{scenario_id:03d}_{model_slug}"


def normalize_label(value: object) -> str:
    """Normalize label to YTA, NTA, or UNKNOWN."""
    label = str(value).strip().upper()
    valid_labels = {"YTA", "NTA"}
    return label if label in valid_labels else "UNKNOWN"


def normalize_reddit_label(value: object) -> str:
    """Normalize Reddit label."""
    marker = str(value).strip().upper()
    reddit_yta_markers = {"1", "TRUE", "YTA"}
    reddit_nta_markers = {"0", "FALSE", "NTA"}
    if marker in reddit_yta_markers:
        return "YTA"
    if marker in reddit_nta_markers:
        return "NTA"
    return "UNKNOWN"


def build_prompt_title(row: pd.Series) -> str:
    """Build prompt title from Reddit and AI labels."""
    reddit_label = normalize_reddit_label(row.get("reddit_is_asshole"))
    model_label = normalize_label(row.get("model_is_asshole"))
    return f"R: {reddit_label} AI: {model_label}"


def build_metadata(_: pd.Series) -> dict[str, str]:
    """Build metadata dict."""
    return {
        "source": DEFAULT_SOURCE,
        "version": DEFAULT_VERSION,
        "topic": DEFAULT_TOPIC,
    }


def _build_quota(items: list[int | str], total: int) -> dict[int | str, int]:
    """Build quota dictionary distributing total across items."""
    base, remainder = divmod(total, len(items))
    quota = {item: base for item in items}
    for idx in range(remainder):
        quota[items[idx]] += 1
    return quota


def _prepare_model_scenarios(
    frame: pd.DataFrame, models: list[str], random_state: int
) -> dict[str, deque[int]]:
    """Prepare deque of scenarios per model, shuffled with different seeds."""
    lookup: dict[str, deque[int]] = {}
    for offset, model in enumerate(models):
        scenarios = (
            frame.loc[frame["model_name"] == model, "scenario_number"]
            .drop_duplicates()
            .sample(frac=1.0, random_state=random_state + offset)
            .tolist()
        )
        lookup[model] = deque(int(val) for val in scenarios)
    return lookup


def _index_rows_by_model_and_scenario(
    frame: pd.DataFrame,
) -> dict[tuple[str, int], list[int]]:
    """Index rows by (model_name, scenario_number) -> list of row indices."""
    index: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in frame.iterrows():
        index[(row["model_name"], int(row["scenario_number"]))].append(idx)
    return index


def sample_balanced_models_and_scenarios_dual_quota(
    frame: pd.DataFrame,
    total_n: int,
    random_state: int,
    exclude_identifiers: set[str],
) -> pd.DataFrame:
    """Select rows balancing quotas across model_name and scenario_number.

    Excludes records that match existing identifiers.
    """
    if total_n <= 0:
        raise ValueError("total_n must be positive")

    # Filter out existing identifiers first
    frame = frame.copy()
    frame["identifier"] = frame.apply(build_identifier, axis=1)
    frame = frame[~frame["identifier"].isin(exclude_identifiers)].copy()

    if len(frame) == 0:
        print("Warning: No records available after excluding existing identifiers")
        return frame.iloc[0:0].copy()

    print(f"After excluding existing identifiers: {len(frame)} records available")

    # Filter models with minimum responses
    model_counts = frame["model_name"].value_counts()
    valid_models = model_counts[model_counts >= MIN_MODEL_RESPONSES].index.tolist()
    frame = frame[frame["model_name"].isin(valid_models)].copy()

    if len(frame) == 0:
        print("Warning: No records available after model filtering")
        return frame.iloc[0:0].copy()

    print(f"After model filtering (>= {MIN_MODEL_RESPONSES} responses): {len(frame)} records")
    print(f"Valid models: {', '.join(valid_models)}")

    # Reset index to ensure row indices match filtered frame
    frame = frame.reset_index(drop=True)

    models = sorted(frame["model_name"].unique())
    scenarios = sorted(frame["scenario_number"].unique())

    if not models or not scenarios:
        return frame.iloc[0:0].copy()

    print(f"Available models: {len(models)}")
    print(f"Available scenarios: {len(scenarios)}")

    model_quota = _build_quota(models, total_n)
    scenario_quota = _build_quota(scenarios, total_n)
    model_to_scenarios = _prepare_model_scenarios(frame, models, random_state)
    row_lookup = _index_rows_by_model_and_scenario(frame)

    picked_indices: list[int] = []
    model_idx = 0

    def select_scenario(model: str, enforce_quota: bool) -> int | None:
        """Select a scenario for the given model, respecting quota if enforce_quota=True."""
        queue = model_to_scenarios[model]
        if not queue:
            return None
        for _ in range(len(queue)):
            candidate = queue[0]
            queue.rotate(-1)
            if (model, candidate) not in row_lookup:
                continue
            if not enforce_quota or scenario_quota.get(candidate, 0) > 0:
                return candidate
        return None

    while len(picked_indices) < total_n and any(val > 0 for val in model_quota.values()):
        model = models[model_idx % len(models)]
        model_idx += 1
        if model_quota[model] <= 0:
            continue

        chosen = select_scenario(model, enforce_quota=True)
        if chosen is None:
            chosen = select_scenario(model, enforce_quota=False)
            if chosen is None:
                model_quota[model] = 0
                continue

        pool = row_lookup[(model, chosen)]
        # Use same deterministic selection as original
        position = (model_quota[model] + scenario_quota.get(chosen, 0) + random_state) % len(pool)
        picked_indices.append(pool[position])

        model_quota[model] -= 1
        if scenario_quota.get(chosen, 0) > 0:
            scenario_quota[chosen] -= 1

    if len(picked_indices) < total_n:
        print(f"Warning: Only selected {len(picked_indices)} records (target was {total_n})")

    return frame.loc[picked_indices].reset_index(drop=True)


def prepare_seed_export(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for seed export."""
    export_df = df.copy()
    export_df["identifier"] = export_df.apply(build_identifier, axis=1)
    export_df["prompt_title"] = export_df.apply(build_prompt_title, axis=1)
    export_df["prompt_body"] = export_df["scenario_text"].astype(str).str.strip()
    export_df["model_response_text"] = export_df["model_response"].astype(str)
    export_df["model_id"] = export_df["model_name"].astype(str)
    export_df["run_id"] = SEED_RUN_ID
    export_df["metadata"] = export_df.apply(build_metadata, axis=1)

    ordered = export_df.sort_values(["scenario_number", "model_name"]).reset_index(drop=True)
    columns = [
        "identifier",
        "prompt_title",
        "prompt_body",
        "model_response_text",
        "model_id",
        "run_id",
        "metadata",
    ]
    return ordered[columns]


def export_to_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Export dataframe to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = df.to_dict("records")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            # Ensure metadata is a dict (not string)
            if isinstance(record.get("metadata"), str):
                try:
                    metadata = json.loads(record["metadata"])
                except json.JSONDecodeError:
                    metadata = record["metadata"]
                record["metadata"] = metadata
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} records to {output_path}")


def main() -> None:
    """Main execution."""
    print("=" * 60)
    print("AITA Sampling for Scoring Script")
    print("=" * 60)
    print()

    # Step 1: Load existing seed identifiers from both files
    print("Step 1: Loading existing seed identifiers to exclude...")
    existing_identifiers_1 = load_existing_seed_identifiers(EXISTING_SEED_FILE_1)
    existing_identifiers_2 = load_existing_seed_identifiers(EXISTING_SEED_FILE_2)

    # Merge identifiers from both files
    existing_identifiers = existing_identifiers_1 | existing_identifiers_2
    print(f"Total unique identifiers to exclude: {len(existing_identifiers)}")
    print()

    # Step 2: Load compiled CSV
    print("Step 2: Loading compiled AITA outputs...")
    if not COMPILED_CSV.exists():
        print(f"Error: Compiled CSV not found: {COMPILED_CSV}")
        sys.exit(1)

    df = pd.read_csv(COMPILED_CSV)
    print(f"Loaded {len(df)} records from compiled CSV")
    print(f"Columns: {', '.join(df.columns)}")
    print()

    # Step 3: Validate required columns
    required_cols = {
        "scenario_number",
        "scenario_text",
        "model_response",
        "model_name",
        "model_is_asshole",
        "reddit_is_asshole",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)

    # Step 4: Sample new records
    print("Step 3: Sampling new records (excluding all existing)...")
    print(f"Target sample size: {TARGET_SAMPLE_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    print("Using dual-quota algorithm")
    print()

    sample_df = sample_balanced_models_and_scenarios_dual_quota(
        df,
        total_n=TARGET_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        exclude_identifiers=existing_identifiers,
    )

    print()
    print("New sample distribution:")
    print(f"  Total records: {len(sample_df)}")
    print(f"  Unique scenarios: {sample_df['scenario_number'].nunique()}")
    print(f"  Unique models: {sample_df['model_name'].nunique()}")
    print()
    print("Model distribution:")
    model_counts = sample_df["model_name"].value_counts().sort_index()
    for model, count in model_counts.items():
        print(f"  {model}: {count}")
    print()

    # Step 5: Prepare and export
    print("Step 4: Preparing seed export...")
    seed_export_df = prepare_seed_export(sample_df)

    # Step 6: Export to file
    print(f"Output file: {OUTPUT_FILE}")
    export_to_jsonl(seed_export_df, OUTPUT_FILE)

    print()
    print("=" * 60)
    print("Sampling complete!")
    print(f"New sample file: {OUTPUT_FILE}")
    print(f"  - New records sampled: {len(seed_export_df)}")
    print(f"  - Existing records excluded: {len(existing_identifiers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
