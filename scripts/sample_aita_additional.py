#!/usr/bin/env python3
"""Sample additional AITA responses for labeling, excluding existing seed records.

This script:
1. Loads existing AITA seed file to identify already-sampled records
2. Loads compiled AITA outputs CSV
3. Filters out records that match existing identifiers
4. Performs stratified sampling across models and scenarios
5. Exports new sample to a versioned seed file
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
EXISTING_SEED_FILE = PROJECT_ROOT / "data" / "humanLabel" / "seeds" / "aita_seed.jsonl"
CONSENSUS_CSV = PROJECT_ROOT / "data" / "humanLabel" / "reviews" / "final_consensus_export.csv"
COMPILED_CSV = PROJECT_ROOT / "data" / "humanLabel" / "raw" / "aita_compiled_llm_outputs.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "humanLabel" / "seeds"
TARGET_SAMPLE_SIZE = 150  # New records to sample
RANDOM_STATE = 42
MIN_MODEL_RESPONSES = 20  # Minimum responses required for a model to be included

# Constants for seed formatting
DEFAULT_SOURCE = "reddit-aita-v2"
DEFAULT_VERSION = "2025-01"
DEFAULT_TOPIC = "general"
SEED_RUN_ID = "aita-balanced-v1"  # Keep same run_id for consistency


def load_existing_seed_identifiers(seed_file: Path) -> Set[str]:
    """Load identifiers from existing seed file to exclude from new sample."""
    if not seed_file.exists():
        print(f"Warning: Existing seed file not found: {seed_file}")
        print("Proceeding without exclusion list (first time sampling)")
        return set()
    
    identifiers = set()
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    identifiers.add(record.get("identifier", ""))
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(identifiers)} existing identifiers from {seed_file.name}")
    return identifiers


def load_identifiers_and_responses_from_consensus_csv(csv_file: Path) -> tuple[Set[str], Set[str]]:
    """Load identifiers and response texts from consensus CSV.
    
    Returns:
        - Set of identifiers (from database query)
        - Set of normalized response texts (for content-based exclusion)
    """
    if not csv_file.exists():
        print(f"Warning: Consensus CSV not found: {csv_file}")
        return set(), set()
    
    # Load response_ids and response texts from CSV
    response_ids = set()
    response_texts = set()
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    response_id = int(row["response_id"])
                    response_ids.add(response_id)
                    
                    # Also extract response text for content-based exclusion
                    response_text = row.get("model_response_text", "").strip()
                    if response_text:
                        # Normalize for comparison (remove extra whitespace)
                        normalized = " ".join(response_text.split())
                        response_texts.add(normalized)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return set(), set()
    
    if not response_ids:
        print(f"No response_ids found in {csv_file.name}")
        return set(), set()
    
    print(f"Found {len(response_ids)} response_ids and {len(response_texts)} unique responses in consensus CSV")
    
    # Query database to get identifiers
    identifiers = set()
    try:
        import os
        os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
        os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))
        
        from src.labeling_app.core.models import Dataset
        from src.labeling_app.db import client_scope
        from src.labeling_app.settings import get_settings
        
        settings = get_settings()
        
        with client_scope(settings) as client:
            # Query in chunks
            chunk_size = 50
            response_ids_list = list(response_ids)
            
            for i in range(0, len(response_ids_list), chunk_size):
                chunk = response_ids_list[i:i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                
                query = f"""
                    SELECT identifier
                    FROM llm_responses
                    WHERE dataset = ? AND id IN ({placeholders})
                """
                
                params = [Dataset.AITA.value] + chunk
                result = client.execute(query, params)
                
                for row in result.rows:
                    if row and row[0]:
                        identifiers.add(row[0])
        
        print(f"Loaded {len(identifiers)} identifiers from database (from consensus CSV)")
    except Exception as e:
        print(f"Warning: Could not query database for identifiers: {e}")
        print("Proceeding without identifier exclusion (database may not be configured)")
    
    return identifiers, response_texts




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


def build_metadata(_: pd.Series) -> Dict[str, str]:
    """Build metadata dict."""
    return {
        "source": DEFAULT_SOURCE,
        "version": DEFAULT_VERSION,
        "topic": DEFAULT_TOPIC,
    }


def _build_quota(items: List[int | str], total: int) -> Dict[int | str, int]:
    """Build quota dictionary distributing total across items."""
    base, remainder = divmod(total, len(items))
    quota = {item: base for item in items}
    for idx in range(remainder):
        quota[items[idx]] += 1
    return quota


def _prepare_model_scenarios(
    frame: pd.DataFrame, models: List[str], random_state: int
) -> Dict[str, Deque[int]]:
    """Prepare deque of scenarios per model, shuffled with different seeds."""
    lookup: Dict[str, Deque[int]] = {}
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
) -> Dict[Tuple[str, int], List[int]]:
    """Index rows by (model_name, scenario_number) -> list of row indices."""
    index: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for idx, row in frame.iterrows():
        index[(row["model_name"], int(row["scenario_number"]))].append(idx)
    return index


def sample_balanced_models_and_scenarios_dual_quota(
    frame: pd.DataFrame,
    total_n: int,
    random_state: int,
    exclude_identifiers: Set[str],
    exclude_response_texts: Set[str] | None = None,
) -> pd.DataFrame:
    """Select rows balancing quotas across model_name and scenario_number.
    
    This is the EXACT same algorithm as used in the original notebook.
    Excludes records that match existing identifiers OR response text content.
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
    
    # Also filter out by response text content if provided
    if exclude_response_texts:
        initial_count = len(frame)
        # Normalize response texts for comparison
        frame["normalized_response"] = frame["model_response"].astype(str).apply(
            lambda x: " ".join(x.strip().split())
        )
        frame = frame[~frame["normalized_response"].isin(exclude_response_texts)].copy()
        frame = frame.drop(columns=["normalized_response"])
        
        excluded_by_content = initial_count - len(frame)
        if excluded_by_content > 0:
            print(f"After excluding by response content: {len(frame)} records available (excluded {excluded_by_content} duplicates)")
    
    if len(frame) == 0:
        print("Warning: No records available after all exclusions")
        return frame.iloc[0:0].copy()
    
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
    
    picked_indices: List[int] = []
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
                    record["metadata"] = json.loads(record["metadata"])
                except json.JSONDecodeError:
                    pass
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Exported {len(records)} records to {output_path}")




def main() -> None:
    """Main execution."""
    print("=" * 60)
    print("AITA Additional Sampling Script")
    print("=" * 60)
    print()
    
    # Step 1: Load existing seed identifiers (all of them) to exclude from new sample
    print("Step 1: Loading all existing seed identifiers to exclude...")
    existing_identifiers = load_existing_seed_identifiers(EXISTING_SEED_FILE)
    print(f"Will exclude all {len(existing_identifiers)} existing identifiers from new sample")
    print()
    
    # Step 1b: Load identifiers and response texts from consensus export (already reviewed records)
    print("Step 1b: Loading identifiers and responses from consensus export (already reviewed)...")
    consensus_identifiers, consensus_response_texts = load_identifiers_and_responses_from_consensus_csv(CONSENSUS_CSV)
    if consensus_identifiers:
        print(f"Will exclude {len(consensus_identifiers)} identifiers from consensus export")
        # Merge with existing identifiers
        existing_identifiers.update(consensus_identifiers)
    if consensus_response_texts:
        print(f"Will exclude {len(consensus_response_texts)} response texts from consensus export (content-based exclusion)")
    if consensus_identifiers or consensus_response_texts:
        print(f"Total identifiers to exclude: {len(existing_identifiers)}")
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
    
    # Step 4: Sample new records using the EXACT same algorithm as original
    print("Step 3: Sampling new records (excluding all existing)...")
    print(f"Target sample size: {TARGET_SAMPLE_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Using dual-quota algorithm (same as original notebook)")
    print()
    
    sample_df = sample_balanced_models_and_scenarios_dual_quota(
        df,
        total_n=TARGET_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        exclude_identifiers=existing_identifiers,
        exclude_response_texts=consensus_response_texts,
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
    output_file = OUTPUT_DIR / "aita_seed_v2.jsonl"
    
    print(f"Output file: {output_file.name}")
    export_to_jsonl(seed_export_df, output_file)
    
    print()
    print("=" * 60)
    print("Sampling complete!")
    print(f"New seed file: {output_file}")
    print(f"  - New records sampled: {len(seed_export_df)}")
    print(f"  - Existing records excluded: {len(existing_identifiers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

