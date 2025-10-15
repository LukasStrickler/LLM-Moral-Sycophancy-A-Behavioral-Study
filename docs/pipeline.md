# Data Pipeline

This document describes the end‑to‑end flow for our study: training a RoBERTa regression scorer on human‑labeled Reddit AITA data, and applying it to benchmark LLM responses in equalized landlord/tenant scenarios with a human audit for validation.

```mermaid
%%{init: { "theme": "neutral" }}%%
flowchart TD
  subgraph TRAINING[Model Training]
    T1["Data Ingestion<br/>(Reddit AITA)"]:::data --> T2["Human Labeling<br/>(Streamlit, scale -1..1)"]:::human
    T2 --> T3["Model Training<br/>(RoBERTa regression)"]:::model
  end

  subgraph BENCHMARKING[Benchmarking]
    B0["Prompt Generation<br/>(scenarios)"]:::script --> B1["Response Collection<br/>(LLMs)"]:::script
    B1 --> B2["Scoring<br/>(RoBERTa inference)"]:::model
    B2 --> B3["Human Audit<br/>(sample relabel)"]:::human
    B3 --> B4["Analysis<br/>(human vs model)"]:::eval
  end

  T3 -. provides model .-> B2

  %% Classes
  classDef data fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1;
  classDef script fill:#FFF8E1,stroke:#FB8C00,color:#E65100;
  classDef model fill:#E8F5E9,stroke:#43A047,color:#1B5E20;
  classDef human fill:#FFEBEE,stroke:#E53935,color:#B71C1C;
  classDef eval fill:#F3E5F5,stroke:#8E24AA,color:#4A148C;
```

## Steps

### 1. Data Ingestion

- Source: LLM responses to r/AmITheAsshole prompts (Cheng et al., 2025)
- Sample: random 1,000 examples for initial labeling
- Stratification: balance by topic/theme, response length buckets, stance polarity
- Integrity: keep raw pulls immutable and versioned (date, commit, source URL)
- Storage: separate raw vs. derived folders to prevent accidental edits

### 2. Human Labeling

- Interface: Streamlit slider in [-1, 1] with optional notes
- Raters: 2–3 independent scorers per item (committee protocol)
- Decision rule: same sign → average; mixed signs → discuss and record consensus
- Recording: store raw rater scores, consensus score, rater IDs, timestamps
- Quality: compute inter‑rater agreement (e.g., Pearson/Spearman, SD) on overlap

### 3. Model Training

- Model: RoBERTa base with a regression head (target ∈ [-1, 1])
- Data splits: training on consensus labels; strict held‑out evaluation set
- Reproducibility: fixed seeds; log config and checkpoint path

### 4. Prompt Generation

- Equalization: matched landlord/tenant pairs with mirrored content and structure
- Options: include neutral prompts for distribution checks (not used for human scoring)

### 5. Response Collection

- Execution: `scripts/run_benchmark.py` over the equalized prompt grid
- Output: `outputs/runs/RUN_ID/run.jsonl` with one record per model–prompt
- Metadata: log model/version, parameters (temperature, max tokens), seeds

### 6. Scoring

- Engine: apply the trained RoBERTa model locally (no API calls)
- Output: `run_scored.jsonl` with a continuous score per response

### 7. Human Audit

- Sample: randomly select 100–150 newly scored responses
- Blinding: raters do not see model scores in advance
- Protocol: same as labeling (independent → consensus); store raw/consensus/agreement
- Purpose: validate scoring quality and surface edge cases for rubric tuning

### 8. Analysis

- Metrics: compare model vs. human (MAE, Pearson) per model and perspective
- Diagnostics: analyze disagreements and residuals by scenario factors
- Reporting: highlight systematic gaps; recommend calibration/rubric adjustments
