"""
Analyze and compare sycophancy scores from two specific models (OLMo vs Grok) stored in
results_full.csv. The script writes the top prompt divergences to text/CSV and saves a
scatter plot of all scores aligned by prompt.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_FULL_PATH = "results_full.csv"

OLMO_ID = "allenai/olmo-3-32b-think"
GROK_ID = "x-ai/grok-4.1-fast:free"

TOP_K = 3

# If you have multiple rows per (prompt_body, model_id), choose an aggregation:
# - "mean": average score per prompt per model
# - "first": take first occurrence
AGG = "mean"

OUT_TXT = "divergence.txt"
OUT_CSV = "top3_divergent_prompts_olmo_vs_grok.csv"
OUT_PLOT = "olmo_vs_grok_all_scores.png"


def load_and_prepare(path: str):
    """
    Load results_full.csv, keep only the two target models, and optionally aggregate
    multiple generations per prompt/model into a single score.
    """
    df = pd.read_csv(path)

    required = {"model_id", "prompt_body", "response_text", "sycophancy_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df[df["model_id"].isin([OLMO_ID, GROK_ID])].copy()
    df["sycophancy_score"] = pd.to_numeric(df["sycophancy_score"], errors="coerce")
    df = df.dropna(subset=["sycophancy_score", "prompt_body", "response_text"])

    # Prefer prompt_identifier when available; it reduces the risk of collisions if prompt_body repeats.
    join_keys = ["prompt_body"]
    if "prompt_identifier" in df.columns:
        join_keys = ["prompt_identifier", "prompt_body"]

    if AGG == "mean":
        df = (
            df.groupby(join_keys + ["model_id"], as_index=False)
            .agg(
                sycophancy_score=("sycophancy_score", "mean"),
                response_text=("response_text", "first"),
            )
        )
    elif AGG == "first":
        df = (
            df.sort_values(join_keys)
            .drop_duplicates(subset=join_keys + ["model_id"], keep="first")
        )
    else:
        raise ValueError("AGG must be either 'mean' or 'first'.")

    return df, join_keys


def compute_top_divergences(df: pd.DataFrame, join_keys):
    """Find the TOP_K prompts where the two models disagree most in score."""
    olmo = (
        df[df["model_id"] == OLMO_ID]
        .drop(columns=["model_id"])
        .rename(
            columns={
                "sycophancy_score": "score_olmo",
                "response_text": "response_olmo",
            }
        )
    )

    grok = (
        df[df["model_id"] == GROK_ID]
        .drop(columns=["model_id"])
        .rename(
            columns={
                "sycophancy_score": "score_grok",
                "response_text": "response_grok",
            }
        )
    )

    merged = pd.merge(olmo, grok, on=join_keys, how="inner")

    merged["abs_diff"] = (merged["score_olmo"] - merged["score_grok"]).abs()
    merged["signed_diff"] = merged["score_olmo"] - merged["score_grok"]

    top = (
        merged.sort_values("abs_diff", ascending=False)
        .head(TOP_K)
        .reset_index(drop=True)
    )
    return top


def write_divergence_txt(top: pd.DataFrame, join_keys, path: str):
    """Write a readable report with prompt text, both responses, and both scores."""
    lines = []
    lines.append(f"Top-{len(top)} prompt divergences between:")
    lines.append(f"  OLMO: {OLMO_ID}")
    lines.append(f"  GROK: {GROK_ID}")
    lines.append("")

    for i, row in top.iterrows():
        lines.append("=" * 110)
        lines.append(f"Rank: {i + 1}")
        for k in join_keys:
            lines.append(f"{k}: {row[k]}")
        lines.append(f"abs_diff: {row['abs_diff']:.6f}")
        lines.append(f"signed_diff (olmo - grok): {row['signed_diff']:.6f}")
        lines.append("")

        lines.append(f"[{OLMO_ID}] score: {row['score_olmo']:.6f}")
        lines.append("response_text:")
        lines.append(str(row["response_olmo"]))
        lines.append("")

        lines.append(f"[{GROK_ID}] score: {row['score_grok']:.6f}")
        lines.append("response_text:")
        lines.append(str(row["response_grok"]))
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_all_scores(df: pd.DataFrame, join_keys, path: str):
    """
    Scatter plot of all scores, aligned so both models share the same x-position
    for the same prompt (one x-index per unique prompt key).
    """
    plot_df = df.copy()

    prompt_key = plot_df[join_keys].astype(str).agg("||".join, axis=1)
    plot_df["prompt_idx"], _ = pd.factorize(prompt_key, sort=True)  # stable ordering [web:77]

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 6))

    palette = {OLMO_ID: "#1f77b4", GROK_ID: "#ff7f0e"}

    sns.scatterplot(
        data=plot_df,
        x="prompt_idx",
        y="sycophancy_score",
        hue="model_id",
        palette=palette,
        alpha=0.7,
        s=35,
        legend="full",
    )

    plt.axhline(0, color="black", linewidth=1, alpha=0.5)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Prompt index")
    plt.ylabel("Sycophancy score")
    plt.title("All sycophancy scores: OLMO vs Grok")

    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    df, join_keys = load_and_prepare(RESULTS_FULL_PATH)

    plot_all_scores(df, join_keys, OUT_PLOT)

    top = compute_top_divergences(df, join_keys)
    top.to_csv(OUT_CSV, index=False)
    write_divergence_txt(top, join_keys, OUT_TXT)


if __name__ == "__main__":
    main()
