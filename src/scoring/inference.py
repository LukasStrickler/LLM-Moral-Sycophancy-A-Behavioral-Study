"""
Sycophancy scoring + analysis for LLM responses.

This script loads responses from `results.csv`, scores each response using the
classifier stored in `MODEL_DIR`, writes a per-model summary to `eval.txt`, and
saves several plots into `{MODEL_DIR}/plots`.

Quick expectations:
- `results.csv` includes: `model_id`, `response_text`
- The classifier returns a single scalar score in [-1, 1]
- Every `model_id` in the CSV must appear in `MODEL_TO_GROUP` (script fails fast)
"""

import csv
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# ==========================================
# STYLE CONFIGURATION (Analytical Style)
# ==========================================
sns.set_style('whitegrid')
sns.set_context("notebook", font_scale=1.1)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['lines.linewidth'] = 2

torch.set_float32_matmul_precision("highest")

# Paths
# Input: CSV with model outputs
CSV_PATH = "results.csv"

# Input: local Hugging Face model directory (tokenizer + weights + config)
MODEL_DIR = "modernbert_chosen_consensus_advanced"

# Output: summary file written inside the model folder
OUTPUT_PATH = Path(MODEL_DIR) / "eval.txt"

# Output: plots go under {MODEL_DIR}/plots
PLOTS_DIR = Path(MODEL_DIR) / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Count rows up front so tqdm progress is accurate
df = pd.read_csv(CSV_PATH)
total_rows = len(df)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# model_id -> list of sycophancy scores
scores_by_model = defaultdict(list)

# Read CSV and score each response_text
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, total=total_rows, desc="Scoring responses"):
        model_id = row["model_id"]
        text = row["response_text"]

        if not text:
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()

        # The model is trained to output the score directly
        score = float(logits.item())
        scores_by_model[model_id].append(score)

# Write per-model summary + a global aggregate row
all_scores = []
avg_scores = {}

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    out.write("model_id,count,avg_sycophancy\n")
    for model_id, scores in sorted(scores_by_model.items()):
        count = len(scores)
        avg_score = sum(scores) / count
        avg_scores[model_id] = avg_score
        all_scores.extend(scores)
        out.write(f"{model_id},{count},{avg_score:.6f}\n")

    if all_scores:
        global_avg = sum(all_scores) / len(all_scores)
        out.write("\n")
        out.write(f"ALL_MODELS,{len(all_scores)},{global_avg:.6f}\n")

# ==========================================
# GROUPING CONFIGURATION
# ==========================================
# Used for grouped plots and consistent coloring.
MODEL_TO_GROUP = {
    # Flagship Models (5)
    "anthropic/claude-opus-4.5": "Flagship Models",
    "google/gemini-3-pro-preview": "Flagship Models",
    "openai/gpt-5.1": "Flagship Models",
    "qwen/qwen3-max": "Flagship Models",
    "mistralai/mistral-medium-3.1": "Flagship Models",

    # Open-Weight Models (4)
    "deepseek/deepseek-r1": "Open-Weight Models",
    "allenai/olmo-3-32b-think": "Open-Weight Models",
    "openai/gpt-oss-120b": "Open-Weight Models",
    "moonshotai/kimi-k2-thinking": "Open-Weight Models",

    # Specialized Models (3)
    "x-ai/grok-4.1-fast:free": "Specialized Models",
    "amazon/nova-premier-v1": "Specialized Models",
    "google/gemma-3n-e4b-it": "Specialized Models",

    # Efficient Models (2)
    "google/gemini-2.5-flash": "Efficient Models",
    "anthropic/claude-sonnet-4.5": "Efficient Models",

    # Historical Models (2)
    "google/gemini-2.5-pro": "Historical Models",
    "anthropic/claude-sonnet-4": "Historical Models",
}

GROUP_ORDER = [
    "Flagship Models",
    "Open-Weight Models",
    "Specialized Models",
    "Efficient Models",
    "Historical Models",
]

GROUP_PALETTE = dict(
    zip(
        GROUP_ORDER,
        sns.color_palette("Set2", n_colors=len(GROUP_ORDER))
    )
)

# ==========================================
# PLOTTING SECTION
# ==========================================

# Long-form dataframe for seaborn plots
plot_data = []
for m, s in scores_by_model.items():
    for val in s:
        plot_data.append({"Model": m, "Score": val})
df_plot = pd.DataFrame(plot_data)

if not df_plot.empty:
    df_plot["Group"] = df_plot["Model"].map(MODEL_TO_GROUP)

    # This guard is intentional: it prevents silently dropping or mis-grouping models.
    missing = sorted(df_plot.loc[df_plot["Group"].isna(), "Model"].unique().tolist())
    if missing:
        raise ValueError(
            "Some model_id values are not present in MODEL_TO_GROUP mapping:\n"
            + "\n".join(missing)
        )

    df_plot["Group"] = pd.Categorical(df_plot["Group"], categories=GROUP_ORDER, ordered=True)
    df_plot = df_plot.sort_values(["Group", "Model"])

# 01) Global histogram over all models
if all_scores:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        all_scores,
        bins=30,
        kde=True,
        stat="density",
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        line_kws={'linewidth': 2, 'color': 'navy'}
    )
    plt.xlim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xlabel("Sycophancy Score")
    plt.ylabel("Density")
    plt.title("Global Score Distribution", pad=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_global_hist.png", dpi=300)
    plt.close()

# 02) Bar chart of average scores per model (alphabetical)
if avg_scores:
    sorted_items = sorted(avg_scores.items(), key=lambda x: x[0])
    sorted_models = [k for k, _ in sorted_items]
    sorted_values = np.array([v for _, v in sorted_items])

    df_avg = pd.DataFrame({"Model": sorted_models, "AvgScore": sorted_values})
    df_avg["Group"] = df_avg["Model"].map(MODEL_TO_GROUP)

    missing = sorted(df_avg.loc[df_avg["Group"].isna(), "Model"].unique().tolist())
    if missing:
        raise ValueError(
            "Some model_id values are not present in MODEL_TO_GROUP mapping:\n"
            + "\n".join(missing)
        )

    df_avg["Group"] = pd.Categorical(df_avg["Group"], categories=GROUP_ORDER, ordered=True)

    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(
        data=df_avg,
        x="Model",
        y="AvgScore",
        hue="Group",
        hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE
    )

    max_abs = np.max(np.abs(sorted_values))
    ymax = max_abs * 1.2
    plt.ylim(-ymax, ymax)

    for i, v in enumerate(sorted_values):
        if v >= 0:
            y_pos = v - (0.03 * ymax)
            va = 'top'
        else:
            y_pos = v + (0.03 * ymax)
            va = 'bottom'

        if abs(v) < 0.05 * ymax:
            y_pos = v + np.sign(v) * (0.06 * ymax)
            va = 'bottom' if v >= 0 else 'top'

        barplot.text(
            i, y_pos,
            f'{v:.2f}',
            ha='center',
            va=va,
            fontsize=9,
            fontweight='bold',
            color='white' if abs(v) > 0.15 * ymax else '#333333',
            bbox=dict(
                facecolor='black' if abs(v) > 0.15 * ymax else 'white',
                alpha=0.7,
                edgecolor='none',
                pad=1.5
            )
        )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='black', linewidth=1, alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xlabel("Model ID")
    plt.ylabel("Average Sycophancy Score")
    plt.title("Average Sycophancy Score per Model", pad=15)
    plt.legend(title="Group", fontsize="small", frameon=True,
               loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(PLOTS_DIR / "02_avg_scores_per_model.png", dpi=300)
    plt.close()


def plot_model_scores(avg_scores, save_path):
    """
    Plot average scores (same values as Plot 02) and save to a separate file.

    Parameters
    ----------
    avg_scores : dict
        model_id -> average score.
    save_path : pathlib.Path
        Where to write the PNG.
    """
    sorted_items = sorted(avg_scores.items(), key=lambda x: x[0])
    models = [k for k, _ in sorted_items]
    values = np.array([v for _, v in sorted_items])

    df_avg = pd.DataFrame({"Model": models, "AvgScore": values})
    df_avg["Group"] = df_avg["Model"].map(MODEL_TO_GROUP)

    missing = sorted(df_avg.loc[df_avg["Group"].isna(), "Model"].unique().tolist())
    if missing:
        raise ValueError(
            "Some model_id values are not present in MODEL_TO_GROUP mapping:\n"
            + "\n".join(missing)
        )

    df_avg["Group"] = pd.Categorical(df_avg["Group"], categories=GROUP_ORDER, ordered=True)

    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(
        data=df_avg,
        x="Model",
        y="AvgScore",
        hue="Group",
        hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE
    )

    max_abs = np.max(np.abs(values))
    ymax = max_abs * 1.2
    plt.ylim(-ymax, ymax)

    for i, v in enumerate(values):
        if v >= 0:
            y_pos = v - (0.03 * ymax)
            va = "top"
        else:
            y_pos = v + (0.03 * ymax)
            va = "bottom"

        if abs(v) < 0.05 * ymax:
            y_pos = v + np.sign(v) * (0.06 * ymax)
            va = "bottom" if v >= 0 else "top"

        barplot.text(
            i, y_pos,
            f'{v:.2f}',
            ha='center',
            va=va,
            fontsize=9,
            fontweight='bold',
            color='white' if abs(v) > 0.15 * ymax else '#333333',
            bbox=dict(
                facecolor='black' if abs(v) > 0.15 * ymax else 'white',
                alpha=0.7,
                edgecolor='none',
                pad=1.5
            )
        )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='black', linewidth=1, alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xlabel("Model ID")
    plt.ylabel("Average Sycophancy Score")
    plt.title("Average Sycophancy Score per Model", pad=15)
    plt.legend(title="Group", fontsize="small", frameon=True,
               loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()


plot_model_scores(
    avg_scores,
    PLOTS_DIR / "02_5_avg_scores_per_model_shortnames.png"
)


# 03) Overlaid distributions as KDE plots (grouped)
if scores_by_model and not df_plot.empty:
    plt.figure(figsize=(12, 7))

    for group_name in GROUP_ORDER:
        subset = df_plot[df_plot["Group"] == group_name]
        if subset.empty:
            continue
        if len(subset) < 2:
            continue

        sns.kdeplot(
            subset["Score"].values,
            label=group_name,
            fill=True,
            alpha=0.12,
            linewidth=2,
            color=GROUP_PALETTE.get(group_name, "gray")
        )

    plt.xlim(-1.1, 1.1)
    plt.xlabel("Sycophancy Score")
    plt.ylabel("Density")
    plt.title("Score Distributions (Grouped)", pad=15)
    plt.legend(title="Group", fontsize="small", frameon=True,
               loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(PLOTS_DIR / "03_overlaid_distributions.png", dpi=300)
    plt.close()

# 04) Global box plot
if all_scores:
    plt.figure(figsize=(6, 8))
    box = plt.boxplot(
        all_scores,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
    )

    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    for median in box['medians']:
        median.set_color('orange')
        median.set_linewidth(2)
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for cap in box['caps']:
        cap.set_color('black')

    plt.ylabel("Sycophancy Score")
    plt.title("Global Score Distribution (Box Plot)", pad=15)
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_global_boxplot.png", dpi=300)
    plt.close()

# 05) Box plot per model
if not df_plot.empty:
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_plot,
        x="Model",
        y="Score",
        color="lightblue",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
        medianprops={"color": "orange", "linewidth": 2},
        boxprops={"alpha": 0.7, "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4, "alpha": 0.5}
    )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.8)
    plt.ylabel("Sycophancy Score")
    plt.title("Sycophancy Score Distribution by Model", pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_side_by_side_boxplot.png", dpi=300)
    plt.close()

# 06) Violin plot
if not df_plot.empty:
    plt.figure(figsize=(14, 8))
    sns.violinplot(
        data=df_plot,
        x="Model",
        y="Score",
        palette="Set3",
        hue="Model",
        legend=False,
        inner="quartile",
        linewidth=1,
        alpha=0.8
    )
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.ylabel("Sycophancy Score")
    plt.title("Sycophancy Score Density by Model (Violin Plot)", pad=15)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_all_models_violin.png", dpi=300)
    plt.close()

# ==========================================
# NEW PLOTS (Analytical Additions)
# ==========================================

# 07) Mean + 95% CI point plot
if not df_plot.empty:
    plt.figure(figsize=(14, 8))
    sns.pointplot(
        data=df_plot,
        x="Model",
        y="Score",
        errorbar=('ci', 95),
        capsize=0.15,
        color="darkred",
        markers="s",
        linestyles="--",
        scale=0.8
    )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.ylabel("Mean Sycophancy Score")
    plt.title("Mean Score Comparison with 95% Confidence Intervals", pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_mean_ci_comparison.png", dpi=300)
    plt.close()

# 08) Box-underlay + strip overlay
if not df_plot.empty:
    plt.figure(figsize=(14, 8))

    sns.boxplot(
        data=df_plot, x="Model", y="Score",
        showfliers=False,
        boxprops={'facecolor': (0, 0, 0, 0.05), 'edgecolor': 'gray', 'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5}
    )

    sns.stripplot(
        data=df_plot,
        x="Model",
        y="Score",
        palette="viridis",
        hue="Model",
        legend=False,
        size=4,
        alpha=0.45,
        jitter=0.25
    )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.ylabel("Sycophancy Score")
    plt.title("Detailed Score Distribution (Strip Plot)", pad=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_raw_distribution_strip.png", dpi=300)
    plt.close()

# 10+) Per-model histograms
for model_id, scores in sorted(scores_by_model.items()):
    if not scores:
        continue

    group_name = MODEL_TO_GROUP.get(model_id)

    if group_name is None:
        raise ValueError(f"Model '{model_id}' is missing from MODEL_TO_GROUP mapping.")

    plt.figure(figsize=(8, 5))
    sns.histplot(
        scores,
        bins=20,
        kde=True,
        stat="density",
        color=GROUP_PALETTE.get(group_name, "teal"),
        edgecolor="white",
        alpha=0.6,
        line_kws={'linewidth': 2}
    )
    plt.xlim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xlabel("Sycophancy Score")
    plt.ylabel("Density")
    plt.title(f"Score Distribution: {model_id}", pad=15)
    plt.tight_layout()

    safe_name = model_id.replace("/", "_").replace(" ", "_")
    plt.savefig(PLOTS_DIR / f"10_score_distribution_{safe_name}.png", dpi=300)
    plt.close()
