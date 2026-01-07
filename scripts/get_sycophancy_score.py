"""
Score and analyze LLM responses for sycophancy.

Reads `results.csv`, assigns a sycophancy score to each response using the model in
`MODEL_DIR`, writes per-model averages to `eval.txt`, and saves several comparison
plots under `{MODEL_DIR}/plots`.

Assumptions:
- The classifier returns a single scalar score in [-1, 1].
- Every `model_id` in the CSV must exist in `MODEL_TO_GROUP` (the script fails fast otherwise).
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
# Set the specific style requested: Whitegrid with clean fonts
sns.set_style('whitegrid')
sns.set_context("notebook", font_scale=1.1)

# Professional Analytical Defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.5      # Visible but subtle grid
plt.rcParams['lines.linewidth'] = 2   # Clear lines

torch.set_float32_matmul_precision("highest")

# Paths
CSV_PATH = "results.csv"          # path to the CSV with logs
MODEL_DIR = "modernbert_chosen_consensus_advanced"         # folder with the trained BERT model
OUTPUT_PATH = Path(MODEL_DIR) / "eval.txt"          # where to write evaluation results

# Plots folder is inside the model directory
PLOTS_DIR = Path(MODEL_DIR) / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Use pandas to get true number of records
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

# Read CSV and get scores
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

        # Model already outputs a sycophancy score in [-1, 1]
        score = float(logits.item())
        scores_by_model[model_id].append(score)

# Compute per-model stats and global average, then write eval.txt
all_scores = []
avg_scores = {}  # for global bar chart

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
# GROUPING CONFIGURATION (NEW)
# ==========================================
# Mapped to the *actual* model_id strings visible in your plot.
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

# Color palette per group (consistent across plots)
GROUP_PALETTE = dict(
    zip(
        GROUP_ORDER,
        sns.color_palette("Set2", n_colors=len(GROUP_ORDER))
    )
)

# ==========================================
# PLOTTING SECTION (Updated Styles)
# ==========================================

# Prepare Dataframe for Seaborn (easier for side-by-side plotting)
plot_data = []
for m, s in scores_by_model.items():
    for val in s:
        plot_data.append({"Model": m, "Score": val})
df_plot = pd.DataFrame(plot_data)
if not df_plot.empty:
    df_plot["Group"] = df_plot["Model"].map(MODEL_TO_GROUP)

    # Enforce: no Unknowns allowed
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
        line_kws={'linewidth': 2, 'color': 'navy'}  # Darker line for contrast
    )
    plt.xlim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xlabel("Sycophancy Score")
    plt.ylabel("Density")
    plt.title("Global Score Distribution", pad=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_global_hist.png", dpi=300)
    plt.close()


# 02) Bar chart of average scores per model (UPDATED & ALPHABETICAL)
if avg_scores:
    # Sort models alphabetically
    sorted_items = sorted(avg_scores.items(), key=lambda x: x[0])
    sorted_models = [k for k, _ in sorted_items]
    sorted_values = np.array([v for _, v in sorted_items])

    # Build a grouped dataframe for plotting (keeps your ordering logic intact)
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

    # Use seaborn barplot (now colored by Group)
    barplot = sns.barplot(
        data=df_avg,
        x="Model",
        y="AvgScore",
        hue="Group",
        hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE
    )

    # ---------------------------
    # Symmetric y-axis
    # ---------------------------
    max_abs = np.max(np.abs(sorted_values))
    ymax = max_abs * 1.2
    plt.ylim(-ymax, ymax)

    # ---------------------------
    # Improved label positions (inside bars)
    # ---------------------------
    for i, v in enumerate(sorted_values):
        if v >= 0:
            y_pos = v - (0.03 * ymax)
            va = 'top'
        else:
            y_pos = v + (0.03 * ymax)
            va = 'bottom'

        # If the bar is tiny → place label slightly outside
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

    # Formatting
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='black', linewidth=1, alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.xlabel("Model ID")
    plt.ylabel("Average Sycophancy Score")
    plt.title("Average Sycophancy Score per Model", pad=15)

    # Legend outside (right)
    plt.legend(title="Group", fontsize="small", frameon=True,
               loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Leave room on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(PLOTS_DIR / "02_avg_scores_per_model.png", dpi=300)
    plt.close()


def plot_model_scores(avg_scores, save_path):
    """
    Produce Plot 2.5: same as Plot 2 but using original full model names (no shortening).
    """
    # Alphabetical sorting first
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

    # Barplot (now colored by Group)
    barplot = sns.barplot(
        data=df_avg,
        x="Model",
        y="AvgScore",
        hue="Group",
        hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE
    )

    # Symmetric y-axis
    max_abs = np.max(np.abs(values))
    ymax = max_abs * 1.2
    plt.ylim(-ymax, ymax)

    # Labels
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

    # Formatting
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color='black', linewidth=1, alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.xlabel("Model ID")
    plt.ylabel("Average Sycophancy Score")
    plt.title("Average Sycophancy Score per Model", pad=15)

    # Legend outside (right)
    plt.legend(title="Group", fontsize="small", frameon=True,
               loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Leave room on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()


plot_model_scores(
    avg_scores,
    PLOTS_DIR / "02_5_avg_scores_per_model_shortnames.png"
)


# 03) Overlaid distributions (all models) as KDE plots
# Now grouped by "Group" (not individual models) so the legend stays readable.
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

    # Legend outside (right)
    plt.legend(title="Group", fontsize="small", frameon=True,
               loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Leave room on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(PLOTS_DIR / "03_overlaid_distributions.png", dpi=300)
    plt.close()

# 04) Global box plot (Styled like Reference: Light Blue Box, Orange Median)
if all_scores:
    plt.figure(figsize=(6, 8))

    # Using matplotlib directly for precise control over the "reference" look
    box = plt.boxplot(
        all_scores,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
    )

    # Apply the specific colors from your reference image
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

# 05) Box plot for each LLM model (Styled like Reference)
# (Kept in the same reference style: uniform light-blue boxes.)
if not df_plot.empty:
    plt.figure(figsize=(14, 8))

    # Replicating the reference style: Uniform Light Blue boxes, Orange Medians
    sns.boxplot(
        data=df_plot,
        x="Model",
        y="Score",
        color="lightblue",  # Uniform color to match reference style
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
        medianprops={"color": "orange", "linewidth": 2},  # The specific orange line
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
        palette="Set3",  # Softer palette for violins
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

# 07) Point Plot with Confidence Intervals (Standard Analytical View)
if not df_plot.empty:
    plt.figure(figsize=(14, 8))

    # Shows Mean + 95% Confidence Interval
    # This is crucial for determining if differences between models are statistically significant
    sns.pointplot(
        data=df_plot,
        x="Model",
        y="Score",
        errorbar=('ci', 95),  # 95% Confidence Interval
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

# 08) Strip Plot (Raw Data Distribution)
if not df_plot.empty:
    plt.figure(figsize=(14, 8))

    # --- UPDATED BOX PLOT SETTINGS ---
    # Box plot underlay (stronger visibility)
    sns.boxplot(
        data=df_plot, x="Model", y="Score",
        showfliers=False,
        # Changed facecolor to have slight fill (alpha 0.1) instead of none
        # Changed edgecolor to 'black' and increased linewidth for visibility
        boxprops={'facecolor': (0, 0, 0, 0.05), 'edgecolor': 'gray', 'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5}
    )

    # Strip plot overlay: Shows every single data point
    # Important for identifying outliers or bimodal distributions
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

# 10+) Histogram (distribution) of scores for each model
# Colored by group (no extra legend to keep each per-model plot clean).
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
