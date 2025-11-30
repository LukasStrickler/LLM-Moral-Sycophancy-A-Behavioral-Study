import pandas as pd
import numpy as np

#THIS FILE IS USED TO CREATE A COMBINED TABLE FROM TWO INPUT FILES:
# team_review_consensus_NEWSCORES_05-11.csv AND final_consensus_export.csv
# THE OUTPUT FILE IS combined_consensus.csv 


# Input filenames (adjust if needed)
NEWSCORES_PATH = "../data/humanLabel/reviews/team_review_consensus_NEWSCORES_05-11.csv"
FINAL_EXPORT_PATH = "../data/humanLabel/reviews/final_consensus_export.csv"
OUTPUT_PATH = "../data/humanLabel/reviews/combined_consensus.csv"

# Helper to normalize likely column header variants
def _normalize_cols(df):
    df = df.copy()
    # Lowercase stripped names without spaces/underscores for matching
    norm_map = {c: ''.join(str(c).lower().strip().replace(' ', '').replace('_', '')) for c in df.columns}
    def find_col(*candidates):
        canon = [''.join(x.lower().strip().replace(' ', '').replace('_', '')) for x in candidates]
        for orig, norm in norm_map.items():
            if norm in canon:
                return orig
        return None
    return df, find_col

# Load files
df_news = pd.read_csv(NEWSCORES_PATH)
df_final = pd.read_csv(FINAL_EXPORT_PATH)

# Normalize and find required columns in NEWSCORES
df_news, find_news = _normalize_cols(df_news)
news_id_col = find_news("response_id", "responseid", "id")
news_manual_col = find_news("manual_consensus", "manualconsensus")
news_prompt_body_col = find_news("prompt_body", "promptbody")
news_model_text_col = find_news("model_response_text", "modelresponsetext")

# Normalize and find required columns in FINAL
df_final, find_final = _normalize_cols(df_final)
final_id_col = find_final("response_id", "responseid", "id")
final_consensus_col = find_final("final_consensus", "finalconsensus")
final_prompt_body_col = find_final("prompt_body", "promptbody")
final_model_text_col = find_final("model_response_text", "modelresponsetext")

# Validate essential columns
missing = []
if final_id_col is None: missing.append("response_id in final_consensus_export")
if final_consensus_col is None: missing.append("final_consensus in final_consensus_export")
if news_id_col is None: missing.append("response_id in team_review_consensus_NEWSCORES_05-11")
if news_manual_col is None: missing.append("manual_consensus in team_review_consensus_NEWSCORES_05-11")
if missing:
    raise ValueError("Missing required columns: " + ", ".join(missing))

# Cast IDs to numeric safely for exact numeric matching
def to_num(s):
    # Handles strings like "123", "123.0", and NaN safely
    try:
        v = float(s)
        # If it's an integer-like float, cast to int for exact match behavior
        if pd.notna(v) and v.is_integer():
            return int(v)
        return v
    except Exception:
        return np.nan

df_final["_rid"] = df_final[final_id_col].apply(to_num)
df_news["_rid"] = df_news[news_id_col].apply(to_num)

# Build a lookup of manual_consensus by response_id
# If duplicates exist in news, last one wins; change to group/first if needed
news_lookup = df_news.set_index("_rid")[news_manual_col].to_dict()

# Prepare model_response_text and prompt_body sources:
# Priority for these fields is to map "exactly" to the response_id from final, as requested.
# The prompt/model text should come from final if present there; if not, fallback to news if available.
def pick_text(row, final_col, news_col):
    rid = row["_rid"]
    val_final = row.get(final_col) if final_col in row.index else None
    if pd.notna(val_final):
        return val_final
    # fallback to news table for the same response_id
    if news_col is not None and rid in df_news.set_index("_rid"):
        return df_news.set_index("_rid").loc[rid, news_col]
    return np.nan

# Compute the chosen consensus column
def choose_consensus(rid, fallback_value):
    if rid in news_lookup and pd.notna(news_lookup[rid]):
        return news_lookup[rid]
    return fallback_value

# Create output DataFrame in the order of df_final
out = pd.DataFrame()
out["response_id"] = df_final["_rid"]

# Chosen consensus per your rule
out["chosen_consensus"] = [
    choose_consensus(rid, fb) for rid, fb in zip(df_final["_rid"], df_final[final_consensus_col])
]

# Also include the source columns if you want to inspect later
out["manual_consensus_from_news"] = [news_lookup.get(rid, np.nan) for rid in df_final["_rid"]]
out["final_consensus_from_final"] = df_final[final_consensus_col].values

# Map model_response_text and prompt_body exactly to the response_id of the new table.
# Prefer the values from final_consensus_export; if missing, try to fill from the news file.
if final_model_text_col is not None:
    out["model_response_text"] = df_final[final_model_text_col].values
else:
    # Try to align from news only if final lacks the column
    if news_model_text_col is not None:
        news_model_map = df_news.set_index("_rid")[news_model_text_col].to_dict()
        out["model_response_text"] = [news_model_map.get(rid, np.nan) for rid in df_final["_rid"]]
    else:
        out["model_response_text"] = np.nan

if final_prompt_body_col is not None:
    out["prompt_body"] = df_final[final_prompt_body_col].values
else:
    if news_prompt_body_col is not None:
        news_prompt_map = df_news.set_index("_rid")[news_prompt_body_col].to_dict()
        out["prompt_body"] = [news_prompt_map.get(rid, np.nan) for rid in df_final["_rid"]]
    else:
        out["prompt_body"] = np.nan

# Optional: keep original ordering and types neat
# Sort columns in a readable order
preferred_cols = ["response_id", "chosen_consensus", "manual_consensus_from_news",
                  "final_consensus_from_final", "prompt_body", "model_response_text"]
out = out[preferred_cols]

# Write result
out.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")
