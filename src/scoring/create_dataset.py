# THIS SCRIPT CREATES A COMBINED DATASET OF THE PREPERATION AND THE TESTING DATASET FROM MULTIPLE CSV FILES


import pandas as pd
from sklearn.model_selection import train_test_split

# Paths to the data files
FILE_1_PATH = "combined_consensus.csv"
FILE_2_PATH = "final_consensus_export_v2_response.csv"
FILE_3_PATH = "final_consensus_export_v2-1_FINALEDITS20-12.csv"

# 1. Load original dataframes
df1 = pd.read_csv(FILE_1_PATH)
df2 = pd.read_csv(FILE_2_PATH)
df3 = pd.read_csv(FILE_3_PATH)

# 2. Select and normalize columns for the first dataset
# Use 'chosen_consensus' as is
cols_df1 = ["response_id", "prompt_body", "model_response_text", "chosen_consensus"]
df1_clean = df1[cols_df1].copy()

# 3. Select and normalize columns for the second dataset
# Map 'final_consensus' to 'chosen_consensus'
cols_df2_src = ["response_id", "prompt_body", "model_response_text", "final_consensus"]
df2_clean = df2[cols_df2_src].copy()
df2_clean = df2_clean.rename(columns={"final_consensus": "chosen_consensus"})

# 3b. Select and normalize columns for the third dataset
# Map 'final_score' to 'chosen_consensus'
cols_df3_src = ["response_id", "prompt_body", "model_response_text", "final_score"]
df3_clean = df3[cols_df3_src].copy()
df3_clean = df3_clean.rename(columns={"final_score": "chosen_consensus"})

# 4. Combine the datasets
df_combined = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)

# 5. Drop rows with missing required data
required_cols = ["response_id", "prompt_body", "model_response_text", "chosen_consensus"]
df_final = df_combined.dropna(subset=required_cols).reset_index(drop=True)

print(f"Combined dataset size: {len(df_final)} samples")

# --- NEW: Save aggregate file with all cleaned/combined data ---
df_final.to_csv("Aggregate_file.csv", index=False)

# 6. Make sure we have enough data for split
# (Updated check for larger combined dataset)
if len(df_final) < 150:
    raise ValueError(f"Not enough data points ({len(df_final)}) for a robust split.")

# 7. Random split
# Using test_size=0.2 will give you an 80/20 split (approx. 240 train / 60 val for 300 samples)
train_df, val_df = train_test_split(df_final, test_size=0.15, random_state=42, shuffle=True)

# Save splits to CSV
train_df.to_csv("training_data.csv", index=False)
val_df.to_csv("validation_data.csv", index=False)

print(f"Saved training_data.csv with {len(train_df)} samples")
print(f"Saved validation_data.csv with {len(val_df)} samples")
