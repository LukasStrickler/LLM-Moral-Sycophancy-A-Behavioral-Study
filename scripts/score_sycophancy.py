import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Use the merged file produced earlier
CSV_PATH = "../data/humanLabel/reviews/combined_consensus.csv"  # path to the new table with 'chosen_consensus'
MODEL_PATH = "answerdotai/ModernBERT-base"
OUTPUT_DIR = "./modernbert_chosen_consensus_first100"

# Load merged data
df = pd.read_csv(CSV_PATH)

# Validate required columns
required_cols = ["model_response_text", "chosen_consensus"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {CSV_PATH}: {missing}")

# Prepare labels from chosen_consensus (numeric expected in [-1, 1])
df["label_avg"] = pd.to_numeric(df["chosen_consensus"], errors="coerce").clip(-1, 1)

# Select first 100 valid rows
text_col = "model_response_text"
subset = df[[text_col, "label_avg"]].dropna().head(100).reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class TextRegDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=2048)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(float(self.labels[idx]), dtype=torch.float)
        return item

dataset = TextRegDataset(
    texts=subset[text_col].astype(str).tolist(),
    labels=subset["label_avg"].astype(float).tolist(),
    tokenizer=tokenizer,
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)

use_cuda = torch.cuda.is_available()
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    fp16=use_cuda,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
