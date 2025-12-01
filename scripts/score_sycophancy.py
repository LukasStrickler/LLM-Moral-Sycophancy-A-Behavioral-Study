import os
# DISABLE DYNAMO TO FIX "FX symbolically trace" ERROR
os.environ["PYTORCH_ENABLE_TORCHDYNAMO"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1" 

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import warnings
import optuna

# Optional: Optuna transformers pruning callback
try:
    from optuna.integration import TransformersPruningCallback
    _HAS_TRANSFORMERS_PRUNING = True
except Exception:
    _HAS_TRANSFORMERS_PRUNING = False

warnings.filterwarnings("ignore", category=FutureWarning, message=".*is deprecated and will be removed*")
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars*")


# =================================================
#              CONSOLIDATED USER CONFIGURATION
# =================================================

# File paths
TRAIN_CSV_PATH = "training_data.csv"
VAL_CSV_PATH = "validation_data.csv"
MODEL_PATH = "answerdotai/ModernBERT-base"
OUTPUT_DIR = "./modernbert_chosen_consensus_advanced"

# ====== EXECUTION MODE (Toggle here) ======
RUN_SINGLE_MODEL = False                 # True: Single model | False: Optuna hyperparameter search
use_upsampling_grid = True              # True: Grid of upsampling configs | False: Single upsampling config
use_kfold = False                       # True: k-fold CV for Optuna | False: simple train/val split
optuna_prune = False                    # True: Enable Optuna pruning | False: Disable pruning
# ====== STANDARD UPSAMPLING CONFIGURATION (Used when use_upsampling_grid = False) ======
upsample_extreme = True                # True: Enable upsampling | False: Disable upsampling
upsample_threshold = 0.62               # Threshold for extreme values
upsample_factor_positive = 2.8          # Upsampling factor for positive extremes
upsample_factor_negative = 2          # Upsampling factor for negative extremes

# ====== OPTUNA CONFIGURATION (For Optuna mode) ======
n_trials_optuna = 180                    # Number of Optuna trials per configuration
freeze_during_search = True             # True: Freeze backbone during search | False: Train all parameters

# ====== TRAINING CONFIGURATION ======
seed = 42
max_length = 2048
batch_size_eval = 16
gradient_accumulation_steps_default = 1
early_stopping_patience = 2


# ====== SINGLE MODEL PARAMETERS (Used when RUN_SINGLE_MODEL = True) ======
SINGLE_MODEL_PARAMS = {
    "learning_rate": 6.482131165247738e-05,
    "per_device_train_batch_size": 3,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.04807901366051648,
    "num_train_epochs": 4,
    "adam_epsilon": 1.9030368381735818e-07,
    "warmup_ratio": 0.14016725176148134,
    "lr_scheduler_type": "constant"
}

# ====== GRID OF UPSAMPLING CONFIGURATIONS (Used when use_upsampling_grid = True) ======
# Each tuple: (enabled, threshold, positive_factor, negative_factor)
UPSAMPLING_GRID = [
    (False, 0.0, 1.0, 1.0),             # Config 0: No upsampling (baseline)
    (True, 0.72, 2.8, 2.0),             # Config 1: Baseline
    (True, 0.72, 2.8, 2.8),             # Config 2
    (True, 0.72, 1.0, 2.0),             # Config 3
    (True, 0.72, 1.5, 2.1),             # Config 4
    (True, 0.72, 2.8, 3.0),             # Config 5
    (True, 0.72, 2.5, 2.5),              # Config 6
    (True, 0.72, 2.8, 3.4),             # Config 7
    (True, 0.72, 2.3, 2.3),             # Config 8
    (True, 0.72, 2.5, 1.8),             # Config 9
]

# Determine results filename based on mode
if RUN_SINGLE_MODEL:
    if use_upsampling_grid:
        RESULTS_FILE = "results_file_single_model_grid.txt"
    else:
        RESULTS_FILE = "results_file_single_model_standard.txt"
else:
    if use_upsampling_grid:
        RESULTS_FILE = "results_file_grid.txt"
    else:
        RESULTS_FILE = "results_file_standard.txt"

# =================================================
#             SETUP & REPRODUCIBILITY
# =================================================

set_seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# =================================================
#               TOKENIZER & UTILS
# =================================================

# Tokenizer must be created before processing
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def tokenize_texts_no_padding(tokenizer, texts, max_length=max_length):
    """
    Tokenize once WITHOUT padding (so tokenized outputs are lists of variable-length token sequences).
    This allows DataCollatorWithPadding to do dynamic padding per-batch (more efficient).
    """
    return tokenizer(
        texts,
        truncation=True,
        padding=False,  # No padding here (dynamic padding used later)
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
    )


# =================================================
#                 DATASET CLASSES
# =================================================

class EncodedRegDataset(Dataset):
    """
    Dataset class storing pre-tokenized inputs (unpadded) and labels.
    """
    def __init__(self, encodings, labels, texts):
        # Convert tokenizer BatchEncoding to plain dict-of-lists if necessary
        if hasattr(encodings, "data") and isinstance(encodings.data, dict):
            enc_dict = encodings.data
        elif isinstance(encodings, dict):
            enc_dict = encodings
        else:
            raise ValueError("encodings must be a dict or BatchEncoding")
            
        # Ensure values are python lists (not tensors) to save memory before collation
        self.encodings = {k: list(v) for k, v in enc_dict.items()}
        self.labels = list(labels)
        self.texts = list(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return tokenized inputs (unpadded) as lists
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            **({"token_type_ids": self.encodings["token_type_ids"][idx]} if "token_type_ids" in self.encodings else {}),
            "labels": torch.tensor(float(self.labels[idx]), dtype=torch.float),
        }


# =================================================
#             DATA LOADING & PREPARATION
# =================================================

def upsample_extreme_consensus(df, threshold=0.5, upsample_factor_positive=2, upsample_factor_negative=2, stage=""):
    """
    Upsample datapoints where chosen_consensus > threshold (positive) and < -threshold (negative) with different factors.
    
    Args:
        df: DataFrame with 'chosen_consensus' column
        threshold: Absolute threshold (e.g., 0.5 means > 0.5 or < -0.5)
        upsample_factor_positive: How many times to duplicate positive extreme samples (consensus > threshold)
        upsample_factor_negative: How many times to duplicate negative extreme samples (consensus < -threshold)
        stage: String label for logging (e.g., "search", "final")
    
    Returns:
        Upsampled DataFrame
    """
    # Ensure factors are integers
    upsample_factor_positive = int(upsample_factor_positive)
    upsample_factor_negative = int(upsample_factor_negative)
    
    # Identify extreme consensus values
    positive_extreme_mask = df["chosen_consensus"] > threshold
    negative_extreme_mask = df["chosen_consensus"] < -threshold
    
    normal_df = df[~(positive_extreme_mask | negative_extreme_mask)]
    positive_extreme_df = df[positive_extreme_mask]
    negative_extreme_df = df[negative_extreme_mask]
    
    # Upsample positive and negative extreme samples separately
    positive_extreme_df_upsampled = pd.concat([positive_extreme_df] * upsample_factor_positive, ignore_index=True)
    negative_extreme_df_upsampled = pd.concat([negative_extreme_df] * upsample_factor_negative, ignore_index=True)
    
    # Combine and shuffle
    df_balanced = pd.concat([normal_df, positive_extreme_df_upsampled, negative_extreme_df_upsampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Better logging
    stage_label = f" [{stage}]" if stage else ""
    print(f"\n--- Upsampling Details{stage_label} ---")
    print(f"Original dataset size: {len(df)}")
    print(f"Normal samples (|consensus| <= {threshold}): {len(normal_df)}")
    print(f"Positive extreme samples (consensus > {threshold}): {len(positive_extreme_df)} → upsampled {upsample_factor_positive}x → {len(positive_extreme_df_upsampled)}")
    print(f"Negative extreme samples (consensus < {-threshold}): {len(negative_extreme_df)} → upsampled {upsample_factor_negative}x → {len(negative_extreme_df_upsampled)}")
    print(f"Final dataset size: {len(df_balanced)}")
    print(f"Increase: {len(df_balanced) - len(df)} samples ({100*(len(df_balanced)-len(df))/len(df):.1f}%)\n")
    
    return df_balanced


def prepare_datasets(full_train_df, upsample_enabled, upsample_threshold, upsample_factor_positive, upsample_factor_negative):
    """
    Prepare search and final training datasets with given upsampling configuration.
    """
    # 2. PREPARE DATA FOR SEARCH (Split Training Data Internal ONLY)
    if use_kfold:
        search_train_df = full_train_df
        search_val_dataset = None
    else:
        sub_train_df, sub_val_df = train_test_split(
            full_train_df,
            test_size=0.20,
            random_state=seed,
        )
        
        # Upsample ONLY the sub-train portion
        if upsample_enabled:
            sub_train_df = upsample_extreme_consensus(sub_train_df, threshold=upsample_threshold, upsample_factor_positive=upsample_factor_positive, upsample_factor_negative=upsample_factor_negative, stage="search training")
        
        search_train_df = sub_train_df
        
        # Prepare the sub-validation dataset used ONLY for Optuna (NOT upsampled)
        sub_val_texts = sub_val_df["model_response_text"].astype(str).tolist()
        sub_val_labels = sub_val_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
        sub_val_encodings = tokenize_texts_no_padding(tokenizer, sub_val_texts, max_length=max_length)
        search_val_dataset = EncodedRegDataset(sub_val_encodings, sub_val_labels, sub_val_texts)

    # Prepare the training dataset for search
    search_train_texts = search_train_df["model_response_text"].astype(str).tolist()
    search_train_labels = search_train_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
    search_train_encodings = tokenize_texts_no_padding(tokenizer, search_train_texts, max_length=max_length)
    search_train_dataset = EncodedRegDataset(search_train_encodings, search_train_labels, search_train_texts)

    # 3. PREPARE FINAL TRAINING DATA (Use 100% of Training CSV, upsampled)
    full_train_df_for_final = full_train_df.copy()
    if upsample_enabled:
        full_train_df_for_final = upsample_extreme_consensus(full_train_df_for_final, threshold=upsample_threshold, upsample_factor_positive=upsample_factor_positive, upsample_factor_negative=upsample_factor_negative, stage="final training")

    full_train_texts = full_train_df_for_final["model_response_text"].astype(str).tolist()
    full_train_labels = full_train_df_for_final["chosen_consensus"].astype(float).clip(-1, 1).tolist()
    full_train_encodings = tokenize_texts_no_padding(tokenizer, full_train_texts, max_length=max_length)
    full_train_dataset = EncodedRegDataset(full_train_encodings, full_train_labels, full_train_texts)

    return search_train_dataset, search_val_dataset, full_train_dataset


# 1. Load Full Training Data (from CSV) - ONCE at the beginning moved here for upsampling purposes
full_train_df = pd.read_csv(TRAIN_CSV_PATH).dropna(subset=["model_response_text", "chosen_consensus"]).reset_index(drop=True)

# 4. PREPARE FINAL TEST/VALIDATION DATA (Real Held-out Validation) - ONCE at the beginning
final_test_df = pd.read_csv(VAL_CSV_PATH).dropna(subset=["model_response_text", "chosen_consensus"]).reset_index(drop=True)
final_test_texts = final_test_df["model_response_text"].astype(str).tolist()
final_test_labels = final_test_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
final_test_encodings = tokenize_texts_no_padding(tokenizer, final_test_texts, max_length=max_length)
final_test_dataset = EncodedRegDataset(final_test_encodings, final_test_labels, final_test_texts)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")


# =================================================
#               MODEL INITIALIZATION
# =================================================

class MSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute custom loss (MAE / L1Loss)
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        return (loss, outputs) if return_outputs else loss

# Global flag to control backbone freezing during HPO
_FREEZE_BACKBONE = False 

def model_init():
    """
    Create a fresh model for Trainer. 
    If _FREEZE_BACKBONE is True, freeze the base transformer parameters.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=1,
        problem_type="regression",
        trust_remote_code=True,
    )

    # Attempt to enable Flash Attention optimizations
    try:
        model.config.use_flash_attention = True
    except Exception:
        pass
    try:
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

    # Enable gradient checkpointing for long sequences
    if max_length > 1024 and not _FREEZE_BACKBONE:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Freeze backbone if requested
    if _FREEZE_BACKBONE:
        backbone = None
        if hasattr(model, "base_model"):
            backbone = model.base_model
        elif hasattr(model, model.__class__.__name__.lower()):
            backbone = getattr(model, model.__class__.__name__.lower())
        
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = False
        else:
            # Fallback: freeze all parameters except specific heads
            for name, p in model.named_parameters():
                if any(k in name for k in ["classifier", "regressor", "score", "out_proj", "lm_head"]):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
    return model


# =================================================
#               METRICS & UTILITIES
# =================================================

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    
    return {
        "mse": mse, 
        "mae": mae, 
        "r2": r2
    }


def train_evaluate_fold(train_subset, val_subset, params, freeze_backbone_flag=False, early_stop=False):
    """
    Trains a Trainer on the provided subsets and returns the validation MSE.
    """
    global _FREEZE_BACKBONE
    _FREEZE_BACKBONE = freeze_backbone_flag

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="no",
        logging_dir=f"{OUTPUT_DIR}/logs",
        max_grad_norm=1.0,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        weight_decay=params["weight_decay"],
        num_train_epochs=params["num_train_epochs"],
        warmup_ratio=params["warmup_ratio"],
        lr_scheduler_type=params["lr_scheduler_type"],
        adam_epsilon=params["adam_epsilon"],
        load_best_model_at_end=False,
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
        fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
        bf16=(use_cuda and torch.cuda.is_bf16_supported()),
    )

    callbacks = []
    if early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = MSETrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # Train and Evaluate
    trainer.train()
    metrics = trainer.evaluate()
    
    # Restore freeze flag default
    _FREEZE_BACKBONE = False
    return metrics["eval_mse"], metrics


# =================================================
#             OPTUNA SEARCH FUNCTIONS
# =================================================

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-4, log=True),  
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [3, 4, 6]),  
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 2),  
        "weight_decay": trial.suggest_float("weight_decay", 0.02, 0.20),  
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 6),  
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 3e-7, log=True),  
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.20),  
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["cosine_with_restarts", "linear", "constant"]),  
    }


def run_optuna_kfold_search(train_dataset, train_df, n_trials=20):
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=seed), 
        pruner=optuna.pruners.MedianPruner()
    )

    def objective(trial):
        params = optuna_hp_space(trial)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_mses = []

        for train_idx, val_idx in kf.split(range(len(train_df))):
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Prepare fold training dataset
            fold_train_texts = fold_train_df["model_response_text"].astype(str).tolist()
            fold_train_labels = fold_train_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
            fold_train_encodings = tokenize_texts_no_padding(tokenizer, fold_train_texts, max_length=max_length)
            fold_train_subset = EncodedRegDataset(fold_train_encodings, fold_train_labels, fold_train_texts)
            
            # Prepare fold validation dataset
            fold_val_texts = fold_val_df["model_response_text"].astype(str).tolist()
            fold_val_labels = fold_val_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
            fold_val_encodings = tokenize_texts_no_padding(tokenizer, fold_val_texts, max_length=max_length)
            fold_val_subset = EncodedRegDataset(fold_val_encodings, fold_val_labels, fold_val_texts)
            
            mse, _ = train_evaluate_fold(
                fold_train_subset, 
                fold_val_subset, 
                params, 
                freeze_backbone_flag=freeze_during_search
            )
            fold_mses.append(mse)
            
            intermediate = np.mean(fold_mses)
            trial.report(intermediate, len(fold_mses))
            
            if optuna_prune and trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(fold_mses)

    study.optimize(objective, n_trials=n_trials)
    trial = study.best_trial
    
    print("Best trial (kfold optuna):")
    print(f" Value (MSE): {trial.value:.4f}")
    print(" Params:")
    for k, v in trial.params.items():
        print(f" {k}: {v}")
    return trial.params


def run_optuna_simple_search(search_train_dataset, search_val_dataset, n_trials=20):
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=seed), 
        pruner=optuna.pruners.MedianPruner()
    )

    def objective(trial):
        params = optuna_hp_space(trial)
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="no",
            logging_dir=f"{OUTPUT_DIR}/logs",
            max_grad_norm=1.0,
            fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
            bf16=(use_cuda and torch.cuda.is_bf16_supported()),
            **params,
        )

        callbacks = []
        if optuna_prune and _HAS_TRANSFORMERS_PRUNING:
            callbacks.append(TransformersPruningCallback(trial, "eval_mse"))

        global _FREEZE_BACKBONE
        _FREEZE_BACKBONE = freeze_during_search

        trainer = MSETrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=search_train_dataset,
            eval_dataset=search_val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks if callbacks else None,
        )

        trainer.train()
        metrics = trainer.evaluate()
        _FREEZE_BACKBONE = False

        val_mse = metrics["eval_mse"]
        trial.report(val_mse, 0)
        
        if optuna_prune and trial.should_prune():
            raise optuna.TrialPruned()

        return val_mse

    study.optimize(objective, n_trials=n_trials)
    trial = study.best_trial
    print("Best trial (simple optuna):")
    print(f"  Value (MSE): {trial.value:.4f}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
    return trial.params


# =================================================
#             FINAL TRAINING & EVALUATION
# =================================================

def train_final_model(best_params, full_train_dataset):
    global _FREEZE_BACKBONE
    _FREEZE_BACKBONE = False
    
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    final_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        max_grad_norm=1.0,
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        weight_decay=best_params["weight_decay"],
        num_train_epochs=best_params["num_train_epochs"],
        warmup_ratio=best_params["warmup_ratio"],
        lr_scheduler_type=best_params["lr_scheduler_type"],
        adam_epsilon=best_params["adam_epsilon"],
        gradient_accumulation_steps=best_params.get("gradient_accumulation_steps", gradient_accumulation_steps_default),
        load_best_model_at_end=False,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
        bf16=(use_cuda and torch.cuda.is_bf16_supported()),
        report_to=[],
        disable_tqdm=False,
    )

    callbacks = []

    final_trainer = MSETrainer(
        model_init=model_init,
        args=final_args,
        train_dataset=full_train_dataset,
        eval_dataset=final_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print(f"Starting FINAL training for exactly {best_params['num_train_epochs']} epochs (no early stopping)...")
    final_trainer.train()
    final_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return final_trainer


def create_plots(true_labels, preds, output_dir):
    """
    Generates and saves 3 key regression plots:
    1. Scatterplot (Actual vs Predicted) with best-fit line
    2. Residuals Plot (Predicted vs Residuals)
    3. Distribution Plot (Histogram of Actual vs Predicted)
    """
    sns.set(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    # Scatter Plot: Actual vs Predicted with Best-Fit Line (green)
    plt.figure(figsize=(8, 6))
    plt.scatter(true_labels, preds, alpha=0.5, color='blue', edgecolors='k', s=40, label='Predictions')
    
    min_val = min(min(true_labels), min(preds))
    max_val = max(max(true_labels), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
    
    z = np.polyfit(true_labels, preds, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    plt.plot(x_line, p(x_line), 'g-', lw=2.5, label=f'Best Fit (y={z[0]:.2f}x+{z[1]:.2f})')
    
    plt.title("Actual vs. Predicted", fontsize=14)
    plt.xlabel("Actual Consensus", fontsize=12)
    plt.ylabel("Predicted Consensus", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_scatter_actual_vs_pred.png"), dpi=300)
    plt.close()

    # Residual Plot
    residuals = true_labels - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.5, color='purple', edgecolors='k', s=40)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    
    plt.title("Residuals Plot", fontsize=14)
    plt.xlabel("Predicted Values", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_residuals.png"), dpi=300)
    plt.close()

    # Distribution Plot (Histogram)
    plt.figure(figsize=(8, 6))
    sns.histplot(true_labels, color="blue", label="Actual", kde=True, stat="density", alpha=0.4)
    sns.histplot(preds, color="orange", label="Predicted", kde=True, stat="density", alpha=0.4)
    
    plt.title("Distribution of Actual vs. Predicted", fontsize=14)
    plt.xlabel("Consensus Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_distribution.png"), dpi=300)
    plt.close()

    print(f"Graphs saved to {output_dir}")


def evaluate_and_save_results(model, tokenizer, dataset, results_file, best_params, upsample_enabled, upsample_threshold, upsample_factor_positive, upsample_factor_negative, config_idx=None, batch_size=batch_size_eval, is_single_model=False):
    """
    Evaluate the final model, print metrics, generate plots, and PREPEND results to a file.
    """
    model.to(device)
    model.eval()
    
    total = len(dataset)
    true_labels = np.array(dataset.labels)
    preds_list = []

    print("Running final evaluation...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_samples = [dataset[i] for i in range(start, end)]
        batch = data_collator(batch_samples)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            batch_preds = outputs.logits.squeeze(-1).cpu().numpy()
        preds_list.append(batch_preds)

    preds = np.concatenate(preds_list, axis=0)

    # Metrics Calculation
    mse = mean_squared_error(true_labels, preds)
    mae = mean_absolute_error(true_labels, preds)
    r2 = r2_score(true_labels, preds)

    # Generate plots
    create_plots(true_labels, preds, OUTPUT_DIR)

    # Formatting
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
    
    separator = "-" * 50
    
    new_entry = (
        f"{separator}\n"
    )
    
    if config_idx is not None:
        new_entry += f"Config Index: {config_idx}\n"
    
    new_entry += (
        f"Time: {timestamp}\n"
        f"Params: {params_str}\n"
        f"Validation MSE: {mse:.4f}\n"
        f"Validation MAE: {mae:.4f}\n"
        f"Validation R^2: {r2:.4f}\n"
    )
    
    if is_single_model:
        new_entry += f"Mode: Single Model (Predefined Parameters)\n"
    else:
        new_entry += f"Number of runs: {n_trials_optuna:d}\n"
    
    new_entry += (
        f"K-Fold Cross-Validation: {use_kfold}\n"
        f"Upsampling Enabled: {upsample_enabled}\n"
        + (f"Upsampling Threshold: {upsample_threshold}\n"
           f"Upsampling Factor (Positive): {upsample_factor_positive}\n"
           f"Upsampling Factor (Negative): {upsample_factor_negative}\n" if upsample_enabled else "")
    )

    print(new_entry)
    
    # --- Prepend to File ---
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing_content = f.read()
    else:
        existing_content = ""

    with open(results_file, "w") as f:
        f.write(new_entry + "\n" + existing_content)
        
    print(f"Results prepended to {results_file}")


# =================================================
#          SINGLE MODEL EXECUTION
# =================================================

def train_single_model(params, full_train_dataset):
    """
    Train a single model with given parameters and evaluate on test set.
    Does NOT run Optuna search.
    """
    global _FREEZE_BACKBONE
    _FREEZE_BACKBONE = False
    
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        max_grad_norm=1.0,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        weight_decay=params["weight_decay"],
        num_train_epochs=params["num_train_epochs"],
        warmup_ratio=params["warmup_ratio"],
        lr_scheduler_type=params["lr_scheduler_type"],
        adam_epsilon=params["adam_epsilon"],
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", gradient_accumulation_steps_default),
        load_best_model_at_end=False,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
        bf16=(use_cuda and torch.cuda.is_bf16_supported()),
        report_to=[],
        disable_tqdm=False,
    )

    trainer = MSETrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=final_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[],
    )

    print(f"Starting training for {params['num_train_epochs']} epochs...\n")
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return trainer


# =================================================
#                 MAIN EXECUTION
# =================================================

if __name__ == "__main__":
    if RUN_SINGLE_MODEL:
        # ===== SINGLE MODEL MODE =====
        if use_upsampling_grid:
            # Single model with upsampling grid
            print(f"\n{'='*60}")
            print(f"SINGLE MODEL EXECUTION WITH UPSAMPLING GRID")
            print(f"Total configurations: {len(UPSAMPLING_GRID)}")
            print(f"{'='*60}\n")
            
            for config_idx, (upsample_enabled, upsample_threshold, upsample_factor_positive, upsample_factor_negative) in enumerate(UPSAMPLING_GRID):
                print(f"\n{'='*60}")
                print(f"CONFIG {config_idx}/{len(UPSAMPLING_GRID)-1}")
                print(f"Upsampling: {upsample_enabled}")
                if upsample_enabled:
                    print(f"  Threshold: {upsample_threshold}")
                    print(f"  Positive Factor: {upsample_factor_positive}")
                    print(f"  Negative Factor: {upsample_factor_negative}")
                print(f"{'='*60}\n")
                
                # Prepare datasets with current upsampling configuration
                search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
                    full_train_df,
                    upsample_enabled,
                    upsample_threshold,
                    upsample_factor_positive,
                    upsample_factor_negative
                )
                
                # Train single model
                print("Parameters:")
                for k, v in SINGLE_MODEL_PARAMS.items():
                    print(f"  {k}: {v}\n")
                
                trainer = train_single_model(SINGLE_MODEL_PARAMS, full_train_dataset)
                
                # Evaluate and save results
                evaluate_and_save_results(
                    trainer.model,
                    tokenizer,
                    final_test_dataset,
                    results_file=RESULTS_FILE,
                    best_params=SINGLE_MODEL_PARAMS,
                    upsample_enabled=upsample_enabled,
                    upsample_threshold=upsample_threshold,
                    upsample_factor_positive=upsample_factor_positive,
                    upsample_factor_negative=upsample_factor_negative,
                    config_idx=config_idx,
                    batch_size=batch_size_eval,
                    is_single_model=True
                )
                
                print(f"Config {config_idx} COMPLETED")
            
            print(f"\n{'='*60}")
            print(f"ALL CONFIGURATIONS COMPLETED!")
            print(f"Results saved to: {RESULTS_FILE}")
            print(f"{'='*60}\n")
        
        else:
            # Single model with standard upsampling
            print(f"\n{'='*60}")
            print(f"SINGLE MODEL EXECUTION WITH STANDARD UPSAMPLING")
            print(f"{'='*60}\n")
            print(f"Upsampling Enabled: {upsample_extreme}")
            if upsample_extreme:
                print(f"  Threshold: {upsample_threshold}")
                print(f"  Positive Factor: {upsample_factor_positive}")
                print(f"  Negative Factor: {upsample_factor_negative}")
            print(f"{'='*60}\n")
            
            # Prepare datasets with standard upsampling configuration
            search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
                full_train_df,
                upsample_extreme,
                upsample_threshold,
                upsample_factor_positive,
                upsample_factor_negative
            )
            
            # Train single model
            print("Parameters:")
            for k, v in SINGLE_MODEL_PARAMS.items():
                print(f"  {k}: {v}\n")
            
            trainer = train_single_model(SINGLE_MODEL_PARAMS, full_train_dataset)
            
            # Evaluate and save results
            evaluate_and_save_results(
                trainer.model,
                tokenizer,
                final_test_dataset,
                results_file=RESULTS_FILE,
                best_params=SINGLE_MODEL_PARAMS,
                upsample_enabled=upsample_extreme,
                upsample_threshold=upsample_threshold,
                upsample_factor_positive=upsample_factor_positive,
                upsample_factor_negative=upsample_factor_negative,
                batch_size=batch_size_eval,
                is_single_model=True
            )
            
            print(f"\n{'='*60}")
            print(f"SINGLE MODEL RUN COMPLETED!")
            print(f"Results saved to: {RESULTS_FILE}")
            print(f"Model saved to: {OUTPUT_DIR}")
            print(f"{'='*60}\n")
    
    elif use_upsampling_grid:
        # Optuna search with upsampling grid (multiple models)
        print(f"\n{'='*60}")
        print(f"RUNNING UPSAMPLING GRID SEARCH WITH OPTUNA")
        print(f"Total configurations: {len(UPSAMPLING_GRID)}")
        print(f"Trials per config: {n_trials_optuna}")
        print(f"{'='*60}\n")
        
        for config_idx, (upsample_enabled, upsample_threshold, upsample_factor_positive, upsample_factor_negative) in enumerate(UPSAMPLING_GRID):
            print(f"\n{'='*60}")
            print(f"CONFIG {config_idx}/{len(UPSAMPLING_GRID)-1}")
            print(f"Upsampling: {upsample_enabled}")
            if upsample_enabled:
                print(f"  Threshold: {upsample_threshold}")
                print(f"  Positive Factor: {upsample_factor_positive}")
                print(f"  Negative Factor: {upsample_factor_negative}")
            print(f"{'='*60}\n")
            
            # Prepare datasets with current upsampling configuration
            search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
                full_train_df,
                upsample_enabled,
                upsample_threshold,
                upsample_factor_positive,
                upsample_factor_negative
            )
            
            # Run Optuna search
            if use_kfold:
                best_params = run_optuna_kfold_search(search_train_dataset, full_train_df, n_trials=n_trials_optuna)
            else:
                best_params = run_optuna_simple_search(search_train_dataset, search_val_dataset, n_trials=n_trials_optuna)

            print("\n=== Best params chosen ===")
            print(best_params)

            # Final Training and Evaluation
            final_trainer = train_final_model(best_params, full_train_dataset)
            evaluate_and_save_results(
                final_trainer.model, 
                tokenizer, 
                final_test_dataset, 
                results_file=RESULTS_FILE,
                best_params=best_params,
                upsample_enabled=upsample_enabled,
                upsample_threshold=upsample_threshold,
                upsample_factor_positive=upsample_factor_positive,
                upsample_factor_negative=upsample_factor_negative,
                config_idx=config_idx,
                batch_size=batch_size_eval,
                is_single_model=False
            )
            
            print(f"\nConfig {config_idx} COMPLETED")
        
        print(f"\n{'='*60}")
        print(f"ALL CONFIGURATIONS COMPLETED!")
        print(f"Results saved to: {RESULTS_FILE}")
        print(f"{'='*60}\n")
    
    else:
        # Standard Optuna search with single upsampling config
        print(f"\n{'='*60}")
        print(f"RUNNING STANDARD UPSAMPLING WITH OPTUNA")
        print(f"Upsampling Enabled: {upsample_extreme}")
        if upsample_extreme:
            print(f"  Threshold: {upsample_threshold}")
            print(f"  Positive Factor: {upsample_factor_positive}")
            print(f"  Negative Factor: {upsample_factor_negative}")
        print(f"{'='*60}\n")
        
        # Prepare datasets with standard upsampling configuration
        search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
            full_train_df,
            upsample_extreme,
            upsample_threshold,
            upsample_factor_positive,
            upsample_factor_negative
        )
        
        # Run Optuna search
        if use_kfold:
            best_params = run_optuna_kfold_search(search_train_dataset, full_train_df, n_trials=n_trials_optuna)
        else:
            best_params = run_optuna_simple_search(search_train_dataset, search_val_dataset, n_trials=n_trials_optuna)

        print("\n=== Best params chosen ===")
        print(best_params)

        # Final Training and Evaluation
        final_trainer = train_final_model(best_params, full_train_dataset)
        evaluate_and_save_results(
            final_trainer.model, 
            tokenizer, 
            final_test_dataset, 
            results_file=RESULTS_FILE,
            best_params=best_params,
            upsample_enabled=upsample_extreme,
            upsample_threshold=upsample_threshold,
            upsample_factor_positive=upsample_factor_positive,
            upsample_factor_negative=upsample_factor_negative,
            batch_size=batch_size_eval,
            is_single_model=False
        )
        
        print(f"\nRUN COMPLETED!")
        print(f"Results saved to: {RESULTS_FILE}\n")
