# full_rewrite_optuna_only.py

import os
# DISABLE DYNAMO TO FIX "FX symbolically trace" ERROR
os.environ["PYTORCH_ENABLE_TORCHDYNAMO"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1" 

import pandas as pd
import torch
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
#                 USER CONFIGURATION
# =================================================

TRAIN_CSV_PATH = "training_data.csv"
VAL_CSV_PATH = "validation_data.csv"
MODEL_PATH = "answerdotai/ModernBERT-base"
OUTPUT_DIR = "./modernbert_chosen_consensus_advanced"
RESULTS_FILE = "results_file.txt"

# Hyperparameter Search Controls
use_kfold = False           # True = k-fold CV for Optuna; False = simple train/val split for Optuna
n_trials_optuna = 300        # Number of Optuna trials
freeze_during_search = True # Freeze backbone during search for speed/stability

# Training & Model Settings
seed = 42
max_length = 2048
batch_size_eval = 16        # Batch size for manual evaluation
gradient_accumulation_steps_default = 1
early_stopping_patience = 2 # Early stopping patience (epochs) for final training
optuna_prune = True         # Enable Optuna pruning callback


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

# 1. Load Full Training Data (from CSV)
full_train_df = pd.read_csv(TRAIN_CSV_PATH).dropna(subset=["model_response_text", "chosen_consensus"]).reset_index(drop=True)

# 2. PREPARE DATA FOR SEARCH (Split Training Data Internal ONLY)
#    We NEVER touch VAL_CSV_PATH data here.
if use_kfold:
    # For k-fold, we use the entire training set. Cross-validation splits happen inside the search loop.
    search_train_df = full_train_df
    search_val_dataset = None
else:
    # For simple split, we split the TRAINING CSV into (sub-train, sub-val)
    # This ensures we tune hyperparameters on a subset of training data, not the final test set.
    sub_train_df, sub_val_df = train_test_split(
        full_train_df,
        test_size=0.20, # 20% of training data used for validation during search
        random_state=seed,
    )
    search_train_df = sub_train_df
    
    # Prepare the sub-validation dataset used ONLY for Optuna
    sub_val_texts = sub_val_df["model_response_text"].astype(str).tolist()
    sub_val_labels = sub_val_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
    sub_val_encodings = tokenize_texts_no_padding(tokenizer, sub_val_texts, max_length=max_length)
    search_val_dataset = EncodedRegDataset(sub_val_encodings, sub_val_labels, sub_val_texts)

# Prepare the training dataset for search (either full or sub-train depending on k-fold)
search_train_texts = search_train_df["model_response_text"].astype(str).tolist()
search_train_labels = search_train_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
search_train_encodings = tokenize_texts_no_padding(tokenizer, search_train_texts, max_length=max_length)
search_train_dataset = EncodedRegDataset(search_train_encodings, search_train_labels, search_train_texts)


# 3. PREPARE FINAL TRAINING DATA (Use 100% of Training CSV)
#    Once parameters are found, we retrain on ALL training data.
full_train_texts = full_train_df["model_response_text"].astype(str).tolist()
full_train_labels = full_train_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
full_train_encodings = tokenize_texts_no_padding(tokenizer, full_train_texts, max_length=max_length)
full_train_dataset = EncodedRegDataset(full_train_encodings, full_train_labels, full_train_texts)


# 4. PREPARE FINAL TEST/VALIDATION DATA (Real Held-out Validation)
#    This dataset is ONLY used for the final evaluation at the very end.
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

class MAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute custom loss (MAE / L1Loss)
        loss_fct = torch.nn.L1Loss()
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
    # ONLY enable if backbone is NOT frozen to avoid "None of the inputs have requires_grad" warning and slowdowns
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
        eval_strategy="epoch",  # Use eval_strategy for newer transformers versions
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

    trainer = Trainer(
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 8),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "constant", "cosine_with_restarts"]),
    }


def run_optuna_kfold_search(train_dataset, n_trials=20):
    # Here train_dataset is search_train_dataset (100% of training_data.csv)
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=seed), 
        pruner=optuna.pruners.MedianPruner()
    )

    def objective(trial):
        params = optuna_hp_space(trial)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_mses = []

        for train_idx, val_idx in kf.split(range(len(train_dataset))):
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            
            mse, _ = train_evaluate_fold(
                train_subset, 
                val_subset, 
                params, 
                freeze_backbone_flag=freeze_during_search
            )
            fold_mses.append(mse)
            
            # Report intermediate result to Optuna after each fold
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


def run_optuna_simple_search(n_trials=20):
    # Uses separate search_train_dataset and search_val_dataset split from training_data.csv
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
            callbacks.append(TransformersPruningCallback(trial, "eval_mae"))

        # Freeze backbone if desired for search
        global _FREEZE_BACKBONE
        _FREEZE_BACKBONE = freeze_during_search

        trainer = MAETrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=search_train_dataset, # Using search split
            eval_dataset=search_val_dataset,    # Using search split
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks if callbacks else None,
        )

        trainer.train()
        metrics = trainer.evaluate()
        _FREEZE_BACKBONE = False

        # report final metric to optuna
        val_mae = metrics["eval_mae"]
        trial.report(val_mae, 0)
        
        if optuna_prune and trial.should_prune():
            raise optuna.TrialPruned()

        return val_mae

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

def train_final_model(best_params):
    global _FREEZE_BACKBONE
    _FREEZE_BACKBONE = False
    
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    final_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",      # New name for evaluation_strategy      
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
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
        bf16=(use_cuda and torch.cuda.is_bf16_supported()),
        report_to=[],  # Disable wandb/mlflow if not using
        disable_tqdm=False,
    )

    callbacks = [] #EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    # Use FULL training dataset for final training
    # Use FINAL TEST dataset for validation (early stopping check)
    final_trainer = MAETrainer(
        model_init=model_init,
        args=final_args,
        train_dataset=full_train_dataset, # 100% of training data
        eval_dataset=final_test_dataset,  # Real validation data
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print(f"Starting FINAL training for exactly {best_params['num_train_epochs']} epochs (no early stopping)...")
    final_trainer.train()
    # Save the FINAL model (after all epochs)
    final_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return final_trainer


import datetime  # Make sure to import datetime

def evaluate_and_save_results(model, tokenizer, dataset, results_file, best_params, batch_size=batch_size_eval):
    """
    Evaluate the final model, print metrics, and PREPEND results to a file.
    """
    model.to(device)
    model.eval()
    
    total = len(dataset)
    true_labels = np.array(dataset.labels)
    preds_list = []

    # Form batches by slicing and dynamic padding
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

    # --- Formatting for File ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format parameters next to each other (inline)
    params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
    
    separator = "-" * 50
    
    new_entry = (
        f"{separator}\n"
        f"Time: {timestamp}\n"
        f"Params: {params_str}\n"
        f"Validation MSE: {mse:.4f}\n"
        f"Validation MAE: {mae:.4f}\n"
        f"Validation R^2: {r2:.4f}\n"
        f"Number of runs: {n_trials_optuna:.4f}\n"
    )

    print(new_entry)
    
    # --- Prepend to File ---
    # Read existing content first
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing_content = f.read()
    else:
        existing_content = ""

    # Write New Entry + Existing Content
    with open(results_file, "w") as f:
        f.write(new_entry + "\n" + existing_content)
        
    print(f"Results prepended to {results_file}")


# =================================================
#                 MAIN EXECUTION
# =================================================

if __name__ == "__main__":
    # Select search method
    if use_kfold:
        best_params = run_optuna_kfold_search(search_train_dataset, n_trials=n_trials_optuna)
    else:
        best_params = run_optuna_simple_search(n_trials=n_trials_optuna)

    print("\n=== Best params chosen ===")
    print(best_params)

    # Final Training and Evaluation
    final_trainer = train_final_model(best_params)
    evaluate_and_save_results(
        final_trainer.model, 
        tokenizer, 
        final_test_dataset, 
        results_file=RESULTS_FILE,
        best_params=best_params,
        batch_size=batch_size_eval
    )
