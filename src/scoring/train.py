"""
Trains a regression model (ModernBERT) to predict consensus scores from model responses. 
Includes logic for handling imbalanced data via upsampling, optional Optuna hyperparameter optimization (with K-Fold support), 
and generates detailed performance metrics per LLM model.
"""

import os
import time

# Dynamo triggers an FX symbolic trace error with this setup, so we're disabling it 
# to ensure the training loop runs stable.
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

# Grab the pruning callback if Optuna is available, otherwise skip it.
try:
    from optuna.integration import TransformersPruningCallback
    _HAS_TRANSFORMERS_PRUNING = True
except Exception:
    _HAS_TRANSFORMERS_PRUNING = False

warnings.filterwarnings("ignore", category=FutureWarning, message=".*is deprecated and will be removed*")
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars*")


# --- Global Configuration ---

TRAIN_CSV_PATH = "training_data.csv"
VAL_CSV_PATH = "validation_data.csv"
MODEL_PATH = "answerdotai/ModernBERT-base"
OUTPUT_DIR = "./modernbert_chosen_consensus_advanced"

# Toggles for the run mode
RUN_SINGLE_MODEL = True                 # Set to False to run the full Optuna search
use_upsampling_grid = False             # Set to True to iterate through the UPSAMPLING_GRID list
use_kfold = False                         
optuna_prune = False                      

# Data handling
USE_PROMPTS = False                      # Include prompt text in input or just the response

# Default upsampling settings (active if not using the grid)
upsample_extreme = False                 
upsample_threshold = 0.62                
upsample_factor_positive = 2.8           
upsample_factor_negative = 2             

# Search settings
n_trials_optuna = 180                     
freeze_during_search = True              # Speeds up search by freezing the backbone

# Training basics
seed = 42
max_length = 2048
batch_size_eval = 16
gradient_accumulation_steps_default = 1
early_stopping_patience = 2

# Params for a single run (if we aren't searching)
SINGLE_MODEL_PARAMS = {
    "learning_rate": 8.693125771439495e-05,
    "per_device_train_batch_size": 3,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.12583141206455262,
    "num_train_epochs": 5,
    "adam_epsilon": 1.071087092582853e-08,
    "warmup_ratio": 0.18651889995585025,
    "lr_scheduler_type": "constant"
}

# Grid for testing different data balancing strategies
UPSAMPLING_GRID = [
    (False, 0.0, 1.0, 1.0),             # Baseline
    (True, 0.72, 2.8, 2.0),             
    (True, 0.72, 2.8, 2.8),             
    (True, 0.72, 1.0, 2.0),             
    (True, 0.72, 1.5, 2.1),             
    (True, 0.72, 2.8, 3.0),             
    (True, 0.72, 2.5, 2.5),              
    (True, 0.72, 2.8, 3.4),             
    (True, 0.72, 2.3, 2.3),             
    (True, 0.72, 2.5, 1.8),             
]

# Set the output filename based on what we're running so we don't overwrite good results.
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


set_seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# --- Tokenizer ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def tokenize_texts_no_padding(tokenizer, texts, text_pairs=None, max_length=max_length):
    # We tokenize without padding here to save memory; the DataCollator handles padding 
    # dynamically per batch later.
    args = {
        "text": texts,
        "truncation": True,
        "padding": False,
        "max_length": max_length,
        "return_attention_mask": True,
        "return_token_type_ids": True,
    }
    
    if text_pairs is not None:
        args["text_pair"] = text_pairs
        
    return tokenizer(**args)


# --- Dataset ---

class EncodedRegDataset(Dataset):
    # Simple wrapper to hold our lists. Converting to tensors happens at getitem 
    # to keep memory footprint low during init.
    def __init__(self, encodings, labels, texts):
        if hasattr(encodings, "data") and isinstance(encodings.data, dict):
            enc_dict = encodings.data
        elif isinstance(encodings, dict):
            enc_dict = encodings
        else:
            raise ValueError("encodings must be a dict or BatchEncoding")
            
        self.encodings = {k: list(v) for k, v in enc_dict.items()}
        
        if "token_type_ids" not in self.encodings:
             self.encodings.pop("token_type_ids", None)

        self.labels = list(labels)
        self.texts = list(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(float(self.labels[idx]), dtype=torch.float),
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        return item

def get_dataset_labels(dataset):
    # Helper to get labels out of a Subset or the main dataset without crashing
    if isinstance(dataset, Subset):
        return [dataset.dataset.labels[i] for i in dataset.indices]
    elif hasattr(dataset, "labels"):
        return dataset.labels
    else:
        raise ValueError(f"Could not extract labels from dataset type: {type(dataset)}")


# --- Data Prep & Upsampling ---

def upsample_extreme_consensus(df, threshold=0.5, upsample_factor_positive=2, upsample_factor_negative=2, stage=""):
    # The dataset is imbalanced. This isolates the high/low consensus rows, copies them, 
    # and shuffles them back in to help the model learn the tails.
    upsample_factor_positive = int(upsample_factor_positive)
    upsample_factor_negative = int(upsample_factor_negative)
    
    positive_extreme_mask = df["chosen_consensus"] > threshold
    negative_extreme_mask = df["chosen_consensus"] < -threshold
    
    normal_df = df[~(positive_extreme_mask | negative_extreme_mask)]
    positive_extreme_df = df[positive_extreme_mask]
    negative_extreme_df = df[negative_extreme_mask]
    
    positive_extreme_df_upsampled = pd.concat([positive_extreme_df] * upsample_factor_positive, ignore_index=True)
    negative_extreme_df_upsampled = pd.concat([negative_extreme_df] * upsample_factor_negative, ignore_index=True)
    
    df_balanced = pd.concat([normal_df, positive_extreme_df_upsampled, negative_extreme_df_upsampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    stage_label = f" [{stage}]" if stage else ""
    print(f"\n--- Upsampling Details{stage_label} ---")
    print(f"Original: {len(df)} -> Final: {len(df_balanced)}")
    print(f"Added: {len(df_balanced) - len(df)} samples\n")
    
    return df_balanced


def prepare_datasets(full_train_df, upsample_enabled, upsample_threshold, upsample_factor_positive, upsample_factor_negative):
    # Splits data for search vs final training. 
    # Crucially, we only upsample the TRAINING portion. Validation must remain real distribution.
    
    # 1. Search Data Prep
    if use_kfold:
        search_train_df = full_train_df
        search_val_dataset = None
    else:
        sub_train_df, sub_val_df = train_test_split(
            full_train_df,
            test_size=0.32,
            random_state=seed,
        )
        
        if upsample_enabled:
            sub_train_df = upsample_extreme_consensus(sub_train_df, threshold=upsample_threshold, upsample_factor_positive=upsample_factor_positive, upsample_factor_negative=upsample_factor_negative, stage="search training")
        
        search_train_df = sub_train_df
        
        sub_val_texts = sub_val_df["model_response_text"].astype(str).tolist()
        sub_val_labels = sub_val_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
        
        if USE_PROMPTS:
            sub_val_prompts = sub_val_df["prompt_body"].fillna("").astype(str).tolist()
            sub_val_encodings = tokenize_texts_no_padding(tokenizer, sub_val_prompts, text_pairs=sub_val_texts, max_length=max_length)
        else:
            sub_val_encodings = tokenize_texts_no_padding(tokenizer, sub_val_texts, max_length=max_length)
            
        search_val_dataset = EncodedRegDataset(sub_val_encodings, sub_val_labels, sub_val_texts)

    # Tokenize search training data
    search_train_texts = search_train_df["model_response_text"].astype(str).tolist()
    search_train_labels = search_train_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
    
    if USE_PROMPTS:
        search_train_prompts = search_train_df["prompt_body"].fillna("").astype(str).tolist()
        search_train_encodings = tokenize_texts_no_padding(tokenizer, search_train_prompts, text_pairs=search_train_texts, max_length=max_length)
    else:
        search_train_encodings = tokenize_texts_no_padding(tokenizer, search_train_texts, max_length=max_length)
        
    search_train_dataset = EncodedRegDataset(search_train_encodings, search_train_labels, search_train_texts)

    # 2. Final Training Data Prep (Uses 100% of data)
    full_train_df_for_final = full_train_df.copy()
    if upsample_enabled:
        full_train_df_for_final = upsample_extreme_consensus(full_train_df_for_final, threshold=upsample_threshold, upsample_factor_positive=upsample_factor_positive, upsample_factor_negative=upsample_factor_negative, stage="final training")

    full_train_texts = full_train_df_for_final["model_response_text"].astype(str).tolist()
    full_train_labels = full_train_df_for_final["chosen_consensus"].astype(float).clip(-1, 1).tolist()
    
    if USE_PROMPTS:
        full_train_prompts = full_train_df_for_final["prompt_body"].fillna("").astype(str).tolist()
        full_train_encodings = tokenize_texts_no_padding(tokenizer, full_train_prompts, text_pairs=full_train_texts, max_length=max_length)
    else:
        full_train_encodings = tokenize_texts_no_padding(tokenizer, full_train_texts, max_length=max_length)
        
    full_train_dataset = EncodedRegDataset(full_train_encodings, full_train_labels, full_train_texts)

    return search_train_dataset, search_val_dataset, full_train_dataset


# Load raw data once
full_train_df = pd.read_csv(TRAIN_CSV_PATH).dropna(subset=["model_response_text", "chosen_consensus"]).reset_index(drop=True)

# Prepare the held-out validation set once
final_test_df = pd.read_csv(VAL_CSV_PATH).dropna(subset=["model_response_text", "chosen_consensus"]).reset_index(drop=True)
final_test_texts = final_test_df["model_response_text"].astype(str).tolist()
final_test_labels = final_test_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()

if USE_PROMPTS:
    final_test_prompts = final_test_df["prompt_body"].fillna("").astype(str).tolist()
    final_test_encodings = tokenize_texts_no_padding(tokenizer, final_test_prompts, text_pairs=final_test_texts, max_length=max_length)
else:
    final_test_encodings = tokenize_texts_no_padding(tokenizer, final_test_texts, max_length=max_length)
    
final_test_dataset = EncodedRegDataset(final_test_encodings, final_test_labels, final_test_texts)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")


# --- Model Setup ---

class MSETrainer(Trainer):
    # Override loss for regression (MSE) instead of default CrossEntropy
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        return (loss, outputs) if return_outputs else loss

def get_model_init(freeze_backbone=False):
    # Using a closure to bake in the freeze logic. This allows Optuna to re-init 
    # the model cleanly for every trial without global state issues.
    def _model_init():
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=1,
            problem_type="regression",
            trust_remote_code=True,
        )

        # Optimization flags (Flash Attention / Grad Checkpointing)
        try:
            model.config.use_flash_attention = True
        except Exception:
            pass
        try:
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

        if max_length > 1024 and not freeze_backbone:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # Freezing logic for faster hyperparam search
        if freeze_backbone:
            backbone = None
            if hasattr(model, "base_model"):
                backbone = model.base_model
            elif hasattr(model, model.__class__.__name__.lower()):
                backbone = getattr(model, model.__class__.__name__.lower())
            
            if backbone is not None:
                for param in backbone.parameters():
                    param.requires_grad = False
            else:
                for name, p in model.named_parameters():
                    if any(k in name for k in ["classifier", "regressor", "score", "out_proj", "lm_head"]):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        return model
        
    return _model_init


# --- Metrics & Eval ---

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
    # Helper to run a training loop on a specific fold/split
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
        model_init=get_model_init(freeze_backbone=freeze_backbone_flag),
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    metrics = trainer.evaluate()
    
    return metrics["eval_mse"], metrics


# --- Optuna ---

def optuna_hp_space(trial):
    # The search space for hyperparams
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


def run_optuna_kfold_search(train_dataset, train_df, n_trials=20, 
                          upsample_enabled=False, upsample_threshold=0.5, 
                          upsample_factor_positive=2, upsample_factor_negative=2):
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
            
            # Upsample ONLY the fold's training data
            if upsample_enabled:
                fold_train_df = upsample_extreme_consensus(
                    fold_train_df, 
                    threshold=upsample_threshold, 
                    upsample_factor_positive=upsample_factor_positive, 
                    upsample_factor_negative=upsample_factor_negative,
                    stage="k-fold training"
                )
            
            # Convert fold data to datasets
            fold_train_texts = fold_train_df["model_response_text"].astype(str).tolist()
            fold_train_labels = fold_train_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
            
            if USE_PROMPTS:
                fold_train_prompts = fold_train_df["prompt_body"].fillna("").astype(str).tolist()
                fold_train_encodings = tokenize_texts_no_padding(tokenizer, fold_train_prompts, text_pairs=fold_train_texts, max_length=max_length)
            else:
                fold_train_encodings = tokenize_texts_no_padding(tokenizer, fold_train_texts, max_length=max_length)
                
            fold_train_subset = EncodedRegDataset(fold_train_encodings, fold_train_labels, fold_train_texts)
            
            fold_val_texts = fold_val_df["model_response_text"].astype(str).tolist()
            fold_val_labels = fold_val_df["chosen_consensus"].astype(float).clip(-1, 1).tolist()
            
            if USE_PROMPTS:
                fold_val_prompts = fold_val_df["prompt_body"].fillna("").astype(str).tolist()
                fold_val_encodings = tokenize_texts_no_padding(tokenizer, fold_val_prompts, text_pairs=fold_val_texts, max_length=max_length)
            else:
                fold_val_encodings = tokenize_texts_no_padding(tokenizer, fold_val_texts, max_length=max_length)
                
            fold_val_subset = EncodedRegDataset(fold_val_encodings, fold_val_labels, fold_val_texts)
            
            mse, _ = train_evaluate_fold(
                fold_train_subset, 
                fold_val_subset, 
                params, 
                freeze_backbone_flag=freeze_during_search
            )
            fold_mses.append(mse)
            
            # Pruning based on average MSE so far
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

        trainer = MSETrainer(
            model_init=get_model_init(freeze_backbone=freeze_during_search),
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


# --- Final Model Train ---

def train_final_model(best_params, full_train_dataset):
    
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
        model_init=get_model_init(freeze_backbone=False),
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


def plot_input_distributions(train_labels, val_labels, output_dir):
    # Dumps distributions so we can see if training data matches validation data shape.
    sns.set(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.histplot(train_labels, color='green', label='Training Data', kde=False)
    plt.title("Distribution of Training Data", fontsize=14)
    plt.xlabel("Sycophancy", fontsize=12)
    plt.ylabel("Amount", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_distribution_training_data.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(val_labels, color='red', label='Validation Data', kde=False)
    plt.title("Distribution of Validation Data", fontsize=14)
    plt.xlabel("Sycophancy", fontsize=12)
    plt.ylabel("Amount", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_distribution_validation_data.png"), dpi=300)
    plt.close()
    
    print(f"Distribution plots saved to {output_dir}")


def create_plots(true_labels, preds, output_dir):
    # Generates standard regression analysis plots (Actual vs Predicted, Residuals, Density)
    sns.set(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Scatter
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

    # 2. Residuals
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

    # 3. Density
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


def compute_per_model_r2_from_val_csv(df_val: pd.DataFrame, preds: np.ndarray, results_dir: str):
    # Calculates R2 specifically for each LLM model found in the validation data.
    df_val = df_val.copy()

    n = min(len(df_val), len(preds))
    df_val = df_val.iloc[:n]
    preds = preds[:n]

    df_val["chosen_consensus"] = df_val["chosen_consensus"].astype(float)
    preds = np.asarray(preds, dtype=float)

    if "model" not in df_val.columns:
        print("Warning: 'model' column not found in validation dataframe. Skipping per-model R2.")
        return

    r2_by_model = {}
    for m, g in df_val.groupby("model"):
        y_true = g["chosen_consensus"].values
        y_pred = preds[g.index.values]

        if len(y_true) < 2:
            r2_by_model[m] = np.nan
            continue

        if np.isclose(np.var(y_true), 0.0):
            r2_by_model[m] = np.nan
            continue

        r2_by_model[m] = r2_score(y_true, y_pred)

    r2_by_model = pd.Series(r2_by_model).sort_values(ascending=False)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "per_model_r2.csv")
    r2_by_model.to_csv(out_csv, header=["r2"])
    print(f"Saved per-model R2 CSV to: {out_csv}")

    plt.figure(figsize=(10, 5))
    r2_by_model.plot(kind="bar")
    plt.ylabel("R²")
    plt.xlabel("LLM model")
    plt.title("Per-model R² on held-out set")
    plt.tight_layout()
    out_png = os.path.join(results_dir, "per_model_r2.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved per-model R2 plot to: {out_png}")


def compute_per_model_mse_from_val_csv(df_val: pd.DataFrame, preds: np.ndarray, results_dir: str):
    # Calculates MSE specifically for each LLM model found in the validation data.
    df_val = df_val.copy()

    n = min(len(df_val), len(preds))
    df_val = df_val.iloc[:n]
    preds = preds[:n]

    df_val["chosen_consensus"] = df_val["chosen_consensus"].astype(float)
    preds = np.asarray(preds, dtype=float)

    if "model" not in df_val.columns:
        print("Warning: 'model' column not found in validation dataframe. Skipping per-model MSE.")
        return

    mse_by_model = {}
    for m, g in df_val.groupby("model"):
        y_true = g["chosen_consensus"].values
        y_pred = preds[g.index.values]

        mse_by_model[m] = mean_squared_error(y_true, y_pred)

    # Sort so best (lowest MSE) is first
    mse_by_model = pd.Series(mse_by_model).sort_values(ascending=True)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "per_model_mse.csv")
    mse_by_model.to_csv(out_csv, header=["mse"])
    print(f"Saved per-model MSE CSV to: {out_csv}")

    plt.figure(figsize=(10, 5))
    mse_by_model.plot(kind="bar", color='salmon')
    plt.ylabel("MSE")
    plt.xlabel("LLM model")
    plt.title("Per-model MSE on held-out set")
    plt.tight_layout()
    out_png = os.path.join(results_dir, "per_model_mse.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved per-model MSE plot to: {out_png}")


def evaluate_and_save_results(model, tokenizer, dataset, results_file, best_params, upsample_enabled, upsample_threshold, upsample_factor_positive, upsample_factor_negative, config_idx=None, batch_size=batch_size_eval, is_single_model=False, total_time_str=None):
    # Runs the final eval, generates all plots, and appends the metrics to our results file.
    model.to(device)
    model.eval()
    
    total = len(dataset)
    true_labels = np.array(get_dataset_labels(dataset))
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

    # Compute and plot breakdown by model
    compute_per_model_r2_from_val_csv(final_test_df, preds, OUTPUT_DIR)
    compute_per_model_mse_from_val_csv(final_test_df, preds, OUTPUT_DIR)
    
    mse = mean_squared_error(true_labels, preds)
    mae = mean_absolute_error(true_labels, preds)
    r2 = r2_score(true_labels, preds)

    create_plots(true_labels, preds, OUTPUT_DIR)
    plot_input_distributions(full_train_dataset.labels, final_test_dataset.labels, OUTPUT_DIR)

    # Logging text
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
    
    if total_time_str:
        new_entry += f"Total Script Execution Time: {total_time_str}\n"

    if is_single_model:
        new_entry += f"Mode: Single Model (Predefined Parameters)\n"
    else:
        new_entry += f"Number of runs: {n_trials_optuna:d}\n"
    
    new_entry += f"Use Prompts (Input Context): {USE_PROMPTS}\n" 

    new_entry += (
        f"K-Fold Cross-Validation: {use_kfold}\n"
        f"Upsampling Enabled: {upsample_enabled}\n"
        + (f"Upsampling Threshold: {upsample_threshold}\n"
           f"Upsampling Factor (Positive): {upsample_factor_positive}\n"
           f"Upsampling Factor (Negative): {upsample_factor_negative}\n" if upsample_enabled else "")
    )

    print(new_entry)
    
    # Prepend to log file so newest results are at the top
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing_content = f.read()
    else:
        existing_content = ""

    with open(results_file, "w") as f:
        f.write(new_entry + "\n" + existing_content)
        
    print(f"Results prepended to {results_file}")

    # Also save a run-specific log in the output dir
    model_results_path = os.path.join(OUTPUT_DIR, "run_metrics.txt")
    with open(model_results_path, "w") as f:
        f.write(new_entry)
    print(f"Results also saved to: {model_results_path}")


# --- Single Model Execution ---

def train_single_model(params, full_train_dataset):
    # Standard training loop without any Optuna overhead
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
        model_init=get_model_init(freeze_backbone=False),
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


# --- Main ---

if __name__ == "__main__":
    script_start_time = time.time()
    
    if RUN_SINGLE_MODEL:
        # 1. Single Model Mode
        if use_upsampling_grid:
            # Grid of upsampling settings with a fixed model config
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
                
                search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
                    full_train_df,
                    upsample_enabled,
                    upsample_threshold,
                    upsample_factor_positive,
                    upsample_factor_negative
                )
                
                print("Parameters:")
                for k, v in SINGLE_MODEL_PARAMS.items():
                    print(f"  {k}: {v}\n")
                
                trainer = train_single_model(SINGLE_MODEL_PARAMS, full_train_dataset)
                
                elapsed_seconds = time.time() - script_start_time
                execution_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

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
                    is_single_model=True,
                    total_time_str=execution_time_str
                )
                
                print(f"Config {config_idx} COMPLETED")
            
            print(f"\n{'='*60}")
            print(f"ALL CONFIGURATIONS COMPLETED!")
            print(f"Results saved to: {RESULTS_FILE}")
            print(f"{'='*60}\n")
        
        else:
            # Standard single run
            print(f"\n{'='*60}")
            print(f"SINGLE MODEL EXECUTION WITH STANDARD UPSAMPLING")
            print(f"{'='*60}\n")
            print(f"Upsampling Enabled: {upsample_extreme}")
            if upsample_extreme:
                print(f"  Threshold: {upsample_threshold}")
                print(f"  Positive Factor: {upsample_factor_positive}")
                print(f"  Negative Factor: {upsample_factor_negative}")
            print(f"{'='*60}\n")
            
            search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
                full_train_df,
                upsample_extreme,
                upsample_threshold,
                upsample_factor_positive,
                upsample_factor_negative
            )
            
            print("Parameters:")
            for k, v in SINGLE_MODEL_PARAMS.items():
                print(f"  {k}: {v}\n")
            
            trainer = train_single_model(SINGLE_MODEL_PARAMS, full_train_dataset)
            
            elapsed_seconds = time.time() - script_start_time
            execution_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

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
                is_single_model=True,
                total_time_str=execution_time_str 
            )
            
            print(f"\n{'='*60}")
            print(f"SINGLE MODEL RUN COMPLETED!")
            print(f"Results saved to: {RESULTS_FILE}")
            print(f"Model saved to: {OUTPUT_DIR}")
            print(f"{'='*60}\n")
    
    elif use_upsampling_grid:
        # 2. Optuna + Grid Search
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
            
            search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
                full_train_df,
                upsample_enabled,
                upsample_threshold,
                upsample_factor_positive,
                upsample_factor_negative
            )
            
            if use_kfold:
                best_params = run_optuna_kfold_search(
                    search_train_dataset, 
                    full_train_df, 
                    n_trials=n_trials_optuna,
                    upsample_enabled=upsample_enabled,
                    upsample_threshold=upsample_threshold,
                    upsample_factor_positive=upsample_factor_positive,
                    upsample_factor_negative=upsample_factor_negative
                )
            else:
                best_params = run_optuna_simple_search(search_train_dataset, search_val_dataset, n_trials=n_trials_optuna)

            print("\n=== Best params chosen ===")
            print(best_params)

            final_trainer = train_final_model(best_params, full_train_dataset)

            elapsed_seconds = time.time() - script_start_time
            execution_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

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
                is_single_model=False,
                total_time_str=execution_time_str
            )
            
            print(f"\nConfig {config_idx} COMPLETED")
        
        print(f"\n{'='*60}")
        print(f"ALL CONFIGURATIONS COMPLETED!")
        print(f"Results saved to: {RESULTS_FILE}")
        print(f"{'='*60}\n")
    
    else:
        # 3. Standard Optuna Search
        print(f"\n{'='*60}")
        print(f"RUNNING STANDARD UPSAMPLING WITH OPTUNA")
        print(f"Upsampling Enabled: {upsample_extreme}")
        if upsample_extreme:
            print(f"  Threshold: {upsample_threshold}")
            print(f"  Positive Factor: {upsample_factor_positive}")
            print(f"  Negative Factor: {upsample_factor_negative}")
        print(f"{'='*60}\n")
        
        search_train_dataset, search_val_dataset, full_train_dataset = prepare_datasets(
            full_train_df,
            upsample_extreme,
            upsample_threshold,
            upsample_factor_positive,
            upsample_factor_negative
        )
        
        if use_kfold:
            best_params = run_optuna_kfold_search(
                search_train_dataset, 
                full_train_df, 
                n_trials=n_trials_optuna,
                upsample_enabled=upsample_extreme,
                upsample_threshold=upsample_threshold,
                upsample_factor_positive=upsample_factor_positive,
                upsample_factor_negative=upsample_factor_negative
            )
        else:
            best_params = run_optuna_simple_search(search_train_dataset, search_val_dataset, n_trials=n_trials_optuna)

        print("\n=== Best params chosen ===")
        print(best_params)

        final_trainer = train_final_model(best_params, full_train_dataset)

        elapsed_seconds = time.time() - script_start_time
        execution_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

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
            is_single_model=False,
            total_time_str=execution_time_str
        )
        
        print(f"\nRUN COMPLETED!")
        print(f"Results saved to: {RESULTS_FILE}\n")