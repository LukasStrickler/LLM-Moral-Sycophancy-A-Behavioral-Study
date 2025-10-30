import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

class SycophancyDataset(Dataset):
    """
    Custom Dataset class for sycophancy data.
    Tokenizes text inputs and stores associated sycophancy scores (labels).
    """
    def __init__(self, texts, scores, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.scores = scores

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Labels are float sycophancy scores for regression
        item['labels'] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.scores)

def main():
    # Parameters
    model_name = "answerdotai/ModernBERT-base"  # Pretrained ModernBERT model
    csv_path = "../data/humanLabel/raw/aita_compiled_llm_outputs_sycophancy_test.csv"  # Path to your CSV file
    delimiter = ";"  # CSV delimiter
    
    # Step 1: Load pretrained tokenizer and model (with a regression head)
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Regression output
    
    # Step 2: Load dataset from CSV
    print(f"Loading dataset from {csv_path}...")
    data = pd.read_csv(csv_path, delimiter=delimiter, encoding='latin1')  # or encoding='cp1252'
    
    # Sample 3% of the data randomly with a fixed seed for reproducibility (for testing purposes)
    sampled_data = data.sample(frac=0.03, random_state=42).reset_index(drop=True)

    
    # Extract relevant columns - ensure model_response and sycophancy columns exist in your CSV
    texts = sampled_data['model_response'].astype(str).tolist()
    # Convert labels to float and ensure scaling - assumed already between -1 and 1 in your CSV
    scores = sampled_data['sycophancy'].str.replace(",", ".").astype(float).tolist()  # handle comma decimal

    # Step 3: Create dataset for training
    print("Preparing dataset for training...")
    dataset = SycophancyDataset(texts, scores, tokenizer)
    
    # Optional: split into train and validation (here, example is all training)
    # Can implement train_test_split from sklearn or use Hugging Face datasets for advanced splits
    
    # Step 4: Setup training arguments
    training_args = TrainingArguments(
        output_dir='./modernbert_sycophancy_model',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        #evaluation_strategy="no",  # Change to "steps" or "epoch" when validation dataset used
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=200,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        report_to='none'  # No logging integration
    )

    # Step 5: Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=val_dataset,  # Add validation dataset if split
        tokenizer=tokenizer,
        # Define regression loss (default for sequence classification with 1 label is MSELoss)
    )
    
    # Step 6: Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Step 7: Save the fine-tuned model and tokenizer
    print("Saving the fine-tuned model and tokenizer...")
    model.save_pretrained('./modernbert_sycophancy_model')
    tokenizer.save_pretrained('./modernbert_sycophancy_model')

if __name__ == "__main__":
    main()
