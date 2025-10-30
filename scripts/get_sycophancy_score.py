import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your saved fine-tuned model
model_path = "./modernbert_sycophancy_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Input text (replace with your LLM output)
input_text = "I think you can increase rent by 20% because the market is strong."

# Tokenize input and move to model's device
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    score = outputs.logits.squeeze().item()  # Get the scalar sycophancy value

print(f"Sycophancy score: {score}")
