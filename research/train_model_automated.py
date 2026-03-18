import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os

# 1. Load Data
df = pd.read_csv("dataset/sarcasm_dataset_multilingual_10000.csv")
df["text"] = df["text"].astype(str).str.lower().str.strip()

# 2. Split (Using 500 samples for faster demo, but 10k works too)
# We will use the full 10k since the user asked for a real training 
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# 3. Tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = SarcasmDataset(train_df["text"], train_df["label"], tokenizer)
val_dataset = SarcasmDataset(val_df["text"], val_df["label"], tokenizer)

# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

# 5. Training
training_args = TrainingArguments(
    output_dir="./sarcasm_model_checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3, # 3 epochs for speed
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting custom model training...")
trainer.train()

# 6. Save
save_path = "sarcasm_model"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")
