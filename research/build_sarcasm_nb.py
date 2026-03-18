import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ─── Cell 1: Install ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# install / upgrade required packages
# run this cell first if packages are missing
# !pip install transformers datasets scikit-learn torch accelerate
import warnings
warnings.filterwarnings("ignore")
print("imports ready!")
"""))

# ─── Cell 2: Load Dataset ─────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd

# load the dataset
df = pd.read_csv("../dataset/sarcasm_dataset_multilingual_10000.csv")

# quick look
print("Shape:", df.shape)
print("\\nFirst 5 rows:")
print(df.head())
print("\\nLabel distribution:")
print(df["label"].value_counts())
"""))

# ─── Cell 3: Clean + Preprocess ───────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# cleaning the text - nothing fancy, just lowercase + strip
def clean_text(text):
    text = str(text)
    text = text.lower().strip()
    return text

df["text"] = df["text"].apply(clean_text)

# check
print("After cleaning:")
print(df.head())
print("\\nAny nulls?", df.isnull().sum())
"""))

# ─── Cell 4: Train / Val / Test Split ─────────────────────
cells.append(nbf.v4.new_code_cell(
"""from sklearn.model_selection import train_test_split

# split: 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))
"""))

# ─── Cell 5: Tokenizer ────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""from transformers import AutoTokenizer

# using distilbert-multilingual because it supports Hindi + Marathi + Hinglish
model_name = "distilbert-base-multilingual-cased"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_name)
print("Done!")

# test the tokenizer real quick
sample = "Wow tum call nahi karte"
tokens = tok(sample, return_tensors="pt")
print("\\nSample tokenized shape:", tokens["input_ids"].shape)
"""))

# ─── Cell 6: Dataset Class ────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import torch
from torch.utils.data import Dataset

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # tokenize each text
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # return as flat dict
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# create the three datasets
train_dataset = SarcasmDataset(train_df["text"], train_df["label"], tok)
val_dataset   = SarcasmDataset(val_df["text"],   val_df["label"],   tok)
test_dataset  = SarcasmDataset(test_df["text"],  test_df["label"],  tok)

print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))
"""))

# ─── Cell 7: Load Model ───────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""from transformers import AutoModelForSequenceClassification

# 2 labels: 0 = normal, 1 = sarcasm
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("Model loaded!")
print("Total params:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M")
"""))

# ─── Cell 8: Metrics ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# this function is called by the Trainer to compute metrics after each eval
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)     # pick the highest probability class
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

print("compute_metrics function ready!")
"""))

# ─── Cell 9: Training ─────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""from transformers import TrainingArguments, Trainer

# training settings - copied from research papers roughly
training_args = TrainingArguments(
    output_dir="../sarcasm_model",         # where to save the model
    learning_rate=2e-5,                    # small lr for fine-tuning
    per_device_train_batch_size=16,        # 16 samples per batch
    num_train_epochs=5,                    # train 5 times through data
    warmup_steps=200,                      # slowly increase lr at start
    weight_decay=0.01,                     # regularisation
    logging_steps=50,
    evaluation_strategy="epoch",           # evaluate after each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,           # keep the best checkpoint
    report_to="none",                      # dont use wandb
)

# create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training done!")
"""))

# ─── Cell 10: Evaluate on Test Set ────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# evaluate on the test set (unseen data)
print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)

print("\\n=== TEST RESULTS ===")
for k, v in test_results.items():
    print(f"{k}: {round(v, 4)}")
"""))

# ─── Cell 11: Save Model ──────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# save the trained model and tokenizer  
# then we can use it directly in models.py
save_path = "../sarcasm_model"

model.save_pretrained(save_path)
tok.save_pretrained(save_path)

print(f"Model saved to: {save_path}")
print("Now you can load it with:")
print(f'  pipeline("text-classification", model="{save_path}")')
"""))

# ─── Cell 12: Test Predictions ────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""from transformers import pipeline

# load back the saved model
sarcasm_pipe = pipeline("text-classification", model="../sarcasm_model", tokenizer="../sarcasm_model")

# test with some real Hinglish messages
test_msgs = [
    "Haan obviously you didn't call",         # should be sarcasm (1)
    "Wow tumne reply nahi kiya",               # should be sarcasm (1)
    "okay thanks for letting me know",        # should be normal (0)
    "I am fine, how are you?",                # should be normal (0)
    "Rehne do tu mala ignore karto",          # should be sarcasm (1)
]

print("=== PREDICTIONS ===")
for msg in test_msgs:
    result = sarcasm_pipe(msg)[0]
    label_name = "SARCASM" if result["label"] == "LABEL_1" else "NORMAL"
    print(f"\\nMsg: {msg}")
    print(f"  → {label_name} ({result['score']:.2%} confident)")
"""))

# ─── Cell 13: Rule-based Ensemble ─────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ensemble: combine model + rule-based signals for better accuracy
# this is the secret boost mentioned in Phase 6

# sarcastic keywords in Hinglish / English
sarc_keywords = [
    "obviously", "of course", "wow", "great", "perfect", "sure",
    "haan haan", "haan obviously", "bilkul", "acha thik hai",
    "kay mast", "rehne do", "ho na", "vahh", "wah"
]

def smart_sarcasm_detection(text, sarcasm_pipeline):
    text_lower = text.lower()
    
    # rule 1: check keywords
    rule_detected = any(kw in text_lower for kw in sarc_keywords)
    
    # rule 2: model prediction
    result = sarcasm_pipeline(text)[0]
    model_conf = result["score"]
    model_label = result["label"]

    # ensemble logic
    if rule_detected:
        return "sarcasm"
    elif model_label == "LABEL_1" and model_conf > 0.7:
        return "sarcasm"
    else:
        return "no sarcasm"

# test it
for msg in test_msgs:
    ans = smart_sarcasm_detection(msg, sarcasm_pipe)
    print(f"{msg}\\n  → {ans}\\n")
"""))

nb["cells"] = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.10"}

import os
os.makedirs("d:/tomato/sarcasm_ml_model", exist_ok=True)

with open("d:/tomato/sarcasm_ml_model/sarcasm_training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("sarcasm_training.ipynb created successfully!")
