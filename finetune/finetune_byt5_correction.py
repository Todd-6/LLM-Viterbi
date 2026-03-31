import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split

# Configuration
# Robust path handling relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "google/byt5-small"
TRAIN_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/byt5_correction_train_data.jsonl"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/ByT5_correction_finetuned"))
MAX_LENGTH = 512
BATCH_SIZE = 8 # Adjust based on VRAM
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4 # ByT5 often handles higher LR
WARMUP_STEPS = 1000
SAVE_STEPS = 3000
EVAL_STEPS = 3000

class CorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Matches logic in viterbi_lm_decode.py: correct_text_with_lm
        input_text = "grammar: " + item['input_text']
        target_text = item['target_text']

        # Tokenize input
        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            truncation=True,
            padding=False # Dynamic padding by collator
        )

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text, 
                max_length=self.max_length, 
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    data = []
    # Handle both absolute and relative paths
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find data file at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line: {line[:50]}...")
    return data

def main():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # 2. Data
    all_data = load_data(TRAIN_FILE)
    print(f"Total examples: {len(all_data)}")
    
    train_data, val_data = train_test_split(all_data, test_size=0.05, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = CorrectionDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = CorrectionDataset(val_data, tokenizer, MAX_LENGTH)

    # 3. Training Config
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=2,
        fp16=False, # Disable FP16 to avoid numerical instability (NaN gradients)
        report_to="none",
        generation_max_length=MAX_LENGTH,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 4. Train
    print("Starting training...")
    trainer.train()

    # 5. Save
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
