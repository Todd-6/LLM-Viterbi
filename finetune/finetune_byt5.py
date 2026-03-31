import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from sklearn.model_selection import train_test_split

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "google/byt5-small"  # Using small for speed/memory. Change to 'base' if needed.
DATA_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/byt5_training_data_full.jsonl"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/byt5_finetuned"))
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 20  # We are predicting 5 chars, so 20 is plenty
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
EVAL_STEPS = 500
SAVE_STEPS = 1000

class ViterbiCandidateDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prefix = item['prefix']
        target = item['target']
        
        # Use a decoder-only style training setup to match the Viterbi decoder.
        # In that decoder, the encoder input is empty (pad only) and the decoder
        # receives the historical context, so training should mirror that usage.
        
        full_text = prefix + target
        
        # 1. Encoder input: a placeholder PAD token only.
        # This tells the model there is no encoder-side context.
        # ByT5 may still require at least one token as encoder input.
        #
        # 2. Labels: the complete sequence (prefix + target).
        # The model learns to predict each next character from history, so the
        # maximum length needs to cover both prefix and target.
        
        labels = self.tokenizer(
            full_text, 
            max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH, 
            truncation=True, 
            padding=False
        )

        # Build model inputs.
        model_inputs = {}
        model_inputs["labels"] = labels["input_ids"]
        # Encoder input is a single PAD token.
        model_inputs["input_ids"] = [self.tokenizer.pad_token_id]
        
        return model_inputs

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Waiting or using dummy data if testing.")
        return []
        
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

class RankingEvalCallback(TrainerCallback):
    """
    Custom callback to evaluate ranking accuracy on a small subset of validation data
    at the end of each epoch.
    """
    def __init__(self, val_data, tokenizer, model, device, num_samples=100):
        self.val_data = val_data[:num_samples] # Check first N samples
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
    def on_evaluate(self, args, state, control, **kwargs):
        # We run this during evaluation phase
        self.model.eval()
        correct_count = 0
        total = 0
        
        print("\nRunning Ranking Evaluation on Validation Subset...")
        
        with torch.no_grad():
            for item in self.val_data:
                prefix = item['prefix']
                target = item['target']
                candidates = item['candidates']
                
                # Ensure target is in candidates (it should be, but just in case)
                if target not in candidates:
                    candidates.append(target)
                
                # Calculate log-prob for each candidate given prefix
                # To do this efficiently, we can batch it, but simple loop is fine for N=10
                
                candidate_scores = []
                
                # Encoder input: ALL candidates share the same empty encoder input (just PAD)
                # Decoder-only mode requires dummy encoder input
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
                encoder_input_ids = torch.tensor([[pad_token_id]], device=self.device)
                
                for cand in candidates:
                    # Construct full text for Decoder-only evaluation
                    full_text = prefix + cand
                    
                    # Prepare labels (Prefix + Candidate)
                    # We want to score P(Full Sequence)
                    labels = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
                    
                    # Forward pass to get loss
                    # input_ids: [PAD]
                    # labels: [Prefix + Candidate]
                    outputs = self.model(
                        input_ids=encoder_input_ids,
                        labels=labels
                    )
                    
                    # loss is average NLL per token. 
                    # Total NLL = loss * seq_len
                    # Score = -Total NLL (Log Prob of the whole sequence)
                    # Ranking by P(Prefix + Candidate) is equivalent to P(Candidate | Prefix)
                    # because P(Prefix) is constant across all candidates for this item.
                    
                    seq_len = labels.shape[1]
                    log_prob = -outputs.loss.item() * seq_len
                    
                    candidate_scores.append((cand, log_prob))
                
                # Sort by score desc
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Check if top 1 is target
                if candidate_scores[0][0] == target:
                    correct_count += 1
                total += 1
                
        accuracy = correct_count / total if total > 0 else 0
        print(f"Ranking Accuracy: {accuracy:.4f} ({correct_count}/{total})")

def main():
    # 1. Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    
    # 2. Load Data
    all_data = load_data(DATA_FILE)
    if not all_data:
        print("No data found. Please run generate_byt5_training_data.py first.")
        return

    # Split
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    train_dataset = ViterbiCandidateDataset(train_data, tokenizer)
    val_dataset = ViterbiCandidateDataset(val_data, tokenizer)
    
    # DEBUG: Check first item
    print("DEBUG: Sample Item 0:")
    sample = train_dataset[0]
    print(f"  Input IDs: {sample['input_ids'][:10]}... (len {len(sample['input_ids'])})")
    print(f"  Labels: {sample['labels']} (len {len(sample['labels'])})")
    print(f"  Decoded Input: {tokenizer.decode(sample['input_ids'])}")
    print(f"  Decoded Label: {tokenizer.decode(sample['labels'])}")
    
    # 3. Setup Trainer
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
        fp16=False,
        report_to="none"
    )
    
    # Custom callback for ranking
    ranking_callback = RankingEvalCallback(val_data, tokenizer, model, device)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ranking_callback]
    )
    
    # 4. Train
    print("Starting training...")
    
    # Check for checkpoints to resume
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"Found checkpoint: {last_checkpoint}. Resuming training...")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 5. Save
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
