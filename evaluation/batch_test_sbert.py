
import os
import sys
import random
import time
import numpy as np
import torch
import csv
import gc
from sentence_transformers import SentenceTransformer, util
from viterbi_lm_decode import (
    initialize_lm, initialize_correction_lm, 
    text_to_binary, convolutional_encode, modulate_bpsk, add_noise,
    viterbi_decode_standard, viterbi_decode_with_lm, correct_text_with_lm,
    sanitize_text, SNR_DB, LM_CONTEXT_METRIC_SCALE, LM_MODEL_NAME_OR_PATH, LM_MODEL_FALLBACK,
    LM_CONTEXT_PROB_CACHE  # Import cache dict to clear it
)
import viterbi_lm_decode

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "results"))
os.makedirs(_RESULTS_DIR, exist_ok=True)

TEST_FILE = os.path.abspath(os.path.join(_SCRIPT_DIR, "../data/clean_sentences_test_set.txt"))
RESULT_FILE = os.path.join(_RESULTS_DIR, "batch_test_results_sbert.csv")
SNR_RANGE = range(0, 6) # 0 to 5 dB
# Number of randomly sampled test sentences; None means use all sentences.
NUM_SAMPLES = 500

# Load the SBERT model.
sbert_model = None

def load_sbert():
    global sbert_model
    if sbert_model is None:
        print("Loading SBERT model (all-MiniLM-L6-v2)...")
        try:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                sbert_model = sbert_model.to('cuda')
        except Exception as e:
            print(f"Error loading SBERT: {e}")
            sys.exit(1)

def calc_sbert(reference, candidate):
    if not candidate:
        return 0.0
    try:
        with torch.no_grad():
            embeddings = sbert_model.encode([reference, candidate], convert_to_tensor=True)
            score = util.cos_sim(embeddings[0], embeddings[1])
            return score.item()
    except Exception:
        return 0.0

def get_processed_indices(result_file, target_snr):
    """
    Check CSV to find which indices have already been processed for a given SNR.
    Returns a set of 1-based indices (as stored in CSV).
    """
    processed = set()
    if not os.path.exists(result_file):
        return processed
        
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                return processed
                
            # Headers: Index, Original_Text, SNR_dB, ...
            # Index is column 0, SNR_dB is column 2
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    idx_val = int(row[0])
                    snr_val = int(row[2])
                    if snr_val == target_snr:
                        processed.add(idx_val)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing results: {e}")
    
    return processed

def run_test_for_snr(sentences, snr_db, writer, processed_indices):
    print(f"\n[Testing SNR = {snr_db} dB]")
    
    # Update global decoder settings for the current SNR.
    viterbi_lm_decode.SNR_DB = snr_db
    _snr_linear = 10 ** (snr_db / 10.0)
    _noise_variance = 1.0 / _snr_linear
    viterbi_lm_decode.LM_CONTEXT_METRIC_SCALE = 1.0 / (2.0 * _noise_variance)
    
    total = len(sentences)
    
    for idx, original_text in enumerate(sentences):
        current_id = idx + 1
        if current_id in processed_indices:
            # print(f"  Skipping {current_id}/{total} (already processed)")
            continue
            
        if not original_text.strip():
            continue
            
        print(f"  Processing {current_id}/{total}: '{original_text[:30]}...'")
        
        # 1. Encode and add channel noise.
        try:
            bits = text_to_binary(original_text)
            codeword = convolutional_encode(bits)
            modulated = modulate_bpsk(codeword)
            received = add_noise(modulated, snr_db, enable_noise=True)
        except Exception as e:
            print(f"    Error in encoding/noise: {e}")
            continue

        try:
            # Ensure no grad globally for the decoding steps
            with torch.no_grad():
                # 2. Standard Viterbi
                start_t = time.time()
                std_paths = viterbi_decode_standard(received, verbose=False)
                std_time = time.time() - start_t
                
                std_text = std_paths[0] if std_paths else ""
                sbert_std = calc_sbert(original_text, std_text)
                
                # 3. Standard + LLM Correction
                start_t = time.time()
                corr_text = correct_text_with_lm(std_text) if std_text else ""
                corr_time = time.time() - start_t
                sbert_corr = calc_sbert(original_text, corr_text)
                
                # Clear cache before heavy LM decoding to free up memory
                LM_CONTEXT_PROB_CACHE.clear()
                
                # 4. LM-Enhanced Viterbi
                start_t = time.time()
                lm_paths = viterbi_decode_with_lm(received, verbose=False)
                lm_time = time.time() - start_t
                
                # Take the top-1 candidate.
                lm_best_text = lm_paths[0]['text'] if lm_paths else ""
                sbert_lm = calc_sbert(original_text, lm_best_text)
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    WARNING: CUDA OOM at index {current_id}. Clearing cache and skipping this sentence.")
                torch.cuda.empty_cache()
                LM_CONTEXT_PROB_CACHE.clear()
                continue
            else:
                raise e

        # Record results.
        writer.writerow([
            current_id,
            original_text,
            snr_db,
            std_text,
            sbert_std,
            std_time,
            corr_text,
            sbert_corr,
            std_time + corr_time,
            lm_best_text,
            sbert_lm,
            lm_time
        ])
        
        # Periodic cleanup
        if idx % 10 == 0:
            sys.stdout.flush()
            # Aggressive cleanup
            LM_CONTEXT_PROB_CACHE.clear()
            torch.cuda.empty_cache()
            gc.collect()

def main():
    # Initialize models.
    initialize_lm()
    initialize_correction_lm()
    load_sbert()
    
    # Load sentences.
    sentences = []
    if os.path.exists(TEST_FILE):
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
    else:
        print(f"Error: Database file '{TEST_FILE}' not found.")
        return

    total_sentences = len(sentences)
    print(f"Loaded {total_sentences} sentences.")

    # Optionally sample a fixed number of sentences.
    if NUM_SAMPLES is not None:
        if NUM_SAMPLES <= 0:
            print("NUM_SAMPLES must be a positive integer or None.")
            return
        sample_count = min(NUM_SAMPLES, total_sentences)
        sentences = random.sample(sentences, sample_count)
        print(f"Sampled {len(sentences)} sentences randomly.")
    
    # Prepare the output CSV file.
    file_exists = os.path.exists(RESULT_FILE)
    headers = [
        "Index", "Original_Text", "SNR_dB", 
        "Std_Text", "Std_SBERT", "Std_Time",
        "Corr_Text", "Corr_SBERT", "Corr_Time_Total",
        "LM_Text", "LM_SBERT", "LM_Time"
    ]
    
    # Open the CSV file for writing in append mode.
    mode = 'a' if file_exists else 'w'
    with open(RESULT_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        # Iterate over SNR values.
        for snr in SNR_RANGE:
            # Check what's already done
            processed_indices = get_processed_indices(RESULT_FILE, snr)
            if len(processed_indices) >= len(sentences):
                print(f"SNR {snr}dB already fully processed. Skipping.")
                continue
                
            run_test_for_snr(sentences, snr, writer, processed_indices)
            
    print(f"\nBatch test completed. Results saved to {RESULT_FILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
