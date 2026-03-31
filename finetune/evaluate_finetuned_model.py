import sys
import os
import time
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DECODER_DIR = os.path.join(REPO_ROOT, "decoder")
FINETUNED_MODEL_DIR = os.path.join(REPO_ROOT, "models", "byt5_finetuned")

# Ensure the decoder module can be imported regardless of the current working directory.
sys.path.insert(0, DECODER_DIR)

try:
    import viterbi_lm_decode as viterbi
except ImportError:
    print("Error: Could not import viterbi_lm_decode.py from the decoder directory.")
    sys.exit(1)

def run_evaluation(model_path, test_sentences, snr_db=3):
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {model_path}")
    print(f"{'='*60}")
    
    # Reload model
    # Note: viterbi_lm_decode uses a global variable for the model, so we re-init
    try:
        viterbi.initialize_lm(model_path)
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    # Configuration
    # We want to force parameters to match our training/testing conditions
    viterbi.K_BEST_PATHS_PER_STATE = 4
    viterbi.LM_CONTEXT_CHECK_INTERVAL = 5 # As per our data generation
    # Weights might need tuning for the new model, but let's stick to defaults first
    viterbi.LM_CONTEXT_METRIC_WEIGHT = 2.0 
    viterbi.LM_CONTEXT_LOGPROB_WEIGHT = 1.0
    
    correct_count = 0
    total_count = 0
    start_time = time.time()
    
    for text in test_sentences:
        if not text.strip():
            continue
            
        print(f"\nTest Sentence: '{text}'")
        
        # Pipeline
        bits = viterbi.text_to_binary(text)
        codeword = viterbi.convolutional_encode(bits)
        mod_signal = viterbi.modulate_bpsk(codeword)
        recv_signal = viterbi.add_noise(mod_signal, snr_db, enable_noise=True)
        
        # Decode
        decoded_paths = viterbi.viterbi_decode_with_lm(recv_signal, verbose=False)
        
        # Check correctness
        is_correct = False
        if decoded_paths and text in decoded_paths:
            is_correct = True
            
        # Top 1 check (if list is sorted by likelihood, usually last step pruning leaves sorted)
        # viterbi_decode_with_lm returns a list of strings.
        # We can rank them if multiple. 
        # For now, just check if correct answer is in the set of survivors.
        
        print(f"  -> Decoded {len(decoded_paths)} candidates.")
        if is_correct:
            print("  -> CORRECT match found.")
            correct_count += 1
        else:
            print("  -> INCORRECT. Best candidate:")
            if decoded_paths:
                print(f"     '{decoded_paths[0]}'")
            else:
                print("     (No paths)")
        
        total_count += 1
        
    duration = time.time() - start_time
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nResult: Accuracy = {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"Time: {duration:.2f}s")

def main():
    # Test Sentences (Subset)
    # We use a few from the target file that were NOT in the training set (ideally)
    # But for a quick check, we pick a few random ones.
    
    test_sentences = [
        "A team of professionals developing research studies using modern software",
        "The magnificent building stands in the classroom surrounded by trees and flowers",
        "Students exploring physics using hands-on practice in the park",
        "Every weekend, children practicing in the laboratory to create solutions"
    ]
    
    # 1. Baseline (Original ByT5-small)
    print("\n\n>>> BASELINE EVALUATION (google/byt5-small) <<<")
    run_evaluation("google/byt5-small", test_sentences)
    
    # 2. Finetuned (Check if exists)
    finetuned_path = FINETUNED_MODEL_DIR
    if os.path.exists(finetuned_path):
        print("\n\n>>> FINETUNED EVALUATION <<<")
        run_evaluation(finetuned_path, test_sentences)
    else:
        print(f"\nFinetuned model not found at {finetuned_path}. Skipping.")

if __name__ == "__main__":
    main()
