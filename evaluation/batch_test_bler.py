
import sys
import os
import random
import numpy as np
import time
import torch
import string

# Ensure we can import the local module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../decoder")))

import viterbi_lm_decode as vlm

def normalize_for_eval(s):
    """Normalize text for evaluation by stripping whitespace and trailing punctuation."""
    if not s: return ""
    return s.strip().rstrip(string.punctuation)

def run_snr_sweep():
    # Configuration
    DATABASE_PATH = os.path.abspath(os.path.join(current_dir, "../data/clean_sentences_test_set.txt"))
    # SNRs to test
    SNR_RANGE = range(0,5) # 0 to 6 dB
    # Stop condition: Number of LM-Viterbi errors to collect
    TARGET_ERRORS = 3
    # Safety limit for samples per SNR to avoid infinite loops during test
    MAX_SAMPLES_SAFETY = 200000

    print(f"SNR Sweep Configuration:")
    print(f"  Database: {DATABASE_PATH}")
    print(f"  SNR Range: {list(SNR_RANGE)} dB")
    print(f"  Stop Condition: {TARGET_ERRORS} LM-Viterbi errors per SNR")
    print(f"  Safety Sample Limit: {MAX_SAMPLES_SAFETY}")
    print(f"  Note: Evaluation ignores trailing punctuation.")

    # 1. Load Data
    try:
        with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(lines)} sentences from database.")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    # Initialize Models (Once)
    print("Initializing Models...")
    vlm.initialize_lm()
    vlm.initialize_correction_lm()

    overall_results = {}

    # Create result file with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(_results_dir, exist_ok=True)
    output_filename = os.path.join(_results_dir, f"batch_test_results_{timestamp}.txt")

    # Write header to file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Batch Test Started at {timestamp}\n")
        f.write(f"Configuration: Range={list(SNR_RANGE)}dB, TargetErr={TARGET_ERRORS}\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'SNR':<6} | {'Tested':<10} | {'Std Err':<10} | {'Corr Err':<10} | {'LM Err':<10} | {'Std WER':<10} | {'LM WER':<10}\n")
        f.write("-" * 100 + "\n")

    print(f"Results will be saved incrementally to: {output_filename}")

    for snr in SNR_RANGE:
        print(f"\n{'='*80}")
        print(f"Testing SNR: {snr} dB")
        print(f"{'='*80}")

        # Setup Viterbi Decoder Globals for this SNR
        vlm.SNR_DB = snr
        _snr_linear = 10 ** (vlm.SNR_DB / 10.0)
        _noise_variance = 1.0 / _snr_linear
        vlm.LM_CONTEXT_METRIC_SCALE = 1.0 / (2.0 * _noise_variance)

        print(f"  Metric Scale: {vlm.LM_CONTEXT_METRIC_SCALE:.4f}")

        stats = {
            'total_tested': 0,
            'std_errors': 0,
            'corr_errors': 0,
            'lm_errors': 0
        }

        start_time = time.time()

        while stats['lm_errors'] < TARGET_ERRORS:
            if stats['total_tested'] >= MAX_SAMPLES_SAFETY:
                print(f"\n  [WARNING] Reached safety limit of {MAX_SAMPLES_SAFETY} samples without hitting {TARGET_ERRORS} errors.")
                break

            # Random selection with replacement
            text = random.choice(lines)

            # Pipeline
            try:
                # Encoding & Modulation
                bits = vlm.text_to_binary(text)
                codeword = vlm.convolutional_encode(bits)
                modulated = vlm.modulate_bpsk(codeword)

                # Channel (Noise)
                received = vlm.add_noise(modulated, vlm.SNR_DB, enable_noise=True)

                # A. Standard Viterbi
                std_paths = vlm.viterbi_decode_standard(received, verbose=False)
                std_text = std_paths[0] if std_paths else ""
                std_is_correct = (normalize_for_eval(std_text) == normalize_for_eval(text))

                # B. Standard + Correction
                corr_text = vlm.correct_text_with_lm(std_text)
                corr_is_correct = (normalize_for_eval(corr_text) == normalize_for_eval(text))

                # C. LM-Enhanced Viterbi
                lm_results = vlm.viterbi_decode_with_lm(received, verbose=False)

                lm_best_text = ""
                lm_is_correct = False
                if lm_results:
                    lm_best_text = lm_results[0]['text']
                    # Check if any surviving path is correct (Found Correct logic)
                    for res in lm_results:
                        if normalize_for_eval(res['text']) == normalize_for_eval(text):
                            lm_is_correct = True
                            break

                # Update Stats
                stats['total_tested'] += 1
                if not std_is_correct: stats['std_errors'] += 1
                if not corr_is_correct: stats['corr_errors'] += 1
                if not lm_is_correct: stats['lm_errors'] += 1

                # Log Progress
                if stats['total_tested'] % 5 == 0:
                     print(f"\r  Progress: Tested {stats['total_tested']}, LM Err {stats['lm_errors']}/{TARGET_ERRORS}, Std Err {stats['std_errors']}, Corr Err {stats['corr_errors']}", end="")

                # Clean up memory to prevent OOM
                if stats['total_tested'] % 10 == 0:
                    torch.cuda.empty_cache()
                    if hasattr(vlm, 'LM_CONTEXT_PROB_CACHE'):
                        vlm.LM_CONTEXT_PROB_CACHE.clear()

                # Conditional Detail Log: Standard CORRECT but (Corr WRONG or LM WRONG)
                if std_is_correct and (not corr_is_correct or not lm_is_correct):
                    print(f"\n  [Regression Case] Sample {stats['total_tested']}")
                    print(f"    Original:   '{text}'")
                    print(f"    Standard:   '{vlm.sanitize_text(std_text)}' (CORRECT)")
                    if not corr_is_correct:
                        print(f"    Std+Corr:   '{vlm.sanitize_text(corr_text)}' (WRONG)")
                    if not lm_is_correct:
                        print(f"    LM-Viterbi: (WRONG - Correct path not found)")
                        # Print top 1 for reference
                        print(f"       Top1:    '{vlm.sanitize_text(lm_best_text)}'")

            except Exception as e:
                print(f"\n  Error on sample: {e}")
                torch.cuda.empty_cache()
                if hasattr(vlm, 'LM_CONTEXT_PROB_CACHE'):
                    vlm.LM_CONTEXT_PROB_CACHE.clear()

        elapsed = time.time() - start_time
        print(f"\n  SNR {snr}dB Complete in {elapsed:.2f}s.")
        print(f"  Total Tested: {stats['total_tested']}")
        print(f"  Standard Errors: {stats['std_errors']} (WER: {stats['std_errors']/stats['total_tested']:.4f})")
        print(f"  Std+Corr Errors: {stats['corr_errors']} (WER: {stats['corr_errors']/stats['total_tested']:.4f})")
        print(f"  LM-Viterbi Errors: {stats['lm_errors']} (WER: {stats['lm_errors']/stats['total_tested']:.4f})")

        overall_results[snr] = stats

        # Incremental Save
        tested = stats['total_tested']
        std_wer = stats['std_errors'] / tested if tested > 0 else 0
        lm_wer = stats['lm_errors'] / tested if tested > 0 else 0

        try:
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(f"{snr:<6} | {tested:<10} | {stats['std_errors']:<10} | {stats['corr_errors']:<10} | {stats['lm_errors']:<10} | {std_wer:.4f}     | {lm_wer:.4f}\n")
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write results to file: {e}")

    print("\n" + "=" * 100)
    print("FINAL SUMMARY REPORT")
    print("=" * 100)
    print(f"{'SNR':<6} | {'Tested':<10} | {'Std Err':<10} | {'Corr Err':<10} | {'LM Err':<10} | {'Std WER':<10} | {'LM WER':<10}")
    print("-" * 100)
    for snr in SNR_RANGE:
        s = overall_results[snr]
        tested = s['total_tested']
        std_wer = s['std_errors'] / tested if tested > 0 else 0
        lm_wer = s['lm_errors'] / tested if tested > 0 else 0
        print(f"{snr:<6} | {tested:<10} | {s['std_errors']:<10} | {s['corr_errors']:<10} | {s['lm_errors']:<10} | {std_wer:.4f}     | {lm_wer:.4f}")

if __name__ == "__main__":
    run_snr_sweep()
