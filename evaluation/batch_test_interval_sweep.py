import sys
import os
import random
import numpy as np
import time
import torch
import string
import datetime

# Ensure we can import the local module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import viterbi_lm_decode as vlm

def normalize_for_eval(s):
    """Normalize text for evaluation by stripping whitespace and trailing punctuation."""
    if not s: return ""
    return s.strip().rstrip(string.punctuation)

def run_interval_sweep():
    # Configuration
    DATABASE_PATH = "../data/clean_sentences_test_set.txt"
    # SNRs to test
    SNR_RANGE = range(0, 6)  # 0 to 5 dB
    # LM_CONTEXT_CHECK_INTERVAL values to test
    INTERVAL_VALUES = [35]
    # Stop condition: Number of LM-Viterbi errors to collect
    TARGET_ERRORS = 100 
    # Safety limit for samples per SNR to avoid infinite loops during test
    MAX_SAMPLES_SAFETY = 20000
    
    print(f"Interval Sweep Configuration:")
    print(f"  Database: {DATABASE_PATH}")
    print(f"  SNR Range: {list(SNR_RANGE)} dB")
    print(f"  LM_CONTEXT_CHECK_INTERVAL values: {INTERVAL_VALUES}")
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
    # Note: correction_lm is not needed since we're only running LLM-Viterbi
    # vlm.initialize_correction_lm()
    
    # Create result file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(_results_dir, exist_ok=True)
    output_filename = os.path.join(_results_dir, f"interval_sweep_results_{timestamp}.txt")

    # Write header to file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Interval Sweep Test Started at {timestamp}\n")
        f.write(f"Configuration: SNR Range={list(SNR_RANGE)}dB, Intervals={INTERVAL_VALUES}, TargetErr={TARGET_ERRORS}\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Interval':<10} | {'SNR':<6} | {'Tested':<10} | {'LM Err':<10} | {'LM WER':<12} | {'Time(s)':<10}\n")
        f.write("-" * 120 + "\n")
    
    print(f"Results will be saved incrementally to: {output_filename}")
    
    # Store all results for final summary
    all_results = {}
    
    # Create a live progress file for real-time monitoring
    progress_filename = os.path.join(_results_dir, f"interval_sweep_progress_{timestamp}.txt")
    print(f"Live progress will be updated to: {progress_filename}")

    # Outer loop: LM_CONTEXT_CHECK_INTERVAL values
    for interval in INTERVAL_VALUES:
        print(f"\n{'#'*80}")
        print(f"# Testing LM_CONTEXT_CHECK_INTERVAL = {interval}")
        print(f"{'#'*80}")
        
        # Set the interval value
        vlm.LM_CONTEXT_CHECK_INTERVAL = interval
        
        all_results[interval] = {}
        
        # Inner loop: SNR values
        for snr in SNR_RANGE:
            print(f"\n{'='*60}")
            print(f"Interval={interval}, SNR={snr} dB")
            print(f"{'='*60}")
            
            # Setup Viterbi Decoder Globals for this SNR
            vlm.SNR_DB = snr
            _snr_linear = 10 ** (vlm.SNR_DB / 10.0)
            _noise_variance = 1.0 / _snr_linear
            vlm.LM_CONTEXT_METRIC_SCALE = 1.0 / (2.0 * _noise_variance)
            
            print(f"  Metric Scale: {vlm.LM_CONTEXT_METRIC_SCALE:.4f}")
            
            stats = {
                'total_tested': 0,
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
                    
                    # LM-Enhanced Viterbi (Only this one!)
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
                    if not lm_is_correct: 
                        stats['lm_errors'] += 1
                    
                    # Log Progress
                    if stats['total_tested'] % 5 == 0:
                        print(f"\r  Progress: Tested {stats['total_tested']}, LM Err {stats['lm_errors']}/{TARGET_ERRORS}", end="")

                    # Save progress every 100 samples
                    if stats['total_tested'] % 100 == 0:
                        elapsed_so_far = time.time() - start_time
                        tested = stats['total_tested']
                        lm_wer = stats['lm_errors'] / tested if tested > 0 else 0
                        
                        # Update progress file
                        try:
                            with open(progress_filename, "w", encoding="utf-8") as pf:
                                pf.write(f"Live Progress - Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                pf.write("=" * 100 + "\n\n")
                                pf.write(f"CURRENT TEST:\n")
                                pf.write(f"  LM_CONTEXT_CHECK_INTERVAL: {interval}\n")
                                pf.write(f"  SNR: {snr} dB\n")
                                pf.write(f"  Samples Tested: {tested}\n")
                                pf.write(f"  LM Errors: {stats['lm_errors']}\n")
                                pf.write(f"  Current WER: {lm_wer:.6f}\n")
                                pf.write(f"  Elapsed Time: {elapsed_so_far:.2f}s\n")
                                pf.write(f"  Target Errors: {TARGET_ERRORS}\n")
                                pf.write(f"  Progress: {stats['lm_errors']}/{TARGET_ERRORS} ({100*stats['lm_errors']/TARGET_ERRORS:.1f}%)\n")
                                pf.write("\n" + "=" * 100 + "\n")
                                pf.write("COMPLETED RESULTS:\n")
                                pf.write("-" * 100 + "\n")
                                pf.write(f"{'Interval':<10} | {'SNR':<6} | {'Tested':<10} | {'LM Err':<10} | {'LM WER':<12} | {'Time(s)':<10}\n")
                                pf.write("-" * 100 + "\n")
                                
                                # Write all completed results
                                for done_interval in all_results:
                                    for done_snr in all_results[done_interval]:
                                        r = all_results[done_interval][done_snr]
                                        pf.write(f"{done_interval:<10} | {done_snr:<6} | {r['total_tested']:<10} | {r['lm_errors']:<10} | {r['lm_wer']:<12.6f} | {r['elapsed']:<10.2f}\n")
                                
                                pf.flush()
                        except Exception as e:
                            print(f"\n  Warning: Failed to update progress file: {e}")

                    # Clean up memory to prevent OOM
                    if stats['total_tested'] % 10 == 0:
                        torch.cuda.empty_cache()
                        if hasattr(vlm, 'LM_CONTEXT_PROB_CACHE'):
                            vlm.LM_CONTEXT_PROB_CACHE.clear()

                except Exception as e:
                    print(f"\n  Error on sample: {e}")
                    torch.cuda.empty_cache()
                    if hasattr(vlm, 'LM_CONTEXT_PROB_CACHE'):
                        vlm.LM_CONTEXT_PROB_CACHE.clear()
            
            elapsed = time.time() - start_time
            tested = stats['total_tested']
            lm_wer = stats['lm_errors'] / tested if tested > 0 else 0
            
            print(f"\n  Interval={interval}, SNR={snr}dB Complete in {elapsed:.2f}s.")
            print(f"  Total Tested: {tested}")
            print(f"  LM-Viterbi Errors: {stats['lm_errors']} (WER: {lm_wer:.4f})")
            
            all_results[interval][snr] = {
                'total_tested': tested,
                'lm_errors': stats['lm_errors'],
                'lm_wer': lm_wer,
                'elapsed': elapsed
            }
            
            # Incremental Save
            try:
                with open(output_filename, "a", encoding="utf-8") as f:
                    f.write(f"{interval:<10} | {snr:<6} | {tested:<10} | {stats['lm_errors']:<10} | {lm_wer:<12.6f} | {elapsed:<10.2f}\n")
                    f.flush()
            except Exception as e:
                print(f"Warning: Failed to write results to file: {e}")

    # Final Summary Report
    print("\n" + "=" * 120)
    print("FINAL SUMMARY REPORT")
    print("=" * 120)
    
    # Header row with SNR values
    header = f"{'Interval':<10}"
    for snr in SNR_RANGE:
        header += f" | SNR={snr}dB WER"
    print(header)
    print("-" * 120)
    
    # Data rows
    for interval in INTERVAL_VALUES:
        row = f"{interval:<10}"
        for snr in SNR_RANGE:
            wer = all_results[interval][snr]['lm_wer']
            row += f" | {wer:<13.6f}"
        print(row)
    
    # Also write summary to file
    try:
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 120 + "\n")
            f.write("FINAL SUMMARY (WER by Interval and SNR)\n")
            f.write("=" * 120 + "\n")
            
            header = f"{'Interval':<10}"
            for snr in SNR_RANGE:
                header += f" | SNR={snr}dB"
            f.write(header + "\n")
            f.write("-" * 120 + "\n")
            
            for interval in INTERVAL_VALUES:
                row = f"{interval:<10}"
                for snr in SNR_RANGE:
                    wer = all_results[interval][snr]['lm_wer']
                    row += f" | {wer:.6f}"
                f.write(row + "\n")
    except Exception as e:
        print(f"Warning: Failed to write final summary to file: {e}")

    print(f"\nResults saved to: {output_filename}")

if __name__ == "__main__":
    run_interval_sweep()
