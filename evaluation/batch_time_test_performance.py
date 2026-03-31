
import os
import time
import numpy as np
import sys

# Ensure we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../decoder"))
import viterbi_lm_decode

def load_sentences(file_path, limit=None):
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    sentences.append(text)
                if limit and len(sentences) >= limit:
                    break
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return sentences

def run_benchmark(sentences, snr_list, num_repeats=2):
    results = {}  # {snr: {'std_time': [], 'lm_time': []}}
    
    # Initialize LM once
    print("Initializing Language Model...")
    viterbi_lm_decode.initialize_lm()
    
    # Disable extensive logging during benchmark to avoid I/O overhead affecting timing
    viterbi_lm_decode.LM_DEBUG_DUMP_ENABLED = False
    viterbi_lm_decode.STEP_DEBUG_LOG_FILE = None
    viterbi_lm_decode.STANDARD_STEP_DEBUG_LOG_FILE = None
    
    total_start_time = time.time()
    
    for snr in snr_list:
        print(f"\nTesting SNR: {snr} dB")
        results[snr] = {'std_time': [], 'lm_time': []}
        
        # Update SNR-dependent globals in viterbi_lm_decode
        viterbi_lm_decode.SNR_DB = snr
        snr_linear = 10 ** (snr / 10.0)
        noise_variance = 1.0 / snr_linear
        viterbi_lm_decode.LM_CONTEXT_METRIC_SCALE = 1.0 / (2.0 * noise_variance)
        # viterbi_lm_decode.ENABLE_NOISE = True # Ensure noise is on
        
        for idx, text in enumerate(sentences):
            if not text:
                continue
                
            print(f"  Sentence {idx+1}/{len(sentences)}: '{text[:30]}...'")
            
            # Prepare signal
            try:
                bits = viterbi_lm_decode.text_to_binary(text)
                codeword = viterbi_lm_decode.convolutional_encode(bits)
                modulated = viterbi_lm_decode.modulate_bpsk(codeword)
            except Exception as e:
                print(f"    Error preparing sentence: {e}")
                continue
            
            std_times = []
            lm_times = []
            
            for r in range(num_repeats):
                # Add noise
                received = viterbi_lm_decode.add_noise(modulated, snr, enable_noise=True)
                
                # Standard Viterbi + LLM Correction
                start = time.perf_counter()
                std_paths = viterbi_lm_decode.viterbi_decode_standard(received, verbose=False)
                if std_paths:
                    viterbi_lm_decode.correct_text_with_lm(std_paths[0])
                std_dur = time.perf_counter() - start
                std_times.append(std_dur)
                
                # LM-Enhanced Viterbi
                start = time.perf_counter()
                viterbi_lm_decode.viterbi_decode_with_lm(received, verbose=False)
                lm_dur = time.perf_counter() - start
                lm_times.append(lm_dur)
            
            avg_std = np.mean(std_times)
            avg_lm = np.mean(lm_times)
            
            results[snr]['std_time'].append(avg_std)
            results[snr]['lm_time'].append(avg_lm)
            
            print(f"    Avg Time - Std+Corr: {avg_std:.4f}s, LM: {avg_lm:.4f}s")

    print(f"\nBenchmark completed in {time.time() - total_start_time:.2f}s")
    return results

def print_summary(results):
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (Average Time per Sentence)")
    print("="*60)
    print(f"{'SNR (dB)':<10} | {'Std+Corr (s)':<15} | {'LM Time (s)':<15} | {'Increase':<10}")
    print("-" * 60)
    
    for snr in sorted(results.keys()):
        avg_std = np.mean(results[snr]['std_time'])
        avg_lm = np.mean(results[snr]['lm_time'])
        increase = ((avg_lm - avg_std) / avg_std) * 100 if avg_std > 0 else 0
        
        print(f"{snr:<10} | {avg_std:<15.4f} | {avg_lm:<15.4f} | {increase:+.1f}%")
    print("="*60)

if __name__ == "__main__":
    # Parameters
    SENTENCE_FILE = "../data/clean_sentences_test_set.txt"
    SNR_VALUES = [0, 3, 5, 7, 10]
    NUM_SENTENCES = 10  # Limit to 10 sentences for quick testing
    REPEATS = 2
    
    print(f"Loading top {NUM_SENTENCES} sentences from {SENTENCE_FILE}")
    sentences = load_sentences(SENTENCE_FILE, limit=NUM_SENTENCES)
    
    if not sentences:
        print("No sentences loaded.")
        sys.exit(1)
        
    results = run_benchmark(sentences, SNR_VALUES, num_repeats=REPEATS)
    print_summary(results)

