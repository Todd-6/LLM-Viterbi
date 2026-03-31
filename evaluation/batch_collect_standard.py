import sys
import os
import random
import numpy as np
import time

# Ensure we can import the local module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import viterbi_lm_decode as viterbi

# Disable debug logging to speed up processing and save disk space
viterbi.STANDARD_STEP_DEBUG_LOG_FILE = None
viterbi.STEP_DEBUG_LOG_FILE = None
viterbi.LM_DEBUG_DUMP_ENABLED = False

def load_sentences(file_path):
    print(f"Reading sentences from: {file_path}")
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(lines)} sentences.")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    return lines

def run_batch_collection(samples_per_snr=None, output_file=None):
    # Default output to results/ directory
    if output_file is None:
        _results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
        os.makedirs(_results_dir, exist_ok=True)
        output_file = os.path.join(_results_dir, "batch_standard_viterbi_results.txt")

    dataset_path = os.path.abspath(os.path.join(current_dir, "../data/clean_sentences_test_set.txt"))
    dataset_path = os.path.abspath(dataset_path)
    
    sentences = load_sentences(dataset_path)
    if not sentences:
        print("No sentences found.")
        return

    snr_range = range(4, 7) # 4 to 6 dB (inclusive)
    
    if samples_per_snr is None:
        print(f"Starting batch collection: ALL {len(sentences)} samples per SNR")
    else:
        print(f"Starting batch collection: {samples_per_snr} samples per SNR")
        
    print(f"Output file: {output_file}")

    total_records = 0
    
    # Using 'a' (append) mode if we wanted to resume, but 'w' to start fresh as requested "form a file"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header or metadata if needed? User asked for specific line format.
        # "Give what is correct and what the corresponding test result is in one line"
        f.write("Format: SNR | Correct Text | Decoded Text\n")

        for snr in snr_range:
            print(f"\nProcessing SNR = {snr} dB...")
            
            # Select batch
            if samples_per_snr is None:
                current_batch = sentences
            elif len(sentences) >= samples_per_snr:
                current_batch = random.sample(sentences, samples_per_snr)
            else:
                current_batch = random.choices(sentences, k=samples_per_snr)
            
            count = 0
            start_time = time.time()
            
            for text in current_batch:
                try:
                    # 1. Text to Binary
                    bits = viterbi.text_to_binary(text)
                    
                    # 2. Convolutional Encoding
                    codeword = viterbi.convolutional_encode(bits)
                    
                    # 3. Modulation
                    modulated = viterbi.modulate_bpsk(codeword)
                    
                    # 4. Add Noise
                    received = viterbi.add_noise(modulated, snr, enable_noise=True)
                    
                    # 5. Standard Viterbi Decoding
                    # verbose=False to keep console clean
                    decoded_candidates = viterbi.viterbi_decode_standard(received, verbose=False)
                    
                    decoded_text = ""
                    if decoded_candidates:
                        decoded_text = decoded_candidates[0]
                    
                    # Format: SNR: <val> | Correct: <text> | Decoded: <text>
                    # Ensuring "in one line"
                    line = f"SNR: {snr} | Correct: {text} | Decoded: {decoded_text}\n"
                    f.write(line)
                    f.flush() # Ensure data is written
                    
                    count += 1
                    total_records += 1
                    
                    if count % 100 == 0:
                        print(f"  Progress: {count}/{len(current_batch)}", end='\r')
                        
                except Exception as e:
                    print(f"\nError processing text: {text[:30]}... Error: {e}")
                    continue
            
            elapsed = time.time() - start_time
            print(f"  Completed SNR {snr}dB. Processed {count} items in {elapsed:.2f}s")

    print(f"\nBatch processing complete. Total records: {total_records}")
    print(f"Results saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Check for command line args for sample count (useful for testing)
    samples = None # Default to None = All
    if len(sys.argv) > 1:
        try:
            val = int(sys.argv[1])
            if val > 0:
                samples = val
        except ValueError:
            pass
            
    run_batch_collection(samples_per_snr=samples)
