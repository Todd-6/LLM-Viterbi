import numpy as np
import sys
import os
import json
import time
from tqdm import tqdm

# ============== Configuration ==============
SNR_DB = 3
SEGMENT_LEN_CHARS = 5
BITS_PER_CHAR = 8
SEGMENT_LEN_BITS = SEGMENT_LEN_CHARS * BITS_PER_CHAR
K_BEST = 10  # Number of candidates to generate
CONSTRAINT_LENGTH = 3
GENERATOR_POLYNOMIALS = [0o7, 0o5]
FEEDBACK_POLYNOMIAL = 0o7
SOURCE_FILE = "../data/train_sentences.txt"  # Path to SNLI training sentences (plain text, one per line)
OUTPUT_FILE = "../data/byt5_training_data_full.jsonl"
LIMIT_SENTENCES = None  # Process all sentences
RANDOM_SEED = 42

# Set seed
np.random.seed(RANDOM_SEED)

# ============== Core Utility Functions (reused/adapted from viterbi_lm_decode.py) ==============

def text_to_binary(text):
    bits = []
    for char in text:
        ascii_val = ord(char)
        for i in range(7, -1, -1):
            bits.append((ascii_val >> i) & 1)
    return np.array(bits, dtype=int)

def binary_to_text(bits):
    text = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = bits[i:i+8]
            ascii_val = 0
            for bit in byte:
                ascii_val = (ascii_val << 1) | int(bit)
            text.append(chr(ascii_val))
    return ''.join(text)

def get_encoder_state(text):
    """
    Compute the encoder end state after encoding the given text.
    This is used to determine the start state for the next segment.
    """
    bits = text_to_binary(text)
    
    # Convert the polynomial to a bit list.
    def poly_to_binary(poly, length=CONSTRAINT_LENGTH):
        binary = []
        for i in range(length-1, -1, -1):
            binary.append((poly >> i) & 1)
        return binary
    
    fb = poly_to_binary(FEEDBACK_POLYNOMIAL)
    state = [0] * (CONSTRAINT_LENGTH - 1)
    
    for bit in bits:
        feedback = bit
        for i in range(len(state)):
            if fb[i+1]:
                feedback ^= state[i]
        state = [feedback] + state[:-1]
    
    # Convert the state list to an integer index (for example, [1, 0] -> 2).
    # state[0] is the most significant bit (latest feedback bit).
    state_idx = sum([bit << (len(state) - 1 - i) for i, bit in enumerate(state)])
    return state_idx

def convolutional_encode_segment(bits, start_state_idx):
    """
    Encode one bit segment starting from the given encoder state.
    """
    # Reconstruct the state list from the integer state index.
    state = [(start_state_idx >> i) & 1 for i in range(CONSTRAINT_LENGTH - 2, -1, -1)]
    
    def poly_to_binary(poly, length=CONSTRAINT_LENGTH):
        binary = []
        for i in range(length-1, -1, -1):
            binary.append((poly >> i) & 1)
        return binary
        
    g1 = poly_to_binary(GENERATOR_POLYNOMIALS[0])
    g2 = poly_to_binary(GENERATOR_POLYNOMIALS[1])
    fb = poly_to_binary(FEEDBACK_POLYNOMIAL)
    
    codeword = []
    
    for bit in bits:
        feedback = bit
        for i in range(len(state)):
            if fb[i+1]:
                feedback ^= state[i]
        
        output1 = feedback * g1[0]
        for i in range(len(state)):
            if g1[i+1]:
                output1 ^= state[i]
        
        output2 = feedback * g2[0]
        for i in range(len(state)):
            if g2[i+1]:
                output2 ^= state[i]
                
        codeword.append(output1)
        codeword.append(output2)
        state = [feedback] + state[:-1]
        
    return np.array(codeword, dtype=int)

def modulate_bpsk(bits):
    return np.array([1.0 - 2.0 * bit for bit in bits])

def add_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

def generate_trellis():
    num_states = 2 ** (CONSTRAINT_LENGTH - 1)
    trellis = {}
    for state in range(num_states):
        trellis[state] = {}
        for input_bit in [0, 1]:
            state_bits = [(state >> i) & 1 for i in range(CONSTRAINT_LENGTH - 2, -1, -1)]
            feedback = input_bit
            for i, s_bit in enumerate(state_bits):
                if (FEEDBACK_POLYNOMIAL >> (CONSTRAINT_LENGTH - 1 - i - 1)) & 1:
                    feedback ^= s_bit
            
            output1 = feedback * ((GENERATOR_POLYNOMIALS[0] >> (CONSTRAINT_LENGTH - 1)) & 1)
            output2 = feedback * ((GENERATOR_POLYNOMIALS[1] >> (CONSTRAINT_LENGTH - 1)) & 1)
            
            for i, s_bit in enumerate(state_bits):
                if (GENERATOR_POLYNOMIALS[0] >> (CONSTRAINT_LENGTH - 2 - i)) & 1:
                    output1 ^= s_bit
                if (GENERATOR_POLYNOMIALS[1] >> (CONSTRAINT_LENGTH - 2 - i)) & 1:
                    output2 ^= s_bit
            
            new_state_bits = [feedback] + state_bits[:-1]
            next_state = sum([bit << (len(new_state_bits) - 1 - i) for i, bit in enumerate(new_state_bits)])
            
            trellis[state][input_bit] = {
                'next_state': next_state,
                'output': [output1, output2]
            }
    return trellis

TRELLIS = generate_trellis()

class PathState:
    def __init__(self, state, decoded_bits, cumulative_metric):
        self.state = state
        self.decoded_bits = decoded_bits
        self.cumulative_metric = cumulative_metric

def viterbi_decode_segment(received_signal, start_state, k_best=10):
    """
    Run Viterbi decoding on one segment and keep the top-K paths.
    """
    # Initial path set: start from the provided state only.
    paths = {start_state: [PathState(state=start_state, decoded_bits=[], cumulative_metric=0.0)]}
    
    for i in range(0, len(received_signal), 2):
        received_pair = received_signal[i:i+2]
        new_paths = {}
        
        for state, state_paths in paths.items():
            for path in state_paths:
                for input_bit in [0, 1]:
                    transition = TRELLIS[state][input_bit]
                    next_state = transition['next_state']
                    expected_output = transition['output']
                    
                    expected_signal = [1.0 - 2.0 * bit for bit in expected_output]
                    branch_metric = sum((received_pair[j] - expected_signal[j]) ** 2 for j in range(2))
                    
                    new_path_obj = PathState(
                        state=next_state,
                        decoded_bits=path.decoded_bits + [input_bit],
                        cumulative_metric=path.cumulative_metric + branch_metric
                    )
                    
                    if next_state not in new_paths:
                        new_paths[next_state] = []
                    new_paths[next_state].append(new_path_obj)
        
        # K-best pruning per state.
        pruned_paths = {}
        for state, spaths in new_paths.items():
            sorted_paths = sorted(spaths, key=lambda p: p.cumulative_metric)
            pruned_paths[state] = sorted_paths[:k_best]
            
        paths = pruned_paths

    # Collect paths from all ending states.
    final_paths = []
    for state_paths in paths.values():
        final_paths.extend(state_paths)
    
    # Rank globally and keep the top-K paths.
    final_paths.sort(key=lambda p: p.cumulative_metric)
    top_paths = final_paths[:k_best]
    
    results = []
    for p in top_paths:
        try:
            text = binary_to_text(np.array(p.decoded_bits))
            results.append(text)
        except:
            pass # ignore conversion errors
            
    return results

def main():
    print(f"Generating training data from {SOURCE_FILE}")
    print(f"SNR: {SNR_DB} dB")
    print(f"Segment Length: {SEGMENT_LEN_CHARS} chars")
    
    data_records = []
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file {SOURCE_FILE} not found.")
        return

    with open(SOURCE_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    process_lines = lines if LIMIT_SENTENCES is None else lines[:LIMIT_SENTENCES]
    
    # Prepare output directory
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output file (overwrite)
    with open(OUTPUT_FILE, 'w') as f:
        pass
        
    for line_idx, text in enumerate(tqdm(process_lines)):
        # Make sure the sentence is long enough.
        if len(text) < SEGMENT_LEN_CHARS * 2:
            continue
            
        for i in range(SEGMENT_LEN_CHARS, len(text), SEGMENT_LEN_CHARS):
            prefix = text[:i]
            target_suffix = text[i : i + SEGMENT_LEN_CHARS]
            
            if len(target_suffix) < SEGMENT_LEN_CHARS:
                break 
                
            start_state = get_encoder_state(prefix)
            
            target_bits = text_to_binary(target_suffix)
            codeword = convolutional_encode_segment(target_bits, start_state)
            mod_signal = modulate_bpsk(codeword)
            recv_signal = add_noise(mod_signal, SNR_DB)
            
            candidates = viterbi_decode_segment(recv_signal, start_state, k_best=K_BEST)
            
            unique_candidates = []
            seen = set()
            for cand in candidates:
                if cand not in seen:
                    seen.add(cand)
                    unique_candidates.append(cand)
            
            record = {
                "prefix": prefix,
                "target": target_suffix,
                "candidates": unique_candidates,
                "full_text": text
            }
            
            # Append to file incrementally
            with open(OUTPUT_FILE, 'a') as f:
                f.write(json.dumps(record) + "\n")
            
    print(f"Generation complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
