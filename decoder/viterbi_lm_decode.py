"""
Viterbi decoder with language model integration.
Soft-information Viterbi decoding with LM-assisted pruning.
"""

import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from collections import OrderedDict
import time
import difflib
import heapq

# ============== Path Configuration ==============
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEBUG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "debug"))
_MODELS_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "models"))
os.makedirs(_DEBUG_DIR, exist_ok=True)

# ============== Tunable Parameters ==============
# Convolutional code parameters (constraint length K=3, rate 1/2, recursive systematic convolutional code)
# CONSTRAINT_LENGTH = 8
# GENERATOR_POLYNOMIALS = [0o371, 0o247]  # Generator polynomials in octal (K=8, standard setting)
# FEEDBACK_POLYNOMIAL = 0o371  # Feedback polynomial (same as first generator for systematic output)

# CONSTRAINT_LENGTH = 5
# GENERATOR_POLYNOMIALS = [0o35, 0o23]  # Generator polynomials in octal (K=5, rate 1/2)
# FEEDBACK_POLYNOMIAL = 0o35  # Feedback polynomial (same as first generator for systematic output)

CONSTRAINT_LENGTH = 3
GENERATOR_POLYNOMIALS = [0o7, 0o5]  # Generator polynomials in octal
FEEDBACK_POLYNOMIAL = 0o7  # Feedback polynomial

# Modulation and noise parameters
SNR_RANGE = np.arange(0, 11)  # SNR from 0 to 10 dB
ENABLE_NOISE = True  # Noise switch
SNR_DB = 5  # Test SNR (lower noise to show LM advantage)

# Viterbi path retention parameters
K_BEST_PATHS_PER_STATE = 8  # Number of best paths kept per state (for example 1/2/4/8)
LM_CONTEXT_CHECK_INTERVAL = 5  # Run LM context pruning every N generated characters

# Automatically computed weighting term (MAP decoding: combined = log_prob - metric / (2 * sigma^2))
# Compute noise variance sigma^2 from SNR
_snr_linear = 10 ** (SNR_DB / 10.0)
_noise_variance = 1.0 / _snr_linear  # Assume unit signal power
LM_CONTEXT_METRIC_SCALE = 1.0 / (2.0 * _noise_variance)

print(f"DEBUG: SNR={SNR_DB}dB, Metric Scale Factor={LM_CONTEXT_METRIC_SCALE:.4f}")

# Language model configuration
HF_MODEL_REPO_ID = os.getenv("VITERBI_HF_REPO", "todd8642/LLMViterbi_ByT5_finetuned")
HF_MODEL_REVISION = os.getenv("VITERBI_HF_REVISION")

LOCAL_LM_MODEL_PATH = os.path.join(_MODELS_DIR, "byt5_finetuned")
LOCAL_CORRECTION_MODEL_PATH = os.path.join(_MODELS_DIR, "ByT5_correction_finetuned")

LM_MODEL_NAME_OR_PATH = os.getenv("VITERBI_LM_MODEL", HF_MODEL_REPO_ID)
LM_MODEL_SUBFOLDER = os.getenv("VITERBI_LM_SUBFOLDER", "byt5_finetuned")
LM_MODEL_FALLBACK = "google/byt5-small"  # Fallback model if the preferred model cannot be loaded

# Correction model configuration (used for standard Viterbi post-processing)
CORRECTION_MODEL_PATH = os.getenv("VITERBI_CORRECTION_MODEL", HF_MODEL_REPO_ID)
CORRECTION_MODEL_SUBFOLDER = os.getenv("VITERBI_CORRECTION_SUBFOLDER", "ByT5_correction_finetuned")

LM_DEBUG_DUMP_ENABLED = True  # Enable or disable LM debug log writing
LM_DEBUG_OUTPUT_FILE = os.path.join(_DEBUG_DIR, "lm_debug_output.txt")
LM_DEBUG_STOP_SUBSTRING = " player"
STEP_DEBUG_LOG_FILE = os.path.join(_DEBUG_DIR, "step_debug_log_forSingle.txt")
STANDARD_STEP_DEBUG_LOG_FILE = os.path.join(_DEBUG_DIR, "standard_step_debug_log.txt")
LM_MIN_CHAR_PROB = 1e-12  # Prevent log(0)
ENABLE_SEMANTIC_EVALUATION = False  # Whether to enable semantic similarity checks (SBERT)

# Test text
# TEST_TEXT = "A beautiful bride walking on a sidewalk with her new husband."
TEST_TEXT = "A boy is enjoying himself while thinking about sliding down a snow hill with his friends"

# ============== Global LM Objects ==============
lm_model = None
lm_tokenizer = None
correction_lm_model = None
correction_lm_tokenizer = None
DEBUG_LM_OUTPUT = False  # Toggle LM debug console output
LM_CONTEXT_CACHE_MAX_SIZE = 2048  # Increased size to prevent cache thrashing
LM_CONTEXT_PROB_CACHE = OrderedDict()
LM_CACHE_EMPTY_KEY = "<|LM_EMPTY_CONTEXT|>"
LM_TOKEN_ID_TO_TEXT_CACHE = {}  # Cache for token ID to text mapping

# Performance Logging
LM_CALL_LOGS = []
LM_CALL_LOG_FILE = os.path.join(_DEBUG_DIR, "lm_detailed_calls.csv")
LM_USE_KV_CACHE = False  # Flag to enable/disable KV cache optimization

class LegacyCacheWrapper:
    """
    Simple wrapper to make tuple-based past_key_values compatible with 
    newer Transformers (>=4.36) that expect Cache objects.
    CRITICAL: This wrapper must be MUTABLE to support in-place updates by T5Stack.
    """
    def __init__(self, legacy_cache):
        # Convert tuple-of-tuples to list-of-lists for mutability
        # legacy_cache structure: ((sk, sv, ck, cv), ...) for each layer
        self.legacy_cache = []
        if legacy_cache is not None:
            for layer_tuple in legacy_cache:
                self.legacy_cache.append(list(layer_tuple))
        self.is_compileable = False 

    def __getitem__(self, item):
        return tuple(self.legacy_cache[item])

    def __iter__(self):
        for layer_list in self.legacy_cache:
            yield tuple(layer_list)
        
    def __len__(self):
        return len(self.legacy_cache)

    def get_seq_length(self, layer_idx=0):
        if len(self.legacy_cache) > 0:
            if len(self.legacy_cache[0]) > 0:
                # Self Key is at index 0
                return self.legacy_cache[0][0].shape[2]
        return 0
    
    def get_max_length(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # T5 model calls this to update the cache during forward pass.
        # We must append the new states to our legacy storage AND update internal state.
        
        seq_len = key_states.shape[2]
        
        # HEURISTIC to distinguish Self vs Cross Attention
        # We ensure Encoder Input has length 2. Decoder Input usually has 1 or 5.
        
        # Cross Attention (Encoder KV) - Dummy Encoder Length 2
        if seq_len == 2: 
            # Return cached Cross KV (indices 2 and 3)
            # No update needed for static encoder cache
            if len(self.legacy_cache) > layer_idx:
                return self.legacy_cache[layer_idx][2], self.legacy_cache[layer_idx][3]
            else:
                return key_states, value_states
        
        # Self Attention (Decoder KV)
        if len(self.legacy_cache) > layer_idx:
            past_key = self.legacy_cache[layer_idx][0]
            past_value = self.legacy_cache[layer_idx][1]
            
            if past_key.size(0) == key_states.size(0):
                new_key = torch.cat([past_key, key_states], dim=2)
                new_value = torch.cat([past_value, value_states], dim=2)
                
                # UPDATE INTERNAL STATE!
                self.legacy_cache[layer_idx][0] = new_key
                self.legacy_cache[layer_idx][1] = new_value
                
                return new_key, new_value
        
        return key_states, value_states

    def to_legacy_cache(self):
        return tuple(tuple(l) for l in self.legacy_cache)


def sanitize_text(text, max_len=40):
    """Safely truncate text for logging."""
    if not text:
        return ""
    try:
        ascii_text = text.encode('ascii', 'replace').decode('ascii')
    except Exception:
        ascii_text = "[non-ascii]"
    if len(ascii_text) > max_len:
        return ascii_text[:max_len-3] + "..."
    return ascii_text


def bits_tail(bits, limit=24):
    """Return a compact string for the tail of a bit sequence."""
    if not bits:
        return ""
    total = len(bits)
    if total > limit:
        return '...' + ''.join(str(b) for b in bits[-limit:])
    return ''.join(str(b) for b in bits)


def append_step_debug_log(step_event, file_path):
    """Append per-step debug information to a file."""
    if not file_path or not step_event:
        return
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            step = step_event.get('step')
            f.write(f"\n[Step {step}] Debug Summary\n")
            def write_group(title, records):
                f.write(f"{title} ({len(records)}):\n")
                if not records:
                    return
                for rec in records:
                    metric_val = rec.get('metric')
                    metric_display = f"{metric_val:.2f}" if isinstance(metric_val, (int, float)) else "N/A"
                    prob_val = rec.get('prob')
                    prob_display = f"{prob_val:.3e}" if isinstance(prob_val, (int, float)) else "N/A"
                    weight_val = rec.get('lm_weight')
                    weight_display = f"{weight_val:.2f}" if isinstance(weight_val, (int, float)) else "N/A"
                    text_repr = rec.get('text') or ""
                    chars_tail = rec.get('chars_tail') or ""
                    last_char = rec.get('last_char') or ""
                    bits_repr = rec.get('bits_tail') or ""
                    reason = rec.get('reason', '-')
                    f.write(
                        f"  - path_id={rec.get('path_id')} metric={metric_display} "
                        f"prob={prob_display} weight={weight_display} text='{text_repr}' "
                        f"last='{last_char}' chars_tail='{chars_tail}' bits={bits_repr} reason={reason}\n"
                    )
            write_group("K-Best Kept", step_event.get('kbest_kept', []))
            write_group("K-Best Pruned", step_event.get('kbest_pruned', []))
            contexts = step_event.get('lm_contexts') or []
            if contexts:
                f.write("LM Context Scores:\n")
                for ctx in contexts:
                    ctx_text = sanitize_text(ctx.get('context', ''), max_len=400)
                    log_prob = ctx.get('log_prob')
                    avg_prob = ctx.get('avg_char_prob')
                    metric_val = ctx.get('best_metric')
                    combined_val = ctx.get('combined_score')
                    log_display = f"{log_prob:.3f}" if isinstance(log_prob, (int, float)) else "N/A"
                    avg_display = f"{avg_prob:.3e}" if isinstance(avg_prob, (int, float)) else "N/A"
                    metric_display = f"{metric_val:.2f}" if isinstance(metric_val, (int, float)) else "N/A"
                    combined_display = f"{combined_val:.3f}" if isinstance(combined_val, (int, float)) else "N/A"
                    best_marker = " *BEST*" if ctx.get('is_best') else ""
                    
                    suffix_text = sanitize_text(ctx.get('suffix', ''))
                    suffix_lp = ctx.get('suffix_log_prob')
                    suffix_lp_display = f"{suffix_lp:.3f}" if isinstance(suffix_lp, (int, float)) else "N/A"
                    suffix_details = ctx.get('suffix_details', '')
                    
                    f.write(
                        f"  - context='{ctx_text}' log_prob={log_display} "
                        f"suffix='{suffix_text}' suffix_lp={suffix_lp_display} details={suffix_details} "
                        f"avg_char_prob={avg_display} metric={metric_display} "
                        f"combined={combined_display}{best_marker}\n"
                    )
    except Exception as e:
        print(f"Warning: failed to write step debug log: {e}")


def refresh_adjusted_metric(path):
    """Refresh the combined metric stored on a path."""
    path.adjusted_metric = path.cumulative_metric + path.lm_penalty


def _load_seq2seq_model(load_target, subfolder=None, revision=None):
    """Load a tokenizer/model pair from either a local path or a Hugging Face repo."""
    load_kwargs = {}
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    if revision:
        load_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(load_target, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(load_target, **load_kwargs)
    return tokenizer, model


def _build_model_candidates(primary_target, primary_subfolder=None, include_fallback=False):
    """Build a prioritized list of model-loading candidates."""
    candidates = []
    seen = set()

    def add_candidate(target, subfolder=None, revision=None):
        if not target:
            return
        key = (target, subfolder or "", revision or "")
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "target": target,
            "subfolder": subfolder,
            "revision": revision,
        })

    add_candidate(primary_target, primary_subfolder, HF_MODEL_REVISION)

    if primary_subfolder:
        local_equivalent = os.path.join(_MODELS_DIR, primary_subfolder)
        if os.path.isdir(local_equivalent):
            add_candidate(local_equivalent, None, None)

    if include_fallback:
        add_candidate(LM_MODEL_FALLBACK, None, None)

    return candidates


def _default_subfolder_for_target(target, subfolder):
    """Only apply the default subfolder when loading from the shared HF repo."""
    if target == HF_MODEL_REPO_ID:
        return subfolder
    return None


def initialize_lm(model_name_or_path=None):
    """Initialize ByT5 language model (or compatible seq2seq LM)"""
    global lm_model, lm_tokenizer, LM_CONTEXT_PROB_CACHE
    
    requested_path = model_name_or_path or LM_MODEL_NAME_OR_PATH
    successful_path = None
    
    candidate_specs = _build_model_candidates(
        primary_target=requested_path,
        primary_subfolder=_default_subfolder_for_target(requested_path, LM_MODEL_SUBFOLDER),
        include_fallback=True,
    )

    for spec in candidate_specs:
        candidate = spec["target"]
        subfolder = spec["subfolder"]
        revision = spec["revision"]

        display_name = candidate
        if subfolder:
            display_name = f"{candidate} (subfolder='{subfolder}')"
        if revision:
            display_name = f"{display_name}, revision='{revision}'"

        print(f"Loading language model from '{display_name}'...")
        try:
            tokenizer, model = _load_seq2seq_model(candidate, subfolder=subfolder, revision=revision)
            pad_token_added = False
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})
                    pad_token_added = True
            
            if pad_token_added:
                model.resize_token_embeddings(len(tokenizer))
            
            # Move to GPU if available
            if torch.cuda.is_available():
                print("Moving model to GPU (CUDA)...")
                model = model.to("cuda")
            
            model.eval()
            
            lm_tokenizer = tokenizer
            lm_model = model
            LM_CONTEXT_PROB_CACHE.clear()
            
            # Pre-compute token ID to text cache
            print("Pre-computing token ID to text cache...")
            LM_TOKEN_ID_TO_TEXT_CACHE.clear()
            # Iterate through a reasonable range for ByT5 (256 bytes + specials)
            # Safe upper bound for ByT5 is small
            vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
            # Limit to a reasonable number if vocab is huge, but ByT5 is small (~384)
            limit = min(vocab_size, 5000) 
            for tid in range(limit):
                try:
                    text = tokenizer.decode([tid])
                    LM_TOKEN_ID_TO_TEXT_CACHE[tid] = text
                except Exception:
                    pass
            print(f"Token cache built with {len(LM_TOKEN_ID_TO_TEXT_CACHE)} entries.")
            
            print(f"Language model loaded successfully from '{display_name}'")
            successful_path = display_name
            break
        except Exception as e:
            print(f"Warning: language model loading failed for '{display_name}': {e}")
    
    if successful_path is None:
        print("Will use placeholder probability values")
        lm_tokenizer = None
        lm_model = None
        LM_CONTEXT_PROB_CACHE.clear()
    
    return successful_path


def initialize_correction_lm():
    """Initialize the dedicated ByT5 model used for correction."""
    global correction_lm_model, correction_lm_tokenizer
    
    candidate_specs = _build_model_candidates(
        primary_target=CORRECTION_MODEL_PATH,
        primary_subfolder=_default_subfolder_for_target(CORRECTION_MODEL_PATH, CORRECTION_MODEL_SUBFOLDER),
        include_fallback=False,
    )

    for spec in candidate_specs:
        candidate = spec["target"]
        subfolder = spec["subfolder"]
        revision = spec["revision"]

        display_name = candidate
        if subfolder:
            display_name = f"{candidate} (subfolder='{subfolder}')"
        if revision:
            display_name = f"{display_name}, revision='{revision}'"

        print(f"Loading correction model from '{display_name}'...")

        try:
            tokenizer, model = _load_seq2seq_model(candidate, subfolder=subfolder, revision=revision)

            if torch.cuda.is_available():
                print("Moving correction model to GPU (CUDA)...")
                model = model.to("cuda")

            model.eval()

            correction_lm_tokenizer = tokenizer
            correction_lm_model = model
            print(f"Correction model loaded successfully from '{display_name}'")
            return True
        except Exception as e:
            print(f"Warning: correction model loading failed for '{display_name}': {e}")

    correction_lm_model = None
    correction_lm_tokenizer = None
    return False


def text_to_binary(text):
    """
    Convert English text into a binary bit sequence.
    The current implementation uses ASCII encoding (8 bits per character).
    This may later be replaced with LM-based arithmetic coding.
    
    Args:
        text: Input text string
    Returns:
        numpy array of bits (0s and 1s)
    """
    bits = []
    for char in text:
        # Get the ASCII value of the current character.
        ascii_val = ord(char)
        # Convert it to 8 bits, most significant bit first.
        for i in range(7, -1, -1):
            bit = (ascii_val >> i) & 1
            bits.append(bit)
    
    return np.array(bits, dtype=int)


def binary_to_text(bits):
    """
    Convert a binary bit sequence back to text.
    This is the inverse of text_to_binary().
    
    Args:
        bits: numpy array of bits
    Returns:
        decoded text string
    """
    text = []
    # Convert every 8 bits into one character.
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            # Take 8 bits.
            byte = bits[i:i+8]
            # Convert to an ASCII value.
            ascii_val = 0
            for bit in byte:
                ascii_val = (ascii_val << 1) | int(bit)
            # Convert to a character.
            text.append(chr(ascii_val))
    
    return ''.join(text)


def convolutional_encode(bits):
    """
    Encode bits with a recursive systematic convolutional code.
    Constraint length K=3, rate 1/2.
    Generator polynomials: [0o7, 0o5] = [[1,1,1], [1,0,1]]
    Feedback polynomial: 0o7 = [1,1,1]
    
    Args:
        bits: Input bit sequence
    Returns:
        encoded codeword (numpy array)
    """
    # Convert an octal polynomial to a binary coefficient list.
    def poly_to_binary(poly, length=CONSTRAINT_LENGTH):
        binary = []
        for i in range(length-1, -1, -1):
            binary.append((poly >> i) & 1)
        return binary
    
    # Generator and feedback polynomials.
    g1 = poly_to_binary(GENERATOR_POLYNOMIALS[0])  # [1,1,1]
    g2 = poly_to_binary(GENERATOR_POLYNOMIALS[1])  # [1,0,1]
    fb = poly_to_binary(FEEDBACK_POLYNOMIAL)        # [1,1,1]
    
    # Initialize the encoder state (shift register).
    state = [0] * (CONSTRAINT_LENGTH - 1)  # K-1 = 2 memory cells
    
    codeword = []
    
    for bit in bits:
        # Compute the feedback bit.
        feedback = bit
        for i in range(len(state)):
            if fb[i+1]:  # fb[0] corresponds to the current input bit
                feedback ^= state[i]
        
        # Compute output bit 1 using g1.
        output1 = feedback * g1[0]  # Current bit
        for i in range(len(state)):
            if g1[i+1]:
                output1 ^= state[i]
        
        # Compute output bit 2 using g2.
        output2 = feedback * g2[0]  # Current bit
        for i in range(len(state)):
            if g2[i+1]:
                output2 ^= state[i]
        
        # Append to the codeword (systematic bit first, parity bit second).
        codeword.append(output1)
        codeword.append(output2)
        
        # Update the shift-register state.
        state = [feedback] + state[:-1]
    
    return np.array(codeword, dtype=int)


def modulate_bpsk(bits):
    """
    BPSK modulation: 0 -> +1, 1 -> -1.
    
    Args:
        bits: Binary bit sequence
    Returns:
        Modulated signal
    """
    # Map 0 to +1 and 1 to -1.
    signal = np.array([1.0 - 2.0 * bit for bit in bits])
    return signal


def add_noise(signal, snr_db, enable_noise=True):
    """
    Add additive white Gaussian noise (AWGN).
    
    Args:
        signal: Modulated signal
        snr_db: Signal-to-noise ratio in dB
        enable_noise: Whether noise injection is enabled
    Returns:
        Noisy signal, or the original signal if enable_noise=False
    """
    if not enable_noise:
        return signal
    
    # Compute signal power.
    signal_power = np.mean(signal ** 2)
    
    # Convert SNR from dB to linear scale.
    snr_linear = 10 ** (snr_db / 10.0)
    
    # Compute noise power.
    noise_power = signal_power / snr_linear
    
    # Sample Gaussian noise.
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    # Add noise to the signal.
    noisy_signal = signal + noise
    
    return noisy_signal


def lm_text_log_probability(text):
    """
    Compute the log-probability of a full text string under the language model.

    Args:
        text: Text string to evaluate
    Returns:
        Log-probability as a float, or None if the LM is not loaded
    """
    if lm_model is None or lm_tokenizer is None:
        return None
    
    if not text:
        return None
    
    try:
        encoded_tokens = lm_tokenizer.encode(text, add_special_tokens=False)
        if len(encoded_tokens) == 0:
            return None
        
        pad_id = lm_tokenizer.pad_token_id or lm_tokenizer.eos_token_id or 0
        
        device = next(lm_model.parameters()).device
        encoder_ids = torch.tensor([[pad_id, pad_id]], dtype=torch.long, device=device) # Len 2 for heuristic
        decoder_ids = torch.tensor([[pad_id]], dtype=torch.long, device=device)
        
        total_log_prob = 0.0
        with torch.no_grad():
            for token_id in encoded_tokens:
                outputs = lm_model(input_ids=encoder_ids, decoder_input_ids=decoder_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=0)
                token_prob = probs[token_id].item()
                total_log_prob += np.log(token_prob + 1e-10)
                
                next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                decoder_ids = torch.cat([decoder_ids, next_token], dim=1)
        
        return float(total_log_prob)
    except Exception:
        return None


def correct_text_with_lm(text):
    """
    Correct text with an LM using seq2seq generation.
    Prefer the dedicated correction model, and fall back to the pruning LM if needed.
    """
    if not text:
        return ""
        
    # Determine which model to use
    model = correction_lm_model
    tokenizer = correction_lm_tokenizer
    
    # Fallback if correction model not loaded
    if model is None or tokenizer is None:
        # print("Debug: Correction model not loaded, falling back to pruning model for correction.")
        model = lm_model
        tokenizer = lm_tokenizer
        
    if model is None or tokenizer is None:
        return text
    
    try:
        device = next(model.parameters()).device
        
        # Dedicated GEC models such as T5 often expect a task prefix.
        # We use "grammar: " as a lightweight prompt for correction.
        input_text = "grammar: " + text
        
        inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate the corrected text.
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=len(text) + 50, # T5 uses subwords, so length might differ
                num_beams=2, # Beam search is better for correction
                early_stopping=True
            )
        
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Guardrail: reject outputs that drift too far from the input.
        # A model trained for completion rather than correction may hallucinate
        # a fully new sentence and ignore the original text.
        similarity = difflib.SequenceMatcher(None, text, corrected_text).ratio()
        
        # Tunable threshold: 0.6 means at least 60% similarity.
        if similarity < 0.6:
            # print(f"  [LM Correction Warning] Generated text rejected due to low similarity ({similarity:.2f})")
            # print(f"  Rejected: '{sanitize_text(corrected_text)}'")
            return text
            
        return corrected_text
    except Exception as e:
        print(f"Warning: LM correction failed: {e}")
        return text


def dump_lm_debug_tokens(stats, file_path, stop_after_substring=None):
    """
    Write detailed LM character-level inspection records to a file for analysis.
    """
    tokens = stats.get('tokens_evaluated') if isinstance(stats, dict) else None
    if not tokens:
        return
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("[LM character evaluations]\n")
            f.write(f"Total records: {len(tokens)}\n")
            f.write(f"Stop substring: {repr(stop_after_substring)}\n")
            f.write("-" * 120 + "\n")
            
            for idx, token_info in enumerate(tokens, 1):
                text = token_info.get('text', '')
                last_char = token_info.get('last_char', '')
                prob = token_info.get('probability')
                kbest_kept = token_info.get('kbest_kept')
                lm_status = "KEPT" if kbest_kept else "K-BEST-PRUNED"
                metric = token_info.get('cumulative_metric')
                adjusted_metric = token_info.get('adjusted_metric')
                step = token_info.get('step')
                
                try:
                    text_safe = text.encode('ascii', 'replace').decode('ascii')
                except Exception:
                    text_safe = "[non-ascii]"
                
                metric_display = f"{metric:.2f}" if isinstance(metric, (int, float)) else "N/A"
                adj_display = f"{adjusted_metric:.2f}" if isinstance(adjusted_metric, (int, float)) else "N/A"
                prob_display = f"{prob:.6e}" if isinstance(prob, float) else "N/A"
                prune_reason = token_info.get('prune_reason') or '-'
                line = (
                    f"[{idx:04d}] step={step} metric={metric_display} adj={adj_display} "
                    f"prob={prob_display} status={lm_status} kbest={kbest_kept} reason={prune_reason}\n"
                )
                f.write(line)
                f.write(f"       text='{text_safe}' last_char='{last_char}'\n")
                
                if stop_after_substring and stop_after_substring in text:
                    f.write(f"\n[STOP] Reached substring {repr(stop_after_substring)} at record {idx}. Stopping dump.\n")
                    break
    except Exception as e:
        print(f"Warning: failed to write LM debug log to '{file_path}': {e}")


class PathState:
    """
    Path state used during Viterbi decoding.
    Stores all information needed to track a surviving path.
    """
    def __init__(self, state, decoded_bits, cumulative_metric, decoded_text="",
                 last_char_prob=1.0, lm_penalty=0.0, last_lm_weight=0.0):
        self.state = state  # Convolutional encoder state
        self.decoded_bits = decoded_bits  # Decoded bit list
        self.cumulative_metric = cumulative_metric  # Cumulative Euclidean metric
        self.decoded_text = decoded_text  # Partially decoded text
        self.last_char_prob = last_char_prob
        self.lm_penalty = lm_penalty
        self.last_lm_weight = last_lm_weight
        self.adjusted_metric = cumulative_metric + lm_penalty
    
    def copy(self):
        """Create a deep copy of the path state."""
        return PathState(
            self.state,
            self.decoded_bits.copy(),
            self.cumulative_metric,
            self.decoded_text,
            self.last_char_prob,
            self.lm_penalty,
            getattr(self, 'last_lm_weight', 0.0)
        )


def viterbi_decode_standard(received_signal, verbose=True):
    """
    Standard soft-information Viterbi decoding without LM pruning.
    Only the single best path is retained, matching standard Viterbi decoding.
    
    Args:
        received_signal: Received noisy signal
        verbose: Whether to print verbose information
    Returns:
        decoded_text: Decoded text
    """
    # Build the trellis transition table.
    def generate_trellis():
        """Build the convolutional-code state transition table."""
        num_states = 2 ** (CONSTRAINT_LENGTH - 1)  # 4 states
        trellis = {}
        
        for state in range(num_states):
            trellis[state] = {}
            for input_bit in [0, 1]:
                # Simulate the encoder.
                state_bits = [(state >> i) & 1 for i in range(CONSTRAINT_LENGTH - 2, -1, -1)]
                
                # Compute feedback.
                feedback = input_bit
                for i, s_bit in enumerate(state_bits):
                    if (FEEDBACK_POLYNOMIAL >> (CONSTRAINT_LENGTH - 1 - i - 1)) & 1:
                        feedback ^= s_bit
                
                # Compute the two output bits.
                output1 = feedback * ((GENERATOR_POLYNOMIALS[0] >> (CONSTRAINT_LENGTH - 1)) & 1)
                output2 = feedback * ((GENERATOR_POLYNOMIALS[1] >> (CONSTRAINT_LENGTH - 1)) & 1)
                
                for i, s_bit in enumerate(state_bits):
                    if (GENERATOR_POLYNOMIALS[0] >> (CONSTRAINT_LENGTH - 2 - i)) & 1:
                        output1 ^= s_bit
                    if (GENERATOR_POLYNOMIALS[1] >> (CONSTRAINT_LENGTH - 2 - i)) & 1:
                        output2 ^= s_bit
                
                # Next encoder state.
                new_state_bits = [feedback] + state_bits[:-1]
                next_state = sum([bit << (len(new_state_bits) - 1 - i) for i, bit in enumerate(new_state_bits)])
                
                trellis[state][input_bit] = {
                    'next_state': next_state,
                    'output': [output1, output2],
                    'input': input_bit
                }
        
        return trellis
    
    trellis = generate_trellis()
    
    # Standard Viterbi keeps only the single best path.
    k_limit = 1
    
    stats = {
        'total_steps': 0,
        'kbest_pruned_total': 0,
        'path_counts': [],
        'step_events': [],
        'step_log_path': None
    }
    
    step_log_path = STANDARD_STEP_DEBUG_LOG_FILE
    if step_log_path:
        try:
            with open(step_log_path, 'w', encoding='utf-8') as f:
                f.write("[Standard Viterbi step-by-step debug log]\n")
        except Exception as e:
            print(f"Warning: cannot initialize standard step debug log file '{step_log_path}': {e}")
            step_log_path = None
    stats['step_log_path'] = step_log_path
    
    def build_snapshot(path, reason=None):
        decoded = path.decoded_text or ""
        return {
            'path_id': id(path),
            'text': sanitize_text(decoded),
            'chars_tail': sanitize_text(decoded[-8:]) if decoded else "",
            'last_char': decoded[-1] if decoded else "",
            'metric': path.cumulative_metric,
            'bits_tail': bits_tail(path.decoded_bits),
            'prob': None,
            'lm_weight': 0.0,
            'reason': reason or '-'
        }
    
    # Initialize paths from state 0; all other states start infeasible.
    paths = {0: [PathState(state=0, decoded_bits=[], cumulative_metric=0.0, decoded_text="")]}
    
    # K-best Viterbi: keep the top-k_limit paths for each state at every step.
    for i in range(0, len(received_signal), 2):
        if i + 1 >= len(received_signal):
            break
        
        stats['total_steps'] += 1
        received_pair = received_signal[i:i+2]
        new_paths = {}
        step_event = {
            'step': stats['total_steps'],
            'kbest_kept': [],
            'kbest_pruned': []
        }
        kbest_pruned_this_step = 0
        
        for state, state_paths in paths.items():
            if not state_paths:
                continue
            
            for path in state_paths:
                for input_bit in [0, 1]:
                    transition = trellis[state][input_bit]
                    next_state = transition['next_state']
                    expected_output = transition['output']
                    
                    # Compute the Euclidean branch metric.
                    expected_signal = [1.0 - 2.0 * bit for bit in expected_output]
                    branch_metric = sum((received_pair[j] - expected_signal[j]) ** 2 
                                       for j in range(2))
                    
                    new_path = path.copy()
                    new_path.state = next_state
                    new_path.decoded_bits.append(input_bit)
                    new_path.cumulative_metric += branch_metric
                    
                    if len(new_path.decoded_bits) % 8 == 0 and len(new_path.decoded_bits) > 0:
                        last_8_bits = new_path.decoded_bits[-8:]
                        char = bits_to_char(last_8_bits)
                        if char is not None:
                            new_path.decoded_text += char
                    
                    new_path.lm_penalty = 0.0
                    refresh_adjusted_metric(new_path)
                    
                    if next_state not in new_paths:
                        new_paths[next_state] = []
                    new_paths[next_state].append(new_path)
        
        trimmed_paths = {}
        kbest_start = time.perf_counter()
        for state, state_paths in new_paths.items():
            if not state_paths:
                continue
            sorted_state_paths = sorted(state_paths, key=lambda p: p.cumulative_metric)
            trimmed_paths[state] = sorted_state_paths[:k_limit]
            
            for p in trimmed_paths[state]:
                step_event['kbest_kept'].append(build_snapshot(p))
            for p in sorted_state_paths[k_limit:]:
                kbest_pruned_this_step += 1
                step_event['kbest_pruned'].append(build_snapshot(p, reason='kbest_metric'))
        
        paths = trimmed_paths
        stats['kbest_pruned_total'] += kbest_pruned_this_step
        stats['step_events'].append(step_event)
        append_step_debug_log(step_event, stats.get('step_log_path'))
        
        current_path_count = sum(len(state_paths) for state_paths in paths.values())
        stats['path_counts'].append(current_path_count)
        
        if not paths:
            if verbose:
                print("WARNING: All paths have been pruned!")
            break
    
    # Collect the final surviving paths.
    all_final_paths = []
    for state_paths in paths.values():
        all_final_paths.extend(state_paths)
    
    if not all_final_paths:
        if verbose:
            print("No surviving paths for standard decoder.")
        return []
    
    # Sort final paths by cumulative metric and keep the top-k_limit results.
    sorted_final_paths = sorted(all_final_paths, key=lambda p: p.cumulative_metric)
    top_paths = sorted_final_paths[:k_limit]
    
    decoded_results = []
    seen_texts = set()
    for path in top_paths:
        decoded_text = binary_to_text(np.array(path.decoded_bits))
        if decoded_text in seen_texts:
            continue
        seen_texts.add(decoded_text)
        decoded_results.append({
            'state': path.state,
            'metric': path.cumulative_metric,
            'text': decoded_text
        })
    
    if verbose:
        print(f"\n[Standard Viterbi Results]")
        print(f"K-best (k={k_limit}) surviving paths: {len(decoded_results)}")
        for idx, result in enumerate(decoded_results, 1):
            try:
                text_safe = result['text'].encode('ascii', 'replace').decode('ascii')
            except Exception:
                text_safe = "[contains non-ascii characters]"
            print(f"  Path {idx}: state={result['state']} metric={result['metric']:.2f} text='{text_safe}'")
        
        print(f"\n[Standard Viterbi Statistics]")
        print(f"Total decoding steps: {stats['total_steps']}")
        print(f"Total paths pruned by k-best: {stats['kbest_pruned_total']}")
        if stats['path_counts']:
            print(f"Maximum concurrent paths: {max(stats['path_counts'])}")
            print(f"Average concurrent paths: {np.mean(stats['path_counts']):.1f}")
    
    return [entry['text'] for entry in decoded_results]


def viterbi_decode_with_lm(received_signal, verbose=True):
    """
    Soft-information Viterbi decoding with ByT5 character-level LM pruning.

    Decoding strategy:
    1. For each state, keep only the K paths with the best combined metric
       (Euclidean distance plus LM penalty).
    2. Whenever N characters have been generated
       (controlled by LM_CONTEXT_CHECK_INTERVAL), evaluate the LM context
       probability of all surviving paths and prune every path except those
       that match the best-scoring context.
    
    Args:
        received_signal: Received noisy signal
        verbose: Whether to print verbose information
    Returns:
        list of possible paths (decoded text strings)
    """
    
    # Statistics tracking
    overall_start_time = time.perf_counter()
    
    stats = {
        'total_steps': 0,
        'kbest_pruned_total': 0,
        'lm_context_pruned_total': 0,
        'tokens_evaluated': [],
        'path_counts': [],
        '_recorded_paths': set(),  # Avoid duplicate bookkeeping records
        '_token_index_map': {},
        'step_events': [],
        'step_log_path': None,
        'timers': {
            'expand_paths': 0.0,
            'kbest': 0.0,
            'lm_filter': 0.0,
            'lm_filter_build': 0.0,
            'lm_filter_incremental': 0.0,
            'lm_filter_full': 0.0,
            'logging': 0.0,
            # Performance analysis timers
            'lm_incremental_total': 0.0,
            'lm_cache_lookup': 0.0,
            'lm_model_inference': 0.0,
            'lm_dict_access': 0.0,
            'lm_logic_overhead': 0.0,
            'lm_batch_prep': 0.0
        }
    }
    
    last_lm_best_context = None
    last_lm_best_log_prob = None
    
    step_log_path = STEP_DEBUG_LOG_FILE
    if step_log_path:
        try:
            with open(step_log_path, 'w', encoding='utf-8') as f:
                f.write("[Step-by-step debug log]\n")
        except Exception as e:
            print(f"Warning: cannot initialize step debug log file '{step_log_path}': {e}")
            step_log_path = None
    stats['step_log_path'] = step_log_path
    
    # Build the trellis transition table.
    def generate_trellis():
        """Build the convolutional-code state transition table."""
        num_states = 2 ** (CONSTRAINT_LENGTH - 1)  # 4 states
        trellis = {}
        
        for state in range(num_states):
            trellis[state] = {}
            for input_bit in [0, 1]:
                # Simulate the encoder.
                state_bits = [(state >> i) & 1 for i in range(CONSTRAINT_LENGTH - 2, -1, -1)]
                
                # Compute feedback.
                feedback = input_bit
                for i, s_bit in enumerate(state_bits):
                    if (FEEDBACK_POLYNOMIAL >> (CONSTRAINT_LENGTH - 1 - i - 1)) & 1:
                        feedback ^= s_bit
                
                # Compute the two output bits.
                output1 = feedback * ((GENERATOR_POLYNOMIALS[0] >> (CONSTRAINT_LENGTH - 1)) & 1)
                output2 = feedback * ((GENERATOR_POLYNOMIALS[1] >> (CONSTRAINT_LENGTH - 1)) & 1)
                
                for i, s_bit in enumerate(state_bits):
                    if (GENERATOR_POLYNOMIALS[0] >> (CONSTRAINT_LENGTH - 2 - i)) & 1:
                        output1 ^= s_bit
                    if (GENERATOR_POLYNOMIALS[1] >> (CONSTRAINT_LENGTH - 2 - i)) & 1:
                        output2 ^= s_bit
                
                # Next encoder state.
                new_state_bits = [feedback] + state_bits[:-1]
                next_state = sum([bit << (len(new_state_bits) - 1 - i) for i, bit in enumerate(new_state_bits)])
                
                trellis[state][input_bit] = {
                    'next_state': next_state,
                    'output': [output1, output2],
                    'input': input_bit
                }
        
        return trellis
    
    trellis = generate_trellis()
    
    # Initialize paths from state 0.
    paths = {0: [PathState(state=0, decoded_bits=[], cumulative_metric=0.0, decoded_text="")]}
    
    # Process the received symbols in pairs because the code rate is 1/2.
    for i in range(0, len(received_signal), 2):
        if i + 1 >= len(received_signal):
            break
        
        stats['total_steps'] += 1
        step_prune_reasons = {}
        step_event = {
            'step': stats['total_steps'],
            'kbest_kept': [],
            'kbest_pruned': []
        }
        def build_snapshot(path, reason=None, prob_override=None):
            decoded = path.decoded_text or ""
            return {
                'path_id': id(path),
                'text': sanitize_text(decoded),
                'chars_tail': sanitize_text(decoded[-8:]) if decoded else "",
                'last_char': decoded[-1] if decoded else "",
                'metric': path.adjusted_metric,
                'bits_tail': bits_tail(path.decoded_bits),
                'prob': prob_override if prob_override is not None else path.last_char_prob,
                'lm_weight': getattr(path, 'last_lm_weight', 0.0),
                'reason': reason or '-'
            }

        def get_current_char_count(path_dict):
            for state_paths in path_dict.values():
                if state_paths:
                    return len(state_paths[0].decoded_bits) // 8
            return 0

        def apply_lm_context_filter_if_needed(path_dict, force_check=False):
            nonlocal last_lm_best_context, last_lm_best_log_prob
            if LM_CONTEXT_CHECK_INTERVAL <= 0:
                return path_dict, [], []
            
            # Check if we should run the filter
            if not force_check:
                char_count = get_current_char_count(path_dict)
                if char_count == 0 or (char_count % LM_CONTEXT_CHECK_INTERVAL) != 0:
                    return path_dict, [], []
            if lm_model is None or lm_tokenizer is None:
                return path_dict, [], []

            build_start = time.perf_counter()
            context_map = {}
            context_best_metric = {}
            for state, state_paths in path_dict.items():
                for p in state_paths:
                    text = p.decoded_text or ""
                    context_text = text[:-1] if len(text) > 0 else ""
                    context_map.setdefault(context_text, []).append((state, p))
                    metric_val = p.cumulative_metric
                    best_metric = context_best_metric.get(context_text)
                    if best_metric is None or metric_val < best_metric:
                        context_best_metric[context_text] = metric_val

            stats['timers']['lm_filter_build'] += time.perf_counter() - build_start
            if len(context_map) <= 1:
                return path_dict, [], []

            context_scores = {}
            context_prob_map = {}
            context_combined_scores = {}
            context_stats = []
            incremental_eval_time = 0.0
            full_eval_time = 0.0
            shared_suffix_time = 0.0
            
            # --- NEW BATCH SUFFIX EVALUATION ---
            # Instead of token-by-token incremental, we process the full suffix in one batch
            # if we have a common base context.
            
            base_ctx = last_lm_best_context
            base_entry = LM_CONTEXT_PROB_CACHE.get(base_ctx) if base_ctx is not None else None
            
            # Determine if we can use optimized batching
            can_use_batch_optim = False
            base_kv = None
            if base_ctx is None:
                # Initial step: empty context. 
                # We can treat empty string as base if we have no cache yet or if cache has empty key
                can_use_batch_optim = True
                base_ctx = ""
            elif base_entry and isinstance(base_entry, tuple) and base_entry[1] is not None:
                can_use_batch_optim = True
                base_kv = base_entry[1]
                
            # Collect valid suffixes
            suffixes_to_eval = [] # (context_text, suffix_ids)
            direct_eval_contexts = [] # Fallback
            
            if can_use_batch_optim:
                for ctx in context_map.keys():
                    if ctx.startswith(base_ctx):
                        suffix = ctx[len(base_ctx):]
                        if suffix:
                            suffixes_to_eval.append((ctx, suffix))
                        else:
                            # Context is exactly base_ctx (shouldn't happen with interval > 0 but possible)
                            # Just reuse base score?
                            # We need log_prob for the *path so far*.
                            # Wait, we need the log_prob of the whole sequence.
                            # last_lm_best_log_prob is the log_prob of base_ctx.
                            context_scores[ctx] = last_lm_best_log_prob or 0.0
                    else:
                        direct_eval_contexts.append(ctx)
            else:
                direct_eval_contexts = list(context_map.keys())
            
            suffix_stats = {}
            batch_results_map = {}  # Store logits/indices for delayed caching

            # Process Batch Suffixes
            if suffixes_to_eval:
                t_batch = time.perf_counter()
                t_prep = time.perf_counter()
                
                # We need to batch these. 
                # Since T5 uses relative position, we can batch different lengths by padding.
                # However, for simplicity and efficiency with KV cache stacking, 
                # we might want to group by length or just pad.
                # Since we use LegacyCacheWrapper which might be picky, let's look at it.
                # LegacyCacheWrapper.update concatenates.
                # If we have a batch, we need `past_key_values` to match batch size.
                # We have 1 `base_kv`. We need to expand it to match batch size.
                
                device = next(lm_model.parameters()).device
                pad_id = lm_tokenizer.pad_token_id or lm_tokenizer.eos_token_id or 0
                
                # Expand base_kv to batch size
                # base_kv is tuple of tuples.
                num_suffixes = len(suffixes_to_eval)
                
                # Prepare batch inputs
                input_ids_list = []
                max_len = 0
                for _, suffix in suffixes_to_eval:
                    ids = lm_tokenizer.encode(suffix, add_special_tokens=False)
                    input_ids_list.append(ids)
                    max_len = max(max_len, len(ids))
                
                # Pad inputs
                padded_input_ids = []
                for ids in input_ids_list:
                    padded = ids + [pad_id] * (max_len - len(ids))
                    padded_input_ids.append(padded)
                
                decoder_input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
                encoder_input_ids = torch.full((num_suffixes, 2), pad_id, dtype=torch.long, device=device) # Len 2
                
                # Prepare KV cache: Expand singleton base_kv to batch
                if base_kv is not None:
                    # base_kv structure: ((k,v), ...)
                    # We need to repeat each tensor along dim 0
                    expanded_pkv = []
                    for layer_tuple in base_kv:
                        new_layer = []
                        for tensor in layer_tuple:
                            # tensor: (1, heads, seq, dim)
                            # expand to (num_suffixes, ...)
                            new_layer.append(tensor.expand(num_suffixes, -1, -1, -1))
                        expanded_pkv.append(tuple(new_layer))
                    # Pass wrapped_pkv. 
                    # Transformers >= 4.36 requires Cache object with specific methods.
                    # Raw tuple raises "AttributeError: 'tuple' object has no attribute 'get_seq_length'"
                    wrapped_pkv = LegacyCacheWrapper(tuple(expanded_pkv))
                else:
                    wrapped_pkv = None
                    # If no base KV, we are starting from scratch?
                    # If base_ctx was "", then wrapped_pkv is None is correct (if we don't have empty KV).
                    # Actually if base_ctx="", we assume full inference of suffix.
                    # But wait, decoder_input_ids should start with PAD if it's start of seq.
                    # If base_ctx="", then suffix IS the whole text.
                    # Standard T5 generation starts with pad_id.
                    # If we provide past_kv, we don't prepend pad_id (it's in past).
                    # If we don't provide past_kv, we MUST prepend pad_id.
                    if base_ctx == "":
                         # Prepend PAD to decoder inputs
                         decoder_input_ids = torch.cat([
                             torch.full((num_suffixes, 1), pad_id, dtype=torch.long, device=device),
                             decoder_input_ids
                         ], dim=1)
                         # Adjust input_ids_list indices (shift by 1)
                         # This logic handles the "Full Inference" case within this block
                         pass

                stats['timers']['lm_batch_prep'] += time.perf_counter() - t_prep
                t_infer = time.perf_counter()

                # Run Inference
                with torch.no_grad():
                    outputs = lm_model(
                        input_ids=encoder_input_ids,
                        decoder_input_ids=decoder_input_ids,
                        past_key_values=wrapped_pkv,
                        use_cache=True
                    )
                
                stats['timers']['lm_model_inference'] += time.perf_counter() - t_infer
                t_post = time.perf_counter()

                # Process Logits
                # logits: (batch, seq, vocab)
                # For each sample, we want to sum log(P(token[i])) for i in range(len(suffix))
                # Note: logits at index i predict token at index i (if using past_kv, logits correspond to inputs).
                # Wait. T5: 
                # If past present (len P). Input (len S).
                # Output logits (len S).
                # Logit at index 0 is prediction for token AFTER first input token?
                # No. Standard AR: Input [x]. Output Logit P(y|...x).
                # So Logit[i] is P(token[i+1] | token[:i+1]).
                # We want P(token[i] | token[:i]).
                # So we need to align logits and labels.
                
                # Case 1: With Past KV (len P).
                # Input: [t1, t2, t3]. (tokens of suffix)
                # We want P(t1|P), P(t2|P,t1), P(t3|P,t1,t2).
                # To get P(t1|P), we normally feed NOTHING? No, we feed decoder_start_token or last token.
                # BUT here we are feeding [t1, t2, t3].
                # If we feed t1, model sees [P, t1] -> predicts t2.
                # So Logit of t1 predicts t2.
                # Where do we get P(t1)?
                # We need the logit from the PREVIOUS step (the one that produced base_kv).
                # But we don't have it explicitly unless we stored it.
                # `LM_CONTEXT_PROB_CACHE` stores `char_probs` for the *next* token!
                # Yes! `base_entry[0]` contains probs for the next token (which is t1).
                
                # So:
                # P(t1) comes from base_entry[0].
                # P(t2) comes from Logit[0] (corresponding to input t1).
                # P(t3) comes from Logit[1] (corresponding to input t2).
                
                # So we need logits for inputs [:-1].
                # Input [t1, t2, t3]. Logits [L1, L2, L3].
                # L1 predicts t2. L2 predicts t3. L3 predicts t4 (next char for future).
                
                # If base_ctx == "", we prepended PAD.
                # Input [PAD, t1, t2, t3].
                # L_PAD predicts t1.
                # L_t1 predicts t2.
                # ...
                
                logits = outputs.logits # (batch, seq, vocab)
                
                # Move to CPU for efficient scalar access or keep on GPU?
                # GPU gather is better.
                log_softmax = torch.log_softmax(logits, dim=-1)
                
                for i, ((ctx_text, suffix), ids) in enumerate(zip(suffixes_to_eval, input_ids_list)):
                    # ids: [id1, id2, id3]
                    seq_len = len(ids)
                    
                    path_log_prob = 0.0
                    detailed_log_probs = []
                    
                    if base_ctx == "":
                        # Input was [PAD, id1, id2...]
                        # Logits indices: 0 -> predicts id1. 1 -> predicts id2.
                        # We need indices 0 to seq_len-1.
                        # Target tokens: id1, id2...
                        
                        # We can gather all at once.
                        # targets: [id1, id2, ..., id_last]
                        # logit_indices: [0, 1, ..., seq_len-1]
                        
                        seq_targets = torch.tensor(ids, device=device)
                        seq_logits = log_softmax[i, :seq_len, :]
                        # gather
                        token_log_probs = seq_logits.gather(1, seq_targets.unsqueeze(1)).squeeze(1)
                        detailed_log_probs = token_log_probs.tolist()
                        path_log_prob = sum(detailed_log_probs)
                        
                        # Final state (for cache)
                        # We need probs for *next* token (future).
                        # That is logit at index seq_len-1 (last input).
                        #Wait, we input [PAD, ..., id_last]. Length seq_len+1.
                        # Logits length seq_len+1.
                        # Index seq_len (last one) corresponds to input id_last. Predicts next.
                        last_frame_logits = logits[i, seq_len, :]
                        
                    else:
                        # Input was [id1, id2, ...].
                        # P(id1) comes from base_probs (CPU dict).
                        first_char = suffix[0]
                        # We need base_entry[0] dict.
                        base_probs = base_entry[0]
                        p_first = base_probs.get(first_char, LM_MIN_CHAR_PROB)
                        if p_first <= 0: p_first = LM_MIN_CHAR_PROB
                        log_p_first = np.log(p_first)
                        path_log_prob += log_p_first
                        detailed_log_probs.append(log_p_first)
                        
                        if seq_len > 1:
                            # We need P(id2)...P(id_last)
                            # P(id2) from logit of id1 (index 0).
                            # Targets: [id2, ..., id_last]
                            # Logit indices: [0, ..., seq_len-2]
                            
                            targets = ids[1:]
                            target_tensor = torch.tensor(targets, device=device)
                            
                            # Slice logits: 0 to seq_len-2
                            relevant_logits = log_softmax[i, :seq_len-1, :]
                            
                            token_log_probs = relevant_logits.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
                            detailed_log_probs.extend(token_log_probs.tolist())
                            path_log_prob += token_log_probs.sum().item()
                        
                        # Final state for cache:
                        # Logit at index seq_len-1 (corresponds to input id_last).
                        last_frame_logits = logits[i, seq_len-1, :]

                    # Add base log prob
                    total_path_log_prob = (last_lm_best_log_prob or 0.0) + path_log_prob
                    context_scores[ctx_text] = total_path_log_prob
                    
                    suffix_stats[ctx_text] = {
                        'text': suffix, 
                        'log_prob': path_log_prob,
                        'details': [f"{val:.3f}" for val in detailed_log_probs]
                    }
                    
                    # Store info for delayed caching (Optimization: don't copy KV yet)
                    batch_results_map[ctx_text] = {
                        'batch_idx': i,
                        'seq_len': seq_len,
                        'last_logits': last_frame_logits
                    }
                
                stats['timers']['lm_logic_overhead'] += time.perf_counter() - t_post
                stats['timers']['lm_filter_incremental'] += time.perf_counter() - t_batch
                
            # Process remaining contexts (if any) with full eval (slow path)
            for ctx in direct_eval_contexts:
                full_start = time.perf_counter()
                log_prob = lm_text_log_probability(ctx)
                full_eval_time += time.perf_counter() - full_start
                if log_prob is not None:
                    context_scores[ctx] = log_prob
                    
                    # Try to infer suffix stats
                    if last_lm_best_context and ctx.startswith(last_lm_best_context):
                         suffix = ctx[len(last_lm_best_context):]
                         diff_prob = log_prob - (last_lm_best_log_prob or 0.0)
                         suffix_stats[ctx] = {'text': suffix, 'log_prob': diff_prob, 'details': []}
            
            # --- METRIC CALCULATION & PRUNING (Existing logic) ---
            
            for context_text, log_prob in context_scores.items():
                length = max(len(context_text), 1)
                avg_log_prob = log_prob / length
                context_prob = float(np.exp(avg_log_prob))
                context_prob_map[context_text] = max(context_prob, LM_MIN_CHAR_PROB)
                metric_score = context_best_metric.get(context_text, np.inf)
                # MAP-style decoding formula: log_prob - metric / (2*sigma^2)
                # The metric weight has already been precomputed as LM_CONTEXT_METRIC_SCALE.
                combined_score = (
                    log_prob
                    - LM_CONTEXT_METRIC_SCALE * (metric_score if np.isfinite(metric_score) else 0.0)
                )
                context_combined_scores[context_text] = combined_score
                s_stat = suffix_stats.get(context_text, {})
                context_stats.append({
                    'context': context_text,
                    'log_prob': log_prob,
                    'avg_char_prob': context_prob_map[context_text],
                    'best_metric': metric_score,
                    'combined_score': combined_score,
                    'suffix': s_stat.get('text', ''),
                    'suffix_log_prob': s_stat.get('log_prob', 0.0),
                    'suffix_details': s_stat.get('details', [])
                })

            if not context_scores:
                return path_dict, [], []

            best_context = max(context_combined_scores.items(), key=lambda item: item[1])[0]
            filtered_paths = {}
            pruned_snapshots = []

            for state, state_paths in path_dict.items():
                kept = []
                for p in state_paths:
                    text = p.decoded_text or ""
                    context_text = text[:-1] if len(text) > 0 else ""
                    context_prob = context_prob_map.get(context_text)
                    if context_prob is not None:
                        p.last_char_prob = context_prob
                    if context_text == best_context:
                        kept.append(p)
                    else:
                        pruned_snapshots.append(build_snapshot(p, reason='lm_context', prob_override=context_prob))
                if kept:
                    filtered_paths[state] = kept

            if not filtered_paths:
                return path_dict, [], []

            for entry in context_stats:
                entry['is_best'] = entry['context'] == best_context
            
            last_lm_best_context = best_context
            last_lm_best_log_prob = context_scores.get(best_context)
            stats['timers']['lm_filter_full'] += full_eval_time
            stats['timers']['lm_filter_shared_suffix'] = stats['timers'].get('lm_filter_shared_suffix', 0.0) + shared_suffix_time

            # --- OPTIMIZATION: Only cache KV for the WINNING context ---
            if best_context in batch_results_map:
                res = batch_results_map[best_context]
                b_idx = res['batch_idx']
                seq_len = res['seq_len']
                last_logits = res['last_logits']
                
                # 1. Extract KV Cache (Expensive copy)
                # valid_kv_len logic repeated from loop
                # Note: 'base_ctx', 'base_kv' are available from outer scope
                valid_kv_len = (0 if base_ctx == "" else base_kv[0][0].shape[2]) + seq_len + (1 if base_ctx=="" else 0)
                
                new_pkv = []
                # outputs is available from outer scope
                for layer_tuple in outputs.past_key_values:
                    new_layer = []
                    for tensor in layer_tuple:
                        # Slice batch b_idx
                        s = tensor[b_idx:b_idx+1, :, :valid_kv_len, :].contiguous()
                        new_layer.append(s)
                    new_pkv.append(tuple(new_layer))
                new_pkv = tuple(new_pkv)
                
                # 2. Compute Next Char Probs (Vectorized)
                # last_logits is on GPU (from outputs.logits)
                probs = torch.softmax(last_logits, dim=0)
                # Filter > 1e-10
                mask = probs > 1e-10
                indices = torch.nonzero(mask).squeeze(1) # [N]
                valid_probs = probs[indices]
                
                indices_np = indices.cpu().numpy()
                probs_np = valid_probs.detach().cpu().numpy()
                
                next_char_probs = {}
                for tid, val in zip(indices_np, probs_np):
                     t_text = LM_TOKEN_ID_TO_TEXT_CACHE.get(tid)
                     if t_text is None: t_text = lm_tokenizer.decode([tid])
                     if t_text and len(t_text) == 1:
                         next_char_probs[t_text] = float(val)
                
                LM_CONTEXT_PROB_CACHE[best_context] = (next_char_probs, new_pkv)

            return filtered_paths, pruned_snapshots, context_stats
        received_pair = received_signal[i:i+2]
        new_paths = {}
        
        expand_start = time.perf_counter()
        # Expand all current paths.
        for state, path_list in paths.items():
            for path in path_list:
                # Try both possible input bits (0 and 1).
                for input_bit in [0, 1]:
                    transition = trellis[state][input_bit]
                    next_state = transition['next_state']
                    expected_output = transition['output']
                    
                    # Compute the Euclidean metric for soft decisions.
                    expected_signal = [1.0 - 2.0 * bit for bit in expected_output]
                    branch_metric = sum((received_pair[j] - expected_signal[j]) ** 2 
                                       for j in range(2))
                    
                    # Create the new candidate path.
                    new_path = path.copy()
                    new_path.state = next_state
                    new_path.decoded_bits.append(input_bit)
                    new_path.cumulative_metric += branch_metric
                    refresh_adjusted_metric(new_path)
                    
                    # Check whether a new character has just been completed.
                    new_char_completed = len(new_path.decoded_bits) % 8 == 0 and len(new_path.decoded_bits) > 0
                    if new_char_completed:
                        last_8_bits = new_path.decoded_bits[-8:]
                        char = bits_to_char(last_8_bits)
                        if char is not None:
                            new_path.decoded_text += char
                        new_path.last_char_prob = None
                    refresh_adjusted_metric(new_path)
                    
                    # Add the new path to the candidate set.
                    if next_state not in new_paths:
                        new_paths[next_state] = []
                    new_paths[next_state].append(new_path)
        stats['timers']['expand_paths'] += time.perf_counter() - expand_start
        
        # Apply the k-best rule: keep only the K paths with the lowest
        # cumulative metric for each state.
        try:
            k_limit = int(K_BEST_PATHS_PER_STATE)
        except (TypeError, ValueError):
            k_limit = 1
        k_limit = max(1, k_limit)
        
        path_kbest_status = {}
        paths = {}
        kbest_pruned_this_step = 0
        
        kbest_start = time.perf_counter()
        for state, state_paths in new_paths.items():
            if not state_paths:
                continue
            
            # Use heapq.nsmallest for faster top-K selection.
            # Skip storing detailed pruned-path records for speed.
            kept_paths = heapq.nsmallest(k_limit, state_paths, key=lambda p: p.adjusted_metric)
            pruned_paths = [] # Disable detailed pruned_paths recording
            
            if kept_paths:
                paths[state] = kept_paths
            
            for p in kept_paths:
                path_kbest_status[id(p)] = True
                step_event['kbest_kept'].append(build_snapshot(p))
            
            # pruned_paths is intentionally empty here, so this loop does not run.
            # for p in pruned_paths:
            #     path_kbest_status[id(p)] = False
            #     ...
            
            # For accounting purposes, estimate the number pruned as the difference
            # between the incoming path count and the number kept.
            kbest_pruned_this_step += max(0, len(state_paths) - len(kept_paths))
        
        stats['kbest_pruned_total'] += kbest_pruned_this_step
        stats['timers']['kbest'] += time.perf_counter() - kbest_start

        lm_filter_start = time.perf_counter()
        paths, lm_context_snapshots, lm_context_records = apply_lm_context_filter_if_needed(paths)
        stats['timers']['lm_filter'] += time.perf_counter() - lm_filter_start
        if lm_context_snapshots:
            stats['lm_context_pruned_total'] += len(lm_context_snapshots)
            for snapshot in lm_context_snapshots:
                step_event['kbest_pruned'].append(snapshot)
        if lm_context_records:
            step_event['lm_contexts'] = lm_context_records
        
        log_section_start = time.perf_counter()
        # Record the paths passed to the LM, namely the k-best survivors.
        lm_paths = []
        for state in paths:
            lm_paths.extend(paths[state])
        
        if lm_paths:
            for path in lm_paths:
                if len(path.decoded_bits) % 8 == 0 and len(path.decoded_text) > 0:
                    current_text = path.decoded_text
                    last_char = current_text[-1]
                    path_key = (stats['total_steps'], id(path))
                    
                    if path_key not in stats['_recorded_paths']:
                        stats['_recorded_paths'].add(path_key)
                        token_info = {
                            'step': stats['total_steps'],
                            'path_id': id(path),
                            'text': current_text,
                            'last_char': last_char,
                            'probability': path.last_char_prob,
                            'lm_pruned': False,
                            'lm_evaluated': False,
                            'kbest_kept': True,
                            'decoded_bits': path.decoded_bits.copy(),
                            'cumulative_metric': path.cumulative_metric,
                            'adjusted_metric': path.adjusted_metric,
                            'prune_reason': None
                        }
                        stats['_token_index_map'][path_key] = len(stats['tokens_evaluated'])
                        stats['tokens_evaluated'].append(token_info)
        
        stats['step_events'].append(step_event)
        append_step_debug_log(step_event, stats.get('step_log_path'))
        
        # Record the path count for this step.
        current_path_count = sum(len(paths[s]) for s in paths)
        stats['path_counts'].append(current_path_count)
        stats['timers']['logging'] += time.perf_counter() - log_section_start
        
        if len(paths) == 0:
            if verbose:
                print("WARNING: All paths have been pruned!")
            break
    
    # --- FINAL LM CHECK FOR TRAILING CHARACTERS ---
    # If trailing characters remain unchecked by the LM because the total
    # count is not divisible by N, force one final LM pruning pass here.
    if paths:
        char_count_final = get_current_char_count(paths)
        if char_count_final > 0 and (char_count_final % LM_CONTEXT_CHECK_INTERVAL) != 0:
            if verbose:
                print(f"Performing final LM check for trailing characters (total chars: {char_count_final})...")
            
            lm_filter_start = time.perf_counter()
            paths, lm_context_snapshots, lm_context_records = apply_lm_context_filter_if_needed(paths, force_check=True)
            stats['timers']['lm_filter'] += time.perf_counter() - lm_filter_start
            
            if lm_context_snapshots:
                stats['lm_context_pruned_total'] += len(lm_context_snapshots)
                # Attaching this cleanly to a finished step_event is awkward.
                # Leave detailed step logging unchanged to avoid breaking the log structure.
                pass
            
            # Update final path statistics if the last pruning step changes the winner.
            # Note that paths has already passed through the final pruning stage.

    stats['timers']['total'] = time.perf_counter() - overall_start_time
    
    # Collect all final paths.
    final_paths = []
    for state in paths:
        for path in paths[state]:
            decoded_text = binary_to_text(np.array(path.decoded_bits))
            final_paths.append(decoded_text)
    
    # Print decoding statistics.
    if verbose:
        print(f"\n{'='*158}")
        print(f"[DECODING STATISTICS]")
        print(f"{'='*158}")
        print(f"Total decoding steps: {stats['total_steps']}")
        print(f"Total paths pruned by k-best selection: {stats['kbest_pruned_total']}")
        print(f"Total paths pruned by LM context filter: {stats['lm_context_pruned_total']}")
        print(f"Final surviving paths: {len(final_paths)}")
        
        if stats['path_counts']:
            print(f"\nPath Count Statistics:")
            print(f"  Maximum paths (during decoding): {max(stats['path_counts'])}")
            print(f"  Average paths (during decoding): {np.mean(stats['path_counts']):.1f}")
            print(f"  Final paths: {stats['path_counts'][-1] if stats['path_counts'] else 0}")
        
        if stats['tokens_evaluated']:
            # print(f"\n[LM CHARACTER EVALUATION DETAILS - Strategy: Context filter every {LM_CONTEXT_CHECK_INTERVAL} chars]")
            # print(f"{'Step':<8} {'Decoded Text':<25} {'Last Char':<12} {'Bit Path (last 24 bits)':<28} {'Metric':<12} {'Adj Metric':<12} {'Char Prob':<15} {'K-Best':<12} {'Reason'}")
            # print(f"{'-'*170}")
            pass
            # for token_info in stats['tokens_evaluated']:
            #     try:
            #         text_safe = token_info['text'].encode('ascii', 'replace').decode('ascii')
            #         last_char_safe = token_info['last_char'].encode('ascii', 'replace').decode('ascii')
            #         text_display = f"'{text_safe[-20:]}'..." if len(text_safe) > 20 else f"'{text_safe}'"
            #         last_char_display = f"'{last_char_safe}'"
            #     except:
            #         text_display = "'[non-ascii]'"
            #         last_char_display = "'?'"
            #     
            #     bits = token_info['decoded_bits']
            #     if len(bits) > 24:
            #         bit_display = '...' + ''.join(str(b) for b in bits[-24:])
            #     else:
            #         bit_display = ''.join(str(b) for b in bits)
            #     
            #     metric_display = f"{token_info['cumulative_metric']:.2f}" if token_info.get('cumulative_metric') is not None else "N/A"
            #     adj_display = f"{token_info['adjusted_metric']:.2f}" if token_info.get('adjusted_metric') is not None else "N/A"
            #     prob_display = "N/A" if token_info['probability'] is None else f"{token_info['probability']:.3e}"
            #     kbest_status = "KEPT" if token_info['kbest_kept'] else "PRUNED"
            #     reason_display = token_info.get('prune_reason') or "-"
            #     
            #     try:
            #         print(f"{token_info['step']:<8} {text_display:<25} {last_char_display:<12} {bit_display:<28} {metric_display:<12} {adj_display:<12} {prob_display:<15} {kbest_status:<12} {reason_display}")
            #     except:
            #         print(f"{token_info['step']:<8} [encoding error]      [?]          {bit_display:<28} {metric_display:<12} {adj_display:<12} {prob_display:<15} {kbest_status:<12} {reason_display}")
        
        timers = stats.get('timers') or {}
        if timers:
            print(f"\n[Timing Breakdown]")
            print(f"  Expand paths time: {timers.get('expand_paths', 0.0):.3f} s")
            print(f"  K-best pruning time: {timers.get('kbest', 0.0):.3f} s")
            print(f"  LM context filter time: {timers.get('lm_filter', 0.0):.3f} s "
                  f"(build {timers.get('lm_filter_build', 0.0):.3f}s, "
                  f"incremental {timers.get('lm_filter_incremental', 0.0):.3f}s, "
                  f"full_eval {timers.get('lm_filter_full', 0.0):.3f}s, "
                  f"shared_suffix_cache {timers.get('lm_filter_shared_suffix', 0.0):.3f}s)")
            print(f"    -> Detail: "
                  f"Cache={timers.get('lm_cache_lookup', 0.0):.3f}s, "
                  f"ModelInfer={timers.get('lm_model_inference', 0.0):.3f}s, "
                  f"DictAccess={timers.get('lm_dict_access', 0.0):.3f}s, "
                  f"Overhead={timers.get('lm_logic_overhead', 0.0):.3f}s, "
                  f"BatchPrep={timers.get('lm_batch_prep', 0.0):.3f}s")
        
        if LM_CALL_LOGS:
            total_calls = len(LM_CALL_LOGS)
            avg_time = sum(l['duration'] for l in LM_CALL_LOGS) / total_calls if total_calls > 0 else 0
            unique_contexts = len(set(l['context'] for l in LM_CALL_LOGS))
            print(f"  [LM Model Calls Analysis]")
            print(f"    Total Calls: {total_calls}")
            print(f"    Unique Contexts: {unique_contexts}")
            print(f"    Avg Inference Time: {avg_time*1000:.2f} ms")
            
            try:
                import csv
                with open(LM_CALL_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Index', 'ContextLength', 'DurationSec', 'ContextSample', 'Incremental'])
                    for i, log in enumerate(LM_CALL_LOGS):
                        # Safe text for CSV
                        ctx = log['context']
                        safe_ctx = ctx.encode('ascii', 'replace').decode('ascii')
                        is_inc = log.get('incremental', False)
                        writer.writerow([i, log['context_len'], f"{log['duration']:.6f}", safe_ctx, is_inc])
                print(f"    Detailed call logs saved to '{LM_CALL_LOG_FILE}'")
            except Exception as e:
                print(f"    Warning: Failed to save detailed LM logs: {e}")
            print(f"  Logging/bookkeeping time: {timers.get('logging', 0.0):.3f} s")
            print(f"  Total decoding wall time: {timers.get('total', 0.0):.3f} s")
        
        print(f"{'='*158}\n")
    
    if LM_DEBUG_DUMP_ENABLED and LM_DEBUG_OUTPUT_FILE:
        dump_lm_debug_tokens(
            stats,
            LM_DEBUG_OUTPUT_FILE,
            stop_after_substring=LM_DEBUG_STOP_SUBSTRING
        )
    
    # Prepare return value with metrics
    final_paths_map = {}
    for state in paths:
        for path in paths[state]:
            decoded_text = binary_to_text(np.array(path.decoded_bits))
            # Keep the one with best adjusted_metric
            if decoded_text not in final_paths_map or path.adjusted_metric < final_paths_map[decoded_text]['metric']:
                final_paths_map[decoded_text] = {
                    'text': decoded_text,
                    'metric': path.adjusted_metric
                }
    
    return sorted(final_paths_map.values(), key=lambda x: x['metric'])


def bits_to_char(bits):
    """
    Convert 8 bits into a single ASCII character.
    Helper used to detect character boundaries during decoding.
    
    Args:
        bits: List or array containing 8 bits
    Returns:
        character or None if fewer than 8 bits are available
    """
    if len(bits) < 8:
        return None
    
    # Take the first 8 bits.
    byte = bits[:8]
    # Convert to an ASCII value.
    ascii_val = 0
    for bit in byte:
        ascii_val = (ascii_val << 1) | int(bit)
    # Convert to a character.
    return chr(ascii_val)


sbert_model = None

def calculate_semantic_similarity(reference, candidate):
    """
    Compute semantic similarity with Sentence-BERT.
    """
    global sbert_model
    
    if not ENABLE_SEMANTIC_EVALUATION:
        return None
    
    if not candidate or not reference:
        return 0.0
        
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        # print("Warning: sentence-transformers not found. Skipping semantic evaluation.")
        return None
        
    try:
        if sbert_model is None:
            # print("Loading SBERT model (all-MiniLM-L6-v2)...")
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                sbert_model = sbert_model.to('cuda')
        
        # Compute embeddings
        # convert_to_tensor=True returns pytorch tensors on the device
        embeddings = sbert_model.encode([reference, candidate], convert_to_tensor=True)
        
        # Compute cosine similarity
        score = util.cos_sim(embeddings[0], embeddings[1])
        return score.item()
    except Exception as e:
        print(f"Warning: SBERT computation failed: {e}")
        return None


def main():
    """Main function: Complete pipeline demonstration"""
    global DEBUG_LM_OUTPUT
    DEBUG_LM_OUTPUT = True  # Enable debug output when running this file directly
    
    print("=" * 60)
    print("Viterbi Decoder with Language Model Integration")
    print("=" * 60)
    
    # Initialize LM
    initialize_lm()
    initialize_correction_lm()
    
    # Step 1: Text to Binary
    print(f"\nStep 1: Text to Binary")
    print(f"Input text: '{TEST_TEXT}'")
    bits = text_to_binary(TEST_TEXT)
    print(f"Binary bits: {bits}")
    
    # Verification: Binary back to text
    decoded_text = binary_to_text(bits)
    print(f"Verification decode: '{decoded_text}'")
    
    # Step 2: Convolutional Encoding
    print(f"\nStep 2: Convolutional Encoding")
    print(f"Generator polynomials: {[oct(g) for g in GENERATOR_POLYNOMIALS]}")
    print(f"Feedback polynomial: {oct(FEEDBACK_POLYNOMIAL)}")
    codeword = convolutional_encode(bits)
    print(f"Encoded codeword length: {len(codeword)} bits (rate 1/2)")
    
    # Step 3: Modulation
    print(f"\nStep 3: BPSK Modulation")
    modulated_signal = modulate_bpsk(codeword)
    print(f"Modulated signal length: {len(modulated_signal)} symbols")
    
    # Step 4: Add Noise
    print(f"\nStep 4: Add Noise")
    print(f"Noise enabled: {ENABLE_NOISE}")
    print(f"SNR: {SNR_DB} dB")
    received_signal = add_noise(modulated_signal, SNR_DB, ENABLE_NOISE)
    print(f"Received signal length: {len(received_signal)} symbols")
    
    # Step 5A: Standard Viterbi Decoding (for comparison)
    print(f"\n{'='*60}")
    print(f"Step 5A: Standard Viterbi Decoding (No LM)")
    print(f"{'='*60}")
    std_start = time.time()
    standard_paths = viterbi_decode_standard(received_signal)
    standard_decode_time = time.time() - std_start
    print(f"Standard Viterbi decoding time: {standard_decode_time:.3f} s")

    # Step 5A-2: Standard Viterbi + LLM Correction
    print(f"\n{'='*60}")
    print(f"Step 5A-2: Standard Viterbi + LLM Correction")
    print(f"{'='*60}")
    
    corrected_std_text = ""
    correction_time = 0.0
    is_corrected_correct = False
    
    if standard_paths:
        # Take the best path (index 0)
        best_std_text = standard_paths[0]
        print(f"Input to LLM Correction (Standard Viterbi Output): '{sanitize_text(best_std_text)}'")
        
        # Calculate LM Score for Standard Output
        score_std = lm_text_log_probability(best_std_text)
        print(f"  Standard Path LM Score: {score_std:.4f}" if score_std is not None else "  Standard Path LM Score: N/A")
        
        corr_start = time.time()
        corrected_std_text = correct_text_with_lm(best_std_text)
        correction_time = time.time() - corr_start
        
        print(f"LLM Corrected Text: '{sanitize_text(corrected_std_text)}'")
        print(f"Correction time: {correction_time:.3f} s")
        is_corrected_correct = (corrected_std_text == TEST_TEXT)
    else:
        print("No standard path to correct.")
    
    # Step 5B: Viterbi Decoding with LM
    print(f"\n{'='*60}")
    print(f"Step 5B: Viterbi Decoding + LM Pruning")
    print(f"{'='*60}")
    print(f"K-best paths per state: {K_BEST_PATHS_PER_STATE}")
    lm_start = time.time()
    possible_paths_with_metric = viterbi_decode_with_lm(received_signal)
    lm_decode_time = time.time() - lm_start
    print(f"LM-enhanced Viterbi decoding time: {lm_decode_time:.3f} s")
    
    # Output comparison
    print(f"\n" + "=" * 80)
    print(f"COMPARISON: Standard vs Standard+LLM vs LM-Enhanced Viterbi")
    print(f"=" * 80)
    
    print(f"\n[Standard Viterbi Result]")
    if standard_paths:
        for idx, path_text in enumerate(standard_paths, 1):
            try:
                path_safe = path_text.encode('ascii', 'replace').decode('ascii')
            except Exception:
                path_safe = "[contains non-ascii characters]"
            marker = " *** CORRECT ***" if path_text == TEST_TEXT else ""
            print(f"Path {idx}: '{path_safe}'{marker}")
    else:
        print("No surviving paths.")
    
    # Standard Viterbi correctness (Any in K-best)
    correct_match_std = any(path == TEST_TEXT for path in standard_paths)
    std_top1_correct = (standard_paths[0] == TEST_TEXT) if standard_paths else False
    
    print(f"\n[Standard Viterbi + LLM Correction Result]")
    marker_corr = " *** CORRECT ***" if is_corrected_correct else ""
    try:
        corr_safe = corrected_std_text.encode('ascii', 'replace').decode('ascii')
    except:
        corr_safe = "[non-ascii]"
    print(f"Corrected: '{corr_safe}'{marker_corr}")

    print(f"\n[LM-Enhanced Viterbi Results]")
    print(f"Number of possible paths: {len(possible_paths_with_metric)}")
    
    lm_any_correct = False
    for item in possible_paths_with_metric:
        if item['text'] == TEST_TEXT:
            lm_any_correct = True
            break
    
    for i, item in enumerate(possible_paths_with_metric):
        path_text = item['text']
        metric = item['metric']
        # Safely handle paths that may contain non-ASCII characters.
        try:
            path_safe = path_text.encode('ascii', 'replace').decode('ascii')
            is_correct = (path_text == TEST_TEXT)
            marker = " *** CORRECT ***" if is_correct else ""
            best_marker = " [BEST METRIC]" if i == 0 else ""
            print(f"Path {i+1}: '{path_safe}' (metric={metric:.2f}){marker}{best_marker}")
        except:
            print(f"Path {i+1}: [contains non-ascii characters] (metric={metric:.2f})")
    
    print(f"\nCorrect in paths? {lm_any_correct}")
    
    # Calculate Semantic Scores
    bert_std = None
    bert_std_llm = None
    bert_lm_viterbi = None
    
    if standard_paths:
        bert_std = calculate_semantic_similarity(TEST_TEXT, standard_paths[0])
    
    if corrected_std_text:
        bert_std_llm = calculate_semantic_similarity(TEST_TEXT, corrected_std_text)
        
    if possible_paths_with_metric:
        # Best metric path is first
        bert_lm_viterbi = calculate_semantic_similarity(TEST_TEXT, possible_paths_with_metric[0]['text'])

    def fmt_bert(score):
        return f"{score:.4f}" if score is not None else "N/A"

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Original text:        '{TEST_TEXT}'")
    
    if standard_paths:
        best_std = standard_paths[0]
        try:
            best_std_safe = best_std.encode('ascii', 'replace').decode('ascii')
            print(f"Standard Viterbi:     {len(standard_paths)} path(s) - Top1 {'CORRECT' if std_top1_correct else 'INCORRECT'} (Any: {correct_match_std}) ({standard_decode_time:.3f} s) SBERT={fmt_bert(bert_std)} Top1='{best_std_safe}'")
        except Exception:
            print(f"Standard Viterbi:     {len(standard_paths)} path(s) - Top1 {'CORRECT' if std_top1_correct else 'INCORRECT'} ({standard_decode_time:.3f} s) SBERT={fmt_bert(bert_std)} Top1=[non-ascii]")
    else:
        print(f"Standard Viterbi:     0 path(s) - INCORRECT ({standard_decode_time:.3f} s)")
    
    try:
        corr_safe_summ = corrected_std_text.encode('ascii', 'replace').decode('ascii')
    except:
        corr_safe_summ = "[non-ascii]"

    print(f"Standard + LLM:       1 path(s) - {'CORRECT' if is_corrected_correct else 'INCORRECT'} ({standard_decode_time + correction_time:.3f} s) SBERT={fmt_bert(bert_std_llm)} Text='{corr_safe_summ}'")
    
    best_lm_safe = "[N/A]"
    if possible_paths_with_metric:
         try:
            best_lm_safe = possible_paths_with_metric[0]['text'].encode('ascii', 'replace').decode('ascii')
         except:
            best_lm_safe = "[non-ascii]"

    print(f"LM-Enhanced Viterbi:  {len(possible_paths_with_metric)} path(s) - {'CORRECT' if lm_any_correct else 'INCORRECT'} ({lm_decode_time:.3f} s) SBERT={fmt_bert(bert_lm_viterbi)} Top1='{best_lm_safe}'")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()




