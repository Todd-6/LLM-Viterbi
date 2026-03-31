# LLM-Viterbi

## Overview

This repository studies language-model-assisted convolutional decoding for noisy text transmission. The codebase compares three decoding paradigms under additive channel noise:

- standard Viterbi decoding,
- standard Viterbi followed by LM-based correction,
- LM-enhanced Viterbi decoding with ByT5-guided path pruning.

The main implementation is provided in `decoder/viterbi_lm_decode.py`, while `evaluation/` contains experiment scripts for BLER, semantic similarity, and runtime analysis.

## Method Summary

Given an input sentence, the pipeline:

1. converts text into binary symbols,
2. applies convolutional encoding,
3. modulates the encoded bits with BPSK,
4. injects AWGN at a chosen SNR,
5. decodes the received sequence with one of the competing methods.

The central research question is whether a character-level language model can improve decoding quality by pruning unlikely paths during the Viterbi search.

## Setup

Python 3.10+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Model Configuration

The repository is configured to load pretrained models from Hugging Face by default:

- repository: `todd8642/LLMViterbi_ByT5_finetuned`
- pruning LM subfolder: `byt5_finetuned`
- correction LM subfolder: `ByT5_correction_finetuned`

Model source:

- [Hugging Face model repository](https://huggingface.co/todd8642/LLMViterbi_ByT5_finetuned/commit/d55c2f057e023b6770f5dacb59b8a5025345be44)

On first use, `transformers` will download model artifacts into the local cache automatically. The repository does not track local `models/` or `debug/` folders.

Optional environment variables:

- `VITERBI_HF_REPO`
- `VITERBI_HF_REVISION`
- `VITERBI_LM_MODEL`
- `VITERBI_LM_SUBFOLDER`
- `VITERBI_CORRECTION_MODEL`
- `VITERBI_CORRECTION_SUBFOLDER`

If private Hugging Face repositories are used, authenticate first with `huggingface-cli login`.

## Running Experiments

Main BLER evaluation:

```bash
python evaluation/batch_test_bler.py
```

Additional scripts:

- `python evaluation/batch_collect_standard.py`
- `python evaluation/batch_test_sbert.py`
- `python evaluation/batch_test_interval_sweep.py`
- `python evaluation/batch_test_performance.py`
- `python evaluation/batch_time_test_performance.py`

Plot the latest BLER result:

```bash
python scripts/plot_bler_results.py
```

Windows PowerShell example:

```powershell
$env:VITERBI_HF_REPO="todd8642/LLMViterbi_ByT5_finetuned"
python evaluation\batch_test_bler.py
```

## Repository Structure

- `decoder/`: encoding, channel simulation, and decoding algorithms
- `evaluation/`: batch experiments and benchmarking
- `scripts/`: plotting and result utilities
- `data/`: training and test data
- `results/`: generated experiment outputs
- `finetune/`: fine-tuning and model evaluation scripts
