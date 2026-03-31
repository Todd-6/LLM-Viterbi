# LLM-Viterbi

This repository explores convolutional decoding for noisy text transmission with three decoding strategies:

- Standard Viterbi decoding
- Standard Viterbi followed by LM-based correction
- LM-enhanced Viterbi decoding with ByT5-based path pruning

The core implementation lives in `decoder/viterbi_lm_decode.py`. The `evaluation/` scripts run batch experiments, and `scripts/plot_bler_results.py` plots BLER curves from saved results.

## Requirements

- Python 3.10+ recommended
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## Models

The project loads models from Hugging Face by default:

- Shared repository: `todd8642/LLMViterbi_ByT5_finetuned`
- Pruning LM subfolder: `byt5_finetuned`
- Correction LM subfolder: `ByT5_correction_finetuned`

On first run, `transformers` will automatically download the model files into the local cache.

Model source:

- [Hugging Face model repository](https://huggingface.co/todd8642/LLMViterbi_ByT5_finetuned/commit/d55c2f057e023b6770f5dacb59b8a5025345be44)

Optional environment variables:

- `VITERBI_HF_REPO`
- `VITERBI_HF_REVISION`
- `VITERBI_LM_MODEL`
- `VITERBI_LM_SUBFOLDER`
- `VITERBI_CORRECTION_MODEL`
- `VITERBI_CORRECTION_SUBFOLDER`

Example on Windows PowerShell:

```powershell
$env:VITERBI_HF_REPO="todd8642/LLMViterbi_ByT5_finetuned"
python evaluation\batch_test_bler.py
```

## Quick Start

Run the main batch BLER evaluation:

```bash
python evaluation/batch_test_bler.py
```

Other useful scripts:

- `python evaluation/batch_collect_standard.py`
- `python evaluation/batch_test_sbert.py`
- `python evaluation/batch_test_interval_sweep.py`
- `python evaluation/batch_test_performance.py`
- `python evaluation/batch_time_test_performance.py`

To plot the latest BLER result:

```bash
python scripts/plot_bler_results.py
```

## Repository Layout

- `decoder/`: core encoding, channel simulation, and decoding logic
- `evaluation/`: batch evaluation and benchmarking scripts
- `scripts/`: result visualization utilities
- `data/`: test and training data
- `results/`: generated evaluation outputs
- `finetune/`: training and model evaluation scripts

## Notes

- `debug/` and `models/` are intentionally excluded from Git tracking.
- If you use private Hugging Face repositories, log in with `huggingface-cli login` before running the code.
