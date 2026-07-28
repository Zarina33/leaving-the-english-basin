# Supplementary Code and Data

**How Do Large Language Models Encode Emotions Across Languages?**
*A Multilingual Probing Study across English, Russian, and Kyrgyz*

This repository contains the complete reproducible pipeline: all scripts, experimental results, annotations, figures, and tables referenced in the paper.

---

## Project Structure

```
├── scripts/                          # Numbered pipeline scripts
│   ├── 0_download_models.py          # Pre-download all HF models
│   ├── 1_prepare_goemotions.py       # Download & filter GoEmotions dataset
│   ├── 2_translate.py                # Translate EN -> RU, KY via Claude CLI
│   ├── 3_validate_corpus.py          # Validate and clean parallel corpus
│   ├── 4_extract_hidden_states.py    # Extract hidden states — Gemma 4 E4B
│   ├── 5_probing.py                  # Probing classifiers
│   ├── 6_cka_analysis.py             # CKA cross-lingual similarity
│   ├── 7_visualization.py            # UMAP clusters + heatmaps
│   ├── 8_additional_experiments.py   # Permutation test, silhouette, anisotropy
│   ├── 8a_probing_gpu.py             # Layer-wise probing (GPU)
│   ├── 9_cross_lingual_transfer.py   # Cross-lingual transfer probing
│   ├── 10_per_emotion_layers.py      # Per-emotion one-vs-rest
│   ├── 11_extract_qwen3.py           # Extract hidden states — Qwen3-8B
│   ├── 12_compare_models.py          # Qwen3 experiments + comparison figures
│   ├── 13_stats_qwen3.py             # Permutation + paired t-test
│   ├── 14_umap_qwen3.py              # UMAP side-by-side comparison
│   ├── 15_summary_table.py           # Summary tables + LaTeX export
│   ├── 16_extract_llama.py           # Extract hidden states — LLaMA
│   ├── 16b_extract_mistral.py        # Extract hidden states — Mistral
│   ├── 17_llama_pipeline.py          # Full LLaMA experiments
│   ├── 18_multimodel_pipeline.py     # Unified multi-model comparison
│   ├── 19_annotation_tool.py         # Human annotation tool (IAA collection)
│   ├── 20_bootstrap_stats.py         # Bootstrap CIs for probing
│   ├── 21_mlp_probe_control.py       # MLP probe control (vs linear)
│   ├── 22_per_emotion_f1.py          # Per-emotion F1 analysis
│   ├── 23_bootstrap_transfer.py      # Bootstrap CIs for transfer
│   ├── 24_fp16_ablation.py           # FP16 vs 4-bit ablation (Gemma 4)
│   ├── 25_procrustes_rsa.py          # Procrustes + RSA analysis
│   ├── 26_extract_xlmr.py            # Extract hidden states — XLM-R
│   ├── 27_silhouette_all_models.py   # Silhouette scores across models
│   ├── 28_per_emotion_peak_robust.py # Robust per-emotion peak layer
│   └── verify_paper_numbers.py       # Sanity-check numbers reported in paper
│
├── data/
│   ├── annotations/                  # Human annotations for IAA
│   │   ├── annotator_a_annotations.csv
│   │   ├── annotator_b_annotations.csv
│   │   ├── annotator_a_200.csv
│   │   ├── annotator_b_200.csv
│   │   ├── iaa_summary.csv           # Cohen's kappa / agreement metrics
│   │   └── *_template.csv            # Blank annotation templates
│   └── results/
│       ├── figures/                  # All generated figures (fig1–fig24)
│       ├── tables/                   # Summary tables + paper_tables.tex
│       │   ├── bootstrap_*.csv       # Bootstrap CIs (probing, transfer)
│       │   ├── per_emotion_f1.csv
│       │   ├── procrustes_results.csv
│       │   ├── rsa_results.csv
│       │   ├── length_control.csv
│       │   ├── probe_selectivity.csv
│       │   ├── fp16_ablation.csv
│       │   └── table_*.csv           # Per-model summary tables
│       ├── gemma4_fp16/              # FP16 Gemma 4 results (ablation)
│       ├── qwen3/                    # Qwen3-8B results
│       ├── llama/                    # LLaMA results
│       ├── mistral/                  # Mistral results
│       ├── xlmr/                     # XLM-R results
│       ├── probing_detailed.csv      # Gemma 4 layer-wise probing
│       ├── cka_results.csv           # Gemma 4 CKA similarity
│       ├── transfer_results.csv      # Gemma 4 cross-lingual transfer
│       ├── per_emotion_layers.csv    # Gemma 4 per-emotion analysis
│       ├── permutation_baseline.csv  # Permutation null distributions
│       ├── significance.csv          # Statistical tests
│       ├── silhouette.csv
│       ├── silhouette_all_models.csv
│       └── anisotropy.csv
│
└── requirements.txt
```

---

## Reproducing the Experiments

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16 GB VRAM recommended for 4-bit inference; 24 GB+ for FP16 ablation)

```bash
pip install -r requirements.txt
```

Gemma 4 requires Transformers from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

HuggingFace access tokens are required for gated models:
- [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large)

```bash
huggingface-cli login
```

### Pipeline

Run scripts in order. Each script saves its output so you can resume from any step.

```bash
# --- Data preparation ---
python scripts/1_prepare_goemotions.py       # Download GoEmotions, 6 emotions x 250
python scripts/2_translate.py                # Translate EN -> RU, KY
python scripts/3_validate_corpus.py          # Validate, save clean corpus (1480 examples)

# --- Gemma 4 E4B experiments ---
python scripts/4_extract_hidden_states.py    # All 43 layers (~900 MB)
python scripts/8a_probing_gpu.py             # Layer-wise probing
python scripts/6_cka_analysis.py             # CKA cross-lingual similarity
python scripts/7_visualization.py            # UMAP + heatmaps
python scripts/8_additional_experiments.py   # Permutation, silhouette, anisotropy
python scripts/9_cross_lingual_transfer.py   # Cross-lingual transfer
python scripts/10_per_emotion_layers.py      # Per-emotion analysis

# --- Qwen3-8B experiments ---
python scripts/11_extract_qwen3.py           # All 37 layers (~1.1 GB)
python scripts/12_compare_models.py          # Qwen3 experiments + comparison figures
python scripts/13_stats_qwen3.py             # Statistical tests
python scripts/14_umap_qwen3.py              # UMAP comparison
python scripts/15_summary_table.py           # Summary tables + LaTeX

# --- Additional models ---
python scripts/16_extract_llama.py           # LLaMA hidden states
python scripts/16b_extract_mistral.py        # Mistral hidden states
python scripts/26_extract_xlmr.py            # XLM-R hidden states
python scripts/17_llama_pipeline.py          # Full LLaMA pipeline
python scripts/18_multimodel_pipeline.py     # Unified multi-model comparison

# --- Robustness and ablations ---
python scripts/19_annotation_tool.py         # Human annotation for IAA
python scripts/20_bootstrap_stats.py         # Bootstrap CIs (probing)
python scripts/21_mlp_probe_control.py       # MLP probe vs linear control
python scripts/22_per_emotion_f1.py          # Per-emotion F1
python scripts/23_bootstrap_transfer.py      # Bootstrap CIs (transfer)
python scripts/24_fp16_ablation.py           # FP16 vs 4-bit ablation
python scripts/25_procrustes_rsa.py          # Procrustes + RSA
python scripts/27_silhouette_all_models.py   # Silhouette across all models
python scripts/28_per_emotion_peak_robust.py # Robust peak-layer analysis

# --- Sanity check ---
python scripts/verify_paper_numbers.py       # Verify reported numbers match CSVs
```

Expected total runtime on RTX 4090/5080: ~8–12 hours for the full pipeline with all five models (dominated by hidden state extraction).

---

## Models

| Model | Parameters | Layers | Hidden dim | Quantization |
|-------|-----------|--------|-----------|-------------|
| Gemma 4 E4B | ~4B eff. | 42 | 2,560 | 4-bit NF4 (+ FP16 ablation) |
| Qwen3-8B | 8B | 36 | 4,096 | 4-bit NF4 |
| LLaMA-3.1-8B | 8B | 32 | 4,096 | 4-bit NF4 |
| Mistral-7B-v0.3 | 7B | 32 | 4,096 | 4-bit NF4 |
| XLM-R-large | 560M | 24 | 1,024 | FP16 |

## Dataset

Based on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Demszky et al., ACL 2020).
6 Ekman emotions x 250 examples = 1,500 sentences -> 1,480 after filtering.

| Language | ISO | Family | Script |
|----------|-----|--------|--------|
| English | en | Indo-European | Latin |
| Russian | ru | Indo-European (Slavic) | Cyrillic |
| Kyrgyz | ky | Turkic | Cyrillic |

## Annotations (IAA)

Human annotations for inter-annotator agreement on the Kyrgyz split are provided in `data/annotations/`. Both author and second-reviewer annotations, plus the 200-sentence subset used for Cohen's kappa, and the summary in `iaa_summary.csv`.

---

## Pre-computed Results

All experimental results are included in `data/results/` so figures and tables from the paper can be inspected without re-running the full pipeline. Generated figures (fig1–fig24) are in `data/results/figures/`.

`verify_paper_numbers.py` cross-checks the numbers reported in the paper against the raw CSV outputs.

## License

Code: MIT License.
GoEmotions data: [Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE).
