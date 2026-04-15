# Supplementary Code and Data

**How Do Large Language Models Encode Emotions Across Languages?**
*A Multilingual Probing Study of Gemma 4 and Qwen3*

This repository contains the complete reproducible pipeline: all scripts, experimental results, figures, and tables referenced in the paper.

---

## Project Structure

```
├── scripts/                        # Numbered pipeline scripts (run in order)
│   ├── 1_prepare_goemotions.py     # Download & filter GoEmotions dataset
│   ├── 2_translate.py              # Translate EN → RU, KY via Claude CLI
│   ├── 3_validate_corpus.py        # Validate and clean parallel corpus
│   ├── 4_extract_hidden_states.py  # Extract hidden states — Gemma 4 E4B
│   ├── 5_probing.py                # Probing classifiers
│   ├── 6_cka_analysis.py           # CKA cross-lingual similarity
│   ├── 7_visualization.py          # UMAP clusters + heatmaps
│   ├── 8a_probing_gpu.py           # Layer-wise probing (GPU, Gemma 4)
│   ├── 8_additional_experiments.py # Permutation test, silhouette, anisotropy
│   ├── 9_cross_lingual_transfer.py # Cross-lingual transfer probing (Gemma 4)
│   ├── 10_per_emotion_layers.py    # Per-emotion one-vs-rest (Gemma 4)
│   ├── 11_extract_qwen3.py         # Extract hidden states — Qwen3-8B
│   ├── 12_compare_models.py        # All experiments on Qwen3 + comparison figs
│   ├── 13_stats_qwen3.py           # Permutation test + paired t-test comparison
│   ├── 14_umap_qwen3.py            # UMAP side-by-side comparison
│   └── 15_summary_table.py         # Summary tables + LaTeX export
├── data/results/
│   ├── figures/                    # All generated figures (fig1–fig19)
│   ├── tables/                     # Summary tables + paper_tables.tex
│   ├── qwen3/                      # Qwen3-specific results
│   ├── probing_detailed.csv        # Layer-wise probing results (Gemma 4)
│   ├── cka_results.csv             # CKA similarity matrices
│   ├── transfer_results.csv        # Cross-lingual transfer results
│   ├── per_emotion_layers.csv      # Per-emotion layer analysis
│   ├── permutation_baseline.csv    # Permutation test null distributions
│   ├── significance.csv            # Statistical significance tests
│   ├── silhouette.csv              # Silhouette scores
│   └── anisotropy.csv              # Anisotropy measurements
└── requirements.txt
```

---

## Reproducing the Experiments

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16 GB VRAM recommended for 4-bit inference)

```bash
pip install -r requirements.txt
```

Gemma 4 requires Transformers from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

Both models require HuggingFace access tokens:
- [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

### Pipeline

Run scripts in order. Each script saves its output so you can resume from any step.

```bash
# Data preparation
python scripts/1_prepare_goemotions.py      # Download GoEmotions, filter to 6 emotions x 250
python scripts/2_translate.py               # Translate EN → RU, KY
python scripts/3_validate_corpus.py         # Validate, save clean corpus (1480 examples)

# Gemma 4 E4B experiments
python scripts/4_extract_hidden_states.py   # Extract hidden states (all 43 layers)
python scripts/8a_probing_gpu.py            # Layer-wise probing
python scripts/6_cka_analysis.py            # CKA cross-lingual similarity
python scripts/7_visualization.py           # UMAP + heatmaps
python scripts/8_additional_experiments.py  # Permutation, silhouette, anisotropy
python scripts/9_cross_lingual_transfer.py  # Cross-lingual transfer
python scripts/10_per_emotion_layers.py     # Per-emotion analysis

# Qwen3-8B experiments
python scripts/11_extract_qwen3.py          # Extract hidden states (all 37 layers)
python scripts/12_compare_models.py         # All Qwen3 experiments + comparison figures
python scripts/13_stats_qwen3.py            # Statistical tests
python scripts/14_umap_qwen3.py             # UMAP comparison
python scripts/15_summary_table.py          # Summary tables + LaTeX
```

Expected total runtime on RTX 4090/5080: ~3-4 hours (dominated by hidden state extraction).

---

## Models

| Model | Parameters | Layers | Hidden dim | Quantization |
|-------|-----------|--------|-----------|-------------|
| Gemma 4 E4B | ~4B eff. | 42 | 2,560 | 4-bit NF4 |
| Qwen3-8B | 8B | 36 | 4,096 | 4-bit NF4 |

## Dataset

Based on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Demszky et al., ACL 2020).
6 Ekman emotions x 250 examples = 1,500 sentences -> 1,480 after filtering.

| Language | ISO | Family | Script |
|----------|-----|--------|--------|
| English | en | Indo-European | Latin |
| Russian | ru | Indo-European (Slavic) | Cyrillic |
| Kyrgyz | ky | Turkic | Cyrillic |

---

## Pre-computed Results

All experimental results are included in `data/results/` so that figures and tables from the paper can be inspected without re-running the full pipeline. Generated figures are in `data/results/figures/`.

## License

Code: MIT License.
GoEmotions data: [Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE).
