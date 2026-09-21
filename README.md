# Leaving the English Basin

**A Logit-Lens View of Cross-Lingual Emotion Transfer in Multilingual LLMs**
Zarina Uvalieva — accepted at the 6th Workshop on Multilingual Representation Learning (MRL @ EMNLP 2026).

Code and data repository for the paper: every script, result table, annotation file and figure
the paper refers to. The paper itself is not stored here.

---

## Overview

Code, result tables and figures for a probing study of how multilingual LLMs encode emotion in
English, Russian and Kyrgyz (a low-resource Turkic language), on a parallel corpus of 1,480
sentences with six Ekman labels mapped from GoEmotions.

**Models.** Four decoder LLMs (Gemma 4 E4B, Qwen3-8B, Llama-3.1-8B, Mistral-7B-v0.3) and an
encoder baseline (XLM-R-large); an expanded sample adds mBERT, Qwen2.5-7B and OLMo-2-7B, with
XLM-R-base as a size control.

**Findings.**
1. *Exit from the English basin.* Under a null-calibrated sentence-level logit lens, the mid-network
   English level is reproduced by shuffled vectors; what differs by language is the depth at which the
   input language re-emerges. Kyrgyz exits only in the last third of every decoder the lens applies to
   (d.5 ≈ .97/.86/.97) and never earlier than Russian (.97/.64/.75).
2. *Similarity and transfer dissociate.* Mistral has the highest RU–KY CKA (.800) and the lowest
   Kyrgyz transfer (.239).
3. *Encoder/decoder does not order Kyrgyz transfer.* Under a leakage-free protocol XLM-R leads (.384)
   while mBERT is second-to-last (.198); XLM-R's absolute lead over the best decoder is not
   significant (p = .081).

All eight models beat chance (1/6) in every language.

---

## Project Structure

```
├── scripts/            # Numbered pipeline scripts; the provenance table below maps them to the paper
├── data/
│   ├── annotations/    # Translation-quality ratings of both annotators + agreement summary
│   ├── processed/      # Hidden states .npz (not tracked by git; produced by the extraction scripts)
│   ├── translated/     # Parallel corpus CSV (not tracked by git)
│   └── results/
│       ├── tables/         # CSVs behind the paper's tables (see provenance table)
│       ├── logit_lens/     # Lens curves, shuffled nulls, bootstraps, robustness checks
│       ├── figures/        # Generated figures
│       ├── qwen3/ llama/ mistral/ xlmr/ mbert/ qwen25_7b/ olmo2_7b/ xlmr_base/
│       │                   # Per-model probing, CKA and transfer CSVs (Gemma 4 CSVs sit in results/)
│       ├── *_lasttok/      # Last-token-pooling ablation
│       └── transfer_noleak.csv, human_vs_claude_probing.csv, tokenizer_fertility.csv, ...
├── translation_api/    # Optional NLLB translation client (alternative to scripts/2_translate.py)
├── translation_review/ # Annotation app and samples for the human translation audit
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # Gemma 4 and Llama-3.1 are gated models
```

Hugging Face checkpoints loaded by the scripts:
`google/gemma-4-E4B-it`, `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`,
`FacebookAI/xlm-roberta-large`; expanded sample: `google-bert/bert-base-multilingual-cased`,
`Qwen/Qwen2.5-7B`, `allenai/OLMo-2-1124-7B`; size control: `FacebookAI/xlm-roberta-base`.

Python 3.10+. Hidden-state extraction needs a CUDA GPU (16 GB is enough: decoders are loaded in
4-bit NF4); everything downstream of extraction runs on CPU from the saved hidden states.
Gemma 4 requires a recent Transformers build.

## Running the Pipeline

1. **Corpus** — `1_prepare_goemotions.py`, `2_translate.py`, `3_validate_corpus.py`
   (1,500 sampled sentences → 1,480 after translation validation).
2. **Hidden states** — one extraction script per model: `4_extract_hidden_states.py` (Gemma 4),
   `11_extract_qwen3.py`, `16_extract_llama.py`, `16b_extract_mistral.py`, `26_extract_xlmr.py`,
   `34_extract_mbert.py`, `33_extract_models.py` (Qwen2.5, OLMo-2), `62_xlmr_base.py`.
3. **Analyses** — every number in the paper comes from the scripts in the provenance table below.
   The two that carry the headline results are `43_unified_probing.py` (all within-language probing
   numbers, one run, one seed) and `40_transfer_noleak.py` (leakage-free transfer: disjoint
   train/dev/test sentence splits, target-dev layer selection).
4. **Check** — `python scripts/verify_paper_numbers.py` compares the numbers reported in the paper
   (transcribed inside the script) against the result CSVs.

Scripts numbered below 40 include earlier prototypes; where a later script supersedes one, the
provenance table names the one the paper uses.

## Paper Table/Figure Provenance

Script that produces each table/figure of the paper. Source-data paths are relative to `data/results/`.

| Paper object | Source data | Script |
|---|---|---|
| Table 3 (probing + CIs), Table 9 (selectivity) | `tables/unified_probing.csv` | `scripts/43_unified_probing.py` |
| Tables 5, 22 (leakage-free transfer) | `transfer_noleak.csv` | `scripts/40_transfer_noleak.py` |
| Table 4 (CKA, main decoders), Table 23 (CKA, added models) | `*/cka_results.csv` | `scripts/6_cka_analysis.py` and per-model analogues |
| Table 6 (per-emotion macro-F1) + Table 7 (peak layers) | `tables/per_emotion_f1.csv` | `scripts/22_per_emotion_f1.py`, `scripts/28_per_emotion_peak_robust.py` |
| Table 8 (Procrustes/RSA) | `tables/procrustes_results.csv`, `tables/rsa_results.csv` | `scripts/25_procrustes_rsa.py` |
| Table 10 (sentence-length control) | `tables/length_control.csv` | `scripts/21_mlp_probe_control.py` |
| Table 12 (probing pairwise) | `tables/probing_pairwise_unified.csv` | `scripts/49_probing_pairwise_unified.py` (seed 42; an earlier prototype, `scripts/20_bootstrap_stats.py`, used pre-unified layers) |
| Table 13 (XLM-R vs others) | `tables/xlmr_vs_others_ky.csv` | `scripts/46_xlmr_vs_others_ky_bootstrap.py` |
| Table 14 (mBERT vs decoders) | `tables/mbert_vs_decoders_ky.csv` | `scripts/48_mbert_vs_decoders_bootstrap.py` |
| Table 15 (eta_cc bootstrap) + eta_EN->RU robustness (App. J) | `tables/eta_cc_bootstrap.csv`, `tables/eta_enru_bootstrap.csv` | `scripts/47_eta_cc_bootstrap.py`, `scripts/52_eta_enru_bootstrap.py` |
| Table 17 (translation control) | `human_vs_claude_probing.csv` | `scripts/29_extract_human_control.py`, `scripts/30_human_vs_claude_probing.py` |
| Table 18 (pooling ablation) | `tables/table12_pooling_ablation.csv` | `scripts/31_extract_lasttoken.py`, `scripts/45_regen_pooling_table.py` |
| Table 19 (quantization ablation, NF4 vs 8-bit) | `tables/fp16_ablation.csv` | `scripts/24_fp16_ablation.py` |
| Table 20 (Qwen3 NF4 vs fp16 transfer) | `tables/qwen3_fp16_transfer.csv` | `scripts/54_extract_qwen3_fp16.py`, `scripts/55_transfer_noleak_qwen3_fp16.py` |
| Table 24 (tokenizer fertility + correlation) | `tokenizer_fertility.csv` | `scripts/41_tokenizer_fertility.py` |
| Figure 2 (UMAP, unified layers) | hidden states | `scripts/14_umap_qwen3.py` |
| Figure 4 (logit lens) | `logit_lens/*_english_pivot.csv` | `scripts/38_logit_lens_english_pivot.py`, `scripts/39_plot_english_pivot.py` |
| Logit-lens shuffled-representation null (App. O) | `logit_lens/*_null.csv` | `scripts/51_logit_lens_null.py` |
| Table 11 (TF-IDF transfer baseline) | `tables/tfidf_transfer_baseline.csv` | `scripts/53_tfidf_transfer_baseline.py` |
| Directional DiD interaction (App. H) | `tables/directional_did.csv` | `scripts/56_directional_did_bootstrap.py` |
| Lens AUC sentence-level bootstrap (App. O) | `logit_lens/auc_bootstrap.csv` | `scripts/57_lens_auc_bootstrap.py` |
| Mistral window-drop manner bootstrap (App. O) | `logit_lens/mistral_window_drop.csv` | `scripts/58_mistral_window_drop_bootstrap.py` |
| Text-specified last-third window check (App. O) | `logit_lens/lastthird_drop.csv` | `scripts/59_lens_lastthird_drop.py` |
| Lens robustness: classification variants + last-token (App. O) | `logit_lens/lens_robustness.csv` | `scripts/60_lens_robustness.py` |
| Control-set prediction-distribution audit (Table 17 caption) | `tables/control_pred_distribution.csv` | `scripts/50_control_pred_distribution.py` |
| Human-translation control for transfer (App. L) | `tables/human_control_transfer.csv` | `scripts/61_human_control_transfer.py` |
| XLM-R-base size control (Table 22 row, §6.3, App. P) | `tables/xlmr_base_transfer.csv`, `xlmr_base/probing_detailed.csv` | `scripts/62_xlmr_base.py` |
| d₀.₅ sentence-level bootstrap (§6.4, App. O) | `logit_lens/d50_bootstrap.csv` | `scripts/63_d50_bootstrap.py` |
| XLM-R-base vs mBERT paired bootstrap (§6.3, App. J) | `tables/base_vs_mbert_eta.csv` | `scripts/64_base_vs_mbert_eta.py` |
| Matched-probe permutation null, Mistral-KY cell (§5) | `tables/matched_permutation.csv` | `scripts/65_matched_permutation.py` |
| Table 16 (transfer pairwise tests) | `tables/transfer_pairwise_noleak.csv` | `scripts/44_transfer_pairwise_noleak.py` |
| Permutation tests, 10,000 shuffles (§5) | `permutation_10k.csv` | `scripts/37_permutation_10k.py` |
| TF-IDF within-language baseline (§6.5) | `lexical_baseline.csv` | `scripts/42_lexical_baseline.py` |
| Figure 1 (silhouette, all models) | `silhouette_all_models.csv` | `scripts/27_silhouette_all_models.py` |
| Emotion confusions (App. S, §6.6) | `tables/confusion_matrices.csv`, `tables/confusion_pairs*.csv`, `tables/confusion_rank_agreement.csv` | `scripts/66_confusion_matrices.py` |

---

## Annotations

`data/annotations/` holds the translation-quality ratings of the Kyrgyz split (4-point scale:
OK / Minor edit / Major edit / Reject) from both annotators (`annotator_a_*`, `annotator_b_*`), the
200-sentence overlap subset used for agreement, and `iaa_summary.csv` (Cohen's kappa and raw
agreement; the paper additionally reports Gwet's AC1, which corrects for the prevalence paradox).

---

## Dataset

Based on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
(Demszky et al., ACL 2020): 6 Ekman emotions × 250 single-label examples = 1,500 sentences,
1,480 after translation validation (all 20 exclusions fell in *anger*). Russian and Kyrgyz
translations were generated with Claude 3.7 Sonnet, audited by trilingual native speakers, and
checked against an independent 102-sentence human-translated control set.

| Language | ISO | Family | Script |
|----------|-----|--------|--------|
| English | en | Indo-European | Latin |
| Russian | ru | Indo-European (Slavic) | Cyrillic |
| Kyrgyz | ky | Turkic | Cyrillic |

---

## Citation

```bibtex
@inproceedings{uvalieva-2026-english-basin,
  title     = {Leaving the {E}nglish Basin: A Logit-Lens View of Cross-Lingual Emotion Transfer in Multilingual {LLM}s},
  author    = {Uvalieva, Zarina},
  booktitle = {Proceedings of the 6th Workshop on Multilingual Representation Learning (MRL)},
  year      = {2026}
}
```

---

## License

Code: MIT License.
GoEmotions data: [Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE).
