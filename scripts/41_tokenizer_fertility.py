"""
Step 41: Tokenizer fertility on Kyrgyz (and EN/RU) for all 8 models, and its
correlation with leakage-free KY transfer.

Reviewer M4: the paper's "scale + coverage" story is a proxy for the real
quantity — how much of the target language the model actually saw. A cheap,
direct measure is tokenizer fertility: subword tokens per whitespace word.
High KY fertility = the tokenizer fragments Kyrgyz heavily = little Kyrgyz in
pre-training. We compute fertility on the parallel corpus for each language
and correlate KY fertility with each model's leakage-free KY_avg transfer.

Fertility(lang, model) = (# subword tokens over corpus) / (# whitespace words).
Reported as EN/RU/KY fertility and the KY/EN fertility ratio (typology-robust).

Uses ONLY tokenizers (no model weights, no GPU). Fast.

Usage:
    python3 scripts/41_tokenizer_fertility.py
Output:
    data/results/tokenizer_fertility.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

CORPUS = Path("data/translated/parallel_corpus_clean.csv")
TRANSFER = Path("data/results/transfer_noleak.csv")
OUT = Path("data/results/tokenizer_fertility.csv")

# HF tokenizer ids per model key (matches the paper's models).
TOKENIZERS = {
    "gemma4":    "google/gemma-2-2b",          # Gemma tokenizer family (text backbone)
    "qwen3":     "Qwen/Qwen3-8B",
    "llama":     "meta-llama/Llama-3.1-8B",
    "mistral":   "mistralai/Mistral-7B-v0.3",
    "xlmr":      "FacebookAI/xlm-roberta-large",
    "mbert":     "bert-base-multilingual-cased",
    "qwen25_7b": "Qwen/Qwen2.5-7B",
    "olmo2_7b":  "allenai/OLMo-2-1124-7B",
}


def fertility(tok, texts):
    n_tok = 0
    n_word = 0
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        n_word += len(t.split())
    return n_tok / max(n_word, 1)


def main():
    from transformers import AutoTokenizer
    df = pd.read_csv(CORPUS)
    cols = {"en": "text_en", "ru": "text_ru", "ky": "text_ky"}

    rows = []
    for key, tid in TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(tid)
        except Exception as e:
            print(f"[!] {key} ({tid}) tokenizer load failed: {e}")
            continue
        fert = {lg: fertility(tok, df[c].tolist()) for lg, c in cols.items()}
        rows.append({
            "model": key,
            "fert_en": round(fert["en"], 3),
            "fert_ru": round(fert["ru"], 3),
            "fert_ky": round(fert["ky"], 3),
            "ky_over_en": round(fert["ky"] / fert["en"], 3),
        })
        print(f"  {key:11s} EN={fert['en']:.2f} RU={fert['ru']:.2f} "
              f"KY={fert['ky']:.2f}  KY/EN={fert['ky']/fert['en']:.2f}")

    fdf = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")

    # Correlate KY fertility (and KY/EN ratio) with leakage-free KY_avg transfer.
    if TRANSFER.exists():
        tdf = pd.read_csv(TRANSFER)
        ky_dirs = [("en", "ky"), ("ru", "ky"), ("ky", "en"), ("ky", "ru")]
        kyavg = {}
        for m in tdf.model.unique():
            s = tdf[(tdf.model == m) &
                    (tdf[["src", "tgt"]].apply(tuple, axis=1).isin(ky_dirs))]
            kyavg[m] = s.test_acc.mean()
        merged = fdf.assign(ky_avg=fdf.model.map(kyavg)).dropna(subset=["ky_avg"])
        if len(merged) >= 3:
            from scipy.stats import pearsonr, spearmanr
            r1, p1 = pearsonr(merged.fert_ky, merged.ky_avg)
            r2, p2 = spearmanr(merged.ky_over_en, merged.ky_avg)
            print(f"\nCorrelation (n={len(merged)}):")
            print(f"  KY fertility      vs KY_avg transfer: Pearson r={r1:+.3f} p={p1:.3f}")
            print(f"  KY/EN fert ratio  vs KY_avg transfer: Spearman rho={r2:+.3f} p={p2:.3f}")
            print("  (negative = more fragmentation -> worse transfer, as expected)")


if __name__ == "__main__":
    main()
