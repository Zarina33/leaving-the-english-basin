"""
Step 48: Paired bootstrap for mBERT vs. each decoder on KY_avg
(absolute), same protocol as scripts/46 for XLM-R.

Closes the untested load-bearing claim in the paper: after the eta_cc
reframing, the negative claim 'architecture does not order the 8
models' rests on mBERT (second encoder) ranking 7th of 8 on absolute
KY, below every decoder except OLMo-2. We test this here.
"""
from pathlib import Path
import numpy as np

CORR = Path("data/results/transfer_correct")
OUT = Path("data/results/tables/mbert_vs_decoders_ky.csv")

SEED = 42
N_BOOT = 10_000
KY_DIRS = ["en_ky", "ru_ky", "ky_en", "ky_ru"]
DECODERS = ["qwen3", "llama", "gemma4", "mistral", "qwen25_7b", "olmo2_7b"]
DISP = {
    "mbert": "mBERT", "qwen3": "Qwen3", "llama": "Llama-3.1",
    "gemma4": "Gemma 4", "mistral": "Mistral",
    "olmo2_7b": "OLMo-2", "qwen25_7b": "Qwen2.5",
}


def load_ky_avg(m):
    d = np.load(CORR / f"{m}.npz", allow_pickle=True)
    return np.stack([d[k] for k in KY_DIRS], axis=1).mean(axis=1)


def main():
    rng = np.random.default_rng(SEED)
    per_ex = {m: load_ky_avg(m) for m in ["mbert"] + DECODERS}
    n = len(per_ex["mbert"])
    print(f"N per direction: {n}")
    print(f"mBERT KY_avg: {per_ex['mbert'].mean():.4f}")

    rows = []
    # Two-sided p, then Holm-Bonferroni over 6 comparisons
    raw = []
    for dec in DECODERS:
        cb = per_ex["mbert"]
        cd = per_ex[dec]
        real = cb.mean() - cd.mean()
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            boot[i] = cb[idx].mean() - cd[idx].mean()
        if real >= 0:
            p = 2 * min((boot <= 0).mean(), 0.5)
        else:
            p = 2 * min((boot >= 0).mean(), 0.5)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        raw.append({"decoder": DISP[dec], "acc_dec": round(cd.mean(), 4),
                    "delta": round(real, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "p": p})

    # Holm within this 6-comparison family
    sorted_by_p = sorted(range(len(raw)), key=lambda i: raw[i]["p"])
    for rank, orig_idx in enumerate(sorted_by_p):
        raw[orig_idx]["p_holm"] = min(1.0, raw[orig_idx]["p"] * (len(raw) - rank))

    print("\nmBERT vs each decoder on KY_avg (paired bootstrap, Holm within family):")
    for r in raw:
        star = "***" if r["p_holm"]<.001 else ("**" if r["p_holm"]<.01 else ("*" if r["p_holm"]<.05 else "ns"))
        print(f"  mBERT vs {r['decoder']:10s}: acc_dec={r['acc_dec']:.4f}  "
              f"Δ={r['delta']:+.4f}  95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
              f"p={r['p']:.4f}  p_holm={r['p_holm']:.4f}  {star}")
        r["p"] = round(r["p"], 4)
        r["p_holm"] = round(r["p_holm"], 4)
        rows.append(r)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
