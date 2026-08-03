"""
Step 47: Paired bootstrap for the chance-corrected efficiency
  eta_cc = (KY_avg - 1/6) / (EN<->RU - 1/6)
across all 8 models, using the exact per-example correctness vectors
from scripts/40 (same 296 held-out sentences for every model).

Addresses reviewer critique: eta_cc has no CI. Since it is a ratio of
two point estimates on n=296 with a small denominator for near-chance
models (mBERT: .249 - 1/6 = .082), a percentile bootstrap on paired
resamples is the honest interval.

We also compute pairwise differences eta_cc[a] - eta_cc[b] with
bootstrap CIs, focusing on the two comparisons that carry contribution 3:
  XLM-R vs Llama (best decoder under eta_cc)
  XLM-R vs mBERT (encoder-only-does-not-suffice control)
"""
from pathlib import Path
import numpy as np

CORR = Path("data/results/transfer_correct")
OUT = Path("data/results/tables/eta_cc_bootstrap.csv")

SEED = 42
N_BOOT = 10_000
CHANCE = 1.0 / 6.0
KY_DIRS = ["en_ky", "ru_ky", "ky_en", "ky_ru"]
ENRU_DIRS = ["en_ru", "ru_en"]
MODELS = ["xlmr", "qwen3", "llama", "gemma4", "mistral", "mbert", "olmo2_7b", "qwen25_7b"]
DISP = {
    "xlmr": "XLM-R", "qwen3": "Qwen3", "llama": "Llama-3.1",
    "gemma4": "Gemma 4", "mistral": "Mistral", "mbert": "mBERT",
    "olmo2_7b": "OLMo-2", "qwen25_7b": "Qwen2.5",
}


def load_per_ex(model):
    d = np.load(CORR / f"{model}.npz", allow_pickle=True)
    ky = np.stack([d[k] for k in KY_DIRS], axis=1).mean(axis=1)     # per-example KY_avg
    enru = np.stack([d[k] for k in ENRU_DIRS], axis=1).mean(axis=1) # per-example EN<->RU
    return ky, enru


def eta_cc(ky_mean, enru_mean):
    denom = enru_mean - CHANCE
    if denom <= 0:
        return float("nan")
    return (ky_mean - CHANCE) / denom


def main():
    per_ex = {m: load_per_ex(m) for m in MODELS}
    n = len(per_ex["xlmr"][0])
    print(f"N per direction: {n}")

    # Point estimates
    print("\nPer-model point estimates:")
    point = {}
    for m in MODELS:
        ky, enru = per_ex[m]
        e = eta_cc(ky.mean(), enru.mean())
        point[m] = e
        print(f"  {DISP[m]:10s}: KY={ky.mean():.4f}  EN<->RU={enru.mean():.4f}  eta_cc={e:.4f}")

    # Paired bootstrap: resample indices, recompute eta_cc for each model
    rng = np.random.default_rng(SEED)
    boot = {m: np.empty(N_BOOT) for m in MODELS}
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for m in MODELS:
            ky, enru = per_ex[m]
            boot[m][i] = eta_cc(ky[idx].mean(), enru[idx].mean())

    rows = []
    print("\nBootstrap 95% CI for eta_cc:")
    for m in MODELS:
        ci_lo, ci_hi = np.percentile(boot[m], [2.5, 97.5])
        print(f"  {DISP[m]:10s}: {point[m]:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        rows.append({"model": DISP[m], "eta_cc": round(point[m], 4),
                     "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4)})

    # Pairwise: XLM-R vs each other model, paired bootstrap on the difference
    print("\nXLM-R vs others (paired bootstrap on delta eta_cc):")
    delta_rows = []
    for other in MODELS:
        if other == "xlmr":
            continue
        deltas = boot["xlmr"] - boot[other]
        d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
        real_delta = point["xlmr"] - point[other]
        # Two-sided p by sign flip
        if real_delta >= 0:
            p = 2 * min((deltas <= 0).mean(), 0.5)
        else:
            p = 2 * min((deltas >= 0).mean(), 0.5)
        star = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
        print(f"  XLM-R vs {DISP[other]:10s}: Δ={real_delta:+.4f}  95% CI [{d_lo:+.4f}, {d_hi:+.4f}]  p={p:.4f}  {star}")
        delta_rows.append({
            "comparison": f"XLM-R vs {DISP[other]}",
            "delta_eta_cc": round(real_delta, 4),
            "ci_lo": round(d_lo, 4), "ci_hi": round(d_hi, 4),
            "p_two_sided": round(p, 4),
        })

    import pandas as pd
    pd.DataFrame(rows + delta_rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
