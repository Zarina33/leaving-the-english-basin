"""
Step 46: Paired bootstrap for XLM-R vs Qwen3 vs mBERT vs OLMo-2 on KY_avg
(average accuracy across the four KY-involving transfer directions), using
the exact per-example correctness vectors saved by scripts/40.

Addresses reviewer critique: §6.3 claims XLM-R (.384) > Qwen3 (.349) as
'highest absolute KY transfer' but the .035 margin had no significance test
in the paper. This script computes the paired bootstrap on the same held-out
test indices (296 per direction) used for Table 5.
"""
from pathlib import Path
from itertools import combinations
import numpy as np

CORR = Path("data/results/transfer_correct")
OUT = Path("data/results/tables/xlmr_vs_others_ky.csv")

SEED = 42
N_BOOT = 10_000
DIRECTIONS_KY = ["en_ky", "ru_ky", "ky_en", "ky_ru"]
MODELS = ["xlmr", "qwen3", "llama", "gemma4", "mistral", "mbert", "olmo2_7b", "qwen25_7b"]
DISP = {
    "xlmr": "XLM-R", "qwen3": "Qwen3", "llama": "Llama-3.1",
    "gemma4": "Gemma 4", "mistral": "Mistral", "mbert": "mBERT",
    "olmo2_7b": "OLMo-2", "qwen25_7b": "Qwen2.5",
}


def load_ky_avg_correct(model):
    d = np.load(CORR / f"{model}.npz", allow_pickle=True)
    # Stack the 4 KY-involving direction correctness vectors -> (296, 4)
    # KY_avg per test example = mean across 4 directions
    stack = np.stack([d[k] for k in DIRECTIONS_KY], axis=1)  # (N, 4)
    return stack.mean(axis=1)  # per-example KY_avg accuracy in [0,1]


def main():
    rng = np.random.default_rng(SEED)
    correct = {m: load_ky_avg_correct(m) for m in MODELS}
    n = len(correct["xlmr"])
    print(f"N per direction: {n}")
    print(f"Per-model KY_avg accuracy (mean of per-example values):")
    for m in MODELS:
        print(f"  {DISP[m]:10s}: {correct[m].mean():.4f}")

    # Paired bootstrap: resample test indices with replacement
    rows = []
    print("\nPairwise (XLM-R first, all pairs):")
    all_pairs = [("xlmr", other) for other in MODELS if other != "xlmr"] + \
                list(combinations([m for m in MODELS if m != "xlmr"], 2))
    for a, b in all_pairs:
        ca, cb = correct[a], correct[b]
        real_diff = ca.mean() - cb.mean()
        boot_diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            boot_diffs[i] = ca[idx].mean() - cb[idx].mean()
        # Two-sided p: proportion of bootstrap diffs with opposite sign of real
        if real_diff >= 0:
            p = 2 * min((boot_diffs <= 0).mean(), 0.5)
        else:
            p = 2 * min((boot_diffs >= 0).mean(), 0.5)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        rows.append({
            "model_a": DISP[a], "model_b": DISP[b],
            "acc_a": round(ca.mean(), 4), "acc_b": round(cb.mean(), 4),
            "delta": round(real_diff, 4),
            "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
            "p_two_sided": round(p, 4),
        })
        star = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
        print(f"  {DISP[a]:10s} vs {DISP[b]:10s}: Δ={real_diff:+.4f}  "
              f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p:.4f}  {star}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
