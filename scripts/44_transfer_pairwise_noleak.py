"""
Step 44: Leakage-free pairwise transfer comparisons (Table 10 / Appendix E),
computed from the EXACT per-example correctness vectors saved by scripts/40,
so Table 10 is guaranteed consistent with Table 5/14 (same predictions).

For each direction and each C(4,2)=6 decoder pair: Δ = acc_hi - acc_lo, with a
paired bootstrap test on per-example correctness (the four decoders share the
same held-out target TEST indices, so the vectors are paired) and
Holm-Bonferroni correction within each direction.

Requires scripts/40 to have been run first (writes
data/results/transfer_correct/<model>.npz).

Usage:
    python3 scripts/44_transfer_pairwise_noleak.py
Output:
    data/results/tables/transfer_pairwise_noleak.csv
"""

from pathlib import Path
from itertools import combinations
import numpy as np

CORR = Path("data/results/transfer_correct")
OUT = Path("data/results/tables/transfer_pairwise_noleak.csv")

SEED = 42
N_BOOT = 10_000
DECODERS = ["gemma4", "qwen3", "llama", "mistral"]  # Table 10 is decoder-only
DISP = {"gemma4": "Gemma 4", "qwen3": "Qwen3", "llama": "Llama", "mistral": "Mistral"}
DIRECTIONS = [("en", "ru"), ("en", "ky"), ("ru", "en"),
              ("ru", "ky"), ("ky", "en"), ("ky", "ru")]


def holm(pvals):
    order = np.argsort(pvals)
    n = len(pvals); adj = np.empty(n); prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, (n - rank) * pvals[i]); adj[i] = min(prev, 1.0)
    return adj


def main():
    import pandas as pd
    corr = {m: dict(np.load(CORR / f"{m}.npz")) for m in DECODERS}
    rng = np.random.default_rng(SEED)
    rows = []
    for src, tgt in DIRECTIONS:
        k = f"{src}_{tgt}"
        pairs = list(combinations(DECODERS, 2))
        raw = {}
        for a, b in pairs:
            ca, cb = corr[a][k], corr[b][k]
            diff = ca.mean() - cb.mean()
            N = len(ca)
            idxs = rng.integers(0, N, (N_BOOT, N))
            boots = ca[idxs].mean(1) - cb[idxs].mean(1)
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            raw[(a, b)] = (diff, min(max(p, 1.0 / N_BOOT), 1.0))
        padj = holm(np.array([raw[k2][1] for k2 in pairs]))
        for (a, b), pa in zip(pairs, padj):
            diff = raw[(a, b)][0]
            hi, lo = (a, b) if diff >= 0 else (b, a)
            rows.append({"direction": f"{src.upper()}->{tgt.upper()}",
                         "model_hi": DISP[hi], "model_lo": DISP[lo],
                         "delta": round(abs(diff), 3), "p_holm": round(float(pa), 4)})
    df = pd.DataFrame(rows).sort_values(["direction", "delta"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
