"""
Step 20: Bootstrap-based statistical testing for probing results.

Replaces paired t-tests (5 folds, 4 df) with per-example bootstrap (10k
resamples), computes 95% CIs, all pairwise model comparisons, and applies
Holm-Bonferroni correction for multiple comparisons.

Output:
  data/results/tables/bootstrap_probing.csv       — per-model CIs
  data/results/tables/bootstrap_pairwise.csv      — all pairwise tests + Holm
  data/results/tables/bootstrap_transfer.csv      — transfer CIs

Run AFTER scripts 18 (all hidden states + probing results exist).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
HS_DIRS = {
    "gemma4":  Path("data/processed"),
    "qwen3":   Path("data/processed/qwen3"),
    "mistral": Path("data/processed/mistral"),
    "llama":   Path("data/processed/llama"),
}
TABLE_DIR = Path("data/results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_FOLDS      = 5
N_EPOCHS     = 300
LR_RATE      = 1e-2
N_BOOTSTRAP  = 10_000
RANDOM_STATE = 42
CI_LEVEL     = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MODEL_LABELS = {
    "gemma4": "Gemma 4 E4B", "qwen3": "Qwen3-8B",
    "mistral": "Mistral-7B",  "llama": "Llama-3.1-8B",
}

# Best layers from probing results
BEST_LAYERS = {
    "gemma4":  {"en": 0,  "ru": 6,  "ky": 4},
    "qwen3":   {"en": 9,  "ru": 11, "ky": 27},
    "mistral": {"en": 5,  "ru": 9,  "ky": 2},
    "llama":   {"en": 1,  "ru": 7,  "ky": 3},
}


# ── Linear probe ──────────────────────────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def get_per_example_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Run 5-fold CV, return per-example binary correct/incorrect array.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    correct = np.zeros(len(y), dtype=bool)
    n_cls = len(np.unique(y))

    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]),
                           dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),
                           dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)

        m = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()

        m.eval()
        with torch.no_grad():
            preds = m(Xte).argmax(1)
        correct[te] = (preds == yte).cpu().numpy()

    return correct


def bootstrap_accuracy(correct: np.ndarray, n_boot: int = N_BOOTSTRAP,
                       rng: np.random.Generator = None) -> np.ndarray:
    """Bootstrap accuracy from per-example correct/incorrect vector."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    n = len(correct)
    boot_accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_accs[i] = correct[idx].mean()
    return boot_accs


def bootstrap_ci(boot_dist: np.ndarray, level: float = CI_LEVEL):
    """Return (lower, upper) CI from bootstrap distribution."""
    alpha = (1 - level) / 2
    return np.quantile(boot_dist, alpha), np.quantile(boot_dist, 1 - alpha)


def bootstrap_pairwise_test(correct_a: np.ndarray, correct_b: np.ndarray,
                            n_boot: int = N_BOOTSTRAP) -> float:
    """
    Bootstrap test for difference in accuracy.
    Returns p-value: proportion of bootstrap samples where
    the observed difference is <= 0 (two-sided).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(correct_a)
    observed_diff = correct_a.mean() - correct_b.mean()

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()

    # Two-sided p-value
    if observed_diff >= 0:
        p = np.mean(boot_diffs <= 0) * 2
    else:
        p = np.mean(boot_diffs >= 0) * 2
    return min(p, 1.0)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    indices = list(range(n))
    sorted_idx = sorted(indices, key=lambda i: p_values[i])
    adjusted = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(sorted_idx):
        adj_p = p_values[i] * (n - rank)
        adj_p = max(adj_p, prev)  # enforce monotonicity
        adj_p = min(adj_p, 1.0)
        adjusted[i] = adj_p
        prev = adj_p
    return adjusted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    models = [k for k in HS_DIRS if (HS_DIRS[k] / "hidden_states_en.npz").exists()]
    print(f"Models: {models}")

    # ── Step 1: Get per-example predictions at best layers ────────────────────
    print("\n[1] Running probing at best layers (per-example predictions)...")
    predictions = {}   # {(model, lang): np.array of bool}

    for model in models:
        labels = np.load(HS_DIRS[model] / "labels.npy")
        for lang in ["en", "ru", "ky"]:
            layer = BEST_LAYERS[model][lang]
            hs = np.load(HS_DIRS[model] / f"hidden_states_{lang}.npz")["hidden_states"]
            X = hs[:, layer, :]
            print(f"  {MODEL_LABELS[model]:>16} / {lang.upper()} (layer {layer})...",
                  end=" ", flush=True)
            correct = get_per_example_predictions(X, labels)
            predictions[(model, lang)] = correct
            print(f"acc={correct.mean():.3f}")

    # ── Step 2: Bootstrap CIs ─────────────────────────────────────────────────
    print(f"\n[2] Bootstrap CIs ({N_BOOTSTRAP} resamples)...")
    rng = np.random.default_rng(RANDOM_STATE)
    ci_rows = []

    for model in models:
        for lang in ["en", "ru", "ky"]:
            correct = predictions[(model, lang)]
            boot = bootstrap_accuracy(correct, rng=rng)
            lo, hi = bootstrap_ci(boot)
            ci_rows.append({
                "model": MODEL_LABELS[model],
                "lang": lang.upper(),
                "layer": BEST_LAYERS[model][lang],
                "accuracy": round(correct.mean(), 3),
                "ci_lower": round(lo, 3),
                "ci_upper": round(hi, 3),
            })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(TABLE_DIR / "bootstrap_probing.csv", index=False)
    print("\nBootstrap CIs:")
    print(ci_df.to_string(index=False))

    # ── Step 3: All pairwise comparisons + Holm correction ────────────────────
    print(f"\n[3] Pairwise bootstrap tests + Holm correction...")
    pair_rows = []

    for lang in ["en", "ru", "ky"]:
        lang_pairs = list(combinations(models, 2))
        p_values = []
        pair_info = []

        for m1, m2 in lang_pairs:
            c1 = predictions[(m1, lang)]
            c2 = predictions[(m2, lang)]
            diff = c1.mean() - c2.mean()
            p = bootstrap_pairwise_test(c1, c2)
            p_values.append(p)
            pair_info.append((m1, m2, diff, p))

        # Holm correction within each language
        adjusted = holm_bonferroni(p_values)

        for (m1, m2, diff, p_raw), p_adj in zip(pair_info, adjusted):
            sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else
                  ("*" if p_adj < 0.05 else "ns"))
            pair_rows.append({
                "lang": lang.upper(),
                "model_a": MODEL_LABELS[m1],
                "model_b": MODEL_LABELS[m2],
                "diff": round(diff, 3),
                "p_raw": round(p_raw, 4),
                "p_holm": round(p_adj, 4),
                "sig": sig,
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(TABLE_DIR / "bootstrap_pairwise.csv", index=False)
    print("\nPairwise comparisons (Holm-corrected):")
    print(pair_df.to_string(index=False))

    # ── Step 4: Summary for paper ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (copy into Table 3 caption):")
    print("=" * 70)
    for lang in ["en", "ru", "ky"]:
        sig_pairs = pair_df[(pair_df.lang == lang.upper()) & (pair_df.sig != "ns")]
        if len(sig_pairs) == 0:
            print(f"  {lang.upper()}: no significant pairwise differences (Holm)")
        else:
            parts = []
            for _, row in sig_pairs.iterrows():
                a = row.model_a.split()[0]
                b = row.model_b.split()[0]
                parts.append(f"{a} vs {b} p={row.p_holm}{row.sig}")
            print(f"  {lang.upper()}: {'; '.join(parts)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
