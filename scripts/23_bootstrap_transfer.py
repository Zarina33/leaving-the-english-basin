"""
Step 23: Bootstrap CIs for cross-lingual transfer accuracy (Table 5).

For each (model, src→tgt) direction, trains a probe on src language
at the best transfer layer, evaluates on tgt, and bootstraps over
tgt examples to get 95% CI. Also computes pairwise model comparisons
per direction with Holm correction.

Output:
  data/results/tables/bootstrap_transfer.csv
  data/results/tables/bootstrap_transfer_pairwise.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from itertools import combinations

HS_DIRS = {
    "gemma4":  Path("data/processed"),
    "qwen3":   Path("data/processed/qwen3"),
    "mistral": Path("data/processed/mistral"),
    "llama":   Path("data/processed/llama"),
}
RES_DIRS = {
    "gemma4":  Path("data/results"),
    "qwen3":   Path("data/results/qwen3"),
    "mistral": Path("data/results/mistral"),
    "llama":   Path("data/results/llama"),
}
TABLE_DIR = Path("data/results/tables")

N_EPOCHS     = 300
LR_RATE      = 1e-2
N_BOOTSTRAP  = 10_000
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MODEL_LABELS = {
    "gemma4": "Gemma 4 E4B", "qwen3": "Qwen3-8B",
    "mistral": "Mistral-7B",  "llama": "Llama-3.1-8B",
}
DIRECTIONS = [("en","ru"),("en","ky"),("ru","en"),
              ("ru","ky"),("ky","en"),("ky","ru")]


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def get_transfer_predictions(X_tr, y_tr, X_te, y_te):
    """Train on src, return per-example correct/incorrect on tgt."""
    sc = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X_te), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.long, device=DEVICE)

    m = LinClf(Xtr.shape[1], len(np.unique(y_tr))).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    m.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(Xte).argmax(1)
    return (preds == yte).cpu().numpy()


def bootstrap_ci(correct, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(correct)
    accs = np.array([correct[rng.integers(0, n, size=n)].mean()
                     for _ in range(n_boot)])
    return correct.mean(), np.quantile(accs, 0.025), np.quantile(accs, 0.975)


def holm_bonferroni(p_values):
    n = len(p_values)
    sorted_idx = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(sorted_idx):
        adj_p = min(p_values[i] * (n - rank), 1.0)
        adj_p = max(adj_p, prev)
        adjusted[i] = adj_p
        prev = adj_p
    return adjusted


def main():
    models = [k for k in HS_DIRS if (HS_DIRS[k] / "hidden_states_en.npz").exists()]

    # Find best transfer layer per model per direction from existing results
    best_layers = {}
    for model in models:
        tr_df = pd.read_csv(RES_DIRS[model] / "transfer_results.csv")
        for src, tgt in DIRECTIONS:
            sub = tr_df[(tr_df.src == src) & (tr_df.tgt == tgt)]
            best_row = sub.loc[sub.transfer_acc.idxmax()]
            best_layers[(model, src, tgt)] = int(best_row.layer)

    # Get per-example predictions for each model × direction
    print("[1] Computing per-example transfer predictions...\n")
    predictions = {}

    for model in models:
        labels = np.load(HS_DIRS[model] / "labels.npy")
        hs = {lang: np.load(HS_DIRS[model] / f"hidden_states_{lang}.npz")["hidden_states"]
              for lang in ["en", "ru", "ky"]}

        for src, tgt in DIRECTIONS:
            layer = best_layers[(model, src, tgt)]
            X_tr = hs[src][:, layer, :]
            X_te = hs[tgt][:, layer, :]
            correct = get_transfer_predictions(X_tr, labels, X_te, labels)
            predictions[(model, src, tgt)] = correct
            print(f"  {MODEL_LABELS[model]:>16} {src.upper()}→{tgt.upper()} "
                  f"(L{layer}) acc={correct.mean():.3f}")

    # Bootstrap CIs
    print(f"\n[2] Bootstrap CIs ({N_BOOTSTRAP} resamples)...\n")
    ci_rows = []
    for model in models:
        for src, tgt in DIRECTIONS:
            correct = predictions[(model, src, tgt)]
            acc, lo, hi = bootstrap_ci(correct)
            ci_rows.append({
                "model": MODEL_LABELS[model],
                "direction": f"{src.upper()}→{tgt.upper()}",
                "accuracy": round(acc, 3),
                "ci_lower": round(lo, 3),
                "ci_upper": round(hi, 3),
            })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(TABLE_DIR / "bootstrap_transfer.csv", index=False)
    print(ci_df.to_string(index=False))

    # Pairwise comparisons per direction with Holm correction
    print(f"\n[3] Pairwise bootstrap tests + Holm...\n")
    pair_rows = []

    for src, tgt in DIRECTIONS:
        direction = f"{src.upper()}→{tgt.upper()}"
        model_pairs = list(combinations(models, 2))
        p_values = []
        info = []

        for m1, m2 in model_pairs:
            c1 = predictions[(m1, src, tgt)]
            c2 = predictions[(m2, src, tgt)]
            rng = np.random.default_rng(RANDOM_STATE)
            n = len(c1)
            obs_diff = c1.mean() - c2.mean()
            boot_diffs = np.array([
                c1[rng.integers(0, n, n)].mean() - c2[rng.integers(0, n, n)].mean()
                for _ in range(N_BOOTSTRAP)
            ])
            if obs_diff >= 0:
                p = min(np.mean(boot_diffs <= 0) * 2, 1.0)
            else:
                p = min(np.mean(boot_diffs >= 0) * 2, 1.0)
            p_values.append(p)
            info.append((m1, m2, obs_diff, p))

        adjusted = holm_bonferroni(p_values)

        for (m1, m2, diff, p_raw), p_adj in zip(info, adjusted):
            sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else
                  ("*" if p_adj < 0.05 else "ns"))
            pair_rows.append({
                "direction": direction,
                "model_a": MODEL_LABELS[m1],
                "model_b": MODEL_LABELS[m2],
                "diff": round(diff, 3),
                "p_holm": round(p_adj, 4),
                "sig": sig,
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(TABLE_DIR / "bootstrap_transfer_pairwise.csv", index=False)

    # Print significant results
    sig_df = pair_df[pair_df.sig != "ns"]
    print("Significant pairwise differences (Holm-corrected):")
    if len(sig_df):
        print(sig_df.to_string(index=False))
    else:
        print("  None")

    print("\nDone!")


if __name__ == "__main__":
    main()
