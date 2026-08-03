"""
Step 43: Unified probing run — single seed, single protocol — to eliminate the
cross-table number inconsistencies flagged by reviewers (M8).

Previously, Table 3 (bootstrap probing), Table 9 (selectivity), and Table 13
(8-model probing) were produced by independent runs with different CV seeds,
so the SAME quantity (best-layer accuracy) appeared with slightly different
values (differences up to ~0.018). This script computes, in ONE run with ONE
seed, everything those tables need, so they are internally consistent:

  - best-layer mean 5-fold CV accuracy per (model, lang)     [Table 3, 13]
  - 95% bootstrap CI over per-example predictions            [Table 3]
  - control-task accuracy (shuffled labels), selectivity     [Table 9]
  - 2-layer MLP accuracy and MLP-linear gap                  [Table 9]

Runs on saved hidden states (no model weights, CPU-friendly).

Usage:
    python3 scripts/43_unified_probing.py
Output:
    data/results/tables/unified_probing.csv
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

PROCESSED = Path("data/processed")
OUT = Path("data/results/tables/unified_probing.csv")

SEED = 42
N_FOLDS = 5
N_EPOCHS = 300
LR = 1e-2
WD = 1e-4
N_BOOT = 10_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5 main models (Table 3 / 9). Root = gemma4.
MODELS = {
    "Gemma 4 E4B":  PROCESSED,
    "Qwen3-8B":     PROCESSED / "qwen3",
    "Llama-3.1-8B": PROCESSED / "llama",
    "Mistral-7B":   PROCESSED / "mistral",
    "XLM-R":        PROCESSED / "xlmr",
}


class Lin(nn.Module):
    def __init__(self, d, k):
        super().__init__(); self.fc = nn.Linear(d, k)
    def forward(self, x): return self.fc(x)


class MLP(nn.Module):
    def __init__(self, d, k, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(h, k))
    def forward(self, x): return self.net(x)


def cv_predictions(X, y, make_model, seed=SEED):
    """Return per-example predictions from stratified K-fold CV (aligned to y)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        torch.manual_seed(seed)
        m = make_model(X.shape[1], len(np.unique(y))).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        ce = nn.CrossEntropyLoss()
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds[te] = m(Xte).argmax(1).cpu().numpy()
    return preds


def boot_ci(correct, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(correct)
    means = np.array([correct[rng.integers(0, N, N)].mean() for _ in range(N_BOOT)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    import pandas as pd
    rows = []
    for name, d in MODELS.items():
        if not (d / "hidden_states_en.npz").exists():
            print(f"[!] {name}: missing — skip"); continue
        labels = np.load(d / "labels.npy")
        print(f"\n=== {name} ===")
        for lang in ["en", "ru", "ky"]:
            hs = np.load(d / f"hidden_states_{lang}.npz")["hidden_states"]
            n_layers = hs.shape[1]
            # find best layer by linear CV accuracy
            best_layer, best_acc, best_preds = None, -1, None
            for L in range(n_layers):
                p = cv_predictions(hs[:, L, :], labels, Lin)
                a = (p == labels).mean()
                if a > best_acc:
                    best_acc, best_layer, best_preds = a, L, p
            correct = (best_preds == labels).astype(int)
            lo, hi = boot_ci(correct)
            # control task: shuffled labels at best layer
            rng = np.random.default_rng(SEED)
            y_shuf = labels.copy(); rng.shuffle(y_shuf)
            ctrl_preds = cv_predictions(hs[:, best_layer, :], y_shuf, Lin)
            ctrl_acc = (ctrl_preds == y_shuf).mean()
            # MLP at best layer
            mlp_preds = cv_predictions(hs[:, best_layer, :], labels, MLP)
            mlp_acc = (mlp_preds == labels).mean()
            rows.append({
                "model": name, "lang": lang.upper(), "layer": best_layer,
                "linear_acc": round(best_acc, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "control_acc": round(ctrl_acc, 4),
                "selectivity": round(best_acc - ctrl_acc, 4),
                "mlp_acc": round(mlp_acc, 4),
                "mlp_gap": round(mlp_acc - best_acc, 4),
            })
            print(f"  {lang.upper()} L{best_layer}: acc={best_acc:.3f} "
                  f"[{lo:.3f},{hi:.3f}] ctrl={ctrl_acc:.3f} "
                  f"sel={best_acc-ctrl_acc:.3f} mlp={mlp_acc:.3f}")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
