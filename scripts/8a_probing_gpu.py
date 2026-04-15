"""
Step 8a: Re-run layer-wise probing on GPU for methodological consistency.

Overwrites data/results/probing_detailed.csv with GPU-based results
so all experiments use the same classifier (PyTorch logistic regression).

Output:
  data/results/probing_detailed.csv  — per-fold acc + mean/std (GPU)
  data/results/figures/fig4_probing_ci.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
FIG_DIR     = Path("data/results/figures")

N_FOLDS      = 5
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}


# ── GPU logistic regression ───────────────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_probe(X: np.ndarray, y: np.ndarray) -> list[float]:
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    accs  = []
    n_cls = len(np.unique(y))

    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]),
                           dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),
                           dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)

        model = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR_RATE,
                                 weight_decay=1e-4)
        ce    = nn.CrossEntropyLoss()

        model.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad()
            ce(model(Xtr), ytr).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xte).argmax(1)
        accs.append((preds == yte).float().mean().item())

    return accs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.load(DATA_DIR / "labels.npy")
    hs = {lang: np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}
    print(f"Shape: {hs['en'].shape}\n")

    records = []
    for lang in ["en", "ru", "ky"]:
        n_layers = hs[lang].shape[1]
        for layer in tqdm(range(n_layers), desc=f"{lang.upper()}"):
            fold_accs = gpu_probe(hs[lang][:, layer, :], labels)
            records.append({
                "lang":     lang,
                "layer":    layer,
                "acc_mean": np.mean(fold_accs),
                "acc_std":  np.std(fold_accs),
                **{f"fold_{i}": a for i, a in enumerate(fold_accs)},
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "probing_detailed.csv", index=False)
    print("\nSaved probing_detailed.csv")

    # Summary
    print("\nBest layer per language:")
    for lang in ["en", "ru", "ky"]:
        sub  = df[df["lang"] == lang]
        best = sub.loc[sub["acc_mean"].idxmax()]
        print(f"  {lang.upper()}: layer {int(best.layer):2d}  "
              f"acc={best.acc_mean:.3f} ± {best.acc_std:.3f}")

    # Plot with CI
    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in ["en", "ru", "ky"]:
        sub = df[df["lang"] == lang].sort_values("layer")
        m, s = sub["acc_mean"].values, sub["acc_std"].values
        ax.plot(sub["layer"], m, label=LANG_LABELS[lang],
                color=LANG_COLORS[lang], linewidth=2)
        ax.fill_between(sub["layer"], m - s, m + s,
                        color=LANG_COLORS[lang], alpha=0.15)
    ax.axhline(1/6, linestyle="--", color="gray", linewidth=1.2, label="Chance (1/6)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy (mean ± std, 5-fold CV)", fontsize=12)
    ax.set_title("Emotion Probing by Layer — Gemma 4 E4B\n"
                 "GPU logistic regression, shaded = ±1 std", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_probing_ci.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_probing_ci.png")


if __name__ == "__main__":
    main()
