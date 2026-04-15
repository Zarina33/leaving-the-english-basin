"""
Step 9: Cross-lingual transfer probing.

Train probe on language A, test on language B (zero-shot transfer).
This directly measures how language-agnostic emotion representations are.

Conditions:
  EN → RU, EN → KY, RU → KY  (and reverses)

Per each layer computes transfer accuracy.
Compared against within-language accuracy (upper bound)
and chance (lower bound).

Output:
  data/results/transfer_results.csv
  data/results/figures/fig8_transfer.png
  data/results/figures/fig9_transfer_gap.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
FIG_DIR     = Path("data/results/figures")

N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}

TRANSFER_STYLES = {
    ("en", "ru"): {"color": "#E53935", "ls": "-",  "label": "EN → RU"},
    ("en", "ky"): {"color": "#1E88E5", "ls": "-",  "label": "EN → KY"},
    ("ru", "en"): {"color": "#E53935", "ls": "--", "label": "RU → EN"},
    ("ru", "ky"): {"color": "#43A047", "ls": "-",  "label": "RU → KY"},
    ("ky", "en"): {"color": "#1E88E5", "ls": "--", "label": "KY → EN"},
    ("ky", "ru"): {"color": "#43A047", "ls": "--", "label": "KY → RU"},
}


# ── GPU probe: train on A, test on B ─────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def train_and_transfer(X_train: np.ndarray, y_train: np.ndarray,
                       X_test:  np.ndarray, y_test:  np.ndarray) -> float:
    """Train on X_train/y_train, evaluate on X_test/y_test."""
    sc  = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_train),
                       dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X_test),
                       dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y_test,  dtype=torch.long, device=DEVICE)

    n_cls = len(np.unique(y_train))
    model = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    ce    = nn.CrossEntropyLoss()

    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        ce(model(Xtr), ytr).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
    return (preds == yte).float().mean().item()


def within_language_acc(X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2) -> float:
    """Train/test split within same language (matched sample size)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n   = len(X)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    tr, te = idx[:split], idx[split:]
    return train_and_transfer(X[tr], y[tr], X[te], y[te])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.load(DATA_DIR / "labels.npy")
    hs = {lang: np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}
    n_layers = hs["en"].shape[1]
    print(f"Shape: {hs['en'].shape}\n")

    # Load within-language accuracies for reference
    probing_df = pd.read_csv(RESULTS_DIR / "probing_detailed.csv")

    records = []
    transfer_pairs = [("en","ru"), ("en","ky"), ("ru","en"),
                      ("ru","ky"), ("ky","en"), ("ky","ru")]

    for src, tgt in transfer_pairs:
        desc = f"{src.upper()} → {tgt.upper()}"
        for layer in tqdm(range(n_layers), desc=f"  {desc}"):
            X_src = hs[src][:, layer, :]
            X_tgt = hs[tgt][:, layer, :]

            acc = train_and_transfer(X_src, labels, X_tgt, labels)
            records.append({
                "src": src, "tgt": tgt,
                "layer": layer,
                "transfer_acc": acc,
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "transfer_results.csv", index=False)
    print("\nSaved transfer_results.csv")

    # Summary at best within-language layer
    print("\nTransfer accuracy at peak within-language layer:")
    for src, tgt in [("en","ru"), ("en","ky"), ("ru","ky")]:
        # Best layer for source language
        src_best = int(probing_df[probing_df["lang"]==src]
                       .loc[probing_df[probing_df["lang"]==src]["acc_mean"].idxmax(),
                            "layer"])
        row = df[(df["src"]==src) & (df["tgt"]==tgt) &
                 (df["layer"]==src_best)]
        ta  = row["transfer_acc"].values[0]
        wa  = probing_df[(probing_df["lang"]==src) &
                         (probing_df["layer"]==src_best)]["acc_mean"].values[0]
        print(f"  {src.upper()}→{tgt.upper()} at layer {src_best}: "
              f"transfer={ta:.3f}  within={wa:.3f}  "
              f"gap={wa-ta:.3f}")

    # ── Figure 8: Transfer accuracy curves ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Cross-lingual Transfer Probing — Gemma 4 E4B\n"
                 "Trained on source language, tested on target",
                 fontsize=13)

    panel_pairs = [
        ("en", ["ru", "ky"]),
        ("ru", ["en", "ky"]),
        ("ky", ["en", "ru"]),
    ]

    for ax, (src, tgts) in zip(axes, panel_pairs):
        # Within-language accuracy (upper bound)
        within = probing_df[probing_df["lang"]==src].sort_values("layer")
        ax.plot(within["layer"], within["acc_mean"],
                color=LANG_COLORS[src], linewidth=2.5,
                label=f"{LANG_LABELS[src]} → {LANG_LABELS[src]} (within)",
                zorder=3)
        ax.fill_between(within["layer"],
                        within["acc_mean"] - within["acc_std"],
                        within["acc_mean"] + within["acc_std"],
                        color=LANG_COLORS[src], alpha=0.1)

        # Transfer curves
        for tgt in tgts:
            sub = df[(df["src"]==src) & (df["tgt"]==tgt)].sort_values("layer")
            style = TRANSFER_STYLES[(src, tgt)]
            ax.plot(sub["layer"], sub["transfer_acc"],
                    color=style["color"], ls="--", linewidth=1.8,
                    label=f"{LANG_LABELS[src]} → {LANG_LABELS[tgt]}")

        ax.axhline(1/6, ls=":", color="gray", linewidth=1.2, label="Chance")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"Source: {LANG_LABELS[src]}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_transfer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig8_transfer.png")

    # ── Figure 9: Transfer gap (within - transfer) ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    gap_pairs = [("en","ru"), ("en","ky"), ("ru","ky")]
    gap_colors = {"en_ru": "#E53935", "en_ky": "#1E88E5", "ru_ky": "#43A047"}

    for src, tgt in gap_pairs:
        transfer_sub = df[(df["src"]==src) & (df["tgt"]==tgt)].sort_values("layer")
        within_sub   = probing_df[probing_df["lang"]==src].sort_values("layer")
        gap = within_sub["acc_mean"].values - transfer_sub["transfer_acc"].values
        key = f"{src}_{tgt}"
        ax.plot(transfer_sub["layer"], gap,
                color=gap_colors[key], linewidth=2,
                label=f"{src.upper()} → {tgt.upper()} gap")

    ax.axhline(0, ls="--", color="gray", linewidth=1.2,
               label="Zero gap (perfect transfer)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Within − Transfer Accuracy", fontsize=12)
    ax.set_title("Transfer Gap by Layer — Gemma 4 E4B\n"
                 "Smaller gap = more language-agnostic representations",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_transfer_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig9_transfer_gap.png")


if __name__ == "__main__":
    main()
