"""
Step 10: Per-emotion per-layer analysis.

For each emotion, language, and layer — computes one-vs-rest accuracy.
Shows at which layer each specific emotion becomes linearly separable.

Questions answered:
  - Does joy emerge earlier than anger?
  - Is fear consistently easy across languages and layers?
  - At which layer does each emotion "crystallize"?

Output:
  data/results/per_emotion_layers.csv
  data/results/figures/fig10_emotion_emergence.png  — heatmap layer × emotion
  data/results/figures/fig11_emotion_curves.png     — per-emotion curves per language
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

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EMOTION_COLORS = {
    "anger":    "#E53935",
    "disgust":  "#8E24AA",
    "fear":     "#FB8C00",
    "joy":      "#FDD835",
    "sadness":  "#1E88E5",
    "surprise": "#43A047",
}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}


# ── GPU one-vs-rest binary probe ──────────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_ovr_accuracy(X: np.ndarray, y_binary: np.ndarray) -> float:
    """Binary one-vs-rest GPU probe. Returns accuracy."""
    rng   = np.random.default_rng(RANDOM_STATE)
    n     = len(X)
    idx   = rng.permutation(n)
    split = int(n * 0.8)
    tr, te = idx[:split], idx[split:]

    sc  = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X[tr]),
                       dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X[te]),
                       dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_binary[tr], dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y_binary[te], dtype=torch.long, device=DEVICE)

    model = LinClf(Xtr.shape[1], 2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-4)

    # Class weights to handle imbalance (1 emotion vs 5 others)
    n_pos = int(ytr.sum().item())
    n_neg = len(ytr) - n_pos
    w     = torch.tensor([1.0, n_neg / max(n_pos, 1)],
                         dtype=torch.float32, device=DEVICE)
    ce    = nn.CrossEntropyLoss(weight=w)

    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        ce(model(Xtr), ytr).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
    return (preds == yte).float().mean().item()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.load(DATA_DIR / "labels.npy")
    hs = {lang: np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}
    n_layers = hs["en"].shape[1]
    print(f"Shape: {hs['en'].shape}\n")

    records = []
    for lang in ["en", "ru", "ky"]:
        for emotion_idx, emotion in enumerate(EMOTIONS):
            y_binary = (labels == emotion_idx).astype(int)
            desc = f"{lang.upper()} / {emotion}"
            for layer in tqdm(range(n_layers), desc=f"  {desc}", leave=False):
                X   = hs[lang][:, layer, :]
                acc = gpu_ovr_accuracy(X, y_binary)
                records.append({
                    "lang": lang, "emotion": emotion,
                    "layer": layer, "ovr_acc": acc,
                })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "per_emotion_layers.csv", index=False)
    print("Saved per_emotion_layers.csv\n")

    # Summary: best layer per emotion per language
    print("Best layer per emotion (one-vs-rest accuracy):")
    print(f"{'Emotion':<10} {'EN layer':>10} {'EN acc':>8} "
          f"{'RU layer':>10} {'RU acc':>8} {'KY layer':>10} {'KY acc':>8}")
    for emotion in EMOTIONS:
        row = []
        for lang in ["en", "ru", "ky"]:
            sub  = df[(df.lang==lang) & (df.emotion==emotion)]
            best = sub.loc[sub.ovr_acc.idxmax()]
            row += [int(best.layer), best.ovr_acc]
        print(f"{emotion:<10} {row[0]:>10} {row[1]:>8.3f} "
              f"{row[2]:>10} {row[3]:>8.3f} {row[4]:>10} {row[5]:>8.3f}")

    # ── Fig 10: Heatmap layer × emotion per language ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-emotion One-vs-Rest Accuracy by Layer — Gemma 4 E4B",
                 fontsize=13)

    for ax, lang in zip(axes, ["en", "ru", "ky"]):
        sub = df[df.lang==lang].pivot(index="emotion", columns="layer",
                                      values="ovr_acc")
        # Sort emotions by the layer at which they peak
        peak_layers = sub.idxmax(axis=1).sort_values()
        sub = sub.loc[peak_layers.index]

        im = ax.imshow(sub.values, cmap="YlOrRd", aspect="auto",
                       vmin=0.5, vmax=1.0)
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_yticklabels([e.capitalize() for e in sub.index], fontsize=10)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_title(LANG_LABELS[lang], fontsize=12)

        # Mark best layer per emotion
        for i, emotion in enumerate(sub.index):
            best_l = int(sub.loc[emotion].idxmax())
            ax.axvline(best_l, color="white", alpha=0.2, linewidth=0.5)
            ax.plot(best_l, i, "w*", markersize=8)

        plt.colorbar(im, ax=ax, label="OvR Accuracy")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_emotion_emergence.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig10_emotion_emergence.png")

    # ── Fig 11: Per-emotion curves (one subplot per emotion) ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Per-emotion Probing Curves Across Languages — Gemma 4 E4B\n"
                 "One-vs-rest accuracy", fontsize=13)

    LANG_STYLES = {
        "en": {"color": "#1565C0", "ls": "-",  "lw": 2.2},
        "ru": {"color": "#C62828", "ls": "--", "lw": 1.8},
        "ky": {"color": "#2E7D32", "ls": ":",  "lw": 1.8},
    }

    for ax, emotion in zip(axes.flatten(), EMOTIONS):
        for lang in ["en", "ru", "ky"]:
            sub = df[(df.lang==lang) & (df.emotion==emotion)].sort_values("layer")
            s   = LANG_STYLES[lang]
            ax.plot(sub["layer"], sub["ovr_acc"],
                    color=s["color"], ls=s["ls"], lw=s["lw"],
                    label=LANG_LABELS[lang])

        ax.axhline(0.5, ls=":", color="gray", linewidth=1,
                   label="Chance (binary)")
        ax.set_title(emotion.capitalize(), fontsize=12,
                     color=EMOTION_COLORS[emotion], fontweight="bold")
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("OvR Accuracy", fontsize=9)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_emotion_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig11_emotion_curves.png")


if __name__ == "__main__":
    main()
