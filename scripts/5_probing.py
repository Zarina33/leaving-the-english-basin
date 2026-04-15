"""
Step 5: Layer-wise probing classifier (CPU/sklearn — LEGACY).

NOTE: Superseded by scripts/8a_probing_gpu.py which uses GPU-accelerated
PyTorch logistic regression for consistency with all other experiments.
This script is kept for reference only. Use 8a_probing_gpu.py instead.

For each layer and each language, trains a logistic regression on hidden states
and evaluates emotion classification accuracy.

Input:  data/processed/hidden_states_{en,ru,ky}.npz
        data/processed/labels.npy
Output: data/results/probing_results.csv
        data/results/probing_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")
N_FOLDS = 5
MAX_ITER = 1000
RANDOM_STATE = 42

LANG_COLORS = {"en": "#2196F3", "ru": "#F44336", "ky": "#4CAF50"}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}


def probe_layer(X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold CV logistic regression on a single layer. Returns mean acc and F1."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE, C=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accs.append((preds == y_test).mean())
        f1s.append(f1_score(y_test, preds, average="macro"))

    return {"accuracy": np.mean(accs), "f1_macro": np.mean(f1s)}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.load(DATA_DIR / "labels.npy")
    emotions = np.load(DATA_DIR / "emotions.npy", allow_pickle=True)
    n_layers_total = None

    records = []

    for lang in ["en", "ru", "ky"]:
        print(f"\n{'='*50}")
        print(f"Probing {LANG_LABELS[lang]}...")
        hs = np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
        # hs shape: (N, n_layers+1, hidden_dim)
        n_layers_total = hs.shape[1]
        print(f"  Shape: {hs.shape}")

        for layer_idx in tqdm(range(n_layers_total), desc=f"  Layers"):
            X = hs[:, layer_idx, :]  # (N, hidden_dim)
            metrics = probe_layer(X, labels)
            records.append({
                "lang": lang,
                "layer": layer_idx,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "probing_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'probing_results.csv'}")

    # Print best layer per language
    print("\nBest layer per language:")
    for lang in ["en", "ru", "ky"]:
        sub = df[df["lang"] == lang]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {LANG_LABELS[lang]}: layer {int(best['layer'])} "
              f"acc={best['accuracy']:.3f} f1={best['f1_macro']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Emotion Probing Across Layers — Gemma 4 E4B\n"
                 "(GoEmotions: 6 emotions, 1480 parallel EN/RU/KY sentences)",
                 fontsize=13)

    for ax, metric, ylabel in zip(
        axes,
        ["accuracy", "f1_macro"],
        ["Accuracy", "Macro F1"]
    ):
        for lang in ["en", "ru", "ky"]:
            sub = df[df["lang"] == lang].sort_values("layer")
            ax.plot(sub["layer"], sub[metric],
                    label=LANG_LABELS[lang],
                    color=LANG_COLORS[lang],
                    linewidth=2.0,
                    alpha=0.85)

        # Baseline: random = 1/6
        ax.axhline(1/6, linestyle="--", color="gray", linewidth=1.2, label="Chance (1/6)")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    plt.tight_layout()
    plot_path = RESULTS_DIR / "probing_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
