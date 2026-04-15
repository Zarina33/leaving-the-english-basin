"""
Step 6: Cross-lingual representation similarity via CKA and cosine similarity.

CKA (Centered Kernel Alignment) measures how similar two representation spaces are,
regardless of rotation/scale. Values: 0 = completely different, 1 = identical.

Per layer computes:
  - CKA(EN, RU), CKA(EN, KY), CKA(RU, KY)
  - Mean cosine similarity between parallel sentence pairs

Input:  data/processed/hidden_states_{en,ru,ky}.npz
Output: data/results/cka_results.csv
        data/results/cka_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")


# ── CKA ──────────────────────────────────────────────────────────────────────

def center(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices X, Y of shape (N, d).
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    return float(num / (denom + 1e-10))


# ── Cosine similarity ─────────────────────────────────────────────────────────

def mean_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean cosine similarity between paired rows of X and Y."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return float((X_norm * Y_norm).sum(axis=1).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hidden states...")
    hs = {}
    for lang in ["en", "ru", "ky"]:
        hs[lang] = np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
        print(f"  {lang.upper()}: {hs[lang].shape}")

    n_layers = hs["en"].shape[1]
    pairs = [("en", "ru"), ("en", "ky"), ("ru", "ky")]

    records = []
    for layer_idx in tqdm(range(n_layers), desc="Computing CKA per layer"):
        row = {"layer": layer_idx}
        for l1, l2 in pairs:
            X = hs[l1][:, layer_idx, :].astype(np.float32)
            Y = hs[l2][:, layer_idx, :].astype(np.float32)
            row[f"cka_{l1}_{l2}"]    = linear_cka(X, Y)
            row[f"cosine_{l1}_{l2}"] = mean_cosine_similarity(X, Y)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "cka_results.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'cka_results.csv'}")

    # Summary
    print("\nPeak CKA per pair:")
    for l1, l2 in pairs:
        col = f"cka_{l1}_{l2}"
        best_layer = df[col].idxmax()
        print(f"  {l1.upper()}↔{l2.upper()}: max={df[col].max():.3f} at layer {best_layer}  "
              f"(layer 0: {df[col].iloc[0]:.3f})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    PAIR_STYLES = {
        ("en", "ru"): {"color": "#E53935", "label": "EN ↔ RU"},
        ("en", "ky"): {"color": "#1E88E5", "label": "EN ↔ KY"},
        ("ru", "ky"): {"color": "#43A047", "label": "RU ↔ KY"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-lingual Representation Similarity Across Layers — Gemma 4 E4B",
                 fontsize=13)

    for ax, metric, ylabel, title in zip(
        axes,
        ["cka", "cosine"],
        ["Linear CKA", "Mean Cosine Similarity"],
        ["CKA (higher = more similar)", "Cosine Similarity (parallel pairs)"],
    ):
        for (l1, l2), style in PAIR_STYLES.items():
            col = f"{metric}_{l1}_{l2}"
            ax.plot(df["layer"], df[col],
                    label=style["label"],
                    color=style["color"],
                    linewidth=2.0,
                    alpha=0.85)

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cka_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
