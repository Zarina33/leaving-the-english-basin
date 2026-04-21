"""
Step 25: Orthogonal Procrustes + RSA as alternatives to CKA.

Addresses reviewer concern: CKA alone may conflate script-level similarity
with genuine semantic alignment. Procrustes and RSA provide complementary
views:
  - Procrustes: finds optimal rotation between two spaces, reports residual
    distance. Low distance = similar geometry after alignment.
  - RSA (Representational Similarity Analysis): compares pairwise distance
    matrices. Correlation of distance matrices = structural similarity.

Output:
  data/results/tables/procrustes_results.csv
  data/results/tables/rsa_results.csv
  data/results/figures/fig23_procrustes_rsa.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

HS_DIRS = {
    "gemma4":  Path("data/processed"),
    "qwen3":   Path("data/processed/qwen3"),
    "mistral": Path("data/processed/mistral"),
    "llama":   Path("data/processed/llama"),
}
TABLE_DIR = Path("data/results/tables")
FIG_DIR   = Path("data/results/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "gemma4": "Gemma 4 E4B", "qwen3": "Qwen3-8B",
    "mistral": "Mistral-7B",  "llama": "Llama-3.1-8B",
}
PAIRS = [("en", "ru"), ("en", "ky"), ("ru", "ky")]
N_SUBSAMPLE = 500  # subsample for speed (Procrustes on full 1480 is slow)
RANDOM_STATE = 42


def center(X):
    return X - X.mean(axis=0)


def procrustes_distance(X, Y):
    """Orthogonal Procrustes: align Y to X, return normalized residual."""
    X, Y = center(X), center(Y)
    # Match dimensions if different
    d = min(X.shape[1], Y.shape[1])
    X, Y = X[:, :d], Y[:, :d]
    # Normalize
    X = X / np.linalg.norm(X, 'fro')
    Y = Y / np.linalg.norm(Y, 'fro')
    R, _ = orthogonal_procrustes(Y, X)
    Y_aligned = Y @ R
    dist = np.linalg.norm(X - Y_aligned, 'fro')
    # Convert to similarity: 1 - normalized_distance
    similarity = 1 - (dist ** 2) / 2
    return similarity


def rsa_correlation(X, Y):
    """RSA: Spearman correlation between pairwise distance matrices."""
    dX = pdist(X, metric='cosine')
    dY = pdist(Y, metric='cosine')
    rho, _ = spearmanr(dX, dY)
    return rho


def main():
    models = [k for k in HS_DIRS if (HS_DIRS[k] / "hidden_states_en.npz").exists()]
    rng = np.random.default_rng(RANDOM_STATE)

    proc_rows = []
    rsa_rows = []

    for model in models:
        hs = {lang: np.load(HS_DIRS[model] / f"hidden_states_{lang}.npz")["hidden_states"]
              for lang in ["en", "ru", "ky"]}
        n_layers = hs["en"].shape[1]

        # Subsample indices (same for all layers/pairs within a model)
        idx = rng.choice(hs["en"].shape[0], size=N_SUBSAMPLE, replace=False)

        print(f"\n{MODEL_LABELS[model]} ({n_layers} layers, {N_SUBSAMPLE} samples)")

        for layer in tqdm(range(n_layers), desc="  Layers"):
            for l1, l2 in PAIRS:
                X = hs[l1][idx, layer, :].astype(np.float32)
                Y = hs[l2][idx, layer, :].astype(np.float32)

                proc_sim = procrustes_distance(X, Y)
                rsa_rho  = rsa_correlation(X, Y)

                proc_rows.append({
                    "model": MODEL_LABELS[model], "layer": layer,
                    "pair": f"{l1.upper()}–{l2.upper()}",
                    "procrustes_sim": round(proc_sim, 4),
                })
                rsa_rows.append({
                    "model": MODEL_LABELS[model], "layer": layer,
                    "pair": f"{l1.upper()}–{l2.upper()}",
                    "rsa_rho": round(rsa_rho, 4),
                })

    proc_df = pd.DataFrame(proc_rows)
    rsa_df  = pd.DataFrame(rsa_rows)
    proc_df.to_csv(TABLE_DIR / "procrustes_results.csv", index=False)
    rsa_df.to_csv(TABLE_DIR / "rsa_results.csv", index=False)

    # ── Summary: peak values ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Peak Procrustes similarity per model × pair:")
    for model in models:
        for pair in ["EN–RU", "EN–KY", "RU–KY"]:
            sub = proc_df[(proc_df.model == MODEL_LABELS[model]) &
                          (proc_df.pair == pair)]
            best = sub.loc[sub.procrustes_sim.idxmax()]
            print(f"  {MODEL_LABELS[model]:>16} {pair}: "
                  f"{best.procrustes_sim:.3f} (L{int(best.layer)})")

    print("\nPeak RSA (Spearman ρ) per model × pair:")
    for model in models:
        for pair in ["EN–RU", "EN–KY", "RU–KY"]:
            sub = rsa_df[(rsa_df.model == MODEL_LABELS[model]) &
                         (rsa_df.pair == pair)]
            best = sub.loc[sub.rsa_rho.idxmax()]
            print(f"  {MODEL_LABELS[model]:>16} {pair}: "
                  f"{best.rsa_rho:.3f} (L{int(best.layer)})")

    # ── Figure: Procrustes + RSA side by side ─────────────────────────────────
    print("\n[Fig] Procrustes + RSA comparison...")
    pair_colors = {"EN–RU": "#E53935", "EN–KY": "#1E88E5", "RU–KY": "#43A047"}

    fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 8),
                             sharey='row')
    fig.suptitle("Cross-lingual Similarity: Procrustes (top) vs RSA (bottom)",
                 fontsize=13)

    for col, model in enumerate(models):
        for row, (df, metric, ylabel) in enumerate([
            (proc_df, "procrustes_sim", "Procrustes similarity"),
            (rsa_df, "rsa_rho", "RSA (Spearman ρ)"),
        ]):
            ax = axes[row, col]
            sub = df[df.model == MODEL_LABELS[model]]
            n = sub.layer.max() + 1
            for pair in ["EN–RU", "EN–KY", "RU–KY"]:
                psub = sub[sub.pair == pair].sort_values("layer")
                xpos = psub.layer / (n - 1)
                ax.plot(xpos, psub[metric], color=pair_colors[pair],
                        lw=2, label=pair)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel("Normalized Layer", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig23_procrustes_rsa.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
