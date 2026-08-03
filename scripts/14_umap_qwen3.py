"""
Step 14: UMAP/t-SNE visualization for Qwen3-8B + side-by-side comparison with Gemma 4.

Produces:
  Fig 18: UMAP clusters — Gemma4 vs Qwen3 side-by-side (best layer, 3 languages)
  Fig 19: t-SNE grid — Qwen3 best layer per language

Input:  data/processed/hidden_states_{en,ru,ky}.npz
        data/processed/qwen3/hidden_states_{en,ru,ky}.npz
        data/results/probing_detailed.csv
        data/results/qwen3/probing_detailed.csv
Output:
  data/results/figures/fig18_umap_compare.png
  data/results/figures/fig19_tsne_qwen3.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
    print("Using UMAP")
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False
    print("umap-learn not found, using t-SNE")

GEMMA_DIR   = Path("data/processed")
QWEN_DIR    = Path("data/processed/qwen3")
RESULTS_DIR = Path("data/results")
Q_RESULTS   = Path("data/results/qwen3")
FIG_DIR     = Path("data/results/figures")

EMOTION_COLORS = {
    "anger":    "#E53935",
    "disgust":  "#8E24AA",
    "fear":     "#FB8C00",
    "joy":      "#FDD835",
    "sadness":  "#1E88E5",
    "surprise": "#43A047",
}
EMOTIONS    = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}


def reduce_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=30, min_dist=0.1)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed,
                       perplexity=40, n_iter=1000)
    return reducer.fit_transform(Xs)


def scatter_emotions(ax, coords, labels, title, s=10, alpha=0.6):
    for emo_idx, emo in enumerate(EMOTIONS):
        mask = labels == emo_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=EMOTION_COLORS[emo], s=s, alpha=alpha,
                   label=emo.capitalize())
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.load(GEMMA_DIR / "labels.npy")

    gemma_hs = {lang: np.load(GEMMA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
                for lang in ["en", "ru", "ky"]}
    qwen_hs  = {lang: np.load(QWEN_DIR  / f"hidden_states_{lang}.npz")["hidden_states"]
                for lang in ["en", "ru", "ky"]}

    # Best layers per language — from the UNIFIED probing run (Table 3 of the
    # paper), so the figure matches the reported best layers.
    uni = pd.read_csv("data/results/tables/unified_probing.csv")
    def best_layer(model_name, lang):
        sub = uni[(uni["model"] == model_name) & (uni["lang"] == lang.upper())]
        return int(sub["layer"].iloc[0])

    gemma_best = {l: best_layer("Gemma 4 E4B", l) for l in ["en", "ru", "ky"]}
    qwen_best  = {l: best_layer("Qwen3-8B",  l) for l in ["en", "ru", "ky"]}

    print("Best layers — Gemma4:", gemma_best)
    print("Best layers — Qwen3: ", qwen_best)

    # ── Fig 18: Side-by-side UMAP (Gemma4 | Qwen3) × 3 languages ─────────────
    method = "UMAP" if HAS_UMAP else "t-SNE"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Emotion Clusters ({method}) at Best Probing Layer\n"
                 f"Top: Gemma 4 E4B  |  Bottom: Qwen3-8B",
                 fontsize=13)

    legend_patches = [Patch(color=EMOTION_COLORS[e], label=e.capitalize())
                      for e in EMOTIONS]

    for col, lang in enumerate(["en", "ru", "ky"]):
        # Gemma4 row
        gl = gemma_best[lang]
        print(f"\nGemma4 {lang.upper()} layer {gl}  →  reducing...")
        g_coords = reduce_2d(gemma_hs[lang][:, gl, :])
        scatter_emotions(axes[0, col], g_coords, labels,
                         title=f"{LANG_LABELS[lang]} (layer {gl})")

        # Qwen3 row
        ql = qwen_best[lang]
        print(f"Qwen3  {lang.upper()} layer {ql}  →  reducing...")
        q_coords = reduce_2d(qwen_hs[lang][:, ql, :])
        scatter_emotions(axes[1, col], q_coords, labels,
                         title=f"{LANG_LABELS[lang]} (layer {ql})")

    # Row labels
    for ax, label in zip(axes[:, 0], ["Gemma 4 E4B", "Qwen3-8B"]):
        ax.set_ylabel(label, fontsize=13, fontweight="bold", labelpad=10)

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(FIG_DIR / "fig18_umap_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig18_umap_compare.png")

    # ── Fig 19: t-SNE for Qwen3 — best / middle / worst layers ───────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle(f"Qwen3-8B: Emotion Clusters ({method}) at Early / Best / Late Layers",
                 fontsize=13)

    for row, lang in enumerate(["en", "ru", "ky"]):
        n_layers   = qwen_hs[lang].shape[1]
        best_l     = qwen_best[lang]
        early_l    = max(0, best_l // 2)
        late_l     = min(n_layers - 1, (best_l + n_layers) // 2)

        for col, (layer, tag) in enumerate([
            (early_l, "Early"),
            (best_l,  "Best"),
            (late_l,  "Late"),
        ]):
            print(f"Qwen3 {lang.upper()} {tag} layer {layer}  →  reducing...")
            coords = reduce_2d(qwen_hs[lang][:, layer, :])
            scatter_emotions(axes[row, col], coords, labels,
                             title=f"{LANG_LABELS[lang]} — {tag} (layer {layer})")

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIG_DIR / "fig19_tsne_qwen3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig19_tsne_qwen3.png")


if __name__ == "__main__":
    main()
