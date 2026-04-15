"""
Step 7: Publication-quality visualizations.

Produces:
  1. UMAP of emotion clusters at best/worst layers for each language
  2. Per-emotion F1 heatmap (emotion × language)
  3. Combined summary figure (probing + CKA) for paper

Input:  data/processed/hidden_states_{en,ru,ky}.npz
        data/processed/labels.npy, emotions.npy
        data/results/probing_results.csv
        data/results/cka_results.csv
Output: data/results/figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False
    print("umap-learn not found, using t-SNE instead")

DATA_DIR   = Path("data/processed")
RESULTS_DIR = Path("data/results")
FIG_DIR    = Path("data/results/figures")

EMOTION_COLORS = {
    "anger":    "#E53935",
    "disgust":  "#8E24AA",
    "fear":     "#FB8C00",
    "joy":      "#FDD835",
    "sadness":  "#1E88E5",
    "surprise": "#43A047",
}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}
RANDOM_STATE = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def reduce_dim(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    X = StandardScaler().fit_transform(X)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE,
                            n_neighbors=30, min_dist=0.1)
    else:
        reducer = TSNE(n_components=n_components, random_state=RANDOM_STATE,
                       perplexity=40, n_iter=1000)
    return reducer.fit_transform(X)


def per_emotion_f1(X: np.ndarray, y: np.ndarray,
                   emotions: list[str]) -> dict[str, float]:
    """5-fold CV → per-emotion F1."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_preds, all_true = [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_tr, y[train_idx])
        all_preds.extend(clf.predict(X_te))
        all_true.extend(y[test_idx])
    f1s = f1_score(all_true, all_preds, average=None, labels=list(range(len(emotions))))
    return {e: f1s[i] for i, e in enumerate(emotions)}


# ── Figure 1: UMAP emotion clusters ──────────────────────────────────────────

def fig_umap(hs: dict, labels: np.ndarray, emotions: list,
             probing_df: pd.DataFrame):
    method = "UMAP" if HAS_UMAP else "t-SNE"
    print(f"\n[Fig 1] {method} emotion clusters...")

    # Best and worst layer per language (from probing)
    special_layers = {}
    for lang in ["en", "ru", "ky"]:
        sub = probing_df[probing_df["lang"] == lang]
        best  = int(sub.loc[sub["accuracy"].idxmax(), "layer"])
        worst = int(sub.loc[sub["accuracy"].idxmin(), "layer"])
        special_layers[lang] = {"best": best, "worst": worst}
        print(f"  {lang.upper()}: best=layer{best}, worst=layer{worst}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Emotion Representations ({method}) in Gemma 4 E4B\n"
                 "Top row: best probing layer | Bottom row: worst probing layer",
                 fontsize=13)

    for col, lang in enumerate(["en", "ru", "ky"]):
        for row, which in enumerate(["best", "worst"]):
            layer = special_layers[lang][which]
            ax = axes[row][col]

            X = hs[lang][:, layer, :]
            emb = reduce_dim(X)

            for i, emotion in enumerate(emotions):
                mask = labels == i
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=EMOTION_COLORS[emotion], label=emotion,
                           s=12, alpha=0.65, linewidths=0)

            title = f"{LANG_LABELS[lang]}\nLayer {layer} ({which})"
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

            if col == 0 and row == 0:
                legend_elements = [Patch(facecolor=EMOTION_COLORS[e], label=e)
                                   for e in emotions]
                ax.legend(handles=legend_elements, fontsize=7,
                          loc="lower left", framealpha=0.8)

    plt.tight_layout()
    path = FIG_DIR / "fig1_umap_clusters.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 2: Per-emotion F1 heatmap ─────────────────────────────────────────

def fig_emotion_heatmap(hs: dict, labels: np.ndarray, emotions: list,
                        probing_df: pd.DataFrame):
    print("\n[Fig 2] Per-emotion F1 heatmap...")

    data = {}
    for lang in ["en", "ru", "ky"]:
        sub = probing_df[probing_df["lang"] == lang]
        best_layer = int(sub.loc[sub["accuracy"].idxmax(), "layer"])
        X = hs[lang][:, best_layer, :]
        data[lang] = per_emotion_f1(X, labels, list(emotions))
        print(f"  {lang.upper()} (layer {best_layer}): "
              + ", ".join(f"{e}={v:.2f}" for e, v in data[lang].items()))

    matrix = np.array([[data[lang][e] for lang in ["en", "ru", "ky"]]
                        for e in emotions])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1 Score")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["English", "Russian", "Kyrgyz"], fontsize=11)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels([e.capitalize() for e in emotions], fontsize=11)

    for i, e in enumerate(emotions):
        for j, lang in enumerate(["en", "ru", "ky"]):
            val = data[lang][e]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    ax.set_title("Per-emotion F1 at Best Probing Layer — Gemma 4 E4B", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / "fig2_emotion_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 3: Combined probing + CKA (main paper figure) ─────────────────────

def fig_combined(probing_df: pd.DataFrame, cka_df: pd.DataFrame):
    print("\n[Fig 3] Combined probing + CKA figure...")

    LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}
    PAIR_COLORS = {
        "en_ru": "#E53935",
        "en_ky": "#1E88E5",
        "ru_ky": "#43A047",
    }

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: Probing accuracy
    ax1 = fig.add_subplot(gs[0])
    for lang in ["en", "ru", "ky"]:
        sub = probing_df[probing_df["lang"] == lang].sort_values("layer")
        ax1.plot(sub["layer"], sub["accuracy"],
                 label=LANG_LABELS[lang],
                 color=LANG_COLORS[lang], linewidth=2)
    ax1.axhline(1/6, linestyle="--", color="gray", linewidth=1.2, label="Chance")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Probing Accuracy", fontsize=11)
    ax1.set_title("(A) Emotion Probing by Layer", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # Panel B: CKA
    ax2 = fig.add_subplot(gs[1])
    for pair, color in PAIR_COLORS.items():
        l1, l2 = pair.split("_")
        label = f"{l1.upper()} ↔ {l2.upper()}"
        ax2.plot(cka_df["layer"], cka_df[f"cka_{pair}"],
                 label=label, color=color, linewidth=2)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Linear CKA", fontsize=11)
    ax2.set_title("(B) Cross-lingual Similarity (CKA)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # Panel C: Cosine similarity
    ax3 = fig.add_subplot(gs[2])
    for pair, color in PAIR_COLORS.items():
        l1, l2 = pair.split("_")
        label = f"{l1.upper()} ↔ {l2.upper()}"
        ax3.plot(cka_df["layer"], cka_df[f"cosine_{pair}"],
                 label=label, color=color, linewidth=2)
    ax3.set_xlabel("Layer", fontsize=11)
    ax3.set_ylabel("Cosine Similarity", fontsize=11)
    ax3.set_title("(C) Cosine Similarity (Parallel Pairs)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(10))

    fig.suptitle(
        "Emotion Representations Across Layers in Gemma 4 E4B  "
        "(EN / RU / KY, n=1480)",
        fontsize=13, y=1.02
    )

    plt.tight_layout()
    path = FIG_DIR / "fig3_combined.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    labels  = np.load(DATA_DIR / "labels.npy")
    emotions = list(np.load(DATA_DIR / "emotions.npy", allow_pickle=True))
    probing_df = pd.read_csv(RESULTS_DIR / "probing_results.csv")
    cka_df     = pd.read_csv(RESULTS_DIR / "cka_results.csv")

    hs = {}
    for lang in ["en", "ru", "ky"]:
        hs[lang] = np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]

    fig_umap(hs, labels, emotions, probing_df)
    fig_emotion_heatmap(hs, labels, emotions, probing_df)
    fig_combined(probing_df, cka_df)

    print("\nAll figures saved to", FIG_DIR)
    print("\nFiles:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
