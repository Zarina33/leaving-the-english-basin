"""
Step 27: Compute silhouette scores per layer for all 5 models.
Replaces single-model Gemma 4 silhouette figure with cross-model panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
FIG_DIR     = Path("data/results/figures")
CORPUS      = Path("data/translated/parallel_corpus_clean.csv")

RANDOM_STATE = 42
N_SUBSAMPLE  = 500
LAYER_STEP   = 2

LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}

MODELS = [
    ("Gemma 4 E4B", DATA_DIR),
    ("Qwen3-8B",    DATA_DIR / "qwen3"),
    ("Llama-3.1-8B", DATA_DIR / "llama"),
    ("Mistral-7B",  DATA_DIR / "mistral"),
    ("XLM-R-large", DATA_DIR / "xlmr"),
]

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
emo2id   = {e: i for i, e in enumerate(EMOTIONS)}


def load_labels() -> np.ndarray:
    df = pd.read_csv(CORPUS)
    return np.array([emo2id[e] for e in df["emotion"]])


def compute_silhouette(model_dir: Path, labels: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    records = []
    for lang in ["en", "ru", "ky"]:
        path = model_dir / f"hidden_states_{lang}.npz"
        if not path.exists():
            print(f"  [skip] {path}")
            continue
        hs = np.load(path)["hidden_states"]
        n_layers = hs.shape[1]
        for layer in tqdm(range(0, n_layers, LAYER_STEP), desc=f"  {lang.upper()}", leave=False):
            X = StandardScaler().fit_transform(hs[:, layer, :].astype(np.float32))
            idx = rng.choice(len(X), min(N_SUBSAMPLE, len(X)), replace=False)
            score = silhouette_score(X[idx], labels[idx], metric="cosine")
            records.append({"lang": lang, "layer": layer, "silhouette": score})
    return pd.DataFrame(records)


def main():
    labels = load_labels()
    print(f"Loaded {len(labels)} labels")

    all_dfs = {}
    for name, mdir in MODELS:
        print(f"\n[{name}]  dir={mdir}")
        df = compute_silhouette(mdir, labels)
        df["model"] = name
        all_dfs[name] = df

    full = pd.concat(all_dfs.values(), ignore_index=True)
    full.to_csv(RESULTS_DIR / "silhouette_all_models.csv", index=False)
    print(f"\n→ silhouette_all_models.csv  ({len(full)} rows)")

    # 5-panel figure (one per model). Normalised x-axis [0,1] for cross-model
    # comparison; horizontal y=0 reference line.
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for ax, (name, _) in zip(axes, MODELS):
        df = all_dfs[name]
        if df.empty:
            ax.set_visible(False)
            continue
        n_layers = df["layer"].max() + 1
        for lang in ["en", "ru", "ky"]:
            sub = df[df["lang"] == lang].sort_values("layer")
            xs = sub["layer"] / max(n_layers - 1, 1)
            ax.plot(xs, sub["silhouette"],
                    label=LANG_LABELS[lang], color=LANG_COLORS[lang],
                    linewidth=1.6, marker="o", markersize=3)
        ax.axhline(0, ls="--", color="gray", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Normalized depth", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Silhouette score", fontsize=10)
    axes[-1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "fig24_silhouette_all_models.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ {out}")

    # Quick numeric summary
    print("\nMax silhouette per (model, lang):")
    print(full.groupby(["model", "lang"])["silhouette"].max().round(4))


if __name__ == "__main__":
    main()
