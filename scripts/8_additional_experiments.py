"""
Step 8: Additional experiments for journal-quality paper — Gemma 4 E4B.
All probing runs on GPU via PyTorch logistic regression.

NOTE: The probing CI results from this script were superseded by
scripts/8a_probing_gpu.py for methodological consistency.
The permutation test, confusion matrices, silhouette, anisotropy, and
paired t-test sections remain canonical for Gemma 4.
For Qwen3 statistical tests, see scripts/13_stats_qwen3.py.

1. Probing with confidence intervals  → superseded by 8a_probing_gpu.py
2. Permutation test (100 runs — fast on GPU)
3. Confusion matrices per language
4. Silhouette score per layer
5. Anisotropy analysis
6. Statistical significance (paired t-test across languages)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
FIG_DIR     = Path("data/results/figures")

N_FOLDS      = 5
N_PERMUT     = 100
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}
EMOTIONS    = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


# ── GPU logistic regression ───────────────────────────────────────────────────

class _LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_probe(X: np.ndarray, y: np.ndarray,
              return_preds: bool = False):
    """5-fold CV logistic regression on GPU. Returns per-fold accuracies."""
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)
    accs, all_pred, all_true = [], [], []
    n_cls = len(np.unique(y))

    for tr, te in skf.split(X, y):
        sc   = StandardScaler()
        Xtr  = torch.tensor(sc.fit_transform(X[tr]),
                             dtype=torch.float32, device=DEVICE)
        Xte  = torch.tensor(sc.transform(X[te]),
                             dtype=torch.float32, device=DEVICE)
        ytr  = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte  = torch.tensor(y[te], dtype=torch.long, device=DEVICE)

        model = _LinClf(Xtr.shape[1], n_cls).to(DEVICE)
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
        if return_preds:
            all_pred.extend(preds.cpu().numpy())
            all_true.extend(y[te])

    if return_preds:
        return accs, np.array(all_true), np.array(all_pred)
    return accs


# ── 1. Probing with CI ────────────────────────────────────────────────────────

def exp_probing_ci(hs: dict, labels: np.ndarray) -> pd.DataFrame:
    print("\n[1/6] Probing with confidence intervals (GPU)...")
    records = []
    for lang in ["en", "ru", "ky"]:
        for layer in tqdm(range(hs[lang].shape[1]), desc=f"  {lang.upper()}"):
            fold_accs = gpu_probe(hs[lang][:, layer, :], labels)
            records.append({
                "lang": lang, "layer": layer,
                "acc_mean": np.mean(fold_accs),
                "acc_std":  np.std(fold_accs),
                **{f"fold_{i}": a for i, a in enumerate(fold_accs)},
            })
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "probing_detailed.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in ["en", "ru", "ky"]:
        sub = df[df["lang"] == lang].sort_values("layer")
        m, s = sub["acc_mean"].values, sub["acc_std"].values
        ax.plot(sub["layer"], m, label=LANG_LABELS[lang],
                color=LANG_COLORS[lang], linewidth=2)
        ax.fill_between(sub["layer"], m - s, m + s,
                        color=LANG_COLORS[lang], alpha=0.15)
    ax.axhline(1/6, linestyle="--", color="gray", linewidth=1.2, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy (mean ± std, 5-fold)", fontsize=12)
    ax.set_title("Emotion Probing by Layer — Gemma 4 E4B", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_probing_ci.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → probing_detailed.csv + fig4_probing_ci.png")
    return df


# ── 2. Permutation test ───────────────────────────────────────────────────────

def exp_permutation(hs: dict, labels: np.ndarray) -> pd.DataFrame:
    print("\n[2/6] Permutation test (100 runs, GPU)...")
    rng     = np.random.default_rng(RANDOM_STATE)
    records = []
    sample_layers = list(range(0, hs["en"].shape[1], 5))

    for lang in ["en", "ru", "ky"]:
        for layer in tqdm(sample_layers, desc=f"  {lang.upper()}"):
            X = hs[lang][:, layer, :]
            perm_accs = [
                np.mean(gpu_probe(X, rng.permutation(labels)))
                for _ in range(N_PERMUT)
            ]
            records.append({
                "lang": lang, "layer": layer,
                "perm_mean": np.mean(perm_accs),
                "perm_std":  np.std(perm_accs),
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "permutation_baseline.csv", index=False)
    print(f"  → permutation_baseline.csv")
    print(f"  Avg perm accuracy: {df['perm_mean'].mean():.3f} (expected ≈ {1/6:.3f})")
    return df


# ── 3. Confusion matrices ─────────────────────────────────────────────────────

def exp_confusion(hs: dict, labels: np.ndarray,
                  probing_df: pd.DataFrame):
    print("\n[3/6] Confusion matrices per language...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Confusion Matrices at Best Probing Layer — Gemma 4 E4B",
                 fontsize=13)

    for ax, lang in zip(axes, ["en", "ru", "ky"]):
        sub = probing_df[probing_df["lang"] == lang]
        best_layer = int(sub.loc[sub["acc_mean"].idxmax(), "layer"])
        _, y_true, y_pred = gpu_probe(
            hs[lang][:, best_layer, :], labels, return_preds=True)

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        short = [e[:3].capitalize() for e in EMOTIONS]
        ax.set_xticks(range(6)); ax.set_xticklabels(short, fontsize=9)
        ax.set_yticks(range(6)); ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(f"{LANG_LABELS[lang]} (layer {best_layer})", fontsize=11)
        for i in range(6):
            for j in range(6):
                c = "white" if cm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color=c)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig5_confusion.png")


# ── 4. Silhouette ─────────────────────────────────────────────────────────────

def exp_silhouette(hs: dict, labels: np.ndarray) -> pd.DataFrame:
    print("\n[4/6] Silhouette score per layer...")
    records = []
    rng = np.random.default_rng(RANDOM_STATE)

    for lang in ["en", "ru", "ky"]:
        for layer in tqdm(range(0, hs[lang].shape[1], 2),
                          desc=f"  {lang.upper()}"):
            X = StandardScaler().fit_transform(
                hs[lang][:, layer, :].astype(np.float32))
            idx = rng.choice(len(X), min(500, len(X)), replace=False)
            score = silhouette_score(X[idx], labels[idx], metric="cosine")
            records.append({"lang": lang, "layer": layer, "silhouette": score})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "silhouette.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in ["en", "ru", "ky"]:
        sub = df[df["lang"] == lang].sort_values("layer")
        ax.plot(sub["layer"], sub["silhouette"],
                label=LANG_LABELS[lang], color=LANG_COLORS[lang],
                linewidth=2, marker="o", markersize=4)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1.2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Emotion Cluster Separability by Layer — Gemma 4 E4B", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → silhouette.csv + fig6_silhouette.png")
    return df


# ── 5. Anisotropy ─────────────────────────────────────────────────────────────

def exp_anisotropy(hs: dict) -> pd.DataFrame:
    print("\n[5/6] Anisotropy analysis...")
    rng     = np.random.default_rng(RANDOM_STATE)
    N_PAIRS = 1000
    records = []

    for lang in ["en", "ru", "ky"]:
        for layer in tqdm(range(hs[lang].shape[1]), desc=f"  {lang.upper()}"):
            X = hs[lang][:, layer, :].astype(np.float32)
            X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            i1  = rng.integers(0, len(X), N_PAIRS)
            i2  = rng.integers(0, len(X), N_PAIRS)
            mask = i1 != i2
            avg_cos = float((X_n[i1[mask]] * X_n[i2[mask]]).sum(axis=1).mean())
            records.append({"lang": lang, "layer": layer,
                             "avg_random_cosine": avg_cos})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "anisotropy.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for lang in ["en", "ru", "ky"]:
        sub = df[df["lang"] == lang].sort_values("layer")
        ax.plot(sub["layer"], sub["avg_random_cosine"],
                label=LANG_LABELS[lang], color=LANG_COLORS[lang], linewidth=2)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Avg Cosine (random pairs)", fontsize=12)
    ax.set_title("Representational Anisotropy by Layer — Gemma 4 E4B\n"
                 "High = anisotropic (vectors in narrow cone)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_anisotropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → anisotropy.csv + fig7_anisotropy.png")
    return df


# ── 6. Statistical significance ──────────────────────────────────────────────

def exp_significance(probing_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[6/6] Statistical significance (paired t-test)...")
    fold_cols = [f"fold_{i}" for i in range(N_FOLDS)]
    records   = []

    for layer in probing_df["layer"].unique():
        row = {"layer": layer}
        for l1, l2 in [("en", "ru"), ("en", "ky"), ("ru", "ky")]:
            a1 = probing_df[(probing_df["lang"] == l1) &
                            (probing_df["layer"] == layer)][fold_cols].values.flatten()
            a2 = probing_df[(probing_df["lang"] == l2) &
                            (probing_df["layer"] == layer)][fold_cols].values.flatten()
            t, p = stats.ttest_rel(a1, a2)
            row[f"tstat_{l1}_{l2}"] = round(t, 4)
            row[f"pval_{l1}_{l2}"]  = round(p, 6)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "significance.csv", index=False)
    alpha = 0.05
    print(f"  EN>RU significant (p<{alpha}): "
          f"{(df['pval_en_ru'] < alpha).sum()}/{len(df)} layers")
    print(f"  EN>KY significant (p<{alpha}): "
          f"{(df['pval_en_ky'] < alpha).sum()}/{len(df)} layers")
    print(f"  RU>KY significant (p<{alpha}): "
          f"{(df['pval_ru_ky'] < alpha).sum()}/{len(df)} layers")
    print("  → significance.csv")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hidden states...")
    labels = np.load(DATA_DIR / "labels.npy")
    hs = {lang: np.load(DATA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}
    print(f"  Shape: {hs['en'].shape}  |  device: {DEVICE}")

    # Skip step 1 if already done
    p = RESULTS_DIR / "probing_detailed.csv"
    if p.exists():
        print("\n[1/6] Loading existing probing_detailed.csv...")
        probing_df = pd.read_csv(p)
    else:
        probing_df = exp_probing_ci(hs, labels)

    exp_permutation(hs, labels)
    exp_confusion(hs, labels, probing_df)
    exp_silhouette(hs, labels)
    exp_anisotropy(hs)
    exp_significance(probing_df)

    print("\n✓ All done!")
    print("\nCSV results:")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  {f.name}")
    print("\nFigures:")
    for f in sorted(FIG_DIR.glob("fig[4-7]*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
