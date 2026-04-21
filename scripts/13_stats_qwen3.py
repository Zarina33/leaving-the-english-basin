"""
Step 13: Statistical tests for Qwen3-8B + comparison with Gemma 4.

1. Permutation test (1000 runs) for Qwen3 probing significance
2. Paired t-test: Gemma4 vs Qwen3 per language at best layers
3. Fig 16: Permutation baseline comparison (both models)
4. Fig 17: Significance heatmap across layers

Output:
  data/results/qwen3/permutation_baseline.csv
  data/results/qwen3/significance.csv
  data/results/figures/fig16_permutation_compare.png
  data/results/figures/fig17_model_significance.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

GEMMA_DIR   = Path("data/processed")
QWEN_DIR    = Path("data/processed/qwen3")
RESULTS_DIR = Path("data/results")
Q_RESULTS   = Path("data/results/qwen3")
FIG_DIR     = Path("data/results/figures")

N_FOLDS      = 5
N_PERMUT     = 1000
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_probe_single(X: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    """Single 80/20 split probe (fast, for permutation test)."""
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    tr, te = idx[:split], idx[split:]

    sc  = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X[te]),     dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)

    n_cls = len(np.unique(y))
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


def permutation_test(X: np.ndarray, y: np.ndarray,
                     true_acc: float, n_perm: int = N_PERMUT) -> tuple:
    """Returns (null_mean, null_std, p_value)."""
    null_accs = []
    for i in range(n_perm):
        y_shuf = np.random.default_rng(i).permutation(y)
        null_accs.append(gpu_probe_single(X, y_shuf, seed=i))
    null_accs = np.array(null_accs)
    p_val = (null_accs >= true_acc).mean()
    return null_accs.mean(), null_accs.std(), p_val


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Q_RESULTS.mkdir(parents=True, exist_ok=True)

    labels = np.load(GEMMA_DIR / "labels.npy")

    # Load hidden states
    gemma_hs = {lang: np.load(GEMMA_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
                for lang in ["en", "ru", "ky"]}
    qwen_hs  = {lang: np.load(QWEN_DIR  / f"hidden_states_{lang}.npz")["hidden_states"]
                for lang in ["en", "ru", "ky"]}

    # Load probing results
    gemma_prob = pd.read_csv(RESULTS_DIR / "probing_detailed.csv")
    qwen_prob  = pd.read_csv(Q_RESULTS   / "probing_detailed.csv")

    # ── 1. Permutation test for Qwen3 at best layers ──────────────────────────
    print("\n[1] Permutation test for Qwen3...")
    perm_records = []
    for lang in ["en", "ru", "ky"]:
        sub = qwen_prob[qwen_prob["lang"] == lang]
        best_layer = int(sub.loc[sub["acc_mean"].idxmax(), "layer"])
        true_acc   = sub.loc[sub["acc_mean"].idxmax(), "acc_mean"]

        X = qwen_hs[lang][:, best_layer, :]
        print(f"  {lang.upper()} layer {best_layer}  true_acc={true_acc:.3f}")
        null_mean, null_std, p_val = permutation_test(X, labels, true_acc)
        perm_records.append({
            "model": "qwen3", "lang": lang,
            "best_layer": best_layer,
            "true_acc": true_acc,
            "null_mean": null_mean,
            "null_std":  null_std,
            "p_value":   p_val,
            "significant": p_val < 0.05,
        })
        print(f"    null={null_mean:.3f}±{null_std:.3f}  p={p_val:.4f}")

    perm_df = pd.DataFrame(perm_records)
    perm_df.to_csv(Q_RESULTS / "permutation_baseline.csv", index=False)
    print("  Saved qwen3/permutation_baseline.csv")

    # ── 2. Paired t-test: Gemma4 vs Qwen3 per language ───────────────────────
    print("\n[2] Paired t-test Gemma4 vs Qwen3...")
    sig_records = []

    for lang in ["en", "ru", "ky"]:
        # Get per-layer accuracies for both models (use acc_mean across layers)
        g_sub  = gemma_prob[gemma_prob["lang"] == lang].sort_values("layer")
        q_sub  = qwen_prob [qwen_prob ["lang"] == lang].sort_values("layer")

        # Normalize to same number of data points (use fold accs at best layer)
        g_best = g_sub.loc[g_sub["acc_mean"].idxmax()]
        q_best = q_sub.loc[q_sub["acc_mean"].idxmax()]

        # Compare fold accuracies at best layer (5 folds each)
        fold_cols = [c for c in g_sub.columns if c.startswith("fold_")]
        g_folds = g_best[fold_cols].values.astype(float)
        q_folds = q_best[fold_cols].values.astype(float)

        t_stat, p_val = stats.ttest_rel(q_folds, g_folds)
        sig_records.append({
            "lang": lang,
            "gemma_layer": int(g_best["layer"]),
            "qwen_layer":  int(q_best["layer"]),
            "gemma_acc":   g_best["acc_mean"],
            "qwen_acc":    q_best["acc_mean"],
            "t_stat":      t_stat,
            "p_value":     p_val,
            "significant": p_val < 0.05,
            "winner":      "qwen3" if q_best["acc_mean"] > g_best["acc_mean"] else "gemma4",
        })
        print(f"  {lang.upper()}: Gemma4={g_best['acc_mean']:.3f}  "
              f"Qwen3={q_best['acc_mean']:.3f}  "
              f"t={t_stat:.3f}  p={p_val:.4f}  "
              f"{'*' if p_val < 0.05 else 'n.s.'}")

    sig_df = pd.DataFrame(sig_records)
    sig_df.to_csv(Q_RESULTS / "significance.csv", index=False)
    print("  Saved qwen3/significance.csv")

    # ── 3. Layer-wise p-values for both models ────────────────────────────────
    print("\n[3] Layer-wise significance (vs permutation null)...")
    gemma_perm = pd.read_csv(RESULTS_DIR / "permutation_baseline.csv")

    layer_sig_records = []
    for lang in ["en", "ru", "ky"]:
        g_null = float(gemma_perm[gemma_perm["lang"] == lang]["perm_mean"].values[0])
        g_null_std = float(gemma_perm[gemma_perm["lang"] == lang]["perm_std"].values[0])

        g_layers = gemma_prob[gemma_prob["lang"] == lang].sort_values("layer")
        q_layers = qwen_prob [qwen_prob ["lang"] == lang].sort_values("layer")

        for _, grow in g_layers.iterrows():
            z = (grow["acc_mean"] - g_null) / max(g_null_std, 1e-6)
            layer_sig_records.append({
                "model": "gemma4", "lang": lang,
                "layer": int(grow["layer"]),
                "acc":   grow["acc_mean"],
                "z_score": z,
            })

        # For Qwen3 use its own permutation null
        q_perm_row = perm_df[perm_df["lang"] == lang].iloc[0]
        q_null     = q_perm_row["null_mean"]
        q_null_std = q_perm_row["null_std"]

        for _, qrow in q_layers.iterrows():
            z = (qrow["acc_mean"] - q_null) / max(q_null_std, 1e-6)
            layer_sig_records.append({
                "model": "qwen3", "lang": lang,
                "layer": int(qrow["layer"]),
                "acc":   qrow["acc_mean"],
                "z_score": z,
            })

    layer_sig_df = pd.DataFrame(layer_sig_records)
    layer_sig_df.to_csv(Q_RESULTS / "layer_significance.csv", index=False)

    # ── Fig 16: Permutation baseline comparison ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Probing Accuracy vs. Permutation Null — Gemma 4 E4B vs Qwen3-8B",
                 fontsize=13)

    gemma_perm_full = {}
    qwen_perm_full  = {}
    for lang in ["en", "ru", "ky"]:
        g_best_layer = int(gemma_prob[gemma_prob["lang"]==lang]
                           .loc[gemma_prob[gemma_prob["lang"]==lang]["acc_mean"].idxmax(), "layer"])
        q_best_layer = int(qwen_prob[qwen_prob["lang"]==lang]
                           .loc[qwen_prob[qwen_prob["lang"]==lang]["acc_mean"].idxmax(), "layer"])

        # Collect null distributions
        g_nulls = []
        for i in range(N_PERMUT):
            y_shuf = np.random.default_rng(i).permutation(labels)
            g_nulls.append(gpu_probe_single(gemma_hs[lang][:, g_best_layer, :], y_shuf, seed=i))
        gemma_perm_full[lang] = np.array(g_nulls)

        q_nulls = []
        for i in range(N_PERMUT):
            y_shuf = np.random.default_rng(i).permutation(labels)
            q_nulls.append(gpu_probe_single(qwen_hs[lang][:, q_best_layer, :], y_shuf, seed=i))
        qwen_perm_full[lang] = np.array(q_nulls)

    for ax, lang in zip(axes, ["en", "ru", "ky"]):
        g_true = float(gemma_prob[gemma_prob["lang"]==lang]["acc_mean"].max())
        q_true = float(qwen_prob [qwen_prob ["lang"]==lang]["acc_mean"].max())
        g_null = gemma_perm_full[lang]
        q_null = qwen_perm_full[lang]

        bins = np.linspace(min(g_null.min(), q_null.min()) - 0.01,
                           max(g_true, q_true) + 0.02, 30)
        ax.hist(g_null, bins=bins, alpha=0.5, color="#1565C0", label="Gemma4 null")
        ax.hist(q_null, bins=bins, alpha=0.5, color="#E53935", label="Qwen3 null")
        ax.axvline(g_true, color="#1565C0", linewidth=2.5,
                   linestyle="-", label=f"Gemma4 true ({g_true:.3f})")
        ax.axvline(q_true, color="#E53935", linewidth=2.5,
                   linestyle="--", label=f"Qwen3 true ({q_true:.3f})")
        ax.set_title(LANG_LABELS[lang], fontsize=12)
        ax.set_xlabel("Accuracy", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig16_permutation_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved fig16_permutation_compare.png")

    # ── Fig 17: Z-score heatmap (layer × language, both models) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Statistical Significance by Layer (Z-score vs Permutation Null)\n"
                 "Gemma 4 E4B vs Qwen3-8B", fontsize=13)

    for ax, model_name in zip(axes, ["gemma4", "qwen3"]):
        sub = layer_sig_df[layer_sig_df["model"] == model_name]
        pivot = sub.pivot(index="lang", columns="layer", values="z_score")
        pivot.index = [LANG_LABELS[l] for l in pivot.index]

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=pivot.values.max())
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=11)
        ax.set_xlabel("Layer", fontsize=11)
        model_label = "Gemma 4 E4B" if model_name == "gemma4" else "Qwen3-8B"
        ax.set_title(model_label, fontsize=12)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        plt.colorbar(im, ax=ax, label="Z-score")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig17_model_significance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig17_model_significance.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("STATISTICAL SUMMARY")
    print("="*55)
    print("\nPaired t-test (Qwen3 vs Gemma4 at best layers):")
    print(sig_df[["lang","gemma_acc","qwen_acc","t_stat","p_value","significant","winner"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
