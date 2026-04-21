"""
Step 17: Run full analysis for Llama-3.1-8B and produce 3-model comparison.

Runs: probing CI, CKA, cross-lingual transfer, per-emotion analysis
Saves: data/results/llama/
Figures: 3-model side-by-side comparisons (Gemma 4 E4B / Qwen3-8B / Llama-3.1-8B)

Run AFTER:
  scripts/4_extract_hidden_states.py   (Gemma 4)
  scripts/8a_probing_gpu.py            (Gemma 4 probing)
  scripts/11_extract_qwen3.py          (Qwen3)
  scripts/12_compare_models.py         (Qwen3 probing/CKA/transfer)
  scripts/16_extract_llama.py          (Llama)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
GEMMA_HS_DIR  = Path("data/processed")
QWEN_HS_DIR   = Path("data/processed/qwen3")
LLAMA_HS_DIR  = Path("data/processed/llama")

GEMMA_RES_DIR = Path("data/results")
QWEN_RES_DIR  = Path("data/results/qwen3")
LLAMA_RES_DIR = Path("data/results/llama")

FIG_DIR = Path("data/results/figures")

# ── Constants ─────────────────────────────────────────────────────────────────
N_FOLDS      = 5
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EMOTIONS    = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}
LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}

MODEL_META = {
    "gemma4": {"label": "Gemma 4 E4B",    "ls": "-",   "lw": 2.2, "marker": "o"},
    "qwen3":  {"label": "Qwen3-8B",       "ls": "--",  "lw": 1.8, "marker": "s"},
    "llama":  {"label": "Llama-3.1-8B",   "ls": "-.",  "lw": 1.8, "marker": "^"},
}


# ── GPU logistic regression ───────────────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


def gpu_probe(X: np.ndarray, y: np.ndarray) -> list[float]:
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    accs  = []
    n_cls = len(np.unique(y))
    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]),
                           dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),
                           dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)
        m   = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
        ce  = nn.CrossEntropyLoss()
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds = m(Xte).argmax(1)
        accs.append((preds == yte).float().mean().item())
    return accs


def train_transfer(X_tr, y_tr, X_te, y_te) -> float:
    sc  = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X_te),     dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.long, device=DEVICE)
    m   = LinClf(Xtr.shape[1], len(np.unique(y_tr))).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    m.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(Xte).argmax(1)
    return (preds == yte).float().mean().item()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num   = np.linalg.norm(Y.T @ X, "fro") ** 2
    denom = (np.linalg.norm(X.T @ X, "fro") *
             np.linalg.norm(Y.T @ Y, "fro"))
    return float(num / (denom + 1e-10))


# ── Run all experiments for one model ─────────────────────────────────────────

def run_all(hs_dir: Path, res_dir: Path, model_key: str) -> tuple:
    res_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(hs_dir / "labels.npy")
    hs = {lang: np.load(hs_dir / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}
    n_layers = hs["en"].shape[1]
    label    = MODEL_META[model_key]["label"]

    print(f"\n{'='*60}")
    print(f"  {label}  |  layers={n_layers}  hidden={hs['en'].shape[2]}")
    print(f"{'='*60}")

    # 1. Probing
    probe_path = res_dir / "probing_detailed.csv"
    if probe_path.exists():
        print("\n[1] Loading existing probing results...")
        probing_df = pd.read_csv(probe_path)
    else:
        print("\n[1] Layer-wise probing...")
        records = []
        for lang in ["en", "ru", "ky"]:
            for layer in tqdm(range(n_layers), desc=f"  {lang.upper()}"):
                fa = gpu_probe(hs[lang][:, layer, :], labels)
                records.append({
                    "lang": lang, "layer": layer,
                    "acc_mean": np.mean(fa), "acc_std": np.std(fa),
                    **{f"fold_{i}": a for i, a in enumerate(fa)},
                })
        probing_df = pd.DataFrame(records)
        probing_df.to_csv(probe_path, index=False)
        print(f"  Saved {probe_path.name}")

    # 2. CKA
    cka_path = res_dir / "cka_results.csv"
    if cka_path.exists():
        print("\n[2] Loading existing CKA results...")
        cka_df = pd.read_csv(cka_path)
    else:
        print("\n[2] Linear CKA cross-lingual...")
        records = []
        for layer in tqdm(range(n_layers), desc="  Layers"):
            row = {"layer": layer}
            for l1, l2 in [("en", "ru"), ("en", "ky"), ("ru", "ky")]:
                X = hs[l1][:, layer, :].astype(np.float32)
                Y = hs[l2][:, layer, :].astype(np.float32)
                row[f"cka_{l1}_{l2}"] = linear_cka(X, Y)
            records.append(row)
        cka_df = pd.DataFrame(records)
        cka_df.to_csv(cka_path, index=False)
        print(f"  Saved {cka_path.name}")

    # 3. Cross-lingual transfer
    transfer_path = res_dir / "transfer_results.csv"
    if transfer_path.exists():
        print("\n[3] Loading existing transfer results...")
        transfer_df = pd.read_csv(transfer_path)
    else:
        print("\n[3] Cross-lingual transfer probing...")
        records = []
        for src, tgt in [("en", "ru"), ("en", "ky"), ("ru", "en"),
                         ("ru", "ky"), ("ky", "en"), ("ky", "ru")]:
            for layer in tqdm(range(n_layers),
                              desc=f"  {src.upper()}→{tgt.upper()}", leave=False):
                acc = train_transfer(hs[src][:, layer, :], labels,
                                     hs[tgt][:, layer, :], labels)
                records.append({"src": src, "tgt": tgt,
                                 "layer": layer, "transfer_acc": acc})
        transfer_df = pd.DataFrame(records)
        transfer_df.to_csv(transfer_path, index=False)
        print(f"  Saved {transfer_path.name}")

    # 4. Per-emotion one-vs-rest
    emotion_path = res_dir / "per_emotion_layers.csv"
    if emotion_path.exists():
        print("\n[4] Loading existing per-emotion results...")
        emotion_df = pd.read_csv(emotion_path)
    else:
        print("\n[4] Per-emotion one-vs-rest probing...")
        records = []
        for lang in ["en", "ru", "ky"]:
            for eidx, emotion in enumerate(EMOTIONS):
                y_bin = (labels == eidx).astype(int)
                for layer in tqdm(range(n_layers),
                                  desc=f"  {lang.upper()}/{emotion}", leave=False):
                    X   = hs[lang][:, layer, :]
                    rng = np.random.default_rng(RANDOM_STATE)
                    idx = rng.permutation(len(X))
                    sp  = int(len(X) * 0.8)
                    tr, te = idx[:sp], idx[sp:]
                    sc  = StandardScaler()
                    Xtr = torch.tensor(sc.fit_transform(X[tr]),
                                       dtype=torch.float32, device=DEVICE)
                    Xte = torch.tensor(sc.transform(X[te]),
                                       dtype=torch.float32, device=DEVICE)
                    ytr = torch.tensor(y_bin[tr], dtype=torch.long, device=DEVICE)
                    yte = torch.tensor(y_bin[te], dtype=torch.long, device=DEVICE)
                    n_pos = int(ytr.sum().item())
                    n_neg = len(ytr) - n_pos
                    w   = torch.tensor([1.0, n_neg / max(n_pos, 1)],
                                       dtype=torch.float32, device=DEVICE)
                    m   = LinClf(Xtr.shape[1], 2).to(DEVICE)
                    opt = torch.optim.Adam(m.parameters(), lr=LR_RATE,
                                          weight_decay=1e-4)
                    ce  = nn.CrossEntropyLoss(weight=w)
                    m.train()
                    for _ in range(N_EPOCHS):
                        opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
                    m.eval()
                    with torch.no_grad():
                        preds = m(Xte).argmax(1)
                    acc = (preds == yte).float().mean().item()
                    records.append({"lang": lang, "emotion": emotion,
                                    "layer": layer, "ovr_acc": acc})
        emotion_df = pd.DataFrame(records)
        emotion_df.to_csv(emotion_path, index=False)
        print(f"  Saved {emotion_path.name}")

    return probing_df, cka_df, transfer_df, emotion_df


# ── 3-model comparison figures ────────────────────────────────────────────────

def _norm(layer_series, df):
    """Normalize layer index to [0, 1]."""
    n = df["layer"].max() + 1
    return layer_series / (n - 1)


def fig_probing_3models(dfs: dict):
    """Layer-wise probing accuracy: 3 models × 3 languages."""
    print("\n[Fig] 3-model probing comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    fig.suptitle(
        "Emotion Probing Accuracy by Normalized Layer\n"
        "Gemma 4 E4B · Qwen3-8B · Llama-3.1-8B",
        fontsize=13,
    )

    for ax, lang in zip(axes, ["en", "ru", "ky"]):
        for model_key, df in dfs.items():
            meta  = MODEL_META[model_key]
            sub   = df[df["lang"] == lang].sort_values("layer")
            xpos  = _norm(sub["layer"], sub)
            m, s  = sub["acc_mean"].values, sub["acc_std"].values
            color = LANG_COLORS[lang]
            ax.plot(xpos, m,
                    label=meta["label"], ls=meta["ls"],
                    lw=meta["lw"], color=color, alpha=0.9)
            ax.fill_between(xpos, m - s, m + s, color=color, alpha=0.07)

        ax.axhline(1 / 6, ls=":", color="gray", lw=1.2, label="Chance (1/6)")
        ax.set_xlabel("Normalized Layer (0=embed, 1=last)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(LANG_LABELS[lang], fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig20_probing_3models.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def fig_transfer_3models(dfs: dict):
    """Peak cross-lingual transfer accuracy bar chart: 3 models."""
    print("\n[Fig] 3-model transfer bar chart...")

    directions = ["en→ru", "en→ky", "ru→en", "ru→ky", "ky→en", "ky→ru"]
    src_tgt    = [d.split("→") for d in directions]

    x   = np.arange(len(directions))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (model_key, df) in enumerate(dfs.items()):
        meta  = MODEL_META[model_key]
        peaks = []
        for src, tgt in src_tgt:
            sub   = df[(df.src == src) & (df.tgt == tgt)]
            peaks.append(sub["transfer_acc"].max() if len(sub) else 0.0)
        ax.bar(x + i * w, peaks, width=w, label=meta["label"],
               color=f"C{i}", alpha=0.8)

    ax.axhline(1 / 6, ls="--", color="gray", lw=1.2, label="Chance")
    ax.set_xticks(x + w)
    ax.set_xticklabels([d.upper() for d in directions], fontsize=10)
    ax.set_ylabel("Peak Transfer Accuracy", fontsize=11)
    ax.set_title("Zero-shot Cross-lingual Transfer: 3-Model Comparison",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = FIG_DIR / "fig21_transfer_3models.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def fig_cka_3models(dfs: dict):
    """CKA per language pair: 1 subplot per model."""
    print("\n[Fig] 3-model CKA...")
    pairs        = [("en", "ru"), ("en", "ky"), ("ru", "ky")]
    pair_colors  = {"en_ru": "#E53935", "en_ky": "#1E88E5", "ru_ky": "#43A047"}
    pair_labels  = {"en_ru": "EN↔RU", "en_ky": "EN↔KY", "ru_ky": "RU↔KY"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    fig.suptitle("Linear CKA Cross-lingual Similarity by Layer\n"
                 "Gemma 4 E4B · Qwen3-8B · Llama-3.1-8B", fontsize=13)

    for ax, (model_key, df) in zip(axes, dfs.items()):
        n    = df["layer"].max() + 1
        xpos = df["layer"] / (n - 1)
        for l1, l2 in pairs:
            col   = f"cka_{l1}_{l2}"
            color = pair_colors[f"{l1}_{l2}"]
            label = pair_labels[f"{l1}_{l2}"]
            ax.plot(xpos, df[col], color=color, lw=2, label=label)
        ax.set_xlabel("Normalized Layer", fontsize=10)
        ax.set_ylabel("Linear CKA", fontsize=10)
        ax.set_title(MODEL_META[model_key]["label"], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig22_cka_3models.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def print_summary(probe_dfs: dict):
    """Print best probing accuracy and statistical comparison table."""
    print("\n" + "=" * 70)
    print("SUMMARY: Best probing accuracy per language (best layer)")
    header = f"{'Lang':<6}" + "".join(
        f"  {MODEL_META[k]['label']:>16}" for k in probe_dfs
    )
    print(header)

    for lang in ["en", "ru", "ky"]:
        row = f"{lang.upper():<6}"
        fold_cols = [f"fold_{i}" for i in range(N_FOLDS)]
        best_folds = {}
        for model_key, df in probe_dfs.items():
            sub  = df[df["lang"] == lang]
            best = sub.loc[sub["acc_mean"].idxmax()]
            row += f"  layer {int(best.layer):2d}  {best.acc_mean:.3f}±{best.acc_std:.3f}"
            best_folds[model_key] = best[fold_cols].values.astype(float)
        print(row)

        # Paired t-test: Llama vs Gemma4 and Llama vs Qwen3
        if "llama" in best_folds and "gemma4" in best_folds:
            t, p = stats.ttest_rel(best_folds["llama"], best_folds["gemma4"])
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
                   ("*" if p < 0.05 else "ns"))
            print(f"       Llama vs Gemma4:  t={t:.2f}  p={p:.4f}  {sig}")
        if "llama" in best_folds and "qwen3" in best_folds:
            t, p = stats.ttest_rel(best_folds["llama"], best_folds["qwen3"])
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
                   ("*" if p < 0.05 else "ns"))
            print(f"       Llama vs Qwen3:   t={t:.2f}  p={p:.4f}  {sig}")

    print("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA_RES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing Gemma 4 and Qwen3 results
    print("\nLoading Gemma 4 results...")
    gemma_probe    = pd.read_csv(GEMMA_RES_DIR / "probing_detailed.csv")
    gemma_cka      = pd.read_csv(GEMMA_RES_DIR / "cka_results.csv")
    gemma_transfer = pd.read_csv(GEMMA_RES_DIR / "transfer_results.csv")
    gemma_emotion  = pd.read_csv(GEMMA_RES_DIR / "per_emotion_layers.csv")

    print("Loading Qwen3 results...")
    qwen_probe    = pd.read_csv(QWEN_RES_DIR / "probing_detailed.csv")
    qwen_cka      = pd.read_csv(QWEN_RES_DIR / "cka_results.csv")
    qwen_transfer = pd.read_csv(QWEN_RES_DIR / "transfer_results.csv")
    qwen_emotion  = pd.read_csv(QWEN_RES_DIR / "per_emotion_layers.csv")

    # Check that Llama hidden states exist before proceeding
    missing = [LLAMA_HS_DIR / f"hidden_states_{lang}.npz"
               for lang in ["en", "ru", "ky"]
               if not (LLAMA_HS_DIR / f"hidden_states_{lang}.npz").exists()]
    if missing:
        print("\nERROR: Llama hidden states not found. Run extraction first:")
        print("  python scripts/16_extract_llama.py")
        print("\nMissing files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    # Run Llama experiments (skips any already computed)
    llama_probe, llama_cka, llama_transfer, llama_emotion = run_all(
        LLAMA_HS_DIR, LLAMA_RES_DIR, "llama"
    )

    # ── 3-model comparison figures ────────────────────────────────────────────
    print("\nBuilding 3-model comparison figures...")

    probe_dfs    = {"gemma4": gemma_probe,    "qwen3": qwen_probe,
                    "llama":  llama_probe}
    cka_dfs      = {"gemma4": gemma_cka,      "qwen3": qwen_cka,
                    "llama":  llama_cka}
    transfer_dfs = {"gemma4": gemma_transfer, "qwen3": qwen_transfer,
                    "llama":  llama_transfer}

    fig_probing_3models(probe_dfs)
    fig_transfer_3models(transfer_dfs)
    fig_cka_3models(cka_dfs)

    print_summary(probe_dfs)
    print("\nAll done!")


if __name__ == "__main__":
    main()
