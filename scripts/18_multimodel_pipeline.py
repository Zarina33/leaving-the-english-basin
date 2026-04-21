"""
Step 18: Full 4-model comparison pipeline.

Models: Gemma 4 E4B · Qwen3-8B · Mistral-7B-v0.3 · Llama-3.1-8B
Runs all experiments for models whose hidden states exist,
skips models whose data is not yet available.

Prerequisites (run in order):
  scripts/4_extract_hidden_states.py   → data/processed/
  scripts/8a_probing_gpu.py
  scripts/11_extract_qwen3.py          → data/processed/qwen3/
  scripts/12_compare_models.py
  scripts/16b_extract_mistral.py       → data/processed/mistral/
  scripts/16_extract_llama.py          → data/processed/llama/

Outputs:
  data/results/mistral/
  data/results/llama/
  data/results/figures/fig20_probing_4models.png
  data/results/figures/fig21_transfer_4models.png
  data/results/figures/fig22_cka_4models.png
  data/results/tables/table_probing_4models.csv
  data/results/tables/table_transfer_4models.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HS_DIRS = {
    "gemma4":  Path("data/processed"),
    "qwen3":   Path("data/processed/qwen3"),
    "mistral": Path("data/processed/mistral"),
    "llama":   Path("data/processed/llama"),
    "xlmr":    Path("data/processed/xlmr"),
}
RES_DIRS = {
    "gemma4":  Path("data/results"),
    "qwen3":   Path("data/results/qwen3"),
    "mistral": Path("data/results/mistral"),
    "llama":   Path("data/results/llama"),
    "xlmr":    Path("data/results/xlmr"),
}
FIG_DIR   = Path("data/results/figures")
TABLE_DIR = Path("data/results/tables")

# ── Model metadata ─────────────────────────────────────────────────────────────
MODEL_META = {
    "gemma4":  {"label": "Gemma 4 E4B",    "ls": "-",   "lw": 2.2, "color": "#1565C0"},
    "qwen3":   {"label": "Qwen3-8B",       "ls": "--",  "lw": 1.8, "color": "#C62828"},
    "mistral": {"label": "Mistral-7B-v0.3","ls": "-.",  "lw": 1.8, "color": "#E65100"},
    "llama":   {"label": "Llama-3.1-8B",   "ls": ":",   "lw": 2.0, "color": "#2E7D32"},
    "xlmr":    {"label": "XLM-R-large",    "ls": (0,(3,1,1,1)), "lw": 1.8, "color": "#6A1B9A"},
}

# ── Constants ──────────────────────────────────────────────────────────────────
N_FOLDS      = 5
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EMOTIONS    = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}
LANG_COLORS = {"en": "#1565C0", "ru": "#C62828", "ky": "#2E7D32"}


# ── GPU logistic regression ────────────────────────────────────────────────────

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

def run_all(model_key: str) -> tuple | None:
    hs_dir  = HS_DIRS[model_key]
    res_dir = RES_DIRS[model_key]

    # Check hidden states exist
    missing = [hs_dir / f"hidden_states_{lang}.npz"
               for lang in ["en", "ru", "ky"]
               if not (hs_dir / f"hidden_states_{lang}.npz").exists()]
    if missing:
        print(f"\n  [{model_key}] Hidden states not found — skipping.")
        print(f"  Run: python scripts/16{'b' if model_key == 'mistral' else ''}"
              f"_extract_{model_key}.py")
        return None

    res_dir.mkdir(parents=True, exist_ok=True)
    labels   = np.load(hs_dir / "labels.npy")
    hs       = {lang: np.load(hs_dir / f"hidden_states_{lang}.npz")["hidden_states"]
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


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_probing(active: dict):
    print("\n[Fig] Probing accuracy comparison...")
    n_models = len(active)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    model_list = ", ".join(MODEL_META[k]["label"] for k in active)
    fig.suptitle(f"Emotion Probing Accuracy by Normalized Layer\n{model_list}",
                 fontsize=12)

    for ax, lang in zip(axes, ["en", "ru", "ky"]):
        for model_key, (probe_df, *_) in active.items():
            meta = MODEL_META[model_key]
            sub  = probe_df[probe_df["lang"] == lang].sort_values("layer")
            n    = sub["layer"].max() + 1
            xpos = sub["layer"] / (n - 1)
            m, s = sub["acc_mean"].values, sub["acc_std"].values
            ax.plot(xpos, m, label=meta["label"],
                    ls=meta["ls"], lw=meta["lw"], color=meta["color"], alpha=0.9)
            ax.fill_between(xpos, m - s, m + s,
                            color=meta["color"], alpha=0.07)

        ax.axhline(1 / 6, ls=":", color="gray", lw=1.2, label="Chance (1/6)")
        ax.set_xlabel("Normalized Layer (0=embed, 1=last)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(LANG_LABELS[lang], fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / f"fig20_probing_{n_models}models.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def fig_transfer_bars(active: dict):
    print("\n[Fig] Transfer accuracy bar chart...")
    directions = ["en→ru", "en→ky", "ru→en", "ru→ky", "ky→en", "ky→ru"]
    src_tgt    = [d.split("→") for d in directions]
    n_models   = len(active)

    x   = np.arange(len(directions))
    w   = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (model_key, (_, __, transfer_df, ___)) in enumerate(active.items()):
        meta  = MODEL_META[model_key]
        peaks = []
        for src, tgt in src_tgt:
            sub = transfer_df[(transfer_df.src == src) & (transfer_df.tgt == tgt)]
            peaks.append(sub["transfer_acc"].max() if len(sub) else 0.0)
        offset = (i - (n_models - 1) / 2) * w
        ax.bar(x + offset, peaks, width=w,
               label=meta["label"], color=meta["color"], alpha=0.85)

    ax.axhline(1 / 6, ls="--", color="gray", lw=1.2, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in directions], fontsize=10)
    ax.set_ylabel("Peak Transfer Accuracy", fontsize=11)
    ax.set_title(f"Zero-shot Cross-lingual Transfer: {n_models}-Model Comparison",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = FIG_DIR / f"fig21_transfer_{n_models}models.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def fig_cka(active: dict):
    print("\n[Fig] CKA cross-lingual similarity...")
    pairs       = [("en", "ru"), ("en", "ky"), ("ru", "ky")]
    pair_colors = {"en_ru": "#E53935", "en_ky": "#1E88E5", "ru_ky": "#43A047"}
    pair_labels = {"en_ru": "EN↔RU", "en_ky": "EN↔KY", "ru_ky": "RU↔KY"}
    n_models    = len(active)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Linear CKA Cross-lingual Similarity by Layer", fontsize=13)

    for ax, (model_key, (_, cka_df, __, ___)) in zip(axes, active.items()):
        n    = cka_df["layer"].max() + 1
        xpos = cka_df["layer"] / (n - 1)
        for l1, l2 in pairs:
            col   = f"cka_{l1}_{l2}"
            color = pair_colors[f"{l1}_{l2}"]
            ax.plot(xpos, cka_df[col], color=color, lw=2,
                    label=pair_labels[f"{l1}_{l2}"])
        ax.set_xlabel("Normalized Layer", fontsize=10)
        ax.set_ylabel("Linear CKA", fontsize=10)
        ax.set_title(MODEL_META[model_key]["label"], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / f"fig22_cka_{n_models}models.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


# ── Summary tables ─────────────────────────────────────────────────────────────

def save_probing_table(active: dict):
    """Save best probing accuracy table (for paper)."""
    fold_cols = [f"fold_{i}" for i in range(N_FOLDS)]
    rows = []
    for lang in ["en", "ru", "ky"]:
        row = {"lang": lang.upper()}
        for model_key, (probe_df, *_) in active.items():
            sub  = probe_df[probe_df["lang"] == lang]
            best = sub.loc[sub["acc_mean"].idxmax()]
            row[f"{model_key}_layer"] = int(best["layer"])
            row[f"{model_key}_acc"]   = round(best["acc_mean"], 3)
            row[f"{model_key}_std"]   = round(best["acc_std"],  3)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = TABLE_DIR / "table_probing_4models.csv"
    df.to_csv(path, index=False)
    print(f"\n  Table saved → {path.name}")
    return df


def save_transfer_table(active: dict):
    """Save peak cross-lingual transfer table (for paper)."""
    directions = [("en","ru"),("en","ky"),("ru","en"),
                  ("ru","ky"),("ky","en"),("ky","ru")]
    rows = []
    for src, tgt in directions:
        row = {"direction": f"{src.upper()}→{tgt.upper()}"}
        for model_key, (_, __, transfer_df, ___) in active.items():
            sub  = transfer_df[(transfer_df.src == src) & (transfer_df.tgt == tgt)]
            peak = sub["transfer_acc"].max() if len(sub) else float("nan")
            row[model_key] = round(peak, 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = TABLE_DIR / "table_transfer_4models.csv"
    df.to_csv(path, index=False)
    print(f"  Table saved → {path.name}")
    return df


def print_summary(active: dict):
    print("\n" + "=" * 72)
    print("PROBING SUMMARY: Best accuracy per language")
    header = f"{'Lang':<6}" + "".join(
        f"  {MODEL_META[k]['label']:>18}" for k in active)
    print(header)

    fold_cols = [f"fold_{i}" for i in range(N_FOLDS)]
    best_folds_all = {}

    for lang in ["en", "ru", "ky"]:
        row = f"{lang.upper():<6}"
        best_folds = {}
        for model_key, (probe_df, *_) in active.items():
            sub  = probe_df[probe_df["lang"] == lang]
            best = sub.loc[sub["acc_mean"].idxmax()]
            row += (f"  layer {int(best.layer):2d}  "
                    f"{best.acc_mean:.3f}±{best.acc_std:.3f}")
            best_folds[model_key] = best[fold_cols].values.astype(float)
        best_folds_all[lang] = best_folds
        print(row)

        # Paired t-tests vs Gemma4 (baseline)
        baseline_key = "gemma4" if "gemma4" in best_folds else list(best_folds)[0]
        for model_key in best_folds:
            if model_key == baseline_key:
                continue
            t, p = stats.ttest_rel(best_folds[model_key],
                                   best_folds[baseline_key])
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
                   ("*" if p < 0.05 else "ns"))
            label = MODEL_META[model_key]["label"]
            print(f"         vs {label:<18}: "
                  f"t={t:+.2f}  p={p:.4f}  {sig}")

    print("=" * 72)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load pre-existing Gemma 4 and Qwen3 results directly
    print("\nLoading Gemma 4 results...")
    gemma_results = (
        pd.read_csv(RES_DIRS["gemma4"] / "probing_detailed.csv"),
        pd.read_csv(RES_DIRS["gemma4"] / "cka_results.csv"),
        pd.read_csv(RES_DIRS["gemma4"] / "transfer_results.csv"),
        pd.read_csv(RES_DIRS["gemma4"] / "per_emotion_layers.csv"),
    )

    print("Loading Qwen3 results...")
    qwen_results = (
        pd.read_csv(RES_DIRS["qwen3"] / "probing_detailed.csv"),
        pd.read_csv(RES_DIRS["qwen3"] / "cka_results.csv"),
        pd.read_csv(RES_DIRS["qwen3"] / "transfer_results.csv"),
        pd.read_csv(RES_DIRS["qwen3"] / "per_emotion_layers.csv"),
    )

    # Run remaining models (skips those whose hidden states are missing)
    print("\nProcessing remaining models...")
    mistral_results = run_all("mistral")
    llama_results   = run_all("llama")
    xlmr_results    = run_all("xlmr")

    # Collect only available models
    active = {"gemma4": gemma_results, "qwen3": qwen_results}
    if mistral_results:
        active["mistral"] = mistral_results
    if llama_results:
        active["llama"] = llama_results
    if xlmr_results:
        active["xlmr"] = xlmr_results

    n = len(active)
    print(f"\nActive models ({n}): {', '.join(MODEL_META[k]['label'] for k in active)}")

    # Figures
    print("\nBuilding comparison figures...")
    fig_probing(active)
    fig_transfer_bars(active)
    fig_cka(active)

    # Tables
    print("\nSaving paper tables...")
    probing_table  = save_probing_table(active)
    transfer_table = save_transfer_table(active)

    print("\nProbing table:")
    print(probing_table.to_string(index=False))
    print("\nTransfer table:")
    print(transfer_table.to_string(index=False))

    print_summary(active)
    print("\nAll done!")


if __name__ == "__main__":
    main()
