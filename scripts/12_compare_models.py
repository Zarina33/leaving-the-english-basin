"""
Step 12: Run all experiments on Qwen3-8B and compare with Gemma 4 E4B.

Runs: probing CI, CKA, cross-lingual transfer, per-emotion analysis
Saves: data/results/qwen3/
Figures: side-by-side Gemma4 vs Qwen3 comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
GEMMA_HS_DIR  = Path("data/processed")
QWEN_HS_DIR   = Path("data/processed/qwen3")
GEMMA_RES_DIR = Path("data/results")
QWEN_RES_DIR  = Path("data/results/qwen3")
FIG_DIR       = Path("data/results/figures")

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
LANG_LS     = {"en": "-", "ru": "--", "ky": ":"}

MODEL_STYLES = {
    "gemma4": {"color_offset": 0,   "marker": "o", "lw": 2.2},
    "qwen3":  {"color_offset": 0.4, "marker": "s", "lw": 1.8},
}
MODEL_LABELS = {"gemma4": "Gemma 4 E4B", "qwen3": "Qwen3-8B"}


# ── GPU logistic regression ───────────────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_probe(X, y, return_preds=False):
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    accs, all_pred, all_true = [], [], []
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
        if return_preds:
            all_pred.extend(preds.cpu().numpy())
            all_true.extend(y[te])
    if return_preds:
        return accs, np.array(all_true), np.array(all_pred)
    return accs


def train_transfer(X_tr, y_tr, X_te, y_te):
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


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num   = np.linalg.norm(Y.T @ X, "fro") ** 2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (denom + 1e-10))


def mean_cosine(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return float((Xn * Yn).sum(1).mean())


# ── Run all experiments for one model ─────────────────────────────────────────

def run_all(hs_dir: Path, res_dir: Path, model_name: str):
    res_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(hs_dir / "labels.npy")
    hs = {lang: np.load(hs_dir / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}
    n_layers = hs["en"].shape[1]
    print(f"\n{'='*55}")
    print(f"  {MODEL_LABELS[model_name]}  |  layers={n_layers}  "
          f"hidden={hs['en'].shape[2]}")
    print(f"{'='*55}")

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
                records.append({"lang": lang, "layer": layer,
                                 "acc_mean": np.mean(fa), "acc_std": np.std(fa),
                                 **{f"fold_{i}": a for i, a in enumerate(fa)}})
        probing_df = pd.DataFrame(records)
        probing_df.to_csv(probe_path, index=False)
        print(f"  Saved {probe_path.name}")

    # 2. CKA
    cka_path = res_dir / "cka_results.csv"
    if cka_path.exists():
        print("\n[2] Loading existing CKA results...")
        cka_df = pd.read_csv(cka_path)
    else:
        print("\n[2] CKA + cosine...")
        records = []
        for layer in tqdm(range(n_layers), desc="  Layers"):
            row = {"layer": layer}
            for l1, l2 in [("en","ru"),("en","ky"),("ru","ky")]:
                X = hs[l1][:, layer, :].astype(np.float32)
                Y = hs[l2][:, layer, :].astype(np.float32)
                row[f"cka_{l1}_{l2}"]    = linear_cka(X, Y)
                row[f"cosine_{l1}_{l2}"] = mean_cosine(X, Y)
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
        print("\n[3] Cross-lingual transfer...")
        records = []
        for src, tgt in [("en","ru"),("en","ky"),("ru","en"),
                         ("ru","ky"),("ky","en"),("ky","ru")]:
            for layer in tqdm(range(n_layers),
                              desc=f"  {src.upper()}→{tgt.upper()}", leave=False):
                acc = train_transfer(hs[src][:, layer, :], labels,
                                     hs[tgt][:, layer, :], labels)
                records.append({"src": src, "tgt": tgt,
                                 "layer": layer, "transfer_acc": acc})
        transfer_df = pd.DataFrame(records)
        transfer_df.to_csv(transfer_path, index=False)
        print(f"  Saved {transfer_path.name}")

    # 4. Per-emotion
    emotion_path = res_dir / "per_emotion_layers.csv"
    if emotion_path.exists():
        print("\n[4] Loading existing per-emotion results...")
        emotion_df = pd.read_csv(emotion_path)
    else:
        print("\n[4] Per-emotion one-vs-rest...")
        records = []
        for lang in ["en", "ru", "ky"]:
            for eidx, emotion in enumerate(EMOTIONS):
                y_bin = (labels == eidx).astype(int)
                for layer in tqdm(range(n_layers),
                                  desc=f"  {lang.upper()}/{emotion}", leave=False):
                    X  = hs[lang][:, layer, :]
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
                    w   = torch.tensor([1.0, n_neg/max(n_pos,1)],
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


# ── Comparison figures ────────────────────────────────────────────────────────

def normalize_layer(layer, n_layers):
    """Normalize layer index to [0,1] for cross-model comparison."""
    return layer / (n_layers - 1)


def fig_probing_compare(gemma_df, qwen_df):
    print("\n[Fig] Probing comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Emotion Probing Accuracy by Normalized Layer\n"
                 "Gemma 4 E4B (42 layers) vs Qwen3-8B (36 layers)",
                 fontsize=13)

    n_gemma = gemma_df["layer"].max() + 1
    n_qwen  = qwen_df["layer"].max() + 1

    for ax, lang in zip(axes, ["en", "ru", "ky"]):
        for df, model, ls, alpha in [
            (gemma_df, "Gemma 4 E4B", "-",  0.9),
            (qwen_df,  "Qwen3-8B",    "--", 0.9),
        ]:
            sub  = df[df["lang"]==lang].sort_values("layer")
            n    = sub["layer"].max() + 1
            xpos = sub["layer"] / (n - 1)
            m, s = sub["acc_mean"].values, sub["acc_std"].values
            color = LANG_COLORS[lang]
            lw    = 2.2 if "Gemma" in model else 1.8
            ax.plot(xpos, m, label=model, ls=ls, color=color, lw=lw, alpha=alpha)
            ax.fill_between(xpos, m-s, m+s, color=color, alpha=0.08)

        ax.axhline(1/6, ls=":", color="gray", lw=1.2, label="Chance")
        ax.set_xlabel("Normalized Layer (0=embed, 1=last)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(LANG_LABELS[lang], fontsize=12)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_probing_compare.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig12_probing_compare.png")


def fig_cka_compare(gemma_cka, qwen_cka):
    print("\n[Fig] CKA comparison...")
    pairs  = [("en","ru"), ("en","ky"), ("ru","ky")]
    colors = {"en_ru": "#E53935", "en_ky": "#1E88E5", "ru_ky": "#43A047"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-lingual CKA: Gemma 4 E4B vs Qwen3-8B\n"
                 "(normalized layer axis)", fontsize=13)

    for ax, (df_g, df_q, title) in zip(axes, [
        (gemma_cka, qwen_cka, "Linear CKA"),
    ] * 2):
        for df, model, ls, lw in [
            (df_g, "Gemma 4 E4B", "-",  2.2),
            (df_q, "Qwen3-8B",    "--", 1.8),
        ]:
            n    = df["layer"].max() + 1
            xpos = df["layer"] / (n - 1)
            for l1, l2 in pairs:
                col   = f"cka_{l1}_{l2}"
                color = colors[f"{l1}_{l2}"]
                label = f"{l1.upper()}↔{l2.upper()} ({model})" if ax == axes[0] else None
                ax.plot(xpos, df[col], color=color, ls=ls, lw=lw,
                        alpha=0.85, label=label)

        ax.set_xlabel("Normalized Layer", fontsize=11)
        ax.set_ylabel("Linear CKA", fontsize=11)
        ax.set_title(f"{'Gemma 4 E4B' if ax==axes[0] else 'Qwen3-8B'} — CKA",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=8, ncol=2)

    # Actually use both axes correctly
    for ax, (df, model) in zip(axes, [(df_g, "Gemma 4 E4B"), (df_q, "Qwen3-8B")]):
        ax.cla()
        n    = df["layer"].max() + 1
        xpos = df["layer"] / (n - 1)
        for l1, l2 in pairs:
            col   = f"cka_{l1}_{l2}"
            color = colors[f"{l1}_{l2}"]
            ax.plot(xpos, df[col], color=color, lw=2,
                    label=f"{l1.upper()}↔{l2.upper()}")
        ax.set_xlabel("Normalized Layer", fontsize=11)
        ax.set_ylabel("Linear CKA", fontsize=11)
        ax.set_title(model, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_cka_compare.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig13_cka_compare.png")


def fig_transfer_compare(gemma_tr, qwen_tr):
    print("\n[Fig] Transfer comparison...")
    pairs  = [("en","ru"), ("en","ky"), ("ru","ky")]
    colors = {"en_ru": "#E53935", "en_ky": "#1E88E5", "ru_ky": "#43A047"}
    labels_map = {"en_ru": "EN→RU", "en_ky": "EN→KY", "ru_ky": "RU→KY"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-lingual Transfer Probing: Gemma 4 E4B vs Qwen3-8B",
                 fontsize=13)

    for ax, (df, model) in zip(axes, [(gemma_tr, "Gemma 4 E4B"),
                                      (qwen_tr,  "Qwen3-8B")]):
        n = df["layer"].max() + 1
        for src, tgt in pairs:
            sub   = df[(df.src==src)&(df.tgt==tgt)].sort_values("layer")
            xpos  = sub["layer"] / (n - 1)
            key   = f"{src}_{tgt}"
            ax.plot(xpos, sub["transfer_acc"],
                    color=colors[key], lw=2, label=labels_map[key])
        ax.axhline(1/6, ls=":", color="gray", lw=1.2, label="Chance")
        ax.set_xlabel("Normalized Layer", fontsize=11)
        ax.set_ylabel("Transfer Accuracy", fontsize=11)
        ax.set_title(model, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_transfer_compare.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig14_transfer_compare.png")


def fig_emotion_compare(gemma_em, qwen_em):
    print("\n[Fig] Per-emotion best layer comparison...")

    # Table: best layer (normalized) per emotion per language per model
    rows = []
    for emotion in EMOTIONS:
        row = {"emotion": emotion}
        for model, df in [("gemma4", gemma_em), ("qwen3", qwen_em)]:
            n = df["layer"].max() + 1
            for lang in ["en", "ru", "ky"]:
                sub  = df[(df.lang==lang)&(df.emotion==emotion)]
                best = sub.loc[sub.ovr_acc.idxmax()]
                row[f"{model}_{lang}_layer"] = int(best.layer)
                row[f"{model}_{lang}_layer_norm"] = best.layer / (n-1)
                row[f"{model}_{lang}_acc"]   = round(best.ovr_acc, 3)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(QWEN_RES_DIR / "emotion_comparison_summary.csv", index=False)

    # Heatmap: best accuracy per emotion × language × model
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Per-emotion OvR Accuracy: Gemma 4 E4B vs Qwen3-8B\n"
                 "Solid = Gemma 4, Dashed = Qwen3", fontsize=13)

    EMOTION_COLORS = {
        "anger": "#E53935", "disgust": "#8E24AA", "fear": "#FB8C00",
        "joy": "#F9A825",   "sadness": "#1E88E5", "surprise": "#43A047",
    }

    for ax, emotion in zip(axes.flatten(), EMOTIONS):
        for lang in ["en", "ru", "ky"]:
            color = LANG_COLORS[lang]
            for df, model, ls, lw in [(gemma_em, "Gemma 4", "-", 2.2),
                                      (qwen_em,  "Qwen3",  "--", 1.8)]:
                sub  = df[(df.lang==lang)&(df.emotion==emotion)].sort_values("layer")
                n    = sub["layer"].max() + 1
                xpos = sub["layer"] / (n - 1)
                ax.plot(xpos, sub["ovr_acc"], color=color, ls=ls, lw=lw,
                        label=f"{LANG_LABELS[lang]} ({model})")

        ax.axhline(0.5, ls=":", color="gray", lw=1)
        ax.set_title(emotion.capitalize(), fontsize=12,
                     color=EMOTION_COLORS[emotion], fontweight="bold")
        ax.set_xlabel("Normalized Layer", fontsize=9)
        ax.set_ylabel("OvR Accuracy", fontsize=9)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_emotion_compare.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig15_emotion_compare.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    QWEN_RES_DIR.mkdir(parents=True, exist_ok=True)

    # Load Gemma 4 results (already computed)
    print("\nLoading Gemma 4 results...")
    gemma_probe    = pd.read_csv(GEMMA_RES_DIR / "probing_detailed.csv")
    gemma_cka      = pd.read_csv(GEMMA_RES_DIR / "cka_results.csv")
    gemma_transfer = pd.read_csv(GEMMA_RES_DIR / "transfer_results.csv")
    gemma_emotion  = pd.read_csv(GEMMA_RES_DIR / "per_emotion_layers.csv")

    # Run Qwen3 experiments
    qwen_probe, qwen_cka, qwen_transfer, qwen_emotion = run_all(
        QWEN_HS_DIR, QWEN_RES_DIR, "qwen3")

    # Build comparison figures
    print("\nBuilding comparison figures...")
    fig_probing_compare(gemma_probe,    qwen_probe)
    fig_cka_compare(gemma_cka,          qwen_cka)
    fig_transfer_compare(gemma_transfer, qwen_transfer)
    fig_emotion_compare(gemma_emotion,   qwen_emotion)

    # Print summary comparison table
    print("\n" + "="*60)
    print("SUMMARY: Best probing accuracy per language")
    print(f"{'Lang':<6} {'Gemma4 layer':>13} {'Gemma4 acc':>11} "
          f"{'Qwen3 layer':>12} {'Qwen3 acc':>10}")
    for lang in ["en", "ru", "ky"]:
        g = gemma_probe[gemma_probe.lang==lang]
        q = qwen_probe[qwen_probe.lang==lang]
        gb = g.loc[g.acc_mean.idxmax()]
        qb = q.loc[q.acc_mean.idxmax()]
        print(f"{lang.upper():<6} {int(gb.layer):>13} {gb.acc_mean:>11.3f} "
              f"{int(qb.layer):>12} {qb.acc_mean:>10.3f}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
