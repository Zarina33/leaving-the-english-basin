"""
Step 32: Layer-wise probing on LAST-TOKEN-pooled hidden states + comparison
with the mean-pooling results used in the paper.

Answers the ablation question: does the "early-peaking" depth pattern
(Gemma 4 ≈ layer 0, Llama ≈ layer 1) survive under last-token pooling, or is
it an artifact of mean pooling?

Prerequisite:  scripts/31_extract_lasttoken.py  (--model all)
  → data/processed/{model}_lasttok/hidden_states_{en,ru,ky}.npz

Outputs:
  data/results/{model}_lasttok/probing_detailed.csv   (per model)
  data/results/lasttoken_vs_mean.csv                  (comparison table)
  data/results/figures/fig25_pooling_ablation.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_ROOT = Path("data/processed")
RES_ROOT  = Path("data/results")
FIG_DIR   = RES_ROOT / "figures"

N_FOLDS, N_EPOCHS, LR_RATE, WD, SEED = 5, 300, 1e-2, 1e-4, 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (key, last-token HS dir, mean-pooling results dir, label)
MODELS = [
    ("gemma4",  "gemma4_lasttok",  "",        "Gemma 4 E4B"),
    ("qwen3",   "qwen3_lasttok",   "qwen3",   "Qwen3-8B"),
    ("llama",   "llama_lasttok",   "llama",   "Llama-3.1-8B"),
    ("mistral", "mistral_lasttok", "mistral", "Mistral-7B-v0.3"),
]


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_probe(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_cls, accs = len(np.unique(y)), []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),     dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)
        m = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=WD)
        ce = nn.CrossEntropyLoss()
        torch.manual_seed(SEED)
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            accs.append((m(Xte).argmax(1) == yte).float().mean().item())
    return accs


def probe_all_layers(hs_dir):
    labels = np.load(hs_dir / "labels.npy")
    hs = {l: np.load(hs_dir / f"hidden_states_{l}.npz")["hidden_states"]
          for l in ["en", "ru", "ky"]}
    n_layers = hs["en"].shape[1]
    records = []
    for lang in ["en", "ru", "ky"]:
        for layer in tqdm(range(n_layers), desc=f"  {lang.upper()}", leave=False):
            fa = gpu_probe(hs[lang][:, layer, :], labels)
            records.append({"lang": lang, "layer": layer,
                            "acc_mean": np.mean(fa), "acc_std": np.std(fa)})
    return pd.DataFrame(records)


def best_row(df, lang):
    sub = df[df["lang"] == lang]
    return sub.loc[sub["acc_mean"].idxmax()]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    comparison = []
    lasttok_dfs = {}

    for key, lt_dir, mean_dir, label in MODELS:
        lt_path = DATA_ROOT / lt_dir
        if not (lt_path / "hidden_states_en.npz").exists():
            print(f"[{key}] last-token HS missing — run scripts/31_extract_lasttoken.py first. Skipping.")
            continue

        print(f"\n=== {label}: probing last-token states ===")
        out_dir = RES_ROOT / lt_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        lt_probe_path = out_dir / "probing_detailed.csv"
        if lt_probe_path.exists():
            lt_df = pd.read_csv(lt_probe_path)
        else:
            lt_df = probe_all_layers(lt_path)
            lt_df.to_csv(lt_probe_path, index=False)
        lasttok_dfs[key] = lt_df

        # Mean-pooling results (from paper pipeline)
        mean_probe_path = (RES_ROOT / mean_dir / "probing_detailed.csv") if mean_dir \
                          else (RES_ROOT / "probing_detailed.csv")
        mean_df = pd.read_csv(mean_probe_path)

        for lang in ["en", "ru", "ky"]:
            n_layers = lt_df["layer"].max() + 1
            mb, lb = best_row(mean_df, lang), best_row(lt_df, lang)
            comparison.append({
                "model": label, "lang": lang.upper(),
                "mean_best_layer":  int(mb["layer"]),
                "mean_best_layer_norm": round(mb["layer"] / (n_layers - 1), 2),
                "mean_acc":  round(mb["acc_mean"], 3),
                "last_best_layer":  int(lb["layer"]),
                "last_best_layer_norm": round(lb["layer"] / (n_layers - 1), 2),
                "last_acc":  round(lb["acc_mean"], 3),
                "layer_shift": int(lb["layer"]) - int(mb["layer"]),
                "acc_shift":  round(lb["acc_mean"] - mb["acc_mean"], 3),
            })

    comp_df = pd.DataFrame(comparison)
    comp_path = RES_ROOT / "lasttoken_vs_mean.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\nComparison table → {comp_path}")
    print(comp_df.to_string(index=False))

    # ── figure: normalized best-layer, mean vs last-token ────────────────────
    if not comp_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, lang in zip(axes, ["EN", "RU", "KY"]):
            sub = comp_df[comp_df["lang"] == lang]
            x = np.arange(len(sub))
            ax.bar(x - 0.2, sub["mean_best_layer_norm"], width=0.4, label="Mean pooling", color="#1565C0")
            ax.bar(x + 0.2, sub["last_best_layer_norm"], width=0.4, label="Last-token", color="#E65100")
            ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=30, ha="right", fontsize=8)
            ax.set_title(lang); ax.set_ylabel("Normalized best layer"); ax.set_ylim(0, 1)
            ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        plt.suptitle("Best probing layer: mean vs last-token pooling", fontsize=12)
        plt.tight_layout()
        path = FIG_DIR / "fig25_pooling_ablation.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure → {path}")

    print("\nInterpretation guide:")
    print("  If `last_best_layer_norm` stays LOW for Gemma 4 / Llama → early-peaking holds.")
    print("  If it shifts to mid/late under last-token → pattern was pooling-dependent.")


if __name__ == "__main__":
    main()
