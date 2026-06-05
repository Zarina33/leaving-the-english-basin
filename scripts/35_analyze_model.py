"""
Step 35: Generic single-model analysis — probing, CKA, cross-lingual transfer,
and per-emotion one-vs-rest — for ANY model whose hidden states already exist.

Use this for newly added models (script 33) and mBERT (script 34) without
editing the hard-coded 4-model pipeline (script 18). Same classifier and
metrics as scripts/8a and 18, so results are directly comparable.

Usage:
    python3 scripts/35_analyze_model.py --model llama32_3b
    python3 scripts/35_analyze_model.py --model qwen25_7b
    python3 scripts/35_analyze_model.py --model olmo2_7b
    python3 scripts/35_analyze_model.py --model mbert

Reads:  data/processed/{model}/hidden_states_{en,ru,ky}.npz + labels.npy
Writes: data/results/{model}/{probing_detailed,cka_results,transfer_results,
        per_emotion_layers}.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_ROOT = Path("data/processed")
RES_ROOT  = Path("data/results")

N_FOLDS, N_EPOCHS, LR_RATE, WD, SEED = 5, 300, 1e-2, 1e-4, 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def _fit(Xtr, ytr, n_cls, weight=None):
    m = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=WD)
    ce = nn.CrossEntropyLoss(weight=weight)
    torch.manual_seed(SEED)
    m.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
    m.eval()
    return m


def gpu_probe(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_cls, accs = len(np.unique(y)), []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),     dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)
        m = _fit(Xtr, ytr, n_cls)
        with torch.no_grad():
            accs.append((m(Xte).argmax(1) == yte).float().mean().item())
    return accs


def transfer(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X_te),     dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.long, device=DEVICE)
    m = _fit(Xtr, ytr, len(np.unique(y_tr)))
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (den + 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="subdir under data/processed/")
    ap.add_argument("--force", action="store_true", help="recompute even if CSVs exist")
    args = ap.parse_args()

    hs_dir  = DATA_ROOT / args.model
    res_dir = RES_ROOT / args.model
    res_dir.mkdir(parents=True, exist_ok=True)

    missing = [l for l in ["en", "ru", "ky"]
               if not (hs_dir / f"hidden_states_{l}.npz").exists()]
    if missing:
        raise SystemExit(f"Missing hidden states for {args.model}: {missing}. "
                         f"Run extraction (script 33/34) first.")

    labels = np.load(hs_dir / "labels.npy")
    hs = {l: np.load(hs_dir / f"hidden_states_{l}.npz")["hidden_states"]
          for l in ["en", "ru", "ky"]}
    n_layers = hs["en"].shape[1]
    print(f"{args.model}: layers={n_layers}, hidden={hs['en'].shape[2]}")

    # 1. Probing
    p = res_dir / "probing_detailed.csv"
    if p.exists() and not args.force:
        print("[1] probing exists — skip")
    else:
        print("[1] layer-wise probing...")
        rec = []
        for lang in ["en", "ru", "ky"]:
            for layer in tqdm(range(n_layers), desc=f"  {lang.upper()}", leave=False):
                fa = gpu_probe(hs[lang][:, layer, :], labels)
                rec.append({"lang": lang, "layer": layer,
                            "acc_mean": np.mean(fa), "acc_std": np.std(fa),
                            **{f"fold_{i}": a for i, a in enumerate(fa)}})
        pd.DataFrame(rec).to_csv(p, index=False)
        print(f"  saved {p}")

    # 2. CKA
    p = res_dir / "cka_results.csv"
    if p.exists() and not args.force:
        print("[2] cka exists — skip")
    else:
        print("[2] linear CKA...")
        rec = []
        for layer in tqdm(range(n_layers), desc="  layers", leave=False):
            row = {"layer": layer}
            for a, b in [("en", "ru"), ("en", "ky"), ("ru", "ky")]:
                row[f"cka_{a}_{b}"] = linear_cka(
                    hs[a][:, layer, :].astype(np.float32),
                    hs[b][:, layer, :].astype(np.float32))
            rec.append(row)
        pd.DataFrame(rec).to_csv(p, index=False)
        print(f"  saved {p}")

    # 3. Transfer
    p = res_dir / "transfer_results.csv"
    if p.exists() and not args.force:
        print("[3] transfer exists — skip")
    else:
        print("[3] cross-lingual transfer...")
        rec = []
        for src, tgt in [("en","ru"),("en","ky"),("ru","en"),("ru","ky"),("ky","en"),("ky","ru")]:
            for layer in tqdm(range(n_layers), desc=f"  {src}->{tgt}", leave=False):
                acc = transfer(hs[src][:, layer, :], labels, hs[tgt][:, layer, :], labels)
                rec.append({"src": src, "tgt": tgt, "layer": layer, "transfer_acc": acc})
        pd.DataFrame(rec).to_csv(p, index=False)
        print(f"  saved {p}")

    # 4. Per-emotion OvR (class-weighted)
    p = res_dir / "per_emotion_layers.csv"
    if p.exists() and not args.force:
        print("[4] per-emotion exists — skip")
    else:
        print("[4] per-emotion one-vs-rest...")
        rec = []
        for lang in ["en", "ru", "ky"]:
            for eidx, emo in enumerate(EMOTIONS):
                y_bin = (labels == eidx).astype(int)
                for layer in tqdm(range(n_layers), desc=f"  {lang}/{emo}", leave=False):
                    X = hs[lang][:, layer, :]
                    rng = np.random.default_rng(SEED)
                    idx = rng.permutation(len(X)); sp = int(len(X) * 0.8)
                    tr, te = idx[:sp], idx[sp:]
                    sc = StandardScaler()
                    Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
                    Xte = torch.tensor(sc.transform(X[te]),     dtype=torch.float32, device=DEVICE)
                    ytr = torch.tensor(y_bin[tr], dtype=torch.long, device=DEVICE)
                    yte = torch.tensor(y_bin[te], dtype=torch.long, device=DEVICE)
                    n_pos = int(ytr.sum().item()); n_neg = len(ytr) - n_pos
                    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
                    m = _fit(Xtr, ytr, 2, weight=w)
                    with torch.no_grad():
                        acc = (m(Xte).argmax(1) == yte).float().mean().item()
                    rec.append({"lang": lang, "emotion": emo, "layer": layer, "ovr_acc": acc})
        pd.DataFrame(rec).to_csv(p, index=False)
        print(f"  saved {p}")

    # Quick summary
    pf = pd.read_csv(res_dir / "probing_detailed.csv")
    print("\nBest probing layer per language:")
    for lang in ["en", "ru", "ky"]:
        sub = pf[pf["lang"] == lang]
        best = sub.loc[sub["acc_mean"].idxmax()]
        print(f"  {lang.upper()}: layer {int(best['layer'])}/{n_layers-1} "
              f"(norm {best['layer']/(n_layers-1):.2f}) acc={best['acc_mean']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
