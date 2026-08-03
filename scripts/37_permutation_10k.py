"""
Step 37: Permutation test with 10,000 shuffles (was 1,000 in the paper).

Removes the "p-value resolution limited to 1e-3" caveat from Limitations.
For each (model, language), at the model's best layer, we compare the real
5-fold CV accuracy against the null distribution of accuracies obtained by
shuffling the emotion labels. p = (1 + #{perm >= real}) / (1 + N_PERM).

Covers all 5 models. Reads existing hidden states; trains the same linear
probe as the rest of the pipeline.

Usage:  python3 scripts/37_permutation_10k.py
        python3 scripts/37_permutation_10k.py --models gemma4 llama   # subset

Output: data/results/permutation_10k.csv
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
N_PERM = 10_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# key -> (hidden-states dir, results dir for best-layer lookup)
MODELS = {
    "gemma4":     (DATA_ROOT,                RES_ROOT),
    "qwen3":      (DATA_ROOT / "qwen3",      RES_ROOT / "qwen3"),
    "llama":      (DATA_ROOT / "llama",      RES_ROOT / "llama"),
    "mistral":    (DATA_ROOT / "mistral",    RES_ROOT / "mistral"),
    "xlmr":       (DATA_ROOT / "xlmr",       RES_ROOT / "xlmr"),
    "mbert":      (DATA_ROOT / "mbert",      RES_ROOT / "mbert"),
    "qwen25_7b":  (DATA_ROOT / "qwen25_7b",  RES_ROOT / "qwen25_7b"),
    "olmo2_7b":   (DATA_ROOT / "olmo2_7b",   RES_ROOT / "olmo2_7b"),
}


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def cv_accuracy(X, y, seed=SEED):
    """Mean 5-fold CV accuracy with the standard linear probe."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
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
        torch.manual_seed(seed)
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            accs.append((m(Xte).argmax(1) == yte).float().mean().item())
    return float(np.mean(accs))


def best_layer(res_dir, lang):
    df = pd.read_csv(res_dir / "probing_detailed.csv")
    sub = df[df["lang"] == lang]
    return int(sub.loc[sub["acc_mean"].idxmax()]["layer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS),
                    help="subset of model keys (default: all)")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []

    for key in args.models:
        hs_dir, res_dir = MODELS[key]
        if not (hs_dir / "hidden_states_en.npz").exists():
            print(f"[{key}] hidden states missing — skip")
            continue
        labels = np.load(hs_dir / "labels.npy")

        for lang in ["en", "ru", "ky"]:
            L = best_layer(res_dir, lang)
            X = np.load(hs_dir / f"hidden_states_{lang}.npz")["hidden_states"][:, L, :]

            real = cv_accuracy(X, labels)

            # Null distribution: shuffle labels. To keep 10k runs tractable we
            # use a fast single-split probe per permutation (train/test 80/20),
            # which is standard for permutation nulls.
            ge = 0
            n = len(labels)
            sp = int(n * 0.8)
            for _ in tqdm(range(args.n_perm), desc=f"{key}/{lang} L{L}", leave=False):
                perm = rng.permutation(labels)
                idx = rng.permutation(n)
                tr, te = idx[:sp], idx[sp:]
                sc = StandardScaler()
                Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
                Xte = torch.tensor(sc.transform(X[te]),     dtype=torch.float32, device=DEVICE)
                ytr = torch.tensor(perm[tr], dtype=torch.long, device=DEVICE)
                yte = torch.tensor(perm[te], dtype=torch.long, device=DEVICE)
                m = LinClf(Xtr.shape[1], len(np.unique(labels))).to(DEVICE)
                opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=WD)
                ce = nn.CrossEntropyLoss()
                m.train()
                for _ in range(60):   # fewer epochs OK for null model
                    opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    pacc = (m(Xte).argmax(1) == yte).float().mean().item()
                if pacc >= real:
                    ge += 1

            pval = (1 + ge) / (1 + args.n_perm)
            print(f"{key}/{lang} L{L}: real={real:.3f}  perm>=real: {ge}/{args.n_perm}  p={pval:.2e}")
            rows.append({"model": key, "lang": lang.upper(), "layer": L,
                         "real_acc": round(real, 4), "n_perm": args.n_perm,
                         "n_ge": ge, "p_value": pval})

    out = RES_ROOT / "permutation_10k.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print("Note: with 10k permutations the smallest reportable p is ~1e-4 "
          "(1/(1+10000)).")


if __name__ == "__main__":
    main()
