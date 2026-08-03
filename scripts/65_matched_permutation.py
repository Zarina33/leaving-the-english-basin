"""
Step 65: Matched-probe permutation null for one cell.

The 10k permutation nulls (scripts/37) use a lighter probe (single
80/20 split, 60 epochs) than the real measurement (5-fold CV, 300
epochs) -- an asymmetry disclosed in Limitations. Here we rerun one
cell with the EXACT real protocol (5-fold StratifiedKFold seed 42,
StandardScaler, Adam lr 1e-2 wd 1e-4, 300 epochs) on permuted labels.

Cell: Mistral KY at its best layer (L5) -- the weakest real accuracy
among the 15 permutation-tested cells (.451), i.e. the smallest
margin over any plausible null.

N_PERM = 1000 (p-resolution 1e-3; the real margin over the null max
is what carries the claim).
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MODEL, LANG, LAYER = "mistral", "ky", 5
REAL_ACC = 0.451
N_PERM = 1000
N_FOLDS, N_EPOCHS, LR, WD, SEED = 5, 300, 1e-2, 1e-4, 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("data/results/tables/matched_permutation.csv")


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


def cv_acc(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)
        torch.manual_seed(SEED)
        m = LinClf(Xtr.shape[1], len(np.unique(y))).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        ce = nn.CrossEntropyLoss()
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad()
            ce(m(Xtr), ytr).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            accs.append((m(Xte).argmax(1) == yte).float().mean().item())
    return float(np.mean(accs))


def main():
    d = np.load(f"data/processed/{MODEL}/hidden_states_{LANG}.npz")
    X = d["hidden_states"][:, LAYER, :].astype(np.float32)
    y = np.load(f"data/processed/{MODEL}/labels.npy")
    print(f"{MODEL} {LANG} L{LAYER}: X {X.shape}, y {np.bincount(y)}")

    real = cv_acc(X, y)
    print(f"real matched-probe acc: {real:.4f} (paper: {REAL_ACC})")

    rng = np.random.default_rng(SEED)
    null = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        null.append(cv_acc(X, yp))
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{N_PERM}  null max so far {max(null):.4f}", flush=True)
    null = np.array(null)
    p = (1 + (null >= real).sum()) / (1 + N_PERM)
    print(f"null mean {null.mean():.4f}  max {null.max():.4f}  p={p:.4g}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("model,lang,layer,real_acc,null_mean,null_max,n_perm,p\n")
        f.write(f"{MODEL},{LANG},{LAYER},{real:.4f},{null.mean():.4f},{null.max():.4f},{N_PERM},{p:.4g}\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
