"""
Step 49: Paired bootstrap for within-language decoder--decoder probing
differences, using best layers from unified_probing.csv (the single
source of truth for Table 3). scripts/20 used out-of-date best layers
(Qwen3 KY=L27, Mistral KY=L2) which no longer match Table 3.

Outputs:
  data/results/tables/probing_pairwise_unified.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path("data/processed")
RES = Path("data/results")
UNIFIED = RES / "tables" / "unified_probing.csv"
OUT = RES / "tables" / "probing_pairwise_unified.csv"

N_FOLDS, N_EPOCHS, LR, WD, SEED = 5, 300, 1e-2, 1e-4, 42
N_BOOT = 10_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HS_DIRS = {
    "Gemma 4 E4B":   DATA_ROOT,
    "Qwen3-8B":      DATA_ROOT / "qwen3",
    "Llama-3.1-8B":  DATA_ROOT / "llama",
    "Mistral-7B":    DATA_ROOT / "mistral",
}


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def per_example_correct(X, y, seed=SEED):
    """5-fold CV: return per-example bool of correct predictions (aligned to y)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    correct = np.zeros(len(y), dtype=bool)
    n_cls = len(np.unique(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),     dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte_np = y[te]
        m = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        ce = nn.CrossEntropyLoss()
        torch.manual_seed(seed)
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(Xte).argmax(1).cpu().numpy()
        correct[te] = (pred == yte_np)
    return correct


def bootstrap_p(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a)
    real = a.mean() - b.mean()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = a[idx].mean() - b[idx].mean()
    if real >= 0:
        p = 2 * min((boot <= 0).mean(), 0.5)
    else:
        p = 2 * min((boot >= 0).mean(), 0.5)
    return real, min(p, 1.0)


def holm(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, pvals[i] * (n - rank))
        v = max(v, prev)
        adj[i] = v
        prev = v
    return adj


def main():
    unified = pd.read_csv(UNIFIED)
    best = {(r["model"], r["lang"]): int(r["layer"])
            for _, r in unified.iterrows()
            if r["model"] in HS_DIRS}

    correct = {}
    for model, hs_dir in HS_DIRS.items():
        labels = np.load(hs_dir / "labels.npy")
        for lang in ["EN", "RU", "KY"]:
            layer = best[(model, lang)]
            hs = np.load(hs_dir / f"hidden_states_{lang.lower()}.npz")["hidden_states"]
            X = hs[:, layer, :]
            print(f"  {model:>16} / {lang} (L{layer})…", end=" ", flush=True)
            c = per_example_correct(X, labels)
            correct[(model, lang)] = c
            print(f"acc={c.mean():.3f}")

    models = list(HS_DIRS)
    rows = []
    for lang in ["EN", "RU", "KY"]:
        pairs, deltas, ps = [], [], []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                d, p = bootstrap_p(correct[(m1, lang)], correct[(m2, lang)])
                pairs.append((m1, m2)); deltas.append(d); ps.append(p)
        holm_ps = holm(ps)
        for (m1, m2), d, p, ph in zip(pairs, deltas, ps, holm_ps):
            hi, lo = (m1, m2) if d >= 0 else (m2, m1)
            sig = "***" if ph < .001 else ("**" if ph < .01 else ("*" if ph < .05 else "ns"))
            rows.append({"lang": lang, "model_hi": hi, "model_lo": lo,
                         "delta": round(abs(d), 4),
                         "p_raw": round(p, 4), "p_holm": round(ph, 4),
                         "sig": sig})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
