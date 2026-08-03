"""
Step 40: Leakage-free cross-lingual transfer with nested layer selection.

Fixes two reviewer-identified flaws in the original transfer protocol
(scripts/9, 23, 35):

  M1 (parallel-sentence leakage): the corpus is parallel, so training a probe
     on ALL sentences of language A and testing on ALL sentences of language B
     tests the probe on translations of the *same* sentences it trained on.
     The probe can exploit sentence-specific lexical/entity cues rather than
     emotion. FIX: split sentence INDICES into disjoint halves. Train on
     half-1 of the source language, test on half-2 of the target language, so
     no test sentence (in any language) has a parallel twin in training.

  M2 (best-layer selected on the test metric): the peak layer was chosen by
     the same accuracy that is then reported, over 25-43 candidate layers.
     FIX: nested selection. Choose the transfer layer on a DEV split
     (held-out source-language sentences, evaluated cross-lingually on a dev
     slice of the target), then report the FROZEN layer's accuracy on a
     separate test slice.

Protocol per (model, src->tgt direction):
  - Fixed seed. Shuffle the 1480 sentence indices once.
  - Partition indices into TRAIN (60%), DEV (20%), TEST (20%) — disjoint.
  - For each candidate layer: train probe on src[TRAIN], evaluate on tgt[DEV].
  - Pick layer* = argmax dev accuracy.
  - Report accuracy of layer* : train on src[TRAIN], test on tgt[TEST].
  (TRAIN/DEV/TEST are the SAME index sets across languages, but because src and
   tgt use DISJOINT index sets for train vs test, there is no parallel leak:
   test sentences' source-language twins are never in the training set.)

Wait — subtle point: if TRAIN indices are identical for src and tgt, then a
tgt TEST sentence's src twin is in src[TEST], NOT src[TRAIN]. Good. But to be
maximally safe we also ensure the probe never sees a sentence id in test that
it saw (in any language) during training: since train uses index set TR and
test uses index set TE with TR ∩ TE = ∅, and the probe only trains on src[TR],
a tgt[TE] sentence's twin src[te] is never trained on. Leak-free. ✓

Usage:
    python3 scripts/40_transfer_noleak.py --model all
    python3 scripts/40_transfer_noleak.py --model xlmr
Output:
    data/results/transfer_noleak.csv   (model, src, tgt, layer_star, acc, dev_acc)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
OUT = Path("data/results/transfer_noleak.csv")

N_EPOCHS = 300
LR = 1e-2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gemma4 lives at the processed root; others in subdirs.
HS_DIRS = {
    "gemma4":  PROCESSED,
    "qwen3":   PROCESSED / "qwen3",
    "llama":   PROCESSED / "llama",
    "mistral": PROCESSED / "mistral",
    "xlmr":    PROCESSED / "xlmr",
    "mbert":   PROCESSED / "mbert",
    "qwen25_7b": PROCESSED / "qwen25_7b",
    "olmo2_7b":  PROCESSED / "olmo2_7b",
}

DIRECTIONS = [("en", "ru"), ("en", "ky"), ("ru", "en"),
              ("ru", "ky"), ("ky", "en"), ("ky", "ru")]


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def fit_eval(X_tr, y_tr, X_te, y_te, return_correct=False):
    sc = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X_te), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.long, device=DEVICE)
    torch.manual_seed(SEED)
    model = LinClf(Xtr.shape[1], len(np.unique(y_tr))).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad(); ce(model(Xtr), ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
    correct = (preds == yte).cpu().numpy().astype(int)
    if return_correct:
        return correct.mean(), correct
    return correct.mean()


def run_model(key):
    d = HS_DIRS[key]
    if not (d / "hidden_states_en.npz").exists():
        print(f"[!] {key}: no hidden states — skip")
        return []
    labels = np.load(d / "labels.npy")
    hs = {lg: np.load(d / f"hidden_states_{lg}.npz")["hidden_states"]
          for lg in ["en", "ru", "ky"]}
    n, n_layers, _ = hs["en"].shape

    # One fixed disjoint partition of sentence indices, shared across languages.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_tr, n_dev = int(0.6 * n), int(0.2 * n)
    TR = idx[:n_tr]
    DEV = idx[n_tr:n_tr + n_dev]
    TE = idx[n_tr + n_dev:]

    rows = []
    correct_vecs = {}  # (src,tgt) -> per-example correctness on TE
    for src, tgt in DIRECTIONS:
        # nested: pick layer on DEV, report on TE
        dev_accs = []
        for L in range(n_layers):
            a = fit_eval(hs[src][TR][:, L, :], labels[TR],
                         hs[tgt][DEV][:, L, :], labels[DEV])
            dev_accs.append(a)
        L_star = int(np.argmax(dev_accs))
        test_acc, correct = fit_eval(hs[src][TR][:, L_star, :], labels[TR],
                                     hs[tgt][TE][:, L_star, :], labels[TE],
                                     return_correct=True)
        correct_vecs[(src, tgt)] = correct
        rows.append({"model": key, "src": src, "tgt": tgt,
                     "layer_star": L_star,
                     "dev_acc": round(dev_accs[L_star], 4),
                     "test_acc": round(test_acc, 4)})
        print(f"  {key} {src}->{tgt}: L*={L_star} dev={dev_accs[L_star]:.3f} "
              f"TEST={test_acc:.3f}")
    # save correctness vectors for pairwise tests (scripts/44)
    cdir = Path("data/results/transfer_correct")
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cdir / f"{key}.npz",
                        **{f"{s}_{t}": v for (s, t), v in correct_vecs.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    args = ap.parse_args()
    keys = list(HS_DIRS) if args.model == "all" else [args.model]

    import pandas as pd
    all_rows = []
    for k in keys:
        print(f"\n=== {k} ===")
        all_rows += run_model(k)
    df = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # merge with any prior partial results
    if OUT.exists() and args.model != "all":
        prev = pd.read_csv(OUT)
        prev = prev[prev.model != args.model]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT} ({len(df)} rows)")

    # KY-transfer summary table (the claim that matters)
    print("\n=== KY-involving transfer (test_acc), leakage-free ===")
    ky_dirs = [("en","ky"),("ru","ky"),("ky","en"),("ky","ru")]
    for k in df.model.unique():
        sub = df[(df.model==k) & (df[["src","tgt"]].apply(tuple,axis=1).isin(ky_dirs))]
        if len(sub):
            print(f"  {k:10s} KY_avg={sub.test_acc.mean():.3f}  "
                  + " ".join(f"{r.src}->{r.tgt}:{r.test_acc:.3f}" for _,r in sub.iterrows()))


if __name__ == "__main__":
    main()
