"""
Step 30: Probing comparison — Claude-translated vs. human-translated subset.

For each (model, language ∈ {ru, ky}):
  1. Hold out the 102 control IDs from the full corpus.
  2. Train a linear probe on the remaining 1378 Claude-translated examples.
  3. Test on:
     (a) Claude versions of the 102 control IDs → acc_claude
     (b) Human versions of the same 102          → acc_human
  4. Bootstrap 95% CI for the difference (acc_claude − acc_human).

If acc_claude ≈ acc_human → no LLM-translation confound (good).
If acc_claude ≫ acc_human → Claude translations are easier to probe (confound).

Probe = PyTorch linear logistic regression (Adam, lr=1e-2, wd=1e-4, 300 epochs),
identical to scripts/8a_probing_gpu.py.

Layers tested = the paper's reported best layers per (model, lang).

Output: data/results/human_vs_claude_probing.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS     = 300
LR_RATE      = 1e-2
WD           = 1e-4
RANDOM_STATE = 42
N_BOOTSTRAP  = 10_000

# (model_key, dir_name, best layers per lang as reported in Table 2 of paper)
MODELS = [
    ("Gemma 4",   "",        {"ru": 6,  "ky": 4}),
    ("Qwen3",     "qwen3",   {"ru": 11, "ky": 27}),
    ("Llama 3.1", "llama",   {"ru": 7,  "ky": 3}),
    ("Mistral",   "mistral", {"ru": 9,  "ky": 2}),
    ("XLM-R",     "xlmr",    {"ru": 12, "ky": 18}),
]


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def train_probe(X_train, y_train, n_cls, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    sc = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_train), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    model = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=WD)
    ce    = nn.CrossEntropyLoss()

    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        ce(model(Xtr), ytr).backward()
        opt.step()
    model.eval()
    return model, sc


def predict(model, scaler, X_test):
    X = torch.tensor(scaler.transform(X_test), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model(X).argmax(1).cpu().numpy()


def bootstrap_diff_ci(correct_a, correct_b, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Bootstrap 95% CI for the difference of two paired accuracy vectors."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    N = len(correct_a)
    for i in range(n):
        idx = rng.integers(0, N, N)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_model(model_key, subdir, best_layers, control_idx_in_full):
    base = DATA_DIR / subdir if subdir else DATA_DIR
    labels = np.load(base / "labels.npy") if (base / "labels.npy").exists() else np.load(DATA_DIR / "labels.npy")
    n_cls  = len(np.unique(labels))

    print(f"\n=== {model_key} (dir: {subdir or 'root'}) ===")

    rows = []
    for lang in ["ru", "ky"]:
        layer = best_layers[lang]

        # Claude hidden states (full corpus, 1480 sentences)
        hs_claude = np.load(base / f"hidden_states_{lang}.npz")["hidden_states"]
        # Human hidden states (control sample, 102 sentences in same order as control_ids.npy)
        hs_human  = np.load(base / f"hidden_states_human_{lang}.npz")["hidden_states"]

        # Sanity checks
        assert hs_claude.shape[0] == 1480, f"Claude full shape: {hs_claude.shape}"
        assert hs_human.shape[0]  == 102,  f"Human shape: {hs_human.shape}"
        assert hs_claude.shape[1] == hs_human.shape[1], "Layer count mismatch"
        assert hs_claude.shape[2] == hs_human.shape[2], "Hidden dim mismatch"

        # Split: train on 1378 Claude (non-control), test on Claude/human 102
        all_idx     = np.arange(hs_claude.shape[0])
        train_mask  = ~np.isin(all_idx, control_idx_in_full)
        train_idx   = all_idx[train_mask]

        X_train = hs_claude[train_idx, layer, :]
        y_train = labels[train_idx]

        X_test_claude = hs_claude[control_idx_in_full, layer, :]
        y_test        = labels[control_idx_in_full]

        X_test_human  = hs_human[:, layer, :]    # already in control order

        # Train probe
        model, scaler = train_probe(X_train, y_train, n_cls)

        # Predict
        preds_claude = predict(model, scaler, X_test_claude)
        preds_human  = predict(model, scaler, X_test_human)

        correct_claude = (preds_claude == y_test).astype(int)
        correct_human  = (preds_human  == y_test).astype(int)

        acc_claude = correct_claude.mean()
        acc_human  = correct_human.mean()
        diff = acc_claude - acc_human

        ci_low, ci_high = bootstrap_diff_ci(correct_claude, correct_human)

        print(f"  {lang.upper()} layer={layer}: "
              f"Claude={acc_claude:.3f}  Human={acc_human:.3f}  "
              f"Δ={diff:+.3f}  95% CI [{ci_low:+.3f}, {ci_high:+.3f}]")

        rows.append({
            "model":      model_key,
            "lang":       lang.upper(),
            "layer":      layer,
            "n_train":    int(len(train_idx)),
            "n_test":     int(len(control_idx_in_full)),
            "acc_claude": round(acc_claude, 4),
            "acc_human":  round(acc_human, 4),
            "delta":      round(diff, 4),
            "ci95_low":   round(ci_low, 4),
            "ci95_high":  round(ci_high, 4),
        })

    return rows


def main():
    # Load control IDs & find their positions in the full corpus
    control_ids = np.load(DATA_DIR / "control_ids.npy", allow_pickle=True)
    full = pd.read_csv("data/translated/parallel_corpus_clean.csv")
    full_id_to_idx = {row_id: i for i, row_id in enumerate(full["id"].tolist())}
    control_idx_in_full = np.array([full_id_to_idx[cid] for cid in control_ids])
    print(f"Control sample: {len(control_idx_in_full)} sentences, "
          f"indices range [{control_idx_in_full.min()}, {control_idx_in_full.max()}]")

    # Sanity check: labels for these indices match what the human-translated CSV expects
    labels_root = np.load(DATA_DIR / "labels.npy")
    control_csv = pd.read_csv("data/translated/human_control_v2.csv")
    EMO = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    expected_labels = np.array([EMO.index(e) for e in control_csv["emotion"]])
    assert (labels_root[control_idx_in_full] == expected_labels).all(), \
        "Label mismatch between full corpus and control CSV — ordering issue!"
    print("Label sanity check passed.")

    all_rows = []
    for model_key, subdir, best_layers in MODELS:
        all_rows.extend(run_model(model_key, subdir, best_layers, control_idx_in_full))

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "human_vs_claude_probing.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
