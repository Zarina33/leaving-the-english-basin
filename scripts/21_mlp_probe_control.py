"""
Step 21: MLP probe + control task (selectivity analysis).

Addresses reviewer concern: "Are probes learning emotions or memorizing vocabulary?"

1. Linear probe (existing) vs 2-layer MLP probe at the best layer.
   If MLP >> linear, then the linear probe underestimates what the model knows.
   If MLP ≈ linear, then the representation is linearly structured.

2. Control task: probe on SHUFFLED emotion labels.
   Selectivity = real_acc - control_acc.
   High selectivity = probe learned emotions, not just general structure.

3. Sentence-length control: probe accuracy binned by token count,
   to rule out length as a confound for early-peaking.

Output:
  data/results/tables/probe_selectivity.csv
  data/results/tables/length_control.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

HS_DIRS = {
    "gemma4":  Path("data/processed"),
    "qwen3":   Path("data/processed/qwen3"),
    "mistral": Path("data/processed/mistral"),
    "llama":   Path("data/processed/llama"),
}
CORPUS_PATH = Path("data/translated/parallel_corpus_clean.csv")
TABLE_DIR = Path("data/results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS      = 5
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MODEL_LABELS = {
    "gemma4": "Gemma 4 E4B", "qwen3": "Qwen3-8B",
    "mistral": "Mistral-7B",  "llama": "Llama-3.1-8B",
}

BEST_LAYERS = {
    "gemma4":  {"en": 0,  "ru": 6,  "ky": 4},
    "qwen3":   {"en": 9,  "ru": 11, "ky": 27},
    "mistral": {"en": 5,  "ru": 9,  "ky": 2},
    "llama":   {"en": 1,  "ru": 7,  "ky": 3},
}


# ── Probe architectures ──────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, d, k, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, k),
        )
    def forward(self, x):
        return self.net(x)


def run_probe(X, y, probe_cls, **kwargs):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    accs = []
    n_cls = len(np.unique(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]),
                           dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(sc.transform(X[te]),
                           dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = torch.tensor(y[te], dtype=torch.long, device=DEVICE)

        m = probe_cls(Xtr.shape[1], n_cls, **kwargs).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds = m(Xte).argmax(1)
        accs.append((preds == yte).float().mean().item())
    return np.mean(accs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    models = [k for k in HS_DIRS if (HS_DIRS[k] / "hidden_states_en.npz").exists()]
    df = pd.read_csv(CORPUS_PATH)

    # ── Part 1: Linear vs MLP + Control task ──────────────────────────────────
    print("[1] Linear vs MLP probe + control task (shuffled labels)...\n")
    rows = []

    for model in models:
        labels = np.load(HS_DIRS[model] / "labels.npy")
        rng = np.random.default_rng(RANDOM_STATE)
        shuffled_labels = rng.permutation(labels)

        for lang in ["en", "ru", "ky"]:
            layer = BEST_LAYERS[model][lang]
            hs = np.load(HS_DIRS[model] / f"hidden_states_{lang}.npz")["hidden_states"]
            X = hs[:, layer, :]

            print(f"  {MODEL_LABELS[model]:>16} / {lang.upper()} (L{layer})", flush=True)

            linear_acc = run_probe(X, labels, LinearProbe)
            mlp_acc = run_probe(X, labels, MLPProbe)
            control_acc = run_probe(X, shuffled_labels, LinearProbe)
            selectivity = linear_acc - control_acc

            print(f"    Linear={linear_acc:.3f}  MLP={mlp_acc:.3f}  "
                  f"Control={control_acc:.3f}  Selectivity={selectivity:.3f}")

            rows.append({
                "model": MODEL_LABELS[model],
                "lang": lang.upper(),
                "layer": layer,
                "linear_acc": round(linear_acc, 3),
                "mlp_acc": round(mlp_acc, 3),
                "control_acc": round(control_acc, 3),
                "selectivity": round(selectivity, 3),
                "mlp_gap": round(mlp_acc - linear_acc, 3),
            })

    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(TABLE_DIR / "probe_selectivity.csv", index=False)
    print(f"\nSaved → probe_selectivity.csv")
    print(sel_df.to_string(index=False))

    # ── Part 2: Length control ────────────────────────────────────────────────
    print("\n\n[2] Accuracy by sentence length (token bins)...\n")
    from transformers import AutoTokenizer

    length_rows = []
    for model in models:
        labels = np.load(HS_DIRS[model] / "labels.npy")
        lang = "en"
        layer = BEST_LAYERS[model][lang]
        hs = np.load(HS_DIRS[model] / f"hidden_states_{lang}.npz")["hidden_states"]
        X = hs[:, layer, :]

        # Get per-example predictions
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        correct = np.zeros(len(labels), dtype=bool)
        for tr, te in skf.split(X, labels):
            sc = StandardScaler()
            Xtr = torch.tensor(sc.fit_transform(X[tr]),
                               dtype=torch.float32, device=DEVICE)
            Xte = torch.tensor(sc.transform(X[te]),
                               dtype=torch.float32, device=DEVICE)
            ytr = torch.tensor(labels[tr], dtype=torch.long, device=DEVICE)
            yte = torch.tensor(labels[te], dtype=torch.long, device=DEVICE)
            m = LinearProbe(Xtr.shape[1], 6).to(DEVICE)
            opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
            ce = nn.CrossEntropyLoss()
            m.train()
            for _ in range(N_EPOCHS):
                opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                preds = m(Xte).argmax(1)
            correct[te] = (preds == yte).cpu().numpy()

        # Bin by word count (simpler than tokenizer-specific length)
        texts = df["text_en"].tolist()
        word_counts = np.array([len(t.split()) for t in texts])
        bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 100)]

        for lo, hi in bins:
            mask = (word_counts >= lo) & (word_counts <= hi)
            if mask.sum() < 10:
                continue
            acc = correct[mask].mean()
            length_rows.append({
                "model": MODEL_LABELS[model],
                "bin": f"{lo}-{hi}",
                "n": int(mask.sum()),
                "accuracy": round(acc, 3),
            })

    len_df = pd.DataFrame(length_rows)
    len_df.to_csv(TABLE_DIR / "length_control.csv", index=False)
    print("Length control (EN):")
    print(len_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
