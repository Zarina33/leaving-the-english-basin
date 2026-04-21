"""
Step 22: Per-emotion analysis with macro-F1 instead of accuracy.

Addresses reviewer concern: OvR accuracy with 1:5 imbalance is unreliable.
Macro-F1 is balanced and penalizes majority-class bias.

Output:
  data/results/tables/per_emotion_f1.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm

HS_DIRS = {
    "gemma4":  Path("data/processed"),
    "qwen3":   Path("data/processed/qwen3"),
    "mistral": Path("data/processed/mistral"),
    "llama":   Path("data/processed/llama"),
}
TABLE_DIR = Path("data/results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
MODEL_LABELS = {
    "gemma4": "Gemma 4 E4B", "qwen3": "Qwen3-8B",
    "mistral": "Mistral-7B",  "llama": "Llama-3.1-8B",
}


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def ovr_f1(X, y_bin, n_cls=2):
    """Train OvR probe, return macro-F1 on 80/20 split."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(X))
    sp = int(len(X) * 0.8)
    tr, te = idx[:sp], idx[sp:]

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_bin[tr], dtype=torch.long, device=DEVICE)
    yte_np = y_bin[te]

    # Class weights for imbalance
    n_pos = int(ytr.sum().item())
    n_neg = len(ytr) - n_pos
    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

    m = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=w)
    m.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(Xte).argmax(1).cpu().numpy()

    return f1_score(yte_np, preds, average="macro")


def main():
    models = [k for k in HS_DIRS if (HS_DIRS[k] / "hidden_states_en.npz").exists()]

    rows = []
    for model in models:
        labels = np.load(HS_DIRS[model] / "labels.npy")
        for lang in ["en", "ru", "ky"]:
            hs = np.load(HS_DIRS[model] / f"hidden_states_{lang}.npz")["hidden_states"]
            n_layers = hs.shape[1]

            for eidx, emotion in enumerate(EMOTIONS):
                y_bin = (labels == eidx).astype(int)
                best_f1 = 0
                best_layer = 0

                for layer in range(n_layers):
                    f1 = ovr_f1(hs[:, layer, :], y_bin)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_layer = layer

                rows.append({
                    "model": MODEL_LABELS[model],
                    "lang": lang.upper(),
                    "emotion": emotion,
                    "best_f1": round(best_f1, 3),
                    "best_layer": best_layer,
                })
                print(f"  {MODEL_LABELS[model]:>16} {lang.upper()} {emotion:<10} "
                      f"F1={best_f1:.3f} (L{best_layer})")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "per_emotion_f1.csv", index=False)

    # Pivot for paper table
    print("\n\nPivot table (best macro-F1 per emotion):\n")
    for model in models:
        sub = df[df.model == MODEL_LABELS[model]]
        pivot = sub.pivot(index="emotion", columns="lang", values="best_f1")
        pivot = pivot[["EN", "RU", "KY"]]
        print(f"\n{MODEL_LABELS[model]}:")
        print(pivot.to_string())

    print(f"\nSaved → per_emotion_f1.csv")


if __name__ == "__main__":
    main()
