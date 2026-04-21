"""
Step 28: Robust per-emotion peak-layer estimation.

For each (model, language, emotion):
  - Train OvR linear probe at every layer, 5-fold stratified CV.
  - peak layer  = argmax(mean F1 across folds)
  - 1-SE band  = set of layers whose mean F1 is within +/- 1 SE of the peak
  - peak range = [min(1-SE band), max(1-SE band)]

Output:
  data/results/tables/per_emotion_peak_robust.csv

Replaces the single-split argmax in Table 7 with a noise-aware estimate.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

HS_DIRS = {
    "Gemma 4 E4B": Path("data/processed"),
    "Qwen3-8B":    Path("data/processed/qwen3"),
    "Llama-3.1-8B": Path("data/processed/llama"),
    "Mistral-7B":  Path("data/processed/mistral"),
}
TABLE_DIR = Path("data/results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS      = 5
N_EPOCHS     = 200
LR_RATE      = 1e-2
RANDOM_STATE = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def cv_macrof1(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return per-fold macro-F1 array of length N_FOLDS for an OvR probe."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
        ytr_t = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
        yte = y[te]

        n_pos = int(ytr_t.sum().item())
        n_neg = len(ytr_t) - n_pos
        w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

        m = LinClf(Xtr_t.shape[1], 2).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss(weight=w)
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr_t), ytr_t).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds = m(Xte_t).argmax(1).cpu().numpy()
        fold_scores.append(f1_score(yte, preds, average="macro"))
    return np.array(fold_scores)


def main():
    rows = []
    for model_name, mdir in HS_DIRS.items():
        labels_path = mdir / "labels.npy"
        if not labels_path.exists():
            print(f"[skip] {model_name}: no labels at {labels_path}")
            continue
        labels = np.load(labels_path)
        for lang in ["en", "ru", "ky"]:
            hs_path = mdir / f"hidden_states_{lang}.npz"
            if not hs_path.exists():
                continue
            hs = np.load(hs_path)["hidden_states"]
            n_layers = hs.shape[1]

            for eidx, emotion in enumerate(EMOTIONS):
                y = (labels == eidx).astype(int)
                mean_f1 = np.zeros(n_layers)
                se_f1   = np.zeros(n_layers)
                pbar = tqdm(range(n_layers),
                            desc=f"{model_name[:12]:<12} {lang.upper()} {emotion[:8]:<8}",
                            leave=False)
                for layer in pbar:
                    fold_scores = cv_macrof1(hs[:, layer, :].astype(np.float32), y)
                    mean_f1[layer] = fold_scores.mean()
                    se_f1[layer]   = fold_scores.std(ddof=1) / np.sqrt(N_FOLDS)

                peak_layer = int(np.argmax(mean_f1))
                threshold  = mean_f1[peak_layer] - se_f1[peak_layer]
                in_band    = np.where(mean_f1 >= threshold)[0]
                rows.append({
                    "model":      model_name,
                    "lang":       lang.upper(),
                    "emotion":    emotion,
                    "peak_layer": peak_layer,
                    "peak_f1":    round(float(mean_f1[peak_layer]), 4),
                    "peak_se":    round(float(se_f1[peak_layer]), 4),
                    "low_layer":  int(in_band.min()),
                    "high_layer": int(in_band.max()),
                    "n_layers":   n_layers,
                })
                print(f"  {model_name:>14} {lang.upper()} {emotion:<10} "
                      f"L={peak_layer:>2} (range {in_band.min():>2}-{in_band.max():>2}) "
                      f"F1={mean_f1[peak_layer]:.3f}+/-{se_f1[peak_layer]:.3f}")

    df = pd.DataFrame(rows)
    out = TABLE_DIR / "per_emotion_peak_robust.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
