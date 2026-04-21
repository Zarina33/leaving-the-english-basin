"""
Step 24: 8-bit quantization ablation for Gemma 4 E4B.

Extracts hidden states in 8-bit (vs existing 4-bit NF4) and re-runs probing
at all layers. If early-peaking persists under 8-bit, quantization level
does not explain the depth profile.

Gemma 4 E4B in 8-bit requires ~8 GB VRAM (fits in 16 GB GPU).

Output:
  data/processed/gemma4_fp16/hidden_states_{en,ru,ky}.npz
  data/results/gemma4_fp16/probing_detailed.csv
  data/results/tables/fp16_ablation.csv
"""

import os
os.environ["TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

MODEL_ID   = "google/gemma-4-E4B-it"
INPUT_PATH = Path("data/translated/parallel_corpus_clean.csv")
HS_DIR     = Path("data/processed/gemma4_fp16")
RES_DIR    = Path("data/results/gemma4_fp16")
TABLE_DIR  = Path("data/results/tables")
BATCH_SIZE = 2       # smaller batch for fp16
MAX_LENGTH = 128

N_FOLDS      = 5
N_EPOCHS     = 300
LR_RATE      = 1e-2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}


# ── Extract ───────────────────────────────────────────────────────────────────

def extract_fp16():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    labels = np.array([LABEL2ID[e] for e in df["emotion"]])
    np.save(HS_DIR / "labels.npy", labels)

    from transformers import BitsAndBytesConfig

    print(f"Loading {MODEL_ID} in 8-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.eval()

    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    hidden_size = text_cfg.hidden_size
    print(f"Loaded. Layers: {n_layers}, Hidden dim: {hidden_size}")

    for lang, col in [("en", "text_en"), ("ru", "text_ru"), ("ky", "text_ky")]:
        out_path = HS_DIR / f"hidden_states_{lang}.npz"
        if out_path.exists():
            print(f"\n{lang.upper()} already extracted, skipping.")
            continue

        texts = df[col].tolist()
        print(f"\nExtracting {lang.upper()} ({len(texts)} texts, 8-bit)...")
        all_states = []

        for i in tqdm(range(0, len(texts), BATCH_SIZE),
                      desc="  Batches", leave=False):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            mask = inputs["attention_mask"].float().unsqueeze(-1)
            batch_states = []
            for layer_hs in hidden_states:
                pooled = (layer_hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                batch_states.append(pooled.cpu().numpy())
            all_states.append(np.stack(batch_states, axis=1))

        states = np.concatenate(all_states, axis=0)
        print(f"  Shape: {states.shape}")
        np.savez_compressed(out_path, hidden_states=states)
        print(f"  Saved → {out_path}")

    del model
    torch.cuda.empty_cache()


# ── Probe ─────────────────────────────────────────────────────────────────────

class LinClf(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.fc(x)


def gpu_probe(X, y):
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
        m = LinClf(Xtr.shape[1], n_cls).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_RATE, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        m.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); ce(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds = m(Xte).argmax(1)
        accs.append((preds == yte).float().mean().item())
    return accs


def run_probing():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    probe_path = RES_DIR / "probing_detailed.csv"
    if probe_path.exists():
        print("\nLoading existing fp16 probing results...")
        return pd.read_csv(probe_path)

    labels = np.load(HS_DIR / "labels.npy")
    hs = {lang: np.load(HS_DIR / f"hidden_states_{lang}.npz")["hidden_states"]
          for lang in ["en", "ru", "ky"]}

    print(f"\nRunning probing on fp16 hidden states...")
    records = []
    for lang in ["en", "ru", "ky"]:
        n_layers = hs[lang].shape[1]
        for layer in tqdm(range(n_layers), desc=f"  {lang.upper()}"):
            fa = gpu_probe(hs[lang][:, layer, :], labels)
            records.append({
                "lang": lang, "layer": layer,
                "acc_mean": np.mean(fa), "acc_std": np.std(fa),
            })

    df = pd.DataFrame(records)
    df.to_csv(probe_path, index=False)
    print(f"Saved → {probe_path.name}")
    return df


# ── Compare ───────────────────────────────────────────────────────────────────

def compare():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fp16_df = pd.read_csv(RES_DIR / "probing_detailed.csv")
    nf4_df  = pd.read_csv("data/results/probing_detailed.csv")

    print("\n" + "=" * 65)
    print("8-bit vs 4-bit NF4 comparison (Gemma 4 E4B)")
    print("=" * 65)

    rows = []
    for lang in ["en", "ru", "ky"]:
        fp16_sub = fp16_df[fp16_df.lang == lang]
        nf4_sub  = nf4_df[nf4_df.lang == lang]

        fp16_best = fp16_sub.loc[fp16_sub.acc_mean.idxmax()]
        nf4_best  = nf4_sub.loc[nf4_sub.acc_mean.idxmax()]

        row = {
            "lang": lang.upper(),
            "nf4_layer": int(nf4_best.layer),
            "nf4_acc": round(nf4_best.acc_mean, 3),
            "fp16_layer": int(fp16_best.layer),
            "fp16_acc": round(fp16_best.acc_mean, 3),
            "diff": round(fp16_best.acc_mean - nf4_best.acc_mean, 3),
            "same_regime": "early" if fp16_best.layer <= 6 else "mid/late",
        }
        rows.append(row)
        print(f"  {lang.upper()}: NF4 L{row['nf4_layer']}={row['nf4_acc']:.3f}  "
              f"fp16 L{row['fp16_layer']}={row['fp16_acc']:.3f}  "
              f"Δ={row['diff']:+.3f}  regime={row['same_regime']}")

    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "fp16_ablation.csv", index=False)
    print(f"\nSaved → fp16_ablation.csv")

    # Key question: does early-peaking persist in fp16?
    fp16_peaks = [int(result[result.lang == l].fp16_layer.values[0])
                  for l in ["EN", "RU", "KY"]]
    if all(p <= 8 for p in fp16_peaks):
        print("\n✓ Early-peaking PERSISTS in fp16 → not a quantization artifact")
    else:
        print("\n⚠ Peak layers SHIFTED in fp16 → quantization may affect depth profile")


def main():
    extract_fp16()
    run_probing()
    compare()
    print("\nDone!")


if __name__ == "__main__":
    main()
