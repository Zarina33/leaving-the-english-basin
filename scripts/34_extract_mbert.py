"""
Step 34: Extract hidden states from mBERT (bert-base-multilingual-cased)
as a SECOND multilingual encoder baseline alongside XLM-R.

Motivation: the paper's headline encoder-vs-decoder finding (XLM-R wins on
Kyrgyz transfer) rests on a single encoder. Adding mBERT tests whether the
pattern generalizes across encoders or is XLM-R-specific. If mBERT shows the
same Kyrgyz-transfer advantage, the "multilingual pre-training objectives
matter" claim is much stronger.

mBERT: 178M params, 12 layers, hidden=768. Tiny — fits in ~1 GB fp16.
Mean pooling over non-padding tokens — identical to XLM-R (script 26).

Usage:  python3 scripts/34_extract_mbert.py

Output: data/processed/mbert/hidden_states_{en,ru,ky}.npz + labels.npy
After:  python3 scripts/35_analyze_model.py --model mbert
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MODEL_ID   = "bert-base-multilingual-cased"
INPUT_PATH = Path("data/translated/parallel_corpus_clean.csv")
OUTPUT_DIR = Path("data/processed/mbert")
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}


def extract(texts, tokenizer, model):
    all_states = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  batches", leave=False):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].float().unsqueeze(-1)
        layer_pooled = [
            ((lh.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
            for lh in outputs.hidden_states
        ]
        all_states.append(np.stack(layer_pooled, axis=1))
    return np.concatenate(all_states, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} examples")
    np.save(OUTPUT_DIR / "labels.npy", np.array([LABEL2ID[e] for e in df["emotion"]]))

    print(f"Loading {MODEL_ID} in fp16...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()
    print(f"Layers: {mdl.config.num_hidden_layers}, Hidden: {mdl.config.hidden_size}")

    for lang, col in [("en", "text_en"), ("ru", "text_ru"), ("ky", "text_ky")]:
        out_path = OUTPUT_DIR / f"hidden_states_{lang}.npz"
        if out_path.exists():
            print(f"  {lang.upper()} already done — skipping.")
            continue
        print(f"  Extracting {lang.upper()}...")
        states = extract(df[col].tolist(), tok, mdl)
        print(f"    shape: {states.shape}")
        np.savez_compressed(out_path, hidden_states=states)
        print(f"    saved → {out_path}")

    print("\nDone. Next: python3 scripts/35_analyze_model.py --model mbert")


if __name__ == "__main__":
    main()
