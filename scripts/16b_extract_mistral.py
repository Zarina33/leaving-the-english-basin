"""
Step 16b: Extract hidden states from Mistral-7B-v0.3.

Mistral-7B-v0.3: dense 7B decoder-only, 32 transformer layers,
hidden_dim=4096. Publicly available — no HuggingFace approval required.

~5 GB VRAM (4-bit NF4), ~10 GB RAM

Input:  data/translated/parallel_corpus_clean.csv
Output: data/processed/mistral/hidden_states_{en,ru,ky}.npz
        data/processed/mistral/labels.npy
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID   = "mistralai/Mistral-7B-v0.3"
INPUT_PATH = Path("data/translated/parallel_corpus_clean.csv")
OUTPUT_DIR = Path("data/processed/mistral")
BATCH_SIZE = 4
MAX_LENGTH = 128

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}


def load_model():
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.eval()

    n_layers    = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Loaded. Layers: {n_layers}, Hidden dim: {hidden_size}")
    return tokenizer, model


def extract_states(texts: list[str], tokenizer, model) -> np.ndarray:
    """
    Returns array of shape (len(texts), n_layers+1, hidden_dim).
    Layer 0 = embedding, layers 1..N = transformer layers.
    Mean-pooled over non-padding tokens.
    """
    all_states = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc="  Batches", leave=False):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
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

    return np.concatenate(all_states, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} examples")

    labels = np.array([LABEL2ID[e] for e in df["emotion"]])
    np.save(OUTPUT_DIR / "labels.npy", labels)
    print(f"Labels saved: {dict(zip(EMOTION_LABELS, np.bincount(labels)))}")

    tokenizer, model = load_model()

    for lang, col in [("en", "text_en"), ("ru", "text_ru"), ("ky", "text_ky")]:
        out_path = OUTPUT_DIR / f"hidden_states_{lang}.npz"
        if out_path.exists():
            print(f"\n{lang.upper()} already extracted, skipping.")
            continue

        print(f"\nExtracting {lang.upper()} ({len(df)} texts)...")
        states = extract_states(df[col].tolist(), tokenizer, model)
        print(f"  Shape: {states.shape}  (samples × layers × hidden_dim)")
        np.savez_compressed(out_path, hidden_states=states)
        print(f"  Saved → {out_path}")

    print("\nAll done!")
    for f in sorted(OUTPUT_DIR.glob("*.np*")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
