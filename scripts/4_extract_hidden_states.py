"""
Step 4: Extract hidden states from Gemma 4 for all layers.

For each sentence (EN/RU/KY), extracts the [CLS]-equivalent representation
(mean pooling over tokens) from every transformer layer.

Input:  data/translated/parallel_corpus_clean.csv
Output: data/processed/hidden_states_{en,ru,ky}.npz
  Each file: array of shape (N_samples, N_layers, hidden_dim)
  + data/processed/labels.npy — emotion labels as integers

Requirements: ~6GB VRAM (4-bit), ~10GB RAM for saving
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"
INPUT_PATH = Path("data/translated/parallel_corpus_clean.csv")
OUTPUT_DIR = Path("data/processed")
BATCH_SIZE = 4
MAX_LENGTH = 128

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}


def load_model():
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    # Gemma 4 is multimodal — text config is nested
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    hidden_size = text_cfg.hidden_size
    print(f"Model loaded. Layers: {n_layers}, Hidden dim: {hidden_size}")
    return tokenizer, model


def extract_states(texts: list[str], tokenizer, model) -> np.ndarray:
    """
    Returns array of shape (len(texts), n_layers+1, hidden_dim)
    Layer 0 = embedding, layers 1..N = transformer layers.
    """
    all_states = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Move only tensor inputs to device (skip non-tensor keys)
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Gemma 4: hidden_states may be in outputs directly or via language_model
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            # fallback: try language_model submodule
            lm = getattr(model, "language_model", model)
            outputs = lm(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Mean pool over non-padding tokens for each layer
        attention_mask = inputs["attention_mask"].float()
        mask_expanded = attention_mask.unsqueeze(-1)

        batch_states = []
        for layer_hidden in hidden_states:
            summed = (layer_hidden.float() * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()
            batch_states.append(pooled)

        batch_array = np.stack(batch_states, axis=1)
        all_states.append(batch_array)

    return np.concatenate(all_states, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} examples")

    labels = np.array([LABEL2ID[e] for e in df["emotion"]])
    np.save(OUTPUT_DIR / "labels.npy", labels)
    np.save(OUTPUT_DIR / "emotions.npy", np.array(EMOTION_LABELS))
    print(f"Labels saved: {dict(zip(EMOTION_LABELS, np.bincount(labels)))}")

    tokenizer, model = load_model()

    for lang, col in [("en", "text_en"), ("ru", "text_ru"), ("ky", "text_ky")]:
        out_path = OUTPUT_DIR / f"hidden_states_{lang}.npz"
        if out_path.exists():
            print(f"\n{lang.upper()} already extracted, skipping.")
            continue

        texts = df[col].tolist()
        print(f"\nExtracting {lang.upper()} hidden states ({len(texts)} texts)...")

        states = extract_states(texts, tokenizer, model)
        print(f"  Shape: {states.shape}  (samples × layers × hidden_dim)")

        np.savez_compressed(out_path, hidden_states=states)
        print(f"  Saved to {out_path}")

    print("\nAll done!")
    print("Files saved:")
    for f in sorted(OUTPUT_DIR.glob("*.np*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
