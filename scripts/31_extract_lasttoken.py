"""
Step 31: Extract LAST-TOKEN-pooled hidden states for the 4 decoder LLMs.

Ablation for the "mean pooling may distort depth profiles" concern. The paper
uses mean pooling; standard practice for decoder-only LLMs is often last-token
pooling (the final token attends to the whole sequence under the causal mask).
We re-extract with last-token pooling and re-run probing to check whether the
"early-peaking" depth pattern (Gemma 4 ≈ layer 0, Llama ≈ layer 1) survives.

Last token is located robustly via the attention mask (works for both left-
and right-padded tokenizers): the last position where mask == 1.

Usage:
    python3 scripts/31_extract_lasttoken.py --model gemma4
    python3 scripts/31_extract_lasttoken.py --model all

Input:  data/translated/parallel_corpus_clean.csv
Output: data/processed/{subdir}_lasttok/hidden_states_{en,ru,ky}.npz
        data/processed/{subdir}_lasttok/labels.npy
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

INPUT_PATH = Path("data/translated/parallel_corpus_clean.csv")
OUTPUT_ROOT = Path("data/processed")
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}

# Decoder LLMs only — last-token pooling is not meaningful for the bidirectional
# XLM-R encoder, and the depth-profile concern is decoder-specific.
MODEL_REGISTRY = {
    "gemma4":  {"id": "google/gemma-4-E4B-it",    "out": "gemma4_lasttok",  "batch": 4},
    "qwen3":   {"id": "Qwen/Qwen3-8B",            "out": "qwen3_lasttok",   "batch": 4},
    "llama":   {"id": "meta-llama/Llama-3.1-8B",  "out": "llama_lasttok",   "batch": 4},
    "mistral": {"id": "mistralai/Mistral-7B-v0.3","out": "mistral_lasttok", "batch": 4},
}


def load_model(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print(f"Loading {model_id} in 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    ).eval()
    return tok, mdl


def extract_lasttoken(texts, tokenizer, model, batch_size):
    all_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  batches", leave=False):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            if hs is None:
                lm = getattr(model, "language_model", model)
                outputs = lm(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states

        mask = inputs["attention_mask"]                      # (B, T)
        seq_total = mask.shape[1]
        # Robust last-real-token index for both padding sides:
        # last position where mask == 1 = (T-1) - argmax(reversed mask)
        last_idx = (seq_total - 1) - mask.flip(1).argmax(1)  # (B,)

        layer_pooled = []
        for layer_hs in hs:                                  # (B, T, H)
            H = layer_hs.size(-1)
            idx = last_idx.view(-1, 1, 1).expand(-1, 1, H)   # (B, 1, H)
            last = layer_hs.float().gather(1, idx).squeeze(1)  # (B, H)
            layer_pooled.append(last.cpu().numpy())
        all_states.append(np.stack(layer_pooled, axis=1))    # (B, L, H)

    return np.concatenate(all_states, axis=0)


def run_one(model_key):
    cfg = MODEL_REGISTRY[model_key]
    out_dir = OUTPUT_ROOT / cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"\n=== {model_key} ({cfg['id']}) — last-token pooling, N={len(df)} ===")

    labels = np.array([LABEL2ID[e] for e in df["emotion"]])
    np.save(out_dir / "labels.npy", labels)

    tok, mdl = load_model(cfg["id"])

    for lang, col in [("en", "text_en"), ("ru", "text_ru"), ("ky", "text_ky")]:
        out_path = out_dir / f"hidden_states_{lang}.npz"
        if out_path.exists():
            print(f"  {lang.upper()} already done — skipping.")
            continue
        print(f"  Extracting {lang.upper()} (last-token)...")
        states = extract_lasttoken(df[col].tolist(), tok, mdl, cfg["batch"])
        print(f"    shape: {states.shape}")
        np.savez_compressed(out_path, hidden_states=states)
        print(f"    saved → {out_path}")

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()) + ["all"])
    args = ap.parse_args()
    keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    for k in keys:
        run_one(k)
    print("\nAll done.")


if __name__ == "__main__":
    main()
