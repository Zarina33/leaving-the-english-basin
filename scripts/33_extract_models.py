"""
Step 33: Extract MEAN-POOLED hidden states for ADDITIONAL decoder LLMs,
to grow the model sample from 5 to 7-8 and strengthen the "depth trend" claim.

Candidate additions (all fit in <8 GB at 4-bit on a 16 GB GPU):
  - Llama-3.2-3B   (smaller Llama; tests size effect within a family)
  - Qwen2.5-7B     (prior Qwen generation; tests within-family stability)
  - OLMo-2-7B      (fully open pre-training data; useful for contamination claims)

Mean pooling over non-padding tokens — identical to the paper pipeline
(scripts/4, 11, 16, 16b) so results are directly comparable.

Usage:
    python3 scripts/33_extract_models.py --model llama32_3b
    python3 scripts/33_extract_models.py --model qwen25_7b
    python3 scripts/33_extract_models.py --model olmo2_7b
    python3 scripts/33_extract_models.py --model all

Output: data/processed/{key}/hidden_states_{en,ru,ky}.npz + labels.npy

After extraction, analyze with:
    python3 scripts/35_analyze_model.py --model {key}
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

INPUT_PATH  = Path("data/translated/parallel_corpus_clean.csv")
OUTPUT_ROOT = Path("data/processed")
MAX_LENGTH  = 128

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}

# Edit IDs here if you prefer different checkpoints.
MODEL_REGISTRY = {
    "llama32_3b": {"id": "meta-llama/Llama-3.2-3B",   "batch": 8},
    "qwen25_7b":  {"id": "Qwen/Qwen2.5-7B",           "batch": 4},
    "olmo2_7b":   {"id": "allenai/OLMo-2-1124-7B",    "batch": 4},
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
    n_layers = getattr(mdl.config, "num_hidden_layers", "?")
    print(f"Loaded. Layers: {n_layers}, Hidden: {getattr(mdl.config, 'hidden_size', '?')}")
    return tok, mdl


def extract_mean(texts, tokenizer, model, batch_size):
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
                hs = lm(**inputs, output_hidden_states=True).hidden_states
        mask = inputs["attention_mask"].float().unsqueeze(-1)
        layer_pooled = [
            ((lh.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
            for lh in hs
        ]
        all_states.append(np.stack(layer_pooled, axis=1))
    return np.concatenate(all_states, axis=0)


def run_one(key):
    cfg = MODEL_REGISTRY[key]
    out_dir = OUTPUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"\n=== {key} ({cfg['id']}) — mean pooling, N={len(df)} ===")
    np.save(out_dir / "labels.npy", np.array([LABEL2ID[e] for e in df["emotion"]]))

    tok, mdl = load_model(cfg["id"])
    for lang, col in [("en", "text_en"), ("ru", "text_ru"), ("ky", "text_ky")]:
        out_path = out_dir / f"hidden_states_{lang}.npz"
        if out_path.exists():
            print(f"  {lang.upper()} already done — skipping.")
            continue
        print(f"  Extracting {lang.upper()}...")
        states = extract_mean(df[col].tolist(), tok, mdl, cfg["batch"])
        print(f"    shape: {states.shape}")
        np.savez_compressed(out_path, hidden_states=states)
        print(f"    saved → {out_path}")

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY) + ["all"])
    args = ap.parse_args()
    keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    results = {}
    for k in keys:
        try:
            run_one(k)
            results[k] = "ok"
        except Exception as e:
            results[k] = f"FAILED: {type(e).__name__}: {e}"
            print(f"\n[!] {k} extraction failed — continuing with remaining models.\n    Reason: {results[k]}\n")
    print("\n=== summary ===")
    for k, status in results.items():
        print(f"  {k}: {status}")
    print("\nFor successful models, run: python3 scripts/35_analyze_model.py --model <key>")


if __name__ == "__main__":
    main()
