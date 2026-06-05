"""
Step 29: Extract hidden states for the 102-sentence HUMAN-translated control
sample across all 5 models.

This is the validation step for the "Claude translation confound" concern:
- Same 102 source EN sentences are translated by Claude (full corpus) and a
  human translator (control).
- We extract hidden states for the HUMAN translations only (RU + KY).
- A later script (30_human_vs_claude_probing.py) compares probing accuracy
  on Claude vs. human translations of the same 102 sentences.

Usage:
    python3 scripts/29_extract_human_control.py --model gemma4
    python3 scripts/29_extract_human_control.py --model qwen3
    python3 scripts/29_extract_human_control.py --model llama
    python3 scripts/29_extract_human_control.py --model mistral
    python3 scripts/29_extract_human_control.py --model xlmr
    python3 scripts/29_extract_human_control.py --model all   # do all sequentially

Input:  data/translated/human_control_v2.csv  (102 rows)
Output: data/processed/{model}/hidden_states_human_{ru,ky}.npz
        data/processed/control_ids.npy  (the 102 ids, same order for all models)
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CONTROL_CSV = Path("data/translated/human_control_v2.csv")
FULL_CSV    = Path("data/translated/parallel_corpus_clean.csv")
OUTPUT_ROOT = Path("data/processed")
MAX_LENGTH  = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}

MODEL_REGISTRY = {
    "gemma4":  {"id": "google/gemma-4-E4B-it",     "type": "decoder_quant", "subdir": "",        "batch": 4},
    "qwen3":   {"id": "Qwen/Qwen3-8B",              "type": "decoder_quant", "subdir": "qwen3",   "batch": 4},
    "llama":   {"id": "meta-llama/Llama-3.1-8B",    "type": "decoder_quant", "subdir": "llama",   "batch": 4},
    "mistral": {"id": "mistralai/Mistral-7B-v0.3",  "type": "decoder_quant", "subdir": "mistral", "batch": 4},
    "xlmr":    {"id": "xlm-roberta-large",          "type": "encoder_fp16",  "subdir": "xlmr",    "batch": 16},
}


def load_decoder_quant(model_id):
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


def load_encoder_fp16(model_id):
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading {model_id} in fp16...")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
    return tok, mdl


def extract(texts, tokenizer, model, batch_size, is_decoder):
    all_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  batches", leave=False):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            if hs is None and is_decoder:
                lm = getattr(model, "language_model", model)
                outputs = lm(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states

        mask = inputs["attention_mask"].float().unsqueeze(-1)
        layer_pooled = []
        for layer_hs in hs:
            pooled = (layer_hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            layer_pooled.append(pooled.cpu().numpy())
        all_states.append(np.stack(layer_pooled, axis=1))

    return np.concatenate(all_states, axis=0)


def run_one(model_key):
    cfg = MODEL_REGISTRY[model_key]
    out_dir = OUTPUT_ROOT / cfg["subdir"] if cfg["subdir"] else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    control = pd.read_csv(CONTROL_CSV)
    print(f"\n=== {model_key} ({cfg['id']}) — control sample N={len(control)} ===")

    # Sanity check vs full corpus
    full = pd.read_csv(FULL_CSV)
    common_ids = set(control["id"]) & set(full["id"])
    assert len(common_ids) == len(control), \
        f"Control ids not all in full corpus: {len(common_ids)} vs {len(control)}"

    # Save control ids once (same for all models)
    ids_path = OUTPUT_ROOT / "control_ids.npy"
    if not ids_path.exists():
        np.save(ids_path, control["id"].to_numpy())
        print(f"Saved control ids → {ids_path}")

    # Load model
    if cfg["type"] == "decoder_quant":
        tok, mdl = load_decoder_quant(cfg["id"])
        is_decoder = True
    else:
        tok, mdl = load_encoder_fp16(cfg["id"])
        is_decoder = False

    # Extract RU_human and KY_human
    for lang, col in [("ru", "text_ru_human"), ("ky", "text_ky_human")]:
        out_path = out_dir / f"hidden_states_human_{lang}.npz"
        if out_path.exists():
            print(f"  {lang.upper()} already extracted — skipping.")
            continue
        texts = control[col].tolist()
        print(f"  Extracting {lang.upper()} (human) — {len(texts)} sentences...")
        states = extract(texts, tok, mdl, cfg["batch"], is_decoder)
        print(f"    shape: {states.shape}")
        np.savez_compressed(out_path, hidden_states=states)
        print(f"    saved → {out_path}")

    # Free VRAM
    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()) + ["all"])
    args = ap.parse_args()

    if args.model == "all":
        for k in MODEL_REGISTRY:
            run_one(k)
    else:
        run_one(args.model)

    print("\nAll done.")


if __name__ == "__main__":
    main()
