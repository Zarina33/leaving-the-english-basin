"""
Step 38: Logit-lens / token-projection test for the English-pivot hypothesis
(Wendler et al., 2024).

Motivation
----------
The paper observes high EN<->RU cross-lingual transfer for Qwen3 and Llama and
raises (but does not test) the question of whether these decoders maintain
genuinely language-agnostic representations or instead internally *pivot*
through an English-like space. This script tests that directly.

Method (token-projection on pooled hidden states)
-------------------------------------------------
For a non-English input (RU or KY), we take the mean-pooled hidden state at
each layer, apply the model's final norm, and project through the *unembedding*
(lm_head) to obtain a distribution over the vocabulary --- exactly the
logit-lens of nostalgebraist / Wendler et al., but on the sentence-pooled
representation rather than a single token position. We then ask: among the
top-k vocabulary tokens the intermediate representation decodes to, what
fraction are ENGLISH (Latin-script word tokens) vs. tokens in the input
language (Cyrillic)?

If the mid-network layers of a RU/KY input decode predominantly to *English*
tokens --- and only the final layers decode back to the input language --- that
is the signature of English-pivoting (Wendler et al., 2024). If instead the
decoded language tracks the input language throughout, the representation is
more genuinely language-agnostic.

We report, per layer, the mean over N sentences of:
  - p_en  : fraction of top-k tokens that are English/Latin word tokens
  - p_cyr : fraction that are Cyrillic word tokens (input-language family)

Because we only have mean-pooled states (not per-token), this is a
sentence-level logit-lens: it answers "what language does the *aggregate*
sentence representation decode to at depth d", which is exactly the quantity
relevant to the pooled-probing transfer results in the paper.

Usage:
    python3 scripts/38_logit_lens_english_pivot.py --model llama --topk 20
    python3 scripts/38_logit_lens_english_pivot.py --model qwen3 --topk 20
    python3 scripts/38_logit_lens_english_pivot.py --model all

Output:
    data/results/logit_lens/{key}_english_pivot.csv   (per-layer p_en/p_cyr, EN/RU/KY)
    data/results/logit_lens/{key}_examples.txt        (qualitative top-token dumps)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROCESSED = Path("data/processed")
OUT_DIR = Path("data/results/logit_lens")

# Models with high EN<->RU transfer (the ones the English-pivot claim concerns).
# We keep XLM-R out: it is an encoder with no lm_head / autoregressive vocab head.
MODEL_REGISTRY = {
    "llama":  "meta-llama/Llama-3.1-8B",
    "qwen3":  "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-v0.3",
}

N_SAMPLE = 300          # sentences per language (subsample for speed; deterministic)
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
# Kyrgyz-specific Cyrillic letters (distinguish KY from RU when needed)
KY_SPECIFIC = set("өүңӨҮҢ")
WORDISH_RE = re.compile(r"[A-Za-zЀ-ӿ]")  # contains at least one letter


def classify_token(s: str) -> str:
    """Classify a decoded token string as 'en' (Latin), 'cyr' (Cyrillic),
    or 'other' (punctuation, digits, whitespace-only, CJK, byte-fragments)."""
    core = s.strip()
    if not core or not WORDISH_RE.search(core):
        return "other"
    has_latin = bool(LATIN_RE.search(core))
    has_cyr = bool(CYRILLIC_RE.search(core))
    if has_latin and not has_cyr:
        return "en"
    if has_cyr and not has_latin:
        return "cyr"
    return "other"  # mixed / ambiguous


def load_head(model_id):
    """Load the model in 4-bit and return (tokenizer, final_norm, lm_head, device).

    We load the full CausalLM (cheap enough at 4-bit) so we get the *exact*
    final norm + tied/untied unembedding the model uses. We only ever call the
    norm and head, never the transformer stack, so this is fast.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print(f"Loading {model_id} in 4-bit NF4 (head + norm only used)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    ).eval()

    # Locate the decoder stack's final norm and the output head robustly.
    base = getattr(mdl, "model", mdl)          # LlamaModel / Qwen3Model / MistralModel
    final_norm = base.norm                      # RMSNorm applied before lm_head
    lm_head = mdl.get_output_embeddings()       # handles weight tying correctly
    device = lm_head.weight.device
    return tok, final_norm, lm_head, device


@torch.no_grad()
def project(hidden_layer, final_norm, lm_head, device, topk):
    """hidden_layer: (N, H) numpy for one layer.
    Returns top-k token IDs (N, topk) after final_norm + lm_head."""
    x = torch.from_numpy(hidden_layer).to(device=device, dtype=lm_head.weight.dtype)
    # Apply the model's final norm so the vector lives in the space lm_head expects.
    x = final_norm(x)
    logits = lm_head(x)                          # (N, vocab)
    top = torch.topk(logits, k=topk, dim=-1).indices  # (N, topk)
    return top.cpu().numpy()


def run_one(key, topk, n_sample, seed=0):
    model_id = MODEL_REGISTRY[key]
    proc_dir = PROCESSED / key
    if not (proc_dir / "hidden_states_en.npz").exists():
        print(f"[!] {key}: no saved hidden states in {proc_dir} — skipping.")
        return None

    tok, final_norm, lm_head, device = load_head(model_id)

    # Deterministic subsample of sentence indices (shared across languages so the
    # SAME sentences are compared EN vs RU vs KY).
    n_total = np.load(proc_dir / "labels.npy").shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    idx.sort()

    rows = []
    examples = []
    for lang in ["en", "ru", "ky"]:
        hs = np.load(proc_dir / f"hidden_states_{lang}.npz")["hidden_states"]  # (N, L, H)
        hs = hs[idx]                                                            # (n, L, H)
        n, n_layers, _ = hs.shape
        print(f"  {key} {lang.upper()}: {n} sentences x {n_layers} layers")
        for layer in range(n_layers):
            top_ids = project(hs[:, layer, :], final_norm, lm_head, device, topk)  # (n, topk)
            cls = np.empty(top_ids.shape, dtype="<U5")
            # Decode + classify each token (cache per unique id for speed).
            uniq = np.unique(top_ids)
            id2cls = {int(i): classify_token(tok.decode([int(i)])) for i in uniq}
            for a in range(top_ids.shape[0]):
                for b in range(top_ids.shape[1]):
                    cls[a, b] = id2cls[int(top_ids[a, b])]
            p_en = float((cls == "en").mean())
            p_cyr = float((cls == "cyr").mean())
            p_other = float((cls == "other").mean())
            rows.append({
                "model": key, "lang": lang, "layer": layer,
                "layer_frac": round(layer / (n_layers - 1), 4),
                "p_en": round(p_en, 4), "p_cyr": round(p_cyr, 4),
                "p_other": round(p_other, 4),
            })
        # Qualitative dump: top-8 tokens for the first sentence at a few depths.
        sample_layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                    3 * n_layers // 4, n_layers - 1]))
        examples.append(f"\n=== {key} {lang.upper()} (sentence idx {idx[0]}) ===")
        for layer in sample_layers:
            top_ids = project(hs[:1, layer, :], final_norm, lm_head, device, 8)[0]
            toks = [repr(tok.decode([int(i)])) for i in top_ids]
            examples.append(f"  L{layer:2d} (frac {layer/(n_layers-1):.2f}): " + " ".join(toks))

    del lm_head, final_norm
    import gc; gc.collect(); torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"{key}_english_pivot.csv"
    df.to_csv(csv_path, index=False)
    (OUT_DIR / f"{key}_examples.txt").write_text("\n".join(examples), encoding="utf-8")
    print(f"  saved -> {csv_path}")

    # Console summary: for RU and KY, the layer of maximal English dominance,
    # and whether it exceeds the same-language (Cyrillic) share there.
    for lang in ["ru", "ky"]:
        sub = df[df.lang == lang].reset_index(drop=True)
        j = int(sub["p_en"].values.argmax())
        peak = sub.iloc[j]
        final = sub.iloc[-1]
        print(f"  [{key} {lang.upper()}] peak English-decoding at layer {int(peak.layer)} "
              f"(frac {peak.layer_frac}): p_en={peak.p_en} vs p_cyr={peak.p_cyr}; "
              f"final layer p_en={final.p_en} p_cyr={final.p_cyr}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=list(MODEL_REGISTRY) + ["all"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    args = ap.parse_args()
    keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    summary = {}
    for k in keys:
        try:
            df = run_one(k, args.topk, args.n)
            summary[k] = "ok" if df is not None else "skipped (no states)"
        except Exception as e:
            summary[k] = f"FAILED: {type(e).__name__}: {e}"
            print(f"[!] {k} failed: {summary[k]}")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
