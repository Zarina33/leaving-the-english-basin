"""
Step 0: Download all models to HuggingFace cache.

Run this once to pre-download everything. After that,
extraction scripts (4, 11, 16, 16b) work fully offline.

Models:
  - google/gemma-4-E4B-it       (~4B MoE, ~6 GB)
  - Qwen/Qwen3-8B               (~8B dense, ~5 GB)
  - mistralai/Mistral-7B-v0.3   (~7B dense, ~5 GB)
  - meta-llama/Llama-3.1-8B     (~8B dense, ~5 GB)  ← requires HF_TOKEN
"""

import os
import sys
from huggingface_hub import snapshot_download, login, get_token

MODELS = [
    ("google/gemma-4-E4B-it",       False),
    ("Qwen/Qwen3-8B",               False),
    ("mistralai/Mistral-7B-v0.3",   False),
    ("meta-llama/Llama-3.1-8B",     True),   # gated — needs HF_TOKEN
]


def check_auth():
    token = os.environ.get("HF_TOKEN") or get_token()
    if token:
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: authenticated.\n")
        return token
    print("WARNING: HF_TOKEN not set. Llama-3.1-8B will be skipped.")
    print("Set it with:  export HF_TOKEN=hf_your_token_here\n")
    return None


def download(model_id: str, token: str | None, gated: bool):
    if gated and not token:
        print(f"  SKIP {model_id}  (gated, no token)")
        return

    print(f"\n{'─'*60}")
    print(f"  Downloading: {model_id}")
    print(f"{'─'*60}")
    try:
        path = snapshot_download(
            repo_id=model_id,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print(f"  ✓ Saved to: {path}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    token = check_auth()

    print("Downloading models (PyTorch weights only, no TF/Flax)...\n")
    for model_id, gated in MODELS:
        download(model_id, token, gated)

    print("\n" + "="*60)
    print("All downloads complete.")
    print("\nNext steps:")
    print("  python scripts/4_extract_hidden_states.py   # Gemma 4")
    print("  python scripts/11_extract_qwen3.py          # Qwen3")
    print("  python scripts/16b_extract_mistral.py       # Mistral")
    print("  python scripts/16_extract_llama.py          # Llama")
    print("  python scripts/18_multimodel_pipeline.py    # All results")


if __name__ == "__main__":
    main()
