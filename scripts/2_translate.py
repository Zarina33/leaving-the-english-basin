"""
Step 2: Translate English sentences to Russian and Kyrgyz using Claude CLI.
Uses your Claude subscription via `claude -p "..."` subprocess calls.

Input:  data/processed/goemotions_en_1500.csv
Output: data/translated/parallel_corpus.csv
  columns: id, text_en, text_ru, text_ky, emotion

Saves progress after every batch — safe to resume if interrupted.
"""

import re
import subprocess
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

BATCH_SIZE = 10
OUTPUT_PATH = Path("data/translated/parallel_corpus.csv")
INPUT_PATH = Path("data/processed/goemotions_en_1500.csv")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def translate_batch_claude(sentences: list[str], lang_code: str, lang_name: str) -> list[str]:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        f"Translate the following English sentences to {lang_name}. "
        f"These are short informal social media comments. "
        f"Preserve the emotional tone, slang, and register. "
        f"Replace [NAME] with [ИМЯ] in Russian or [АТ] in Kyrgyz. "
        f"Return ONLY the numbered translations, nothing else.\n\n{numbered}"
    )

    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        log(f"  ERROR from claude CLI: {result.stderr[:200]}")
        return [""] * len(sentences)

    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    translations = []
    for line in lines:
        # Strip "1. ", "2. " prefixes
        m = re.match(r"^\d+\.\s*(.*)", line)
        if m:
            translations.append(m.group(1).strip())

    if len(translations) != len(sentences):
        log(f"  WARNING: expected {len(sentences)}, got {len(translations)} translations")
        while len(translations) < len(sentences):
            translations.append("")

    return translations[:len(sentences)]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    log(f"Loaded {len(df)} sentences from {INPUT_PATH}")

    # Resume support
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["id"].tolist())
        log(f"Resuming: {len(done_ids)} already translated, {len(df) - len(done_ids)} remaining")
        df = df[~df["id"].isin(done_ids)]
        results = existing.to_dict("records")
    else:
        done_ids = set()
        results = []

    sentences = df["text"].tolist()
    ids = df["id"].tolist()
    emotions = df["emotion"].tolist()
    total_batches = (len(sentences) + BATCH_SIZE - 1) // BATCH_SIZE

    log(f"Starting translation: {len(sentences)} sentences, {total_batches} batches\n")

    for batch_num, start in enumerate(range(0, len(sentences), BATCH_SIZE), 1):
        batch_en = sentences[start:start + BATCH_SIZE]
        batch_ids = ids[start:start + BATCH_SIZE]
        batch_emotions = emotions[start:start + BATCH_SIZE]
        emotions_in_batch = set(batch_emotions)

        log(f"Batch {batch_num}/{total_batches} | rows {start+1}-{min(start+BATCH_SIZE, len(sentences))} | emotions: {', '.join(emotions_in_batch)}")

        log(f"  → Translating to Russian...")
        t0 = time.time()
        batch_ru = translate_batch_claude(batch_en, "ru", "Russian")
        log(f"  ✓ Russian done in {time.time()-t0:.1f}s")

        log(f"  → Translating to Kyrgyz...")
        t0 = time.time()
        batch_ky = translate_batch_claude(batch_en, "ky", "Kyrgyz")
        log(f"  ✓ Kyrgyz done in {time.time()-t0:.1f}s")

        for i in range(len(batch_en)):
            results.append({
                "id": batch_ids[i],
                "text_en": batch_en[i],
                "text_ru": batch_ru[i],
                "text_ky": batch_ky[i],
                "emotion": batch_emotions[i],
            })

        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

        empty_ru = sum(1 for r in results[-len(batch_en):] if not r["text_ru"])
        empty_ky = sum(1 for r in results[-len(batch_en):] if not r["text_ky"])
        log(f"  Saved {len(results)}/{len(sentences) + len(done_ids)} total | empty RU: {empty_ru} | empty KY: {empty_ky}\n")

    log(f"All done! Parallel corpus saved to {OUTPUT_PATH}")
    log(f"Final distribution:\n{pd.DataFrame(results)['emotion'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
