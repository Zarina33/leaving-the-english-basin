"""
Translate the parallel corpus via translation_api.client (NLLB API).
Drop-in replacement for scripts/2_translate.py.

Per-sentence calls (API has no batch endpoint). If NUM_VARIANTS > 1, all
variants + confidences are saved to extra columns.

Run from project root:
    python -m translation_api.translate
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from translation_api.client import NUM_VARIANTS, translate

INPUT_PATH = Path("data/processed/goemotions_en_1500.csv")
OUTPUT_PATH = Path("data/translated/parallel_corpus.csv")
SAVE_EVERY = 25  # rows


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _make_row(idx, en, ru_res, ky_res, emotion, n):
    row = {
        "id": idx,
        "text_en": en,
        "text_ru": ru_res.best,
        "text_ky": ky_res.best,
        "emotion": emotion,
        "conf_ru": ru_res.variants[0].confidence if ru_res.variants else 0.0,
        "conf_ky": ky_res.variants[0].confidence if ky_res.variants else 0.0,
    }
    if n > 1:
        for v in range(n):
            row[f"text_ru_v{v+1}"] = ru_res.variants[v].text if v < len(ru_res.variants) else ""
            row[f"conf_ru_v{v+1}"] = ru_res.variants[v].confidence if v < len(ru_res.variants) else 0.0
            row[f"text_ky_v{v+1}"] = ky_res.variants[v].text if v < len(ky_res.variants) else ""
            row[f"conf_ky_v{v+1}"] = ky_res.variants[v].confidence if v < len(ky_res.variants) else 0.0
    return row


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = NUM_VARIANTS

    df = pd.read_csv(INPUT_PATH)
    log(f"Loaded {len(df)} sentences from {INPUT_PATH}")
    log(f"Requesting {n} variant(s) per sentence")

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["id"].tolist())
        log(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining")
        df = df[~df["id"].isin(done_ids)].reset_index(drop=True)
        results = existing.to_dict("records")
    else:
        results = []

    total = len(df)
    if total == 0:
        log("Nothing to do.")
        return

    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        en = row["text"]
        try:
            ru_res = translate(en, tgt_lang="ru", num_variants=n)
            ky_res = translate(en, tgt_lang="ky", num_variants=n)
        except Exception as e:
            log(f"  FAILED on id={row['id']}: {e}; skipping")
            continue

        results.append(_make_row(row["id"], en, ru_res, ky_res, row["emotion"], n))

        if i % SAVE_EVERY == 0 or i == total:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            rate = i / (time.time() - t_start)
            eta = (total - i) / rate if rate > 0 else 0
            log(f"  [{i}/{total}] saved | {rate:.2f} sent/s | ETA {eta/60:.1f} min")

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    log(f"Done. Wrote {OUTPUT_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
