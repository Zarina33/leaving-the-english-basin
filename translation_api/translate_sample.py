"""
Translate the 100-sentence sample via NLLB API and merge with the existing
Claude gold draft for side-by-side review.

Output: data/translated/translator_sample_100_review.csv

Columns:
  id, emotion, text_en,
  claude_ru, nllb_ru_v1..vN, conf_ru_v1..vN, final_ru,
  claude_ky, nllb_ky_v1..vN, conf_ky_v1..vN, final_ky,
  notes
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from translation_api.client import NUM_VARIANTS, translate

SAMPLE_PATH = Path("data/translated/translator_sample_100.csv")
GOLD_PATH = Path("data/translated/translator_sample_100_gold.csv")
OUT_PATH = Path("data/translated/translator_sample_100_review.csv")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    n = NUM_VARIANTS
    sample = pd.read_csv(SAMPLE_PATH)
    gold = pd.read_csv(GOLD_PATH).rename(columns={"text_ru": "claude_ru", "text_ky": "claude_ky"})
    df = sample.merge(gold[["id", "claude_ru", "claude_ky"]], on="id", how="left")

    log(f"{len(df)} rows, requesting {n} NLLB variant(s) per language")

    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples(index=False), 1):
        try:
            ru = translate(r.text_en, "ru", num_variants=n)
            ky = translate(r.text_en, "ky", num_variants=n)
        except Exception as e:
            log(f"  FAILED id={r.id}: {e}")
            continue

        out = {
            "id": r.id,
            "emotion": r.emotion,
            "text_en": r.text_en,
            "claude_ru": r.claude_ru,
        }
        for v in range(n):
            out[f"nllb_ru_v{v+1}"] = ru.variants[v].text if v < len(ru.variants) else ""
            out[f"conf_ru_v{v+1}"] = round(ru.variants[v].confidence, 3) if v < len(ru.variants) else ""
        out["final_ru"] = ""
        out["claude_ky"] = r.claude_ky
        for v in range(n):
            out[f"nllb_ky_v{v+1}"] = ky.variants[v].text if v < len(ky.variants) else ""
            out[f"conf_ky_v{v+1}"] = round(ky.variants[v].confidence, 3) if v < len(ky.variants) else ""
        out["final_ky"] = ""
        out["notes"] = ""
        rows.append(out)

        if i % 10 == 0 or i == len(df):
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
            rate = i / (time.time() - t0)
            log(f"  [{i}/{len(df)}] saved | {rate:.2f} sent/s | ETA {(len(df)-i)/rate/60:.1f} min")

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    log(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
