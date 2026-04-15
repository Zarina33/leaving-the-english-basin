"""
Step 1: Download GoEmotions and select balanced subset for probing.

Downloads TSV files directly from Google Research GitHub.
Format: text \t comma-separated label ids \t example_id

Output: data/processed/goemotions_en_1500.csv
  columns: id, text, emotion

We keep 6 basic emotions (Ekman-based) and sample 250 per class = 1500 total.
"""

import io
import urllib.request
import pandas as pd
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
TSV_URLS = [
    f"{BASE_URL}/train.tsv",
    f"{BASE_URL}/dev.tsv",
    f"{BASE_URL}/test.tsv",
]

# 28 GoEmotions labels by index
ALL_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Map GoEmotions labels → 6 Ekman emotions
EMOTION_MAP = {
    "joy":     ["joy", "amusement", "excitement", "gratitude", "love", "relief", "pride", "optimism"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anger":   ["anger", "annoyance", "disapproval"],
    "fear":    ["fear", "nervousness"],
    "surprise":["surprise", "realization"],
    "disgust": ["disgust"],
}

SAMPLES_PER_CLASS = 250
OUTPUT_PATH = Path("data/processed/goemotions_en_1500.csv")


def load_tsvs() -> pd.DataFrame:
    dfs = []
    for i, url in enumerate(TSV_URLS, 1):
        split = url.split("/")[-1].replace(".tsv", "")
        print(f"  Downloading {split}...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep="\t", header=None,
                         names=["text", "label_ids", "id"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_emotion(label_ids_str: str) -> str | None:
    ids = [int(x) for x in str(label_ids_str).split(",")]
    if len(ids) != 1:
        return None  # keep only single-label examples
    label = ALL_LABELS[ids[0]]
    if label == "neutral":
        return None
    for emotion, sources in EMOTION_MAP.items():
        if label in sources:
            return emotion
    return None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading GoEmotions from GitHub...")
    df = load_tsvs()
    print(f"Total rows: {len(df)}")

    print("Mapping to 6 emotions...")
    df["emotion"] = df["label_ids"].apply(get_emotion)
    df = df.dropna(subset=["emotion"])

    print(f"After filtering (single-label, non-neutral): {len(df)}")
    print(df["emotion"].value_counts())

    # Balanced sample
    sampled = (
        df.groupby("emotion")
        .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )

    sampled = sampled[["id", "text", "emotion"]].reset_index(drop=True)

    print(f"\nFinal dataset: {len(sampled)} examples")
    print(sampled["emotion"].value_counts())

    sampled.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
