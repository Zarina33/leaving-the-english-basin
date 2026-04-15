"""
Step 3: Validate and inspect the parallel corpus.

Checks:
- No empty translations
- Reasonable length ratio (translation shouldn't be 10x longer/shorter)
- Prints random samples per emotion for manual review

Output: data/translated/parallel_corpus_clean.csv (with flagged rows removed)
"""

import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/translated/parallel_corpus.csv")
OUTPUT_PATH = Path("data/translated/parallel_corpus_clean.csv")


def length_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s2) / len(s1)


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows\n")

    issues = []

    # Check empty translations
    empty_ru = df["text_ru"].isna() | (df["text_ru"] == "")
    empty_ky = df["text_ky"].isna() | (df["text_ky"] == "")
    print(f"Empty RU translations: {empty_ru.sum()}")
    print(f"Empty KY translations: {empty_ky.sum()}")

    # Check length ratios (EN->RU and EN->KY)
    df["ratio_ru"] = df.apply(lambda r: length_ratio(r["text_en"], str(r["text_ru"])), axis=1)
    df["ratio_ky"] = df.apply(lambda r: length_ratio(r["text_en"], str(r["text_ky"])), axis=1)

    suspicious_ru = df[(df["ratio_ru"] < 0.3) | (df["ratio_ru"] > 4.0)]
    suspicious_ky = df[(df["ratio_ky"] < 0.3) | (df["ratio_ky"] > 4.0)]
    print(f"\nSuspicious RU length ratios: {len(suspicious_ru)}")
    print(f"Suspicious KY length ratios: {len(suspicious_ky)}")

    # Flag rows to remove
    flagged = empty_ru | empty_ky
    print(f"\nFlagged for removal: {flagged.sum()}")

    df_clean = df[~flagged].drop(columns=["ratio_ru", "ratio_ky"])
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Clean corpus: {len(df_clean)} rows → {OUTPUT_PATH}\n")

    # Print samples per emotion for manual inspection
    print("=" * 60)
    print("RANDOM SAMPLES PER EMOTION (for manual review)")
    print("=" * 60)
    for emotion in sorted(df_clean["emotion"].unique()):
        subset = df_clean[df_clean["emotion"] == emotion].sample(min(3, len(df_clean[df_clean["emotion"] == emotion])), random_state=42)
        print(f"\n--- {emotion.upper()} ({len(df_clean[df_clean['emotion'] == emotion])} examples) ---")
        for _, row in subset.iterrows():
            print(f"  EN: {row['text_en']}")
            print(f"  RU: {row['text_ru']}")
            print(f"  KY: {row['text_ky']}")
            print()

    print("\nFinal class distribution:")
    print(df_clean["emotion"].value_counts())


if __name__ == "__main__":
    main()
