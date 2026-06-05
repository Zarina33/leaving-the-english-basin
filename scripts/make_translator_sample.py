"""
Build a stratified 100-sentence sample for human translator validation.

Outputs:
- data/translated/translator_sample_100.csv   (give to translator: EN + emotion + empty RU/KY cols)
- data/translated/translator_sample_100_gold.csv  (keep aside: Claude's RU/KY for later comparison)
"""
import pandas as pd

SRC = "data/translated/parallel_corpus_clean.csv"
N_PER_EMOTION = 17  # 17 * 6 = 102 sentences
SEED = 42

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} sentences")
print("Per emotion:", df["emotion"].value_counts().to_dict())

sample = (
    df.groupby("emotion", group_keys=False)
      .sample(n=N_PER_EMOTION, random_state=SEED)
      .reset_index(drop=True)
)
# Shuffle so the translator doesn't see emotions grouped together
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Translator-facing file (no Claude translations shown)
translator_file = sample[["id", "text_en", "emotion"]].copy()
translator_file["text_ru_human"] = ""
translator_file["text_ky_human"] = ""
translator_file["notes"] = ""
translator_file.to_csv("data/translated/translator_sample_100.csv", index=False)

# Gold file (kept aside for after-translation comparison)
gold = sample[["id", "text_en", "text_ru", "text_ky", "emotion"]].copy()
gold.to_csv("data/translated/translator_sample_100_gold.csv", index=False)

print(f"\nTranslator sample: data/translated/translator_sample_100.csv ({len(translator_file)} rows)")
print(f"Gold (hidden):     data/translated/translator_sample_100_gold.csv")
print("Per class in sample:", translator_file["emotion"].value_counts().to_dict())
