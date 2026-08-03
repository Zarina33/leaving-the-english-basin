"""
Step 42: Lexical (bag-of-words / TF-IDF) baseline for emotion classification.

Reviewer M6: the paper's claim that "emotion-bearing vocabulary alone carries
substantial discriminative signal" (explaining the layer-0/early peak of
Gemma 4 and Llama) is not quantified. The Hewitt-Liang control task
(label shuffling) rules out that the probe fits a random mapping, but it does
NOT test whether a purely lexical model already reaches probe-level accuracy.
This script provides that missing baseline.

For each language we train TF-IDF (word 1-2 grams) + logistic regression with
the SAME 5-fold stratified CV as the neural probes, and report mean accuracy.
If the lexical baseline is close to the best-layer probing accuracy, that
supports "emotion is largely lexical here"; the gap over the baseline is the
part that requires contextual representation.

Uses only the text corpus (no models, no GPU). Fast.

Usage:
    python3 scripts/42_lexical_baseline.py
Output:
    data/results/lexical_baseline.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

CORPUS = Path("data/translated/parallel_corpus_clean.csv")
OUT = Path("data/results/lexical_baseline.csv")

EMO = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
L2I = {e: i for i, e in enumerate(EMO)}
SEED = 42


def cv_accuracy(texts, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    texts = np.array(texts, dtype=object)
    y = np.array(y)
    for tr, te in skf.split(texts, y):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=10.0,
                               class_weight="balanced"),
        )
        clf.fit(texts[tr], y[tr])
        accs.append((clf.predict(texts[te]) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    df = pd.read_csv(CORPUS)
    y = df["emotion"].map(L2I).values
    cols = {"en": "text_en", "ru": "text_ru", "ky": "text_ky"}

    rows = []
    for lg, c in cols.items():
        acc, sd = cv_accuracy(df[c].tolist(), y)
        rows.append({"lang": lg, "tfidf_acc": round(acc, 4),
                     "tfidf_std": round(sd, 4)})
        print(f"  {lg.upper()}: TF-IDF+LR acc = {acc:.3f} ± {sd:.3f}")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")
    print("\nCompare to best-layer probing (bootstrap_probing.csv) to quantify"
          "\nhow much emotion signal is already lexical vs. contextual.")


if __name__ == "__main__":
    main()
