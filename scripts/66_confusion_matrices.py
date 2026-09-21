"""
Step 66: Multiclass confusion matrices at the unified best layers (camera-ready;
reviewer request: which emotion categories are confused with one another).

Re-runs the exact Step-43 linear probe (same seed, folds, epochs) at the best
layer recorded in unified_probing.csv, so the per-example predictions are the
ones behind Table 3. The script asserts that the reproduced accuracy matches
the recorded one before writing anything.

Pair confusion rate for emotions (a, b) is symmetric:
    (C[a,b] + C[b,a]) / (n_a + n_b)
i.e. the share of a-or-b sentences assigned to the other member of the pair.

Usage:
    python3 scripts/66_confusion_matrices.py
Output:
    data/results/tables/confusion_matrices.csv   (long format, all 15 cells)
    data/results/tables/confusion_pairs.csv      (all 15 unordered pairs per cell)
    data/results/tables/confusion_pairs_summary.csv   (5-model mean per pair x lang)
    data/results/tables/confusion_rank_agreement.csv  (cross-language Spearman)
"""

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

spec = importlib.util.spec_from_file_location(
    "unified", Path(__file__).parent / "43_unified_probing.py")
unified = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified)
unified.DEVICE = torch.device("cpu")  # deterministic, matches the Step-43 run

UNIFIED = Path("data/results/tables/unified_probing.csv")
OUT_CM = Path("data/results/tables/confusion_matrices.csv")
OUT_PAIRS = Path("data/results/tables/confusion_pairs.csv")
OUT_SUMMARY = Path("data/results/tables/confusion_pairs_summary.csv")
OUT_RANK = Path("data/results/tables/confusion_rank_agreement.csv")
TOL = 0.005


def emotion_names(d):
    """Label index -> emotion name, read from the corpus the labels came from."""
    corpus = pd.read_csv("data/translated/parallel_corpus_clean.csv")
    labels = np.load(d / "labels.npy")
    names = {}
    for idx, emo in zip(labels, corpus["emotion"]):
        names.setdefault(int(idx), emo)
    assert len(names) == 6 and len(corpus) == len(labels)
    # every index must map to exactly one emotion
    assert all(corpus["emotion"][labels == i].nunique() == 1 for i in names)
    return [names[i] for i in range(6)]


def main():
    best = pd.read_csv(UNIFIED)
    cm_rows, pair_rows = [], []
    for name, d in unified.MODELS.items():
        labels = np.load(d / "labels.npy")
        emos = emotion_names(d)
        for lang in ["en", "ru", "ky"]:
            rec = best[(best.model == name) & (best.lang == lang.upper())].iloc[0]
            L = int(rec.layer)
            hs = np.load(d / f"hidden_states_{lang}.npz")["hidden_states"]
            preds = unified.cv_predictions(hs[:, L, :], labels, unified.Lin)
            acc = (preds == labels).mean()
            assert abs(acc - rec.linear_acc) < TOL, (name, lang, acc, rec.linear_acc)
            C = np.zeros((6, 6), dtype=int)
            np.add.at(C, (labels, preds), 1)
            n = C.sum(1)
            for i in range(6):
                for j in range(6):
                    cm_rows.append({"model": name, "lang": lang.upper(), "layer": L,
                                    "true": emos[i], "pred": emos[j],
                                    "count": int(C[i, j]),
                                    "row_share": round(C[i, j] / n[i], 4)})
            for i, j in combinations(range(6), 2):
                pair_rows.append({"model": name, "lang": lang.upper(),
                                  "pair": f"{emos[i]}-{emos[j]}",
                                  "rate": round((C[i, j] + C[j, i]) / (n[i] + n[j]), 4)})
            print(f"{name} {lang.upper()} L{L}: acc={acc:.4f} (recorded {rec.linear_acc})")
    pd.DataFrame(cm_rows).to_csv(OUT_CM, index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT_PAIRS, index=False)

    # Appendix table: pair rate averaged over the 5 models, per language
    summary = (pairs.pivot_table(index="pair", columns="lang", values="rate")
               [["EN", "RU", "KY"]])
    summary["mean"] = summary.mean(1)
    summary.sort_values("mean", ascending=False).round(4).to_csv(OUT_SUMMARY)

    # Is the confusion structure shared across languages? Spearman over 15 pairs.
    rank_rows = []
    for name, g in pairs.groupby("model", sort=False):
        w = g.pivot(index="pair", columns="lang", values="rate")
        for a, b in [("EN", "RU"), ("EN", "KY"), ("RU", "KY")]:
            rho, p = spearmanr(w[a], w[b])
            rank_rows.append({"model": name, "langs": f"{a}-{b}",
                              "spearman": round(rho, 4), "p": p})
    pd.DataFrame(rank_rows).to_csv(OUT_RANK, index=False)
    print(f"Saved {OUT_CM}, {OUT_PAIRS}, {OUT_SUMMARY}, {OUT_RANK}")


if __name__ == "__main__":
    main()
