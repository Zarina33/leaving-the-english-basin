"""
Step 39: Plot the logit-lens English-pivot curves (Figure for the paper).

Reads data/results/logit_lens/{model}_english_pivot.csv (from step 38) and
produces a per-model figure: for each non-English input language (RU, KY),
the fraction of top-k decoded tokens that are English (Latin) vs. Cyrillic,
as a function of normalized network depth.

The signature of English-pivoting (Wendler et al., 2024): English dominates
the mid/late layers and only the final layer(s) switch back to the input
language. Kyrgyz is expected to leave the English basin earlier / less
cleanly than Russian.

Usage:
    python3 scripts/39_plot_english_pivot.py --models llama qwen3
Output:
    data/results/figures/fig25_english_pivot.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_DIR = Path("data/results/logit_lens")
FIG_PATH = Path("data/results/figures/fig25_english_pivot.png")

EN_COLOR = "#C44E52"   # English (Latin)
CYR_COLOR = "#4C72B0"  # Cyrillic (input language)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "qwen3"])
    args = ap.parse_args()

    models = [m for m in args.models if (CSV_DIR / f"{m}_english_pivot.csv").exists()]
    if not models:
        print("No CSVs found; run step 38 first.")
        return

    langs = ["ru", "ky"]
    fig, axes = plt.subplots(len(models), len(langs),
                             figsize=(4.2 * len(langs), 3.1 * len(models)),
                             squeeze=False, sharex=True, sharey=True)

    for r, model in enumerate(models):
        df = pd.read_csv(CSV_DIR / f"{model}_english_pivot.csv")
        for c, lang in enumerate(langs):
            ax = axes[r][c]
            s = df[df.lang == lang].sort_values("layer_frac")
            ax.plot(s.layer_frac, s.p_en, color=EN_COLOR, lw=2,
                    label="English (Latin)")
            ax.plot(s.layer_frac, s.p_cyr, color=CYR_COLOR, lw=2,
                    label=f"{lang.upper()} (Cyrillic)")
            ax.fill_between(s.layer_frac, s.p_en, color=EN_COLOR, alpha=0.08)
            # mark the switch-back layer: first layer from which cyr > en
            # holds through to the output (sustained crossing)
            xl = None
            vals = s.reset_index(drop=True)
            for i in range(len(vals)):
                tail = vals.iloc[i:]
                if (tail.p_cyr > tail.p_en).all():
                    xl = vals.iloc[i].layer_frac
                    break
            if xl is not None:
                ax.axvline(xl, color="gray", ls="--", lw=1)
                ax.text(xl - 0.02, 0.5, "switch-back", rotation=90,
                        va="center", ha="right", fontsize=7, color="gray")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, 1)
            if r == 0:
                ax.set_title(f"{lang.upper()} input", fontsize=11)
            if c == 0:
                pretty = {"llama": "Llama-3.1-8B", "qwen3": "Qwen3-8B",
                          "mistral": "Mistral-7B"}.get(model, model)
                ax.set_ylabel(f"{pretty}\nfraction of top-20 tokens", fontsize=9)
            if r == len(models) - 1:
                ax.set_xlabel("normalized depth (layer / N)", fontsize=9)
            ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, loc="center left")
    axes[0][1].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {FIG_PATH}")


if __name__ == "__main__":
    main()
