"""
Step 15: Generate summary tables for paper.

1. Per-emotion best-layer accuracy table (both models × 3 languages)
2. Cross-lingual transfer summary table
3. CKA peak similarity table
4. Print LaTeX-ready tables

Output:
  data/results/tables/table_emotion_summary.csv
  data/results/tables/table_transfer_summary.csv
  data/results/tables/table_cka_summary.csv
  data/results/tables/paper_tables.tex  ← copy-paste into LaTeX
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("data/results")
Q_RESULTS   = Path("data/results/qwen3")
TABLES_DIR  = Path("data/results/tables")

EMOTIONS    = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LANG_LABELS = {"en": "English", "ru": "Russian", "ky": "Kyrgyz"}


def bold(val, best_val, fmt=".3f"):
    s = f"{val:{fmt}}"
    return f"\\textbf{{{s}}}" if abs(val - best_val) < 1e-6 else s


def df_to_latex(df, caption, label, bold_cols=None):
    col_fmt = "l" + "r" * (len(df.columns))
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
    ]
    # Header
    lines.append(" & ".join([""] + list(df.columns)) + " \\\\")
    lines.append("\\midrule")
    # Rows
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.values]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all results ───────────────────────────────────────────────────────
    gemma_prob  = pd.read_csv(RESULTS_DIR / "probing_detailed.csv")
    qwen_prob   = pd.read_csv(Q_RESULTS   / "probing_detailed.csv")
    gemma_emo   = pd.read_csv(RESULTS_DIR / "per_emotion_layers.csv")
    qwen_emo    = pd.read_csv(Q_RESULTS   / "per_emotion_layers.csv")
    gemma_cka   = pd.read_csv(RESULTS_DIR / "cka_results.csv")
    qwen_cka    = pd.read_csv(Q_RESULTS   / "cka_results.csv")
    gemma_trans = pd.read_csv(RESULTS_DIR / "transfer_results.csv")
    qwen_trans  = pd.read_csv(Q_RESULTS   / "transfer_results.csv")

    # ── Table 1: Overall probing — best acc per language ──────────────────────
    print("\n" + "="*65)
    print("TABLE 1: Best Probing Accuracy per Language")
    print("="*65)

    t1_rows = []
    for lang in ["en", "ru", "ky"]:
        g = gemma_prob[gemma_prob["lang"] == lang]
        q = qwen_prob [qwen_prob ["lang"] == lang]
        g_best = g.loc[g["acc_mean"].idxmax()]
        q_best = q.loc[q["acc_mean"].idxmax()]
        t1_rows.append({
            "Language":     LANG_LABELS[lang],
            "Gemma4 Layer": int(g_best["layer"]),
            "Gemma4 Acc":   round(g_best["acc_mean"], 3),
            "Gemma4 ±Std":  round(g_best["acc_std"],  3),
            "Qwen3 Layer":  int(q_best["layer"]),
            "Qwen3 Acc":    round(q_best["acc_mean"], 3),
            "Qwen3 ±Std":   round(q_best["acc_std"],  3),
        })
    t1 = pd.DataFrame(t1_rows)
    print(t1.to_string(index=False))
    t1.to_csv(TABLES_DIR / "table_probing_summary.csv", index=False)

    # ── Table 2: Per-emotion best accuracy (both models, EN only for space) ───
    print("\n" + "="*65)
    print("TABLE 2: Per-emotion Best One-vs-Rest Accuracy")
    print("="*65)

    t2_rows = []
    for emo in EMOTIONS:
        row = {"Emotion": emo.capitalize()}
        for model_name, emo_df in [("Gemma4", gemma_emo), ("Qwen3", qwen_emo)]:
            for lang in ["en", "ru", "ky"]:
                sub = emo_df[(emo_df["emotion"] == emo) & (emo_df["lang"] == lang)]
                best_acc   = sub["ovr_acc"].max()
                best_layer = int(sub.loc[sub["ovr_acc"].idxmax(), "layer"])
                row[f"{model_name}/{lang.upper()}"] = f"{best_acc:.3f} (L{best_layer})"
        t2_rows.append(row)

    t2 = pd.DataFrame(t2_rows).set_index("Emotion")
    print(t2.to_string())
    t2.to_csv(TABLES_DIR / "table_emotion_summary.csv")

    # ── Table 3: Cross-lingual transfer peak ──────────────────────────────────
    print("\n" + "="*65)
    print("TABLE 3: Cross-lingual Transfer (peak accuracy per direction)")
    print("="*65)

    t3_rows = []
    for src, tgt in [("en","ru"), ("en","ky"), ("ru","en"),
                     ("ru","ky"), ("ky","en"), ("ky","ru")]:
        g_sub = gemma_trans[(gemma_trans["src"]==src) & (gemma_trans["tgt"]==tgt)]
        q_sub = qwen_trans [(qwen_trans ["src"]==src) & (qwen_trans ["tgt"]==tgt)]
        t3_rows.append({
            "Direction":       f"{src.upper()} → {tgt.upper()}",
            "Gemma4 Peak Acc": round(g_sub["transfer_acc"].max(), 3),
            "Gemma4 Layer":    int(g_sub.loc[g_sub["transfer_acc"].idxmax(), "layer"]),
            "Qwen3 Peak Acc":  round(q_sub["transfer_acc"].max(), 3),
            "Qwen3 Layer":     int(q_sub.loc[q_sub["transfer_acc"].idxmax(), "layer"]),
        })
    t3 = pd.DataFrame(t3_rows)
    print(t3.to_string(index=False))
    t3.to_csv(TABLES_DIR / "table_transfer_summary.csv", index=False)

    # ── Table 4: CKA peak cross-lingual similarity ────────────────────────────
    print("\n" + "="*65)
    print("TABLE 4: Peak CKA Cross-lingual Similarity")
    print("="*65)

    # CKA is in wide format: columns cka_en_ru, cka_en_ky, cka_ru_ky
    cka_pairs = ["en_ru", "en_ky", "ru_ky"]
    t4_rows = []
    for pair in cka_pairs:
        g_col = f"cka_{pair}"
        q_col = f"cka_{pair}"
        t4_rows.append({
            "Pair":            pair.replace("_", " → ").upper(),
            "Gemma4 Peak CKA": round(gemma_cka[g_col].max(), 3),
            "Gemma4 Layer":    int(gemma_cka.loc[gemma_cka[g_col].idxmax(), "layer"]),
            "Qwen3 Peak CKA":  round(qwen_cka [q_col].max(), 3),
            "Qwen3 Layer":     int(qwen_cka .loc[qwen_cka [q_col].idxmax(), "layer"]),
        })
    t4 = pd.DataFrame(t4_rows)
    print(t4.to_string(index=False))
    t4.to_csv(TABLES_DIR / "table_cka_summary.csv", index=False)

    # ── Write LaTeX tables ────────────────────────────────────────────────────
    latex_parts = [
        "% ============================================================",
        "% Auto-generated tables for emotion probing paper",
        "% ============================================================",
        "\\usepackage{booktabs}  % add to preamble",
        "",
    ]

    # Table 1 LaTeX
    latex_parts.append("% --- Table 1: Probing Accuracy ---")
    latex_parts.append("\\begin{table}[ht]")
    latex_parts.append("\\centering")
    latex_parts.append("\\caption{Best layer-wise probing accuracy per language."
                       " Chance level = 1/6 $\\approx$ 0.167.}")
    latex_parts.append("\\label{tab:probing}")
    latex_parts.append("\\begin{tabular}{lrrrrr}")
    latex_parts.append("\\toprule")
    latex_parts.append("Language & \\multicolumn{2}{c}{Gemma 4 E4B} & "
                        "\\multicolumn{2}{c}{Qwen3-8B} \\\\")
    latex_parts.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    latex_parts.append("& Layer & Acc $\\pm$ Std & Layer & Acc $\\pm$ Std \\\\")
    latex_parts.append("\\midrule")
    for _, row in t1.iterrows():
        latex_parts.append(
            f"{row['Language']} & {row['Gemma4 Layer']} & "
            f"{row['Gemma4 Acc']:.3f} $\\pm$ {row['Gemma4 ±Std']:.3f} & "
            f"{row['Qwen3 Layer']} & "
            f"{row['Qwen3 Acc']:.3f} $\\pm$ {row['Qwen3 ±Std']:.3f} \\\\"
        )
    latex_parts += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # Table 3 LaTeX (transfer)
    latex_parts.append("% --- Table 3: Transfer Accuracy ---")
    latex_parts.append("\\begin{table}[ht]")
    latex_parts.append("\\centering")
    latex_parts.append("\\caption{Peak cross-lingual transfer accuracy."
                        " Models trained on source language, tested on target.}")
    latex_parts.append("\\label{tab:transfer}")
    latex_parts.append("\\begin{tabular}{lrrrr}")
    latex_parts.append("\\toprule")
    latex_parts.append("Direction & \\multicolumn{2}{c}{Gemma 4 E4B} & "
                        "\\multicolumn{2}{c}{Qwen3-8B} \\\\")
    latex_parts.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    latex_parts.append("& Peak Acc & Layer & Peak Acc & Layer \\\\")
    latex_parts.append("\\midrule")
    for _, row in t3.iterrows():
        latex_parts.append(
            f"{row['Direction']} & {row['Gemma4 Peak Acc']:.3f} & {row['Gemma4 Layer']} & "
            f"{row['Qwen3 Peak Acc']:.3f} & {row['Qwen3 Layer']} \\\\"
        )
    latex_parts += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    tex_path = TABLES_DIR / "paper_tables.tex"
    tex_path.write_text("\n".join(latex_parts))
    print(f"\nSaved LaTeX tables → {tex_path}")
    print("\nAll tables saved to data/results/tables/")


if __name__ == "__main__":
    main()
