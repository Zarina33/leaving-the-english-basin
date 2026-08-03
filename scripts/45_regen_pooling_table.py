"""
Step 45: Regenerate Table 12 (pooling ablation) with mean-pooling numbers
taken from the unified probing table (Table 3) so the two tables are
consistent, keeping the existing last-token pooling column from
lasttoken_vs_mean.csv (last-token extraction has not changed).

Outputs a small CSV suitable for pasting into paper.tex.
"""

from pathlib import Path
import pandas as pd

unified = pd.read_csv("data/results/tables/unified_probing.csv")
lasttok = pd.read_csv("data/results/lasttoken_vs_mean.csv")

# unified model names -> lasttok model names
LT_NAMES = {
    "Gemma 4 E4B":   "Gemma 4 E4B",
    "Qwen3-8B":      "Qwen3-8B",
    "Llama-3.1-8B":  "Llama-3.1-8B",
    "Mistral-7B":    "Mistral-7B-v0.3",
}
# n_layers per model (for L/N normalization)
N_LAYERS = {
    "Gemma 4 E4B": 43, "Qwen3-8B": 37, "Llama-3.1-8B": 33, "Mistral-7B": 33,
}

rows = []
for uni_name, lt_name in LT_NAMES.items():
    n = N_LAYERS[uni_name] - 1
    for lang in ("EN", "RU", "KY"):
        u = unified[(unified.model == uni_name) & (unified.lang == lang)].iloc[0]
        lt = lasttok[(lasttok.model == lt_name) & (lasttok.lang == lang)].iloc[0]
        rows.append({
            "model": uni_name.split("-")[0].replace(" E4B", ""),
            "lang": lang,
            "mean_L": int(u.layer),
            "mean_LN": round(u.layer / n, 2),
            "mean_acc": round(u.linear_acc, 3),
            "last_L": int(lt.last_best_layer),
            "last_LN": round(lt.last_best_layer_norm, 2),
            "last_acc": round(lt.last_acc, 3),
            "d_acc": round(lt.last_acc - u.linear_acc, 3),
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("data/results/tables/table12_pooling_ablation.csv", index=False)
print(f"\nSaved: data/results/tables/table12_pooling_ablation.csv")
