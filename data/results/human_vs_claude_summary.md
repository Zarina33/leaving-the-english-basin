# Human vs. Claude Translation Probing Control

**Purpose:** validate that the use of Claude 3.7 Sonnet for corpus translation does
not artificially inflate probing accuracy (i.e., that probing is decoding genuine
emotion structure, not LLM-translation distribution).

**Setup:**
- 102 sentences stratified across 6 emotions, sampled from the 1480-sentence parallel
  corpus.
- Independent human translator (trilingual EN/RU/KY native) produced RU and KY
  translations of the same source EN sentences, with explicit prohibition of any
  machine translation. Translations were quality-reviewed before use (v2).
- For each model and each language ∈ {RU, KY}:
  1. Hold out the 102 control IDs from the full corpus.
  2. Train a linear probe (Adam, lr=1e-2, wd=1e-4, 300 epochs) on the remaining
     1378 Claude-translated examples at the model's best layer (per Table 2).
  3. Test on (a) Claude versions of the 102 → `acc_claude`,
              (b) Human versions of the 102 → `acc_human`.
  4. Bootstrap 95% CI (10,000 resamples) for the paired difference.

## Results

| Model     | Lang | Layer | acc_claude | acc_human | Δ      | 95% CI            |
|-----------|------|-------|------------|-----------|--------|-------------------|
| Gemma 4   | RU   | 6     | 0.529      | 0.529     |  0.000 | [ 0.000,  0.000]  |
| Gemma 4   | KY   | 4     | 0.480      | 0.451     | +0.029 | [-0.049, +0.108]  |
| Qwen3     | RU   | 11    | 0.578      | 0.578     |  0.000 | [ 0.000,  0.000]  |
| Qwen3     | KY   | 27    | 0.441      | 0.441     |  0.000 | [-0.069, +0.069]  |
| Llama 3.1 | RU   | 7     | 0.500      | 0.500     |  0.000 | [-0.029, +0.029]  |
| Llama 3.1 | KY   | 3     | 0.451      | 0.431     | +0.020 | [-0.059, +0.098]  |
| Mistral   | RU   | 9     | 0.588      | 0.588     |  0.000 | [ 0.000,  0.000]  |
| Mistral   | KY   | 2     | 0.412      | 0.422     | -0.010 | [-0.069, +0.049]  |
| XLM-R     | RU   | 12    | 0.402      | 0.402     |  0.000 | [ 0.000,  0.000]  |
| XLM-R     | KY   | 18    | 0.480      | 0.422     | +0.059 | [-0.020, +0.137]  |

**Aggregate statistics:**
- Mean |Δ| across 10 cells: 0.012 (1.2 pp)
- Max |Δ|: 0.059 (XLM-R KY, n.s.)
- 10 / 10 95% CIs include zero (no significant difference)
- For 4/5 models on RU, predictions are *identical* between Claude and human
  translations (Δ = 0.000 exact)

## Conclusion

Probing accuracy on the human-translated control set does not differ
significantly from that on the Claude-translated version at any (model, language)
cell. The maximum observed difference is 5.9 pp (XLM-R Kyrgyz), well within
bootstrap noise. We therefore conclude that the use of LLM-generated translations
does not substantially inflate probing performance, and that the probes are
decoding genuine emotion structure rather than translation-distribution
artifacts.
