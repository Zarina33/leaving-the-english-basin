"""
Step 36: Verify the Llama-3.1-8B EN-RU CKA peak at layer 31.

The paper reports Llama's highest EN-RU CKA at layer 31 (the last transformer
layer), which is unusual — cross-lingual alignment typically peaks in middle
layers (Wu & Dredze 2019). This script checks whether L31 is a robust peak or
a numerical/measurement artifact by:

  1. Printing the full per-layer EN-RU CKA curve (full precision).
  2. Reporting the argmax layer and the top-3 layers.
  3. Bootstrap stability: recompute CKA on 50 subsamples (n=1000) per layer
     and report how often each layer is the argmax — a stable peak should win
     most resamples.
  4. Cross-checking against the saved cka_results.csv (paper value).

Read-only on hidden states; writes a short report to
  data/results/llama/cka_l31_check.csv

Usage:  python3 scripts/36_check_llama_cka.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HS_DIR  = Path("data/processed/llama")
RES_DIR = Path("data/results/llama")
N_BOOT  = 50
SUBSAMPLE = 1000
SEED = 42


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (den + 1e-10))


def main():
    en = np.load(HS_DIR / "hidden_states_en.npz")["hidden_states"].astype(np.float32)
    ru = np.load(HS_DIR / "hidden_states_ru.npz")["hidden_states"].astype(np.float32)
    n_samples, n_layers, _ = en.shape
    print(f"Llama HS: {en.shape}  (samples × layers × hidden)")

    # 1. Full-precision per-layer CKA
    cka = np.array([linear_cka(en[:, L, :], ru[:, L, :]) for L in range(n_layers)])
    argmax = int(cka.argmax())
    top3 = np.argsort(cka)[::-1][:3]

    print("\nPer-layer EN-RU CKA:")
    for L in range(n_layers):
        marker = "  <== PEAK" if L == argmax else ""
        print(f"  L{L:2d}: {cka[L]:.4f}{marker}")

    print(f"\nArgmax layer: {argmax}  (CKA={cka[argmax]:.4f})")
    print(f"Top-3 layers: {[int(x) for x in top3]}  "
          f"values={[round(float(cka[x]),4) for x in top3]}")

    # 2. Compare with saved paper value
    saved_path = RES_DIR / "cka_results.csv"
    if saved_path.exists():
        saved = pd.read_csv(saved_path)
        if "cka_en_ru" in saved.columns:
            saved_peak_layer = int(saved["cka_en_ru"].idxmax())
            print(f"\nSaved cka_results.csv EN-RU peak layer: {saved_peak_layer} "
                  f"(value {saved['cka_en_ru'].max():.4f})")
            if saved_peak_layer != argmax:
                print("  ⚠️  MISMATCH between recomputed and saved peak layer!")
            else:
                print("  ✓ matches recomputed argmax")

    # 3. Bootstrap stability of the peak layer
    rng = np.random.default_rng(SEED)
    argmax_counts = np.zeros(n_layers, dtype=int)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_samples, SUBSAMPLE)
        boot = np.array([linear_cka(en[idx][:, L, :], ru[idx][:, L, :])
                         for L in range(n_layers)])
        argmax_counts[boot.argmax()] += 1

    print(f"\nBootstrap peak-layer stability ({N_BOOT} resamples, n={SUBSAMPLE}):")
    for L in np.argsort(argmax_counts)[::-1]:
        if argmax_counts[L] > 0:
            print(f"  L{L:2d}: won argmax {argmax_counts[L]}/{N_BOOT} "
                  f"({100*argmax_counts[L]/N_BOOT:.0f}%)")

    stable = argmax_counts[argmax] / N_BOOT
    verdict = ("ROBUST peak" if stable >= 0.7 else
               "UNSTABLE — likely flat top / artifact" if stable < 0.4 else
               "MODERATELY stable")
    print(f"\nVerdict: layer {argmax} wins {100*stable:.0f}% of resamples → {verdict}")

    # Save report
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "layer": range(n_layers),
        "cka_en_ru": cka,
        "bootstrap_argmax_wins": argmax_counts,
    })
    out.to_csv(RES_DIR / "cka_l31_check.csv", index=False)
    print(f"\nReport → {RES_DIR / 'cka_l31_check.csv'}")
    print("\nInterpretation:")
    print("  - If the curve rises monotonically to L31 and L31 wins most")
    print("    resamples → real (Llama aligns EN/RU late). Discuss vs Wu&Dredze.")
    print("  - If the top is flat (L29-31 within ~0.01) and resamples split →")
    print("    report 'late plateau' rather than a sharp L31 peak.")


if __name__ == "__main__":
    main()
