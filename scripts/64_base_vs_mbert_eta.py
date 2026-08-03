"""
Step 64: Paired bootstrap for eta_cc(XLM-R-base) - eta_cc(mBERT).

The mBERT->XLM-R-base contrast is the matched-depth data-swap in the
size-control decomposition (Section 6.3). Reviewer: the difference has
no CI/test. Same protocol as scripts/47: paired percentile bootstrap
over the 296 held-out sentences, 10,000 resamples, seed 42.
"""
from pathlib import Path
import numpy as np

CORR = Path("data/results/transfer_correct")
OUT = Path("data/results/tables/base_vs_mbert_eta.csv")
CHANCE = 1.0 / 6.0
KY_DIRS = ["en_ky", "ru_ky", "ky_en", "ky_ru"]
ENRU_DIRS = ["en_ru", "ru_en"]


def load(model):
    d = np.load(CORR / f"{model}.npz", allow_pickle=True)
    ky = np.stack([d[k] for k in KY_DIRS], axis=1).mean(axis=1)
    enru = np.stack([d[k] for k in ENRU_DIRS], axis=1).mean(axis=1)
    return ky, enru


def eta(ky, enru):
    denom = enru - CHANCE
    return (ky - CHANCE) / denom if denom > 0 else np.nan


def main():
    base_ky, base_enru = load("xlmr_base")
    mb_ky, mb_enru = load("mbert")
    n = len(base_ky)
    e_base = eta(base_ky.mean(), base_enru.mean())
    e_mb = eta(mb_ky.mean(), mb_enru.mean())
    print(f"n={n}  eta_base={e_base:.3f}  eta_mbert={e_mb:.3f}  diff={e_base-e_mb:.3f}")

    rng = np.random.default_rng(42)
    boots = rng.integers(0, n, size=(10_000, n))
    diffs, kydiffs = [], []
    for b in boots:
        eb = eta(base_ky[b].mean(), base_enru[b].mean())
        em = eta(mb_ky[b].mean(), mb_enru[b].mean())
        diffs.append(eb - em)
        kydiffs.append(base_ky[b].mean() - mb_ky[b].mean())
    diffs = np.array(diffs)
    kydiffs = np.array(kydiffs)
    ok = np.isfinite(diffs)
    print(f"finite resamples: {ok.sum()}/10000")
    d = diffs[ok]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    klo, khi = np.percentile(kydiffs, [2.5, 97.5])
    kp = 2 * min((kydiffs <= 0).mean(), (kydiffs >= 0).mean())
    print(f"eta diff  {e_base-e_mb:.3f} [{lo:.3f},{hi:.3f}] p={p:.4f}")
    print(f"KYavg diff {base_ky.mean()-mb_ky.mean():.3f} [{klo:.3f},{khi:.3f}] p={kp:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("contrast,diff,ci_lo,ci_hi,p,n_finite\n")
        f.write(f"eta_cc base-mbert,{e_base-e_mb:.4f},{lo:.4f},{hi:.4f},{p:.4f},{ok.sum()}\n")
        f.write(f"KYavg base-mbert,{base_ky.mean()-mb_ky.mean():.4f},{klo:.4f},{khi:.4f},{kp:.4f},10000\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
