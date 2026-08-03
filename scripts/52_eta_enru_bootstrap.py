"""
Step 52: Paired bootstrap for the EN->RU-only normalized efficiency
  eta' = (KY_avg - 1/6) / (EN->RU - 1/6)
on the same 296 held-out sentences (per-example vectors from scripts/40).

Motivation: the eta_cc robustness check that normalizes by within-language
EN probing mixes samples (probing: 1480 5-fold CV; transfer: 296 held-out),
so it has no paired CI. Normalizing by the EN->RU direction alone keeps
numerator and denominator on the same 296 examples -> honest paired
bootstrap, reported as the primary normalization-robustness check.

Output: data/results/tables/eta_enru_bootstrap.csv
"""
from pathlib import Path
import numpy as np, csv

CORR=Path("data/results/transfer_correct"); OUT=Path("data/results/tables/eta_enru_bootstrap.csv")
SEED=42; NB=10_000; C=1/6
KY=["en_ky","ru_ky","ky_en","ky_ru"]
MODELS=["xlmr","llama","qwen3","gemma4","qwen25_7b","mistral","mbert","olmo2_7b"]
DISP={"xlmr":"XLM-R","qwen3":"Qwen3","llama":"Llama-3.1","gemma4":"Gemma 4","mistral":"Mistral",
      "mbert":"mBERT","olmo2_7b":"OLMo-2","qwen25_7b":"Qwen2.5"}
pex={}
for m in MODELS:
    d=np.load(CORR/f"{m}.npz",allow_pickle=True)
    pex[m]=(np.stack([d[k] for k in KY],1).mean(1), d["en_ru"].astype(float))
n=len(pex["xlmr"][0]); print("n =",n)
def eta(ky,enru,idx):
    den=enru[idx].mean()-C
    return (ky[idx].mean()-C)/den if den>0 else np.nan
rng=np.random.default_rng(SEED)
boots=rng.integers(0,n,size=(NB,n))
rows=[]
E={m:{ 'point': eta(*pex[m],np.arange(n)) } for m in MODELS}
samples={m:np.array([eta(*pex[m],b) for b in boots]) for m in MODELS}
for m in MODELS:
    s=samples[m]; s=s[~np.isnan(s)]
    rows.append(dict(model=DISP[m],eta_enru=round(E[m]['point'],4),
        ci_lo=round(np.percentile(s,2.5),4),ci_hi=round(np.percentile(s,97.5),4),comparison="",delta="",p_two_sided=""))
    print(rows[-1])
for m in ["llama","qwen3","mbert"]:
    d=samples["xlmr"]-samples[m]
    dd=d[~np.isnan(d)]
    point=E["xlmr"]['point']-E[m]['point']
    p=2*min((dd<=0).mean(),(dd>=0).mean()); p=min(p,1.0)
    rows.append(dict(model="",eta_enru="",ci_lo=round(np.percentile(dd,2.5),4),ci_hi=round(np.percentile(dd,97.5),4),
        comparison=f"XLM-R vs {DISP[m]}",delta=round(point,4),p_two_sided=round(p,4)))
    print(rows[-1])
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("saved",OUT)
