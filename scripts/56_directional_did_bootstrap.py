"""
Step 56: Difference-in-differences test for the directional argument.

The into-KY vs KY-source asymmetry (Section 6.3) was point estimates only.
Here: DiD = [(XLM-R - decoder) on into-KY] - [(XLM-R - decoder) on
KY-source], paired bootstrap over the shared 296 test sentences, for the
two top decoders (Qwen3, Llama) and Qwen3-fp16 (matched precision).

Output: data/results/tables/directional_did.csv
"""
from pathlib import Path
import numpy as np, csv
C=Path("data/results/transfer_correct"); SEED=42; NB=10_000
INTO=["en_ky","ru_ky"]; SRC=["ky_en","ky_ru"]
def load(m):
    d=np.load(C/f"{m}.npz",allow_pickle=True)
    return (np.stack([d[k] for k in INTO],1).mean(1),
            np.stack([d[k] for k in SRC],1).mean(1))
x_into,x_src=load("xlmr")
n=len(x_into); rng=np.random.default_rng(SEED)
boots=rng.integers(0,n,size=(NB,n))
rows=[]
for dec in ["qwen3","llama","qwen3_fp16"]:
    d_into,d_src=load(dec)
    did_pt=(x_into-d_into).mean()-(x_src-d_src).mean()
    stat=(x_into[boots].mean(1)-d_into[boots].mean(1))-(x_src[boots].mean(1)-d_src[boots].mean(1))
    p=2*min((stat<=0).mean(),(stat>=0).mean())
    rows.append(dict(decoder=dec,
        adv_into=round((x_into-d_into).mean(),4),adv_src=round((x_src-d_src).mean(),4),
        did=round(did_pt,4),ci_lo=round(np.percentile(stat,2.5),4),
        ci_hi=round(np.percentile(stat,97.5),4),p_two_sided=round(min(p,1.0),4)))
    print(rows[-1])
with open("data/results/tables/directional_did.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("saved")
