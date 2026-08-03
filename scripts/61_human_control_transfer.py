"""
Step 61: Does the LLM-translation confound inflate TRANSFER (not just
probing)? Train the EN->target transfer probe exactly as in scripts/40
(EN train split, transfer layer L* from transfer_noleak), but evaluate on
the 102 control sentences' target-language states in two versions:
Claude-translated vs human-translated. If Claude translations are easier
to transfer into (shared latent generator with the EN source), acc_C
should exceed acc_H. Paired bootstrap over the 102.

Output: data/results/tables/human_control_transfer.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, csv
from sklearn.preprocessing import StandardScaler
SEED=42; NB=10_000
DATA=Path("data/processed")
dev="cuda" if torch.cuda.is_available() else "cpu"
labels=np.load(DATA/"labels.npy")
cids=np.load(DATA/"control_ids.npy",allow_pickle=True)
full=pd.read_csv("data/translated/parallel_corpus_clean.csv")
idmap={rid:i for i,rid in enumerate(full["id"].tolist())}
ctrl=np.array([idmap[c] for c in cids])
n=len(labels); rng=np.random.default_rng(SEED); idx=rng.permutation(n)
TR=idx[:int(0.6*n)]
TR=np.array([i for i in TR if i not in set(ctrl)])
TL={(r['model'],r['src'],r['tgt']): int(r['layer_star']) for r in csv.DictReader(open("data/results/transfer_noleak.csv"))}
META=[("Gemma 4","","gemma4"),("Qwen3","qwen3","qwen3"),("Llama 3.1","llama","llama"),("Mistral","mistral","mistral"),("XLM-R","xlmr","xlmr")]
def fit(Xtr,ytr):
    sc=StandardScaler(); a=torch.tensor(sc.fit_transform(Xtr),dtype=torch.float32,device=dev)
    ya=torch.tensor(ytr,dtype=torch.long,device=dev)
    torch.manual_seed(SEED)
    m=nn.Linear(a.shape[1],6).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=1e-4)
    ce=nn.CrossEntropyLoss(); m.train()
    for _ in range(300): opt.zero_grad(); ce(m(a),ya).backward(); opt.step()
    m.eval(); return m,sc
rows=[]
for disp,d,key in META:
    base=DATA/d if d else DATA
    hs_en=np.load(base/"hidden_states_en.npz")["hidden_states"]
    for tgt in ["ru","ky"]:
        L=TL[(key,"en",tgt)]
        m,sc=fit(hs_en[TR][:,L,:],labels[TR])
        hsC=np.load(base/f"hidden_states_{tgt}.npz")["hidden_states"][ctrl][:,L,:]
        hH=np.load(base/f"hidden_states_human_{tgt}.npz")["hidden_states"]
        hsH=(hH[:,L,:] if hH.ndim==3 else hH)
        with torch.no_grad():
            pC=m(torch.tensor(sc.transform(hsC),dtype=torch.float32,device=dev)).argmax(1).cpu().numpy()
            pH=m(torch.tensor(sc.transform(hsH),dtype=torch.float32,device=dev)).argmax(1).cpu().numpy()
        y=labels[ctrl]; cC=(pC==y).astype(int); cH=(pH==y).astype(int)
        b=np.random.default_rng(SEED).integers(0,len(y),size=(NB,len(y)))
        dd=cC[b].mean(1)-cH[b].mean(1)
        p=2*min((dd<=0).mean(),(dd>=0).mean())
        rows.append(dict(model=disp,direction=f"en->{tgt}",layer=L,acc_claude=round(cC.mean(),4),acc_human=round(cH.mean(),4),
            delta=round(cC.mean()-cH.mean(),4),ci_lo=round(np.percentile(dd,2.5),4),ci_hi=round(np.percentile(dd,97.5),4),p=round(min(p,1.0),4)))
        print(rows[-1])
pd.DataFrame(rows).to_csv("data/results/tables/human_control_transfer.csv",index=False)
print("DONE")
