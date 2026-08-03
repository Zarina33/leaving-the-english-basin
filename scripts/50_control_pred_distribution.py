"""
Step 50: Prediction-distribution audit for the 102-sentence control set.

Reviewer concern: acc_claude == acc_human exactly (delta=0) in 4/5 RU cells
could indicate a degenerate probe (collapse onto few classes). For each
(model, lang) cell we retrain the identical probe (script 30 protocol) and
report: accuracy on both versions, per-class prediction counts, prediction
entropy, and the number of sentences (of 102) where the C- and H-version
predictions AGREE (identical labels would explain delta=0 without collapse).

Output: data/results/tables/control_pred_distribution.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42); np.random.seed(42)
DATA=Path("data/processed"); OUT=Path("data/results/tables/control_pred_distribution.csv")
best=pd.read_csv("data/results/tables/unified_probing.csv")
BL={ (r.model, r.lang.lower()): int(r.layer) for r in best.itertuples() }
META=[("Gemma 4","","Gemma 4 E4B"),("Qwen3","qwen3","Qwen3-8B"),("Llama 3.1","llama","Llama-3.1-8B"),
      ("Mistral","mistral","Mistral-7B"),("XLM-R","xlmr","XLM-R")]
labels=np.load(DATA/"labels.npy")
cids=np.load(DATA/"control_ids.npy",allow_pickle=True)
full=pd.read_csv("data/translated/parallel_corpus_clean.csv")
idmap={rid:i for i,rid in enumerate(full["id"].tolist())}
ctrl=np.array([idmap[c] for c in cids])
mask=np.zeros(len(labels),bool); mask[ctrl]=True
rows=[]
for disp,d,uni in META:
    base=DATA/d if d else DATA
    for lang in ["ru","ky"]:
        L=BL[(uni,lang)]
        hs=np.load(base/f"hidden_states_{lang}.npz")["hidden_states"][:,L,:]
        hum=np.load(base/f"hidden_states_human_{lang}.npz")["hidden_states"]
        hum=hum[:,L,:] if hum.ndim==3 else hum
        sc=StandardScaler().fit(hs[~mask])
        Xtr=torch.tensor(sc.transform(hs[~mask]),dtype=torch.float32)
        ytr=torch.tensor(labels[~mask],dtype=torch.long)
        Xc=torch.tensor(sc.transform(hs[ctrl]),dtype=torch.float32)
        Xh=torch.tensor(sc.transform(hum),dtype=torch.float32)
        yte=labels[ctrl]
        m=nn.Linear(Xtr.shape[1],6); opt=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=1e-4)
        ce=nn.CrossEntropyLoss()
        for _ in range(300):
            opt.zero_grad(); ce(m(Xtr),ytr).backward(); opt.step()
        with torch.no_grad():
            pc=m(Xc).argmax(1).numpy(); ph=m(Xh).argmax(1).numpy()
        cc=np.bincount(pc,minlength=6); ch=np.bincount(ph,minlength=6)
        pe=lambda c: float(-(c[c>0]/c.sum()*np.log(c[c>0]/c.sum())).sum())
        rows.append(dict(model=disp,lang=lang.upper(),layer=L,
            acc_c=round(float((pc==yte).mean()),4),acc_h=round(float((ph==yte).mean()),4),
            agree=int((pc==ph).sum()),n_classes_c=int((cc>0).sum()),n_classes_h=int((ch>0).sum()),
            counts_c="/".join(map(str,cc)),counts_h="/".join(map(str,ch)),
            entropy_c=round(pe(cc),3),entropy_h=round(pe(ch),3)))
        print(rows[-1])
pd.DataFrame(rows).to_csv(OUT,index=False); print("saved",OUT)
