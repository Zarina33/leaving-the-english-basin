"""
Step 55: Leakage-free transfer for Qwen3-8B fp16 states (scripts/54) --
the precision control promised in Limitations. Exact replica of the
scripts/40 protocol (seed-42 60/20/20 split, nested dev layer selection,
identical linear probe), run on data/processed/qwen3_fp16.

Also runs the paired bootstrap NF4-vs-fp16 on KY_avg (same 296 test
sentences) and recomputes the XLM-R - Qwen3(fp16) gap.

Output: data/results/tables/qwen3_fp16_transfer.csv
        data/results/transfer_correct/qwen3_fp16.npz
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

N_EPOCHS=300; LR=1e-2; SEED=42
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
D=Path("data/processed/qwen3_fp16")
labels=np.load("data/processed/labels.npy")
hs={lg: np.load(D/f"hidden_states_{lg}.npz")["hidden_states"] for lg in ["en","ru","ky"]}
n,n_layers,_=hs["en"].shape
rng=np.random.default_rng(SEED); idx=rng.permutation(n)
n_tr,n_dev=int(0.6*n),int(0.2*n)
TR,DEV,TE=idx[:n_tr],idx[n_tr:n_tr+n_dev],idx[n_tr+n_dev:]

def fit_eval(Xtr,ytr,Xte,yte,ret=False):
    sc=StandardScaler()
    a=torch.tensor(sc.fit_transform(Xtr),dtype=torch.float32,device=DEVICE)
    b=torch.tensor(sc.transform(Xte),dtype=torch.float32,device=DEVICE)
    ya=torch.tensor(ytr,dtype=torch.long,device=DEVICE)
    yb=torch.tensor(yte,dtype=torch.long,device=DEVICE)
    torch.manual_seed(SEED)
    m=nn.Linear(a.shape[1],6).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=1e-4)
    ce=nn.CrossEntropyLoss(); m.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad(); ce(m(a),ya).backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred=m(b).argmax(1)
    corr=(pred==yb).cpu().numpy().astype(int)
    return (corr.mean(),corr) if ret else corr.mean()

rows=[]; cv={}
for src,tgt in [("en","ru"),("en","ky"),("ru","en"),("ru","ky"),("ky","en"),("ky","ru")]:
    dev=[fit_eval(hs[src][TR][:,L,:],labels[TR],hs[tgt][DEV][:,L,:],labels[DEV]) for L in range(n_layers)]
    Ls=int(np.argmax(dev))
    acc,corr=fit_eval(hs[src][TR][:,Ls,:],labels[TR],hs[tgt][TE][:,Ls,:],labels[TE],ret=True)
    cv[(src,tgt)]=corr
    rows.append(dict(model="qwen3_fp16",src=src,tgt=tgt,layer_star=Ls,dev_acc=round(dev[Ls],4),test_acc=round(acc,4)))
    print(rows[-1])
pd.DataFrame(rows).to_csv("data/results/tables/qwen3_fp16_transfer.csv",index=False)
np.savez_compressed("data/results/transfer_correct/qwen3_fp16.npz",**{f"{s}_{t}":v for (s,t),v in cv.items()})

# paired bootstraps
KY=["en_ky","ru_ky","ky_en","ky_ru"]
fp=np.stack([cv[tuple(k.split('_'))] for k in KY],1).mean(1)
nf=np.load("data/results/transfer_correct/qwen3.npz"); nf=np.stack([nf[k] for k in KY],1).mean(1)
xl=np.load("data/results/transfer_correct/xlmr.npz"); xl=np.stack([xl[k] for k in KY],1).mean(1)
bo=np.random.default_rng(SEED).integers(0,len(fp),size=(10000,len(fp)))
for name,a,b in [("fp16-NF4 (Qwen3 KYavg)",fp,nf),("XLM-R - Qwen3fp16 (KYavg)",xl,fp)]:
    d=a[bo].mean(1)-b[bo].mean(1); pt=a.mean()-b.mean()
    p=2*min((d<=0).mean(),(d>=0).mean())
    print(f"{name}: delta={pt:+.4f} CI[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] p={min(p,1.0):.4f}")
print("KYavg fp16 =",round(fp.mean(),4),"| NF4 =",round(nf.mean(),4),"| XLM-R =",round(xl.mean(),4))
