"""
Step 62: XLM-R-base (270M, 12 layers) -- the model-size deconfound for the
XLM-R-vs-mBERT scale contrast. Same corpus and objective family, same data
as XLM-R-large, ~2x mBERT's parameters, 0.5x large's layers. Pipeline:
fp16 extraction (mean-pooled, all 13 layers incl. embeddings), per-layer
5-fold probing, and the exact leakage-free transfer protocol of scripts/40.

Output: data/processed/xlmr_base/hidden_states_{en,ru,ky}.npz
        data/results/xlmr_base/probing_detailed.csv
        rows appended to data/results/tables/xlmr_base_transfer.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_ID="FacebookAI/xlm-roberta-base"
OUT=Path("data/processed/xlmr_base"); OUT.mkdir(parents=True,exist_ok=True)
RES=Path("data/results/xlmr_base"); RES.mkdir(parents=True,exist_ok=True)
BATCH=32; MAXLEN=128; SEED=42; N_EPOCHS=300; LR=1e-2
dev="cuda" if torch.cuda.is_available() else "cpu"
df=pd.read_csv("data/translated/parallel_corpus_clean.csv")
labels=np.load("data/processed/labels.npy")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModel.from_pretrained(MODEL_ID,dtype=torch.float16).to(dev).eval()
for lang in ["en","ru","ky"]:
    out=OUT/f"hidden_states_{lang}.npz"
    if out.exists(): print("skip",lang); continue
    texts=df[f"text_{lang}"].tolist(); allst=[]
    for i in tqdm(range(0,len(texts),BATCH),desc=lang):
        inp=tok(texts[i:i+BATCH],return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(dev)
        with torch.no_grad():
            o=model(**inp,output_hidden_states=True)
        mask=inp["attention_mask"].float().unsqueeze(-1)
        bs=[((hs.float()*mask).sum(1)/mask.sum(1).clamp(min=1e-9)).cpu().numpy() for hs in o.hidden_states]
        allst.append(np.stack(bs,axis=1))
    np.savez_compressed(out,hidden_states=np.concatenate(allst,axis=0))
    print(lang,"saved")
del model; torch.cuda.empty_cache()

def fit_eval(Xtr,ytr,Xte,yte):
    sc=StandardScaler()
    a=torch.tensor(sc.fit_transform(Xtr),dtype=torch.float32,device=dev)
    b=torch.tensor(sc.transform(Xte),dtype=torch.float32,device=dev)
    ya=torch.tensor(ytr,dtype=torch.long,device=dev); yb=torch.tensor(yte,dtype=torch.long,device=dev)
    torch.manual_seed(SEED)
    m=nn.Linear(a.shape[1],6).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
    for _ in range(N_EPOCHS): opt.zero_grad(); ce(m(a),ya).backward(); opt.step()
    with torch.no_grad(): pred=m(b).argmax(1)
    return (pred==yb).float().mean().item()

hs={lg: np.load(OUT/f"hidden_states_{lg}.npz")["hidden_states"] for lg in ["en","ru","ky"]}
n,L,H=hs["en"].shape; print("shape",n,L,H)
rows=[]
for lang in ["en","ru","ky"]:
    for layer in range(L):
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
        accs=[fit_eval(hs[lang][tr][:,layer,:],labels[tr],hs[lang][te][:,layer,:],labels[te]) for tr,te in skf.split(hs[lang][:,layer,:],labels)]
        rows.append(dict(lang=lang,layer=layer,acc_mean=float(np.mean(accs)),acc_std=float(np.std(accs))))
    best=max([r for r in rows if r['lang']==lang],key=lambda r:r['acc_mean'])
    print(f"probing {lang}: best L{best['layer']} {best['acc_mean']:.4f}")
pd.DataFrame(rows).to_csv(RES/"probing_detailed.csv",index=False)

rng=np.random.default_rng(SEED); idx=rng.permutation(n)
n_tr,n_dev=int(0.6*n),int(0.2*n)
TR,DEV,TE=idx[:n_tr],idx[n_tr:n_tr+n_dev],idx[n_tr+n_dev:]
trows=[]; cv={}
for src,tgt in [("en","ru"),("en","ky"),("ru","en"),("ru","ky"),("ky","en"),("ky","ru")]:
    devaccs=[fit_eval(hs[src][TR][:,l,:],labels[TR],hs[tgt][DEV][:,l,:],labels[DEV]) for l in range(L)]
    Ls=int(np.argmax(devaccs))
    sc=StandardScaler()
    a=torch.tensor(sc.fit_transform(hs[src][TR][:,Ls,:]),dtype=torch.float32,device=dev)
    b=torch.tensor(sc.transform(hs[tgt][TE][:,Ls,:]),dtype=torch.float32,device=dev)
    ya=torch.tensor(labels[TR],dtype=torch.long,device=dev)
    torch.manual_seed(SEED)
    m=nn.Linear(a.shape[1],6).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
    for _ in range(N_EPOCHS): opt.zero_grad(); ce(m(a),ya).backward(); opt.step()
    with torch.no_grad(): pred=m(b).argmax(1).cpu().numpy()
    corr=(pred==labels[TE]).astype(int); cv[(src,tgt)]=corr
    trows.append(dict(model="xlmr_base",src=src,tgt=tgt,layer_star=Ls,dev_acc=round(devaccs[Ls],4),test_acc=round(corr.mean(),4)))
    print(trows[-1])
pd.DataFrame(trows).to_csv("data/results/tables/xlmr_base_transfer.csv",index=False)
np.savez_compressed("data/results/transfer_correct/xlmr_base.npz",**{f"{s}_{t}":v for (s,t),v in cv.items()})
ky=np.stack([cv[k] for k in [("en","ky"),("ru","ky"),("ky","en"),("ky","ru")]],1).mean(1)
enru=np.stack([cv[("en","ru")],cv[("ru","en")]],1).mean(1)
print("KYavg",round(ky.mean(),4),"| ENRU",round(enru.mean(),4),"| eta_cc",round((ky.mean()-1/6)/(enru.mean()-1/6),4))
print("DONE")
