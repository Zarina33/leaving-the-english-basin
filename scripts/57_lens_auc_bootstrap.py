"""
Step 57: Sentence-level bootstrap CIs for the logit-lens exit metrics.

The headline AUC[.5,1] contrast (e.g., Llama RU .90 vs KY .74) had no
uncertainty at n=300. Here we recompute the lens per sentence (same 300
sentences, seed 0; lm_head + final norm fetched as in scripts/51), giving
a per-sentence word-like EN share per layer; AUC of the mean curve equals
the mean of per-sentence AUCs (trapezoid is linear), so we paired-bootstrap
sentences for AUC_RU, AUC_KY, their difference, and d_.5 stability.

Output: data/results/logit_lens/auc_bootstrap.csv
"""
import json, re
from pathlib import Path
import numpy as np, torch, csv
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

PROC=Path("data/processed"); OUT=Path("data/results/logit_lens")
MODELS={"llama":("meta-llama/Llama-3.1-8B",{"lm":"model-00004-of-00004.safetensors","norm":"model-00004-of-00004.safetensors"}),
        "qwen3":("Qwen/Qwen3-8B",{"lm":"model-00005-of-00005.safetensors","norm":"model-00004-of-00005.safetensors"}),
        "mistral":("mistralai/Mistral-7B-v0.3",{"lm":"model-00003-of-00003.safetensors","norm":"model-00003-of-00003.safetensors"})}
EPS=1e-5; TOPK=20; NS=300; NB=10_000; SEED=42
LATIN=re.compile(r"[A-Za-z]"); CYR=re.compile(r"[Ѐ-ӿ]"); WORD=re.compile(r"[A-Za-zЀ-ӿ]")
def classify(s):
    c=s.strip()
    if not c or not WORD.search(c): return 0
    la,cy=bool(LATIN.search(c)),bool(CYR.search(c))
    return 1 if (la and not cy) else (2 if (cy and not la) else 0)
fs=HfFileSystem()
def fetch(repo,fname,tname):
    with fs.open(f"{repo}/{fname}","rb") as f:
        hl=int.from_bytes(f.read(8),"little"); hdr=json.loads(f.read(hl))
        m=hdr[tname]; o0,o1=m["data_offsets"]
        f.seek(8+hl+o0); buf=f.read(o1-o0)
    dt={"BF16":torch.bfloat16,"F16":torch.float16}[m["dtype"]]
    return torch.frombuffer(bytearray(buf),dtype=dt).reshape(m["shape"]).float()
torch.set_num_threads(16)
rows=[]
for key,(repo,sh) in MODELS.items():
    tok=AutoTokenizer.from_pretrained(repo)
    W=fetch(repo,sh["lm"],"lm_head.weight"); g=fetch(repo,sh["norm"],"model.norm.weight")
    pdir=PROC/key
    n_total=np.load(pdir/"labels.npy").shape[0]
    idx=np.random.default_rng(0).choice(n_total,size=NS,replace=False); idx.sort()
    share={}
    for lang in ["ru","ky"]:
        hs=np.load(pdir/f"hidden_states_{lang}.npz")["hidden_states"][idx]
        n,L,H=hs.shape
        S=np.zeros((n,L))
        for layer in range(L):
            x=torch.from_numpy(hs[:,layer,:]).float()
            x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g
            top=torch.topk(x@W.T,TOPK,dim=-1).indices.numpy()
            uniq=np.unique(top); id2c={int(i):classify(tok.decode([int(i)])) for i in uniq}
            cls=np.vectorize(lambda i:id2c[int(i)])(top)
            en=(cls==1).sum(1); cy=(cls==2).sum(1)
            tot=np.maximum(en+cy,1)
            S[:,layer]=np.where(en+cy>0, en/tot, 1.0)
        share[lang]=S
        print(key,lang,"done")
    fr=np.array([l/(share['ru'].shape[1]-1) for l in range(share['ru'].shape[1])])
    seg=fr>=0.5
    def auc_rows(S):
        x=fr[seg]; Y=S[:,seg]
        return np.trapz(Y,x,axis=1)/(x[-1]-x[0])
    aR,aK=auc_rows(share['ru']),auc_rows(share['ky'])
    rng=np.random.default_rng(SEED); boots=rng.integers(0,NS,size=(NB,NS))
    dR=aR[boots].mean(1); dK=aK[boots].mean(1); dd=dK-dR
    p=2*min((dd<=0).mean(),(dd>=0).mean())
    rows.append(dict(model=key,auc_ru=round(aR.mean(),3),ru_lo=round(np.percentile(dR,2.5),3),ru_hi=round(np.percentile(dR,97.5),3),
        auc_ky=round(aK.mean(),3),ky_lo=round(np.percentile(dK,2.5),3),ky_hi=round(np.percentile(dK,97.5),3),
        d_ky_minus_ru=round((aK-aR).mean(),3),d_lo=round(np.percentile(dd,2.5),3),d_hi=round(np.percentile(dd,97.5),3),p=round(min(p,1.0),5)))
    print(rows[-1])
with open(OUT/"auc_bootstrap.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("ALL DONE")
