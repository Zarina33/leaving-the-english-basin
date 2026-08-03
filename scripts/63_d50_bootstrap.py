"""
Step 63: Sentence-level bootstrap for d_.5 itself (the threshold statistic
carrying the headline ordering). Recomputes per-sentence word-like EN
shares (as scripts/57), then for each bootstrap resample of the 300
sentences rebuilds the MEAN curve and recomputes sustained d_.5 for RU
and KY; reports percentile CIs for d_.5 and for the KY-RU difference,
plus P(KY >= RU) across resamples.

Output: data/results/logit_lens/d50_bootstrap.csv
"""
import json, re
from pathlib import Path
import numpy as np, torch, csv
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer
PROC=Path("data/processed")
MODELS={"llama":("meta-llama/Llama-3.1-8B",{"lm":"model-00004-of-00004.safetensors","norm":"model-00004-of-00004.safetensors"}),
        "qwen3":("Qwen/Qwen3-8B",{"lm":"model-00005-of-00005.safetensors","norm":"model-00004-of-00005.safetensors"}),
        "mistral":("mistralai/Mistral-7B-v0.3",{"lm":"model-00003-of-00003.safetensors","norm":"model-00003-of-00003.safetensors"})}
EPS=1e-5; TOPK=20; NS=300; NB=2000; SEED=42
LATIN=re.compile(r"[A-Za-z]"); CYR=re.compile(r"[Ѐ-ӿ]"); WORD=re.compile(r"[A-Za-zЀ-ӿ]")
def classify(sd):
    c=sd.strip()
    if not c or not WORD.search(c): return 0
    la,cy=bool(LATIN.search(c)),bool(CYR.search(c))
    return 1 if (la and not cy) else (2 if (cy and not la) else 0)
fs=HfFileSystem()
def fetch(repo,fname,tname):
    for att in range(8):
        try:
            with fs.open(f"{repo}/{fname}","rb") as f:
                hl=int.from_bytes(f.read(8),"little"); hdr=json.loads(f.read(hl))
                m=hdr[tname]; o0,o1=m["data_offsets"]; f.seek(8+hl+o0); buf=f.read(o1-o0)
            dt={"BF16":torch.bfloat16,"F16":torch.float16}[m["dtype"]]
            return torch.frombuffer(bytearray(buf),dtype=dt).reshape(m["shape"]).float()
        except Exception as e: print("retry",att,type(e).__name__)
    raise RuntimeError()
dev="cuda" if torch.cuda.is_available() else "cpu"
def d50_from_mean(mean_curve,fr):
    for i,(f_,v) in enumerate(zip(fr,mean_curve)):
        if f_>0.3 and v<0.5 and all(x<0.5 for x in mean_curve[i:]): return f_
    return 1.0
rows=[]
for key,(repo,sh) in MODELS.items():
    tok=AutoTokenizer.from_pretrained(repo)
    W=fetch(repo,sh["lm"],"lm_head.weight").to(dev); g=fetch(repo,sh["norm"],"model.norm.weight").to(dev)
    n_total=np.load(PROC/key/"labels.npy").shape[0]
    idx=np.random.default_rng(0).choice(n_total,size=NS,replace=False); idx.sort()
    S={}
    for lang in ["ru","ky"]:
        hs=np.load(PROC/key/f"hidden_states_{lang}.npz")["hidden_states"][idx]
        n,L,H=hs.shape
        M=np.zeros((n,L))
        for layer in range(L):
            x=torch.from_numpy(hs[:,layer,:]).float().to(dev)
            x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g
            top=torch.topk(x@W.T,TOPK,dim=-1).indices.cpu().numpy()
            uniq=np.unique(top); mp={int(i):classify(tok.decode([int(i)])) for i in uniq}
            cl=np.vectorize(lambda i:mp[int(i)])(top)
            en=(cl==1).sum(1); cy=(cl==2).sum(1); tot=np.maximum(en+cy,1)
            M[:,layer]=np.where(en+cy>0,en/tot,1.0)
        S[lang]=M
    L=S['ru'].shape[1]; fr=[l/(L-1) for l in range(L)]
    rng=np.random.default_rng(SEED); boots=rng.integers(0,NS,size=(NB,NS))
    dR=np.array([d50_from_mean(S['ru'][b].mean(0),fr) for b in boots])
    dK=np.array([d50_from_mean(S['ky'][b].mean(0),fr) for b in boots])
    diff=dK-dR
    rows.append(dict(model=key,
        d50_ru=round(d50_from_mean(S['ru'].mean(0),fr),3),ru_lo=round(np.percentile(dR,2.5),3),ru_hi=round(np.percentile(dR,97.5),3),
        d50_ky=round(d50_from_mean(S['ky'].mean(0),fr),3),ky_lo=round(np.percentile(dK,2.5),3),ky_hi=round(np.percentile(dK,97.5),3),
        diff_lo=round(np.percentile(diff,2.5),3),diff_hi=round(np.percentile(diff,97.5),3),
        p_ky_ge_ru=round(float((diff>=0).mean()),4)))
    print(rows[-1])
    del W,g
    if dev=="cuda": torch.cuda.empty_cache()
with open("data/results/logit_lens/d50_bootstrap.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("ALL DONE")
