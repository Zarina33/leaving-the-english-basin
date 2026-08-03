"""
Step 59: Pre-registered manner window for the lens, all three decoders.

Reviewer: the .56-.84 window (scripts/58) was chosen on the curve it
tests. Here the window is fixed a priori to the LAST THIRD of depth,
[.67, 1.0] -- the interval the paper itself names for Kyrgyz's gradual
drift -- and applied identically to all three models and both languages.
Statistic: per-sentence drop of the word-like EN share, share(.67) -
share(1.0); paired bootstrap of the RU-KY difference. We also record
share(.67) itself, since drop-to-zero over the last third is dominated
by the level at .67 (the final layer decodes the input language in all
models).

Output: data/results/logit_lens/lastthird_drop.csv
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
EPS=1e-5; TOPK=20; NS=300; NB=10_000; SEED=42
LATIN=re.compile(r"[A-Za-z]"); CYR=re.compile(r"[Ѐ-ӿ]"); WORD=re.compile(r"[A-Za-zЀ-ӿ]")
def classify(s):
    c=s.strip()
    if not c or not WORD.search(c): return 0
    la,cy=bool(LATIN.search(c)),bool(CYR.search(c))
    return 1 if (la and not cy) else (2 if (cy and not la) else 0)
fs=HfFileSystem()
def fetch(repo,fname,tname):
    for attempt in range(6):
        try:
            with fs.open(f"{repo}/{fname}","rb") as f:
                hl=int.from_bytes(f.read(8),"little"); hdr=json.loads(f.read(hl))
                m=hdr[tname]; o0,o1=m["data_offsets"]; f.seek(8+hl+o0); buf=f.read(o1-o0)
            dt={"BF16":torch.bfloat16,"F16":torch.float16}[m["dtype"]]
            return torch.frombuffer(bytearray(buf),dtype=dt).reshape(m["shape"]).float()
        except Exception as e:
            print("retry",attempt,type(e).__name__)
    raise RuntimeError("fetch failed")
torch.set_num_threads(16)
rows=[]
for key,(repo,sh) in MODELS.items():
    tok=AutoTokenizer.from_pretrained(repo)
    W=fetch(repo,sh["lm"],"lm_head.weight"); g=fetch(repo,sh["norm"],"model.norm.weight")
    pdir=PROC/key
    n_total=np.load(pdir/"labels.npy").shape[0]
    idx=np.random.default_rng(0).choice(n_total,size=NS,replace=False); idx.sort()
    def shares(lang,layers):
        hs=np.load(pdir/f"hidden_states_{lang}.npz")["hidden_states"][idx]
        out={}
        for layer in layers:
            x=torch.from_numpy(hs[:,layer,:]).float()
            x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g
            top=torch.topk(x@W.T,TOPK,dim=-1).indices.numpy()
            uniq=np.unique(top); id2c={int(i):classify(tok.decode([int(i)])) for i in uniq}
            cls=np.vectorize(lambda i:id2c[int(i)])(top)
            en=(cls==1).sum(1); cy=(cls==2).sum(1); tot=np.maximum(en+cy,1)
            out[layer]=np.where(en+cy>0,en/tot,1.0)
        return out
    L=np.load(pdir/"hidden_states_ru.npz")["hidden_states"].shape[1]
    fr=np.array([l/(L-1) for l in range(L)])
    la=int(np.argmin(np.abs(fr-0.67))); lb=L-1
    d={}
    for lang in ["ru","ky"]:
        S=shares(lang,(la,lb))
        d[lang]=(S[la]-S[lb], S[la])
        print(key,lang,f"share(.67)={S[la].mean():.3f} share(1.0)={S[lb].mean():.3f} drop={d[lang][0].mean():.3f}")
    rng=np.random.default_rng(SEED); boots=rng.integers(0,NS,size=(NB,NS))
    dd=d['ru'][0][boots].mean(1)-d['ky'][0][boots].mean(1)
    p=2*min((dd<=0).mean(),(dd>=0).mean())
    rows.append(dict(model=key,fr_a=round(fr[la],3),
        drop_ru=round(d['ru'][0].mean(),4),drop_ky=round(d['ky'][0].mean(),4),
        share67_ru=round(d['ru'][1].mean(),4),share67_ky=round(d['ky'][1].mean(),4),
        delta_ru_minus_ky=round(d['ru'][0].mean()-d['ky'][0].mean(),4),
        ci_lo=round(np.percentile(dd,2.5),4),ci_hi=round(np.percentile(dd,97.5),4),p=round(min(p,1.0),5)))
    print(rows[-1])
with open("data/results/logit_lens/lastthird_drop.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("ALL DONE")
