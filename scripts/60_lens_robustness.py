"""
Step 60: Two robustness checks for the logit-lens exit result, one pass.

(A) CLASSIFICATION ROBUSTNESS (reviewer Major 1). The original classifier
sends mixed-script tokens and letterless byte fragments to "other"; since
KY fertility (3.8-4.0) exceeds RU's (2.1-2.7), Cyrillic may be undercounted
more for KY, pushing d_.5 later for KY -- the direction of our conclusion.
Variants, applied to the SAME top-20 decodes:
  A  original: word-like only; mixed -> other
  B  any-Cyrillic: token containing any Cyrillic char -> cyr (mixed -> cyr);
     else any Latin -> en; else other
  C  upper bound: as B, plus replacement-char/letterless fragments -> cyr
     (attributes every undecodable byte fragment to the input language)
(B) LAST-TOKEN LENS (reviewer Major 2). Same projection applied to the
stored last-token states (data/processed/*_lasttok), classifier A and B.

Output: data/results/logit_lens/lens_robustness.csv (per model x lang x
variant x pooling: d_.5 sustained, AUC[.5,1]).
"""
import json, re
from pathlib import Path
import numpy as np, torch, csv
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

PROC=Path("data/processed")
MODELS={"llama":("meta-llama/Llama-3.1-8B",{"lm":"model-00004-of-00004.safetensors","norm":"model-00004-of-00004.safetensors"},"llama_lasttok"),
        "qwen3":("Qwen/Qwen3-8B",{"lm":"model-00005-of-00005.safetensors","norm":"model-00004-of-00005.safetensors"},"qwen3_lasttok"),
        "mistral":("mistralai/Mistral-7B-v0.3",{"lm":"model-00003-of-00003.safetensors","norm":"model-00003-of-00003.safetensors"},"mistral_lasttok")}
EPS=1e-5; TOPK=20; NS=300
LATIN=re.compile(r"[A-Za-z]"); CYR=re.compile(r"[Ѐ-ӿ]"); WORD=re.compile(r"[A-Za-zЀ-ӿ]")
def cls_all(sdec):
    c=sdec.strip()
    la,cy=bool(LATIN.search(c)),bool(CYR.search(c))
    wordish=bool(c) and bool(WORD.search(c))
    A = 1 if (wordish and la and not cy) else (2 if (wordish and cy and not la) else 0)
    B = 2 if cy else (1 if la else 0)
    frag = (not c) or ('�' in c) or not WORD.search(c)
    C = 2 if (cy or frag) else (1 if la else 0)
    return A,B,C
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
    raise RuntimeError("fetch failed")
dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
rows=[]
def dstats(curve,fr):
    cross=None
    for i,(f_,s) in enumerate(zip(fr,curve)):
        if f_>0.3 and s<0.5 and all(v<0.5 for v in curve[i:]): cross=f_; break
    seg=[(a,b) for a,b in zip(fr,curve) if a>=0.5]
    auc=sum((seg[i+1][0]-seg[i][0])*(seg[i][1]+seg[i+1][1])/2 for i in range(len(seg)-1))/(seg[-1][0]-seg[0][0])
    return cross,auc
for key,(repo,sh,ltdir) in MODELS.items():
    tok=AutoTokenizer.from_pretrained(repo)
    W=fetch(repo,sh["lm"],"lm_head.weight").to(dev); g=fetch(repo,sh["norm"],"model.norm.weight").to(dev)
    n_total=np.load(PROC/key/"labels.npy").shape[0]
    idx=np.random.default_rng(0).choice(n_total,size=NS,replace=False); idx.sort()
    for pooling,src in [("mean",PROC/key),("lasttok",PROC/ltdir)]:
        for lang in ["ru","ky"]:
            hs=np.load(src/f"hidden_states_{lang}.npz")["hidden_states"][idx]
            n,L,H=hs.shape
            fr=[l/(L-1) for l in range(L)]
            curves={v:[] for v in "ABC"}
            for layer in range(L):
                x=torch.from_numpy(hs[:,layer,:]).float().to(dev)
                x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g
                top=torch.topk(x@W.T,TOPK,dim=-1).indices.cpu().numpy()
                uniq=np.unique(top); m={int(i):cls_all(tok.decode([int(i)])) for i in uniq}
                for vi,v in enumerate("ABC"):
                    cl=np.vectorize(lambda i:m[int(i)][vi])(top)
                    en=(cl==1).sum(); cy=(cl==2).sum()
                    curves[v].append(en/max(en+cy,1))
            for v in "ABC":
                if pooling=="lasttok" and v=="C": continue
                d,auc=dstats(curves[v],fr)
                rows.append(dict(model=key,pooling=pooling,lang=lang,variant=v,d50=(round(d,3) if d else None),auc=round(auc,3)))
                print(rows[-1])
    del W,g
    if dev=="cuda": torch.cuda.empty_cache()
with open("data/results/logit_lens/lens_robustness.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("ALL DONE")
