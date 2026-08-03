"""
Step 58: Bootstrap for the one *manner* number quoted in 6.4: Mistral's
RU-vs-KY drop of the word-like EN share over the .56-.84 depth window
(.77 vs .22). AUC tests level/timing, not abruptness (and its sign flips
for Llama), so the windowed-drop contrast is tested separately, per
sentence, paired bootstrap. Same lens recomputation as scripts/57.

Output: data/results/logit_lens/mistral_window_drop.csv
"""
import json, re
from pathlib import Path
import numpy as np, torch, csv
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer
PROC=Path("data/processed")
repo="mistralai/Mistral-7B-v0.3"; sh={"lm":"model-00003-of-00003.safetensors","norm":"model-00003-of-00003.safetensors"}
EPS=1e-5; TOPK=20; NS=300; NB=10_000; SEED=42
LATIN=re.compile(r"[A-Za-z]"); CYR=re.compile(r"[Ѐ-ӿ]"); WORD=re.compile(r"[A-Za-zЀ-ӿ]")
def classify(s):
    c=s.strip()
    if not c or not WORD.search(c): return 0
    la,cy=bool(LATIN.search(c)),bool(CYR.search(c))
    return 1 if (la and not cy) else (2 if (cy and not la) else 0)
fs=HfFileSystem()
def fetch(fname,tname):
    with fs.open(f"{repo}/{fname}","rb") as f:
        hl=int.from_bytes(f.read(8),"little"); hdr=json.loads(f.read(hl))
        m=hdr[tname]; o0,o1=m["data_offsets"]; f.seek(8+hl+o0); buf=f.read(o1-o0)
    dt={"BF16":torch.bfloat16,"F16":torch.float16}[m["dtype"]]
    return torch.frombuffer(bytearray(buf),dtype=dt).reshape(m["shape"]).float()
torch.set_num_threads(16)
tok=AutoTokenizer.from_pretrained(repo)
W=fetch(sh["lm"],"lm_head.weight"); g=fetch(sh["norm"],"model.norm.weight")
pdir=PROC/"mistral"
n_total=np.load(pdir/"labels.npy").shape[0]
idx=np.random.default_rng(0).choice(n_total,size=NS,replace=False); idx.sort()
drops={}
for lang in ["ru","ky"]:
    hs=np.load(pdir/f"hidden_states_{lang}.npz")["hidden_states"][idx]
    n,L,H=hs.shape
    fr=np.array([l/(L-1) for l in range(L)])
    la=int(np.argmin(np.abs(fr-0.56))); lb=int(np.argmin(np.abs(fr-0.84)))
    S={}
    for layer in (la,lb):
        x=torch.from_numpy(hs[:,layer,:]).float()
        x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g
        top=torch.topk(x@W.T,TOPK,dim=-1).indices.numpy()
        uniq=np.unique(top); id2c={int(i):classify(tok.decode([int(i)])) for i in uniq}
        cls=np.vectorize(lambda i:id2c[int(i)])(top)
        en=(cls==1).sum(1); cy=(cls==2).sum(1); tot=np.maximum(en+cy,1)
        S[layer]=np.where(en+cy>0,en/tot,1.0)
    drops[lang]=S[la]-S[lb]
    print(lang,"drop mean",round(drops[lang].mean(),4),f"(layers {la}->{lb}, fr {fr[la]:.2f}->{fr[lb]:.2f})")
rng=np.random.default_rng(SEED); boots=rng.integers(0,NS,size=(NB,NS))
dd=drops['ru'][boots].mean(1)-drops['ky'][boots].mean(1)
p=2*min((dd<=0).mean(),(dd>=0).mean())
row=dict(drop_ru=round(drops['ru'].mean(),4),drop_ky=round(drops['ky'].mean(),4),
         delta=round(drops['ru'].mean()-drops['ky'].mean(),4),
         ci_lo=round(np.percentile(dd,2.5),4),ci_hi=round(np.percentile(dd,97.5),4),p=round(min(p,1.0),5))
print(row)
with open("data/results/logit_lens/mistral_window_drop.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=row.keys()); w.writeheader(); w.writerows([row])
print("saved")
