"""
Step 51: Shuffled-representation null for the sentence-level logit lens.

Reviewer concern: projecting mean-pooled states through lm_head may decode to
English merely because the unembedding's high-logit region is English-biased
(frequency prior), not because representations are English-like. Null: take
the SAME pooled vectors (same 300 sentences, seed 0 as script 38), randomly
permute each vector's dimensions (destroys learned structure, preserves norm
and per-vector statistics), apply the model's final RMSNorm and lm_head, and
classify the top-20 tokens with the same classifier as script 38.

Weights are fetched with ranged reads from the HF hub (lm_head + norm only,
fp16/bf16), so no full checkpoint download and no GPU is needed.

Output: data/results/logit_lens/{key}_null.csv
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

PROCESSED=Path("data/processed"); OUT=Path("data/results/logit_lens")
MODELS={"llama":("meta-llama/Llama-3.1-8B",
                 {"lm":"model-00004-of-00004.safetensors","norm":"model-00004-of-00004.safetensors"}),
        "qwen3":("Qwen/Qwen3-8B",
                 {"lm":"model-00005-of-00005.safetensors","norm":"model-00004-of-00005.safetensors"}),
        "mistral":("mistralai/Mistral-7B-v0.3",
                 {"lm":"model-00003-of-00003.safetensors","norm":"model-00003-of-00003.safetensors"})}
EPS=1e-5; TOPK=20; NS=300
LATIN=re.compile(r"[A-Za-z]"); CYR=re.compile(r"[Ѐ-ӿ]"); WORD=re.compile(r"[A-Za-zЀ-ӿ]")
def classify(s):
    c=s.strip()
    if not c or not WORD.search(c): return "other"
    la,cy=bool(LATIN.search(c)),bool(CYR.search(c))
    if la and not cy: return "en"
    if cy and not la: return "cyr"
    return "other"

fs=HfFileSystem()
def fetch_tensor(repo, fname, tname):
    path=f"{repo}/{fname}"
    with fs.open(path,"rb") as f:
        hl=int.from_bytes(f.read(8),"little")
        hdr=json.loads(f.read(hl))
        meta=hdr[tname]
        o0,o1=meta["data_offsets"]; dt=meta["dtype"]; shape=meta["shape"]
        f.seek(8+hl+o0)
        buf=f.read(o1-o0)
    if dt=="BF16":
        t=torch.frombuffer(bytearray(buf),dtype=torch.bfloat16).reshape(shape).float()
    elif dt=="F16":
        t=torch.frombuffer(bytearray(buf),dtype=torch.float16).reshape(shape).float()
    else:
        raise ValueError(dt)
    print(f"  fetched {tname} {shape} {dt}")
    return t

torch.set_num_threads(16)
import sys
only=sys.argv[1] if len(sys.argv)>1 else None
for key,(repo,sh) in MODELS.items():
    if only and key!=only: continue
    print(f"== {key}")
    tok=AutoTokenizer.from_pretrained(repo)
    W=fetch_tensor(repo,sh["lm"],"lm_head.weight")          # (V,H)
    g=fetch_tensor(repo,sh["norm"],"model.norm.weight")     # (H,)
    pdir=PROCESSED/key
    n_total=np.load(pdir/"labels.npy").shape[0]
    idx=np.random.default_rng(0).choice(n_total,size=min(NS,n_total),replace=False); idx.sort()
    rng=np.random.default_rng(123)
    rows=[]
    for lang in ["ru","ky"]:
        hs=np.load(pdir/f"hidden_states_{lang}.npz")["hidden_states"][idx]  # (n,L,H)
        n,L,H=hs.shape
        print(f"  {lang}: {n}x{L}x{H}")
        for layer in range(L):
            X=hs[:,layer,:].copy()
            for a in range(n):                      # per-vector dim permutation
                X[a]=X[a][rng.permutation(H)]
            x=torch.from_numpy(X).float()
            x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g
            logits=x@W.T
            top=torch.topk(logits,TOPK,dim=-1).indices.numpy()
            uniq=np.unique(top)
            id2c={int(i):classify(tok.decode([int(i)])) for i in uniq}
            cls=np.vectorize(lambda i:id2c[int(i)])(top)
            rows.append(dict(model=key,lang=lang,layer=layer,
                layer_frac=round(layer/(L-1),4),
                p_en=round(float((cls=="en").mean()),4),
                p_cyr=round(float((cls=="cyr").mean()),4),
                p_other=round(float((cls=="other").mean()),4)))
        print(f"    done {lang}")
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/f"{key}_null.csv",index=False)
    print(f"  saved {OUT}/{key}_null.csv")
print("ALL DONE")
