"""
Step 54: Extract Qwen3-8B hidden states in fp16 (vs the NF4 states of
scripts/11) for the KY-transfer precision control promised in Limitations.

The 16 GB RTX 5080 cannot hold fp16 8B weights entirely, so we use
device_map="auto" with a GPU memory cap and CPU offload for the remainder.
Identical pooling to scripts/11: mean over non-padding tokens, all layers
(embedding + 36), max_length 128, same corpus order.

Output: data/processed/qwen3_fp16/hidden_states_{en,ru,ky}.npz
"""
import os
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID="Qwen/Qwen3-8B"
OUT=Path("data/processed/qwen3_fp16"); OUT.mkdir(parents=True,exist_ok=True)
BATCH=2; MAXLEN=128
df=pd.read_csv("data/translated/parallel_corpus_clean.csv")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
print("Loading fp16 with GPU cap 9GiB + CPU offload...")
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto",
    max_memory={0:"8GiB","cpu":"48GiB"}, low_cpu_mem_usage=True).eval()
print(model.hf_device_map if hasattr(model,'hf_device_map') else 'no map')
for lang in ["en","ru","ky"]:
    out=OUT/f"hidden_states_{lang}.npz"
    if out.exists(): print("skip",lang); continue
    texts=df[f"text_{lang}"].tolist()
    all_states=[]
    for i in tqdm(range(0,len(texts),BATCH),desc=lang):
        batch=texts[i:i+BATCH]
        inp=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN)
        inp={k:v.to("cuda:0") for k,v in inp.items()}
        with torch.no_grad():
            # call the base transformer directly: skips lm_head entirely
            o=model.model(**inp,output_hidden_states=True)
        mask=inp["attention_mask"].float().unsqueeze(-1)
        bs=[]
        for hs in o.hidden_states:
            hs=hs.to("cuda:0")
            pooled=(hs.float()*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            bs.append(pooled.cpu().numpy())
        all_states.append(np.stack(bs,axis=1))
    states=np.concatenate(all_states,axis=0)
    print(lang,states.shape)
    np.savez_compressed(out,hidden_states=states)
print("DONE")
