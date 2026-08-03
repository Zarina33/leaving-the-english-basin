"""
Step 53: Lexical (TF-IDF) baseline under the leakage-free transfer protocol.

Reviewer concern: "a lexical probe cannot transfer" was asserted, not
measured -- and RU->KY/KY->RU share Cyrillic script plus Russian loanwords
in Kyrgyz, so for that pair it is a hypothesis. Here we run TF-IDF word
1-2 grams + logistic regression under the exact protocol of scripts/40:
same seed-42 60/20/20 sentence split, train on the source-language train
split, evaluate on the target-language test split (the vectorizer is fit
on the source language; target texts are transformed with that vocabulary).

Output: data/results/tables/tfidf_transfer_baseline.csv
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED=42
df=pd.read_csv("data/translated/parallel_corpus_clean.csv")
labels=np.load("data/processed/labels.npy")
n=len(df); assert n==len(labels)
rng=np.random.default_rng(SEED); idx=rng.permutation(n)
n_tr,n_dev=int(0.6*n),int(0.2*n)
TR,TE=idx[:n_tr],idx[n_tr+n_dev:]
col={'en':'text_en','ru':'text_ru','ky':'text_ky'}
rows=[]
for s in ['en','ru','ky']:
    for t in ['en','ru','ky']:
        if s==t: continue
        vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True)
        Xtr=vec.fit_transform(df[col[s]].iloc[TR])
        clf=LogisticRegression(max_iter=2000).fit(Xtr,labels[TR])
        Xte=vec.transform(df[col[t]].iloc[TE])
        acc=float((clf.predict(Xte)==labels[TE]).mean())
        nz=float((Xte.sum(axis=1)!=0).mean())  # share of target test sents with ANY source vocab hit
        rows.append(dict(src=s,tgt=t,acc=round(acc,4),nonzero_share=round(nz,3)))
        print(rows[-1])
pd.DataFrame(rows).to_csv("data/results/tables/tfidf_transfer_baseline.csv",index=False)
print("chance = 1/6 =",round(1/6,4))
