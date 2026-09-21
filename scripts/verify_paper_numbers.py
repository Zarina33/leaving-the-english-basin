"""
Verify that every numerical claim in paper/paper.tex matches the data
in data/results/. Reports MISMATCH if any cell is off by more than the
rounding threshold.

Usage: python3 scripts/verify_paper_numbers.py
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data/results"

def ok(a, b, tol=3e-3):
    return abs(a - b) < tol

fail = 0

# --- Table 2: probing + CI (bootstrap_probing.csv) ---
# Table 3 probing now comes from the UNIFIED run (scripts/43). Paper values
# below are transcribed from paper.tex Table 3; they must match unified_probing.csv.
up = pd.read_csv(RES / "tables/unified_probing.csv")
paper_t2 = {
    ('Gemma 4 E4B','EN'):(0,.603), ('Gemma 4 E4B','RU'):(7,.508), ('Gemma 4 E4B','KY'):(4,.453),
    ('Qwen3-8B','EN'):(10,.605),   ('Qwen3-8B','RU'):(11,.582),   ('Qwen3-8B','KY'):(9,.482),
    ('Llama-3.1-8B','EN'):(1,.625),('Llama-3.1-8B','RU'):(19,.568),('Llama-3.1-8B','KY'):(3,.478),
    ('Mistral-7B','EN'):(5,.611), ('Mistral-7B','RU'):(9,.570), ('Mistral-7B','KY'):(5,.451),
    ('XLM-R','EN'):(11,.551), ('XLM-R','RU'):(21,.482), ('XLM-R','KY'):(14,.459),
}
for (m,l),(lay,acc) in paper_t2.items():
    r = up[(up.model==m)&(up.lang==l)].iloc[0]
    if not (int(r.layer)==lay and ok(r.linear_acc,acc)):
        print(f"T3 MISMATCH {m} {l}: paper={acc}@{lay} vs data={r.linear_acc}@{int(r.layer)}"); fail+=1

# --- Table 3: CKA ---
cka_files = {
    'Gemma 4 E4B': RES/"cka_results.csv", 'Qwen3-8B': RES/"qwen3/cka_results.csv",
    'Llama-3.1-8B': RES/"llama/cka_results.csv", 'Mistral-7B': RES/"mistral/cka_results.csv",
}
cols = {'EN–RU':'cka_en_ru','EN–KY':'cka_en_ky','RU–KY':'cka_ru_ky'}
paper_t3 = {
    ('Gemma 4 E4B','EN–RU'):(.599,12),('Gemma 4 E4B','EN–KY'):(.575,12),('Gemma 4 E4B','RU–KY'):(.599,21),
    ('Qwen3-8B','EN–RU'):(.604,22), ('Qwen3-8B','EN–KY'):(.620,19),('Qwen3-8B','RU–KY'):(.620,19),
    ('Llama-3.1-8B','EN–RU'):(.830,31),('Llama-3.1-8B','EN–KY'):(.713,12),('Llama-3.1-8B','RU–KY'):(.731,2),
    ('Mistral-7B','EN–RU'):(.794,31),('Mistral-7B','EN–KY'):(.736,18),('Mistral-7B','RU–KY'):(.800,1),
}
for (m,p),(v,lay) in paper_t3.items():
    df = pd.read_csv(cka_files[m]); col = cols[p]
    idx = df[col].idxmax(); peak = df[col][idx]; layer = df.layer[idx]
    if not (ok(peak,v) and int(layer)==lay):
        print(f"T3 MISMATCH {m} {p}: paper={v}@L{lay} vs data={peak:.3f}@L{int(layer)}"); fail+=1

# --- Table 5/14: leakage-free transfer (scripts/40) ---
tn = pd.read_csv(RES/"transfer_noleak.csv")
key = {"gemma4":"Gemma 4 E4B","qwen3":"Qwen3-8B","llama":"Llama-3.1-8B",
       "mistral":"Mistral-7B","xlmr":"XLM-R"}
paper_t5 = {  # transcribed from paper.tex Table 5
    ('gemma4',('en','ru')):.497,('gemma4',('en','ky')):.250,('gemma4',('ru','en')):.513,
    ('gemma4',('ru','ky')):.290,('gemma4',('ky','en')):.321,('gemma4',('ky','ru')):.307,
    ('qwen3',('en','ru')):.581,('qwen3',('en','ky')):.294,('qwen3',('ru','en')):.544,
    ('qwen3',('ru','ky')):.277,('qwen3',('ky','en')):.443,('qwen3',('ky','ru')):.382,
    ('llama',('en','ru')):.537,('llama',('en','ky')):.358,('llama',('ru','en')):.513,
    ('llama',('ru','ky')):.304,('llama',('ky','en')):.341,('llama',('ky','ru')):.361,
    ('mistral',('en','ru')):.537,('mistral',('en','ky')):.199,('mistral',('ru','en')):.527,
    ('mistral',('ru','ky')):.240,('mistral',('ky','en')):.280,('mistral',('ky','ru')):.236,
    ('xlmr',('en','ru')):.470,('xlmr',('en','ky')):.392,('xlmr',('ru','en')):.476,
    ('xlmr',('ru','ky')):.358,('xlmr',('ky','en')):.409,('xlmr',('ky','ru')):.378,
}
for (m,(s,t)),v in paper_t5.items():
    r = tn[(tn.model==m)&(tn.src==s)&(tn.tgt==t)].iloc[0]
    if not ok(r.test_acc,v):
        print(f"T5 MISMATCH {key[m]} {s}->{t}: paper={v} vs data={r.test_acc}"); fail+=1

# --- Appendix D: selectivity (now from unified run) ---
sel_names = {'Gemma 4 E4B':'Gemma 4 E4B','Qwen3-8B':'Qwen3-8B',
             'Llama-3.1-8B':'Llama-3.1-8B','Mistral-7B':'Mistral-7B'}
paper_d = {  # transcribed from paper.tex Table 9
    ('Gemma 4 E4B','EN'):(.603,.621,.191,.413,.018),('Gemma 4 E4B','RU'):(.508,.534,.164,.345,.026),
    ('Gemma 4 E4B','KY'):(.453,.485,.162,.292,.032),('Qwen3-8B','EN'):(.605,.611,.165,.441,.005),
    ('Qwen3-8B','RU'):(.582,.607,.172,.410,.025),('Qwen3-8B','KY'):(.482,.506,.183,.299,.024),
    ('Llama-3.1-8B','EN'):(.625,.636,.165,.460,.011),('Llama-3.1-8B','RU'):(.568,.589,.172,.396,.021),
    ('Llama-3.1-8B','KY'):(.478,.482,.169,.309,.004),('Mistral-7B','EN'):(.611,.619,.152,.459,.008),
    ('Mistral-7B','RU'):(.570,.581,.176,.393,.011),('Mistral-7B','KY'):(.451,.455,.149,.301,.005),
}
for (m,l),(lin,mlp,ctrl,sel,gap) in paper_d.items():
    r = up[(up.model==m)&(up.lang==l)].iloc[0]
    vals = {'linear_acc':lin,'mlp_acc':mlp,'control_acc':ctrl,'selectivity':sel,'mlp_gap':gap}
    for f,v in vals.items():
        if not ok(getattr(r,f),v):
            print(f"D MISMATCH {m} {l} {f}: paper={v} vs data={getattr(r,f)}"); fail+=1

# --- Appendix E: leakage-free pairwise Δ (scripts/44); consistency check only ---
pw_path = RES/"tables/transfer_pairwise_noleak.csv"
if pw_path.exists():
    pw = pd.read_csv(pw_path)
    # every pairwise |Δ| must equal the rounded subtraction of the two Table-5 test_accs
    smap = {'Gemma 4':'gemma4','Qwen3':'qwen3','Llama':'llama','Mistral':'mistral'}
    for _,r in pw.iterrows():
        s,t = r.direction.lower().split('->')
        hi = tn[(tn.model==smap[r.model_hi])&(tn.src==s)&(tn.tgt==t)].test_acc.iloc[0]
        lo = tn[(tn.model==smap[r.model_lo])&(tn.src==s)&(tn.tgt==t)].test_acc.iloc[0]
        if abs(round(abs(hi-lo),3) - r.delta) > 1.5e-3:
            print(f"E MISMATCH {r.direction} {r.model_hi} vs {r.model_lo}: "
                  f"Δcsv={r.delta} vs subtract={round(abs(hi-lo),3)}"); fail+=1
else:
    print("E SKIP: transfer_pairwise_noleak.csv not yet generated")

# --- Appendix S: emotion confusions (scripts/66) ---
# Paper values transcribed from paper.tex Table "confusion_pairs" (5-model mean, EN/RU/KY).
cs_path = RES/"tables/confusion_pairs_summary.csv"
if cs_path.exists():
    cs = pd.read_csv(cs_path).set_index("pair")
    paper_s = {
        'anger-disgust':(.168,.154,.153),    'anger-surprise':(.138,.145,.160),
        'anger-sadness':(.148,.137,.148),    'disgust-fear':(.094,.135,.156),
        'sadness-surprise':(.097,.102,.134), 'joy-surprise':(.092,.102,.109),
        'disgust-sadness':(.082,.098,.101),  'fear-sadness':(.061,.079,.095),
        'disgust-surprise':(.066,.074,.082), 'anger-fear':(.054,.073,.088),
        'joy-sadness':(.054,.061,.091),      'fear-surprise':(.050,.070,.082),
        'anger-joy':(.045,.060,.087),        'disgust-joy':(.034,.047,.062),
        'fear-joy':(.026,.040,.062),
    }
    for p, vals in paper_s.items():
        for l, v in zip(['EN','RU','KY'], vals):
            if not ok(cs.loc[p, l], v, tol=6e-4):
                print(f"S MISMATCH {p} {l}: paper={v} vs data={cs.loc[p, l]}"); fail+=1
    # table order in the paper = descending cross-language mean
    if list(paper_s) != list(cs.sort_values("mean", ascending=False).index):
        print("S MISMATCH: pair order in paper != descending cross-language mean"); fail+=1
    # summary must be the 5-model mean of confusion_pairs.csv
    cp = pd.read_csv(RES/"tables/confusion_pairs.csv")
    mean5 = cp.groupby(['pair','lang']).rate.mean().unstack()
    for p in cs.index:
        for l in ['EN','RU','KY']:
            if not ok(mean5.loc[p, l], cs.loc[p, l], tol=6e-4):
                print(f"S MISMATCH {p} {l}: summary={cs.loc[p, l]} vs 5-model mean={mean5.loc[p, l]:.4f}"); fail+=1
    # pair rates must follow from the raw matrices: (C_ab + C_ba) / (n_a + n_b)
    cm = pd.read_csv(RES/"tables/confusion_matrices.csv")
    if cm.groupby(['model','lang']).ngroups != 15:
        print(f"S MISMATCH: {cm.groupby(['model','lang']).ngroups} confusion matrices, expected 15"); fail+=1
    for (m, l), g in cm.groupby(['model','lang']):
        C = g.pivot(index='true', columns='pred', values='count').fillna(0)
        n = C.sum(axis=1)
        # each matrix must be the Table-3 predictions: same layer, 1,480 sentences, same accuracy
        u = up[(up.model==m)&(up.lang==l)].iloc[0]
        acc = sum(C.loc[e, e] for e in C.index) / n.sum()
        if not (int(g.layer.iloc[0])==int(u.layer) and int(n.sum())==1480 and ok(acc, u.linear_acc, tol=6e-5)):
            print(f"S MISMATCH {m} {l}: matrix acc={acc:.4f}@L{int(g.layer.iloc[0])} n={int(n.sum())} "
                  f"vs unified={u.linear_acc}@L{int(u.layer)}"); fail+=1
        for _, r in cp[(cp.model==m)&(cp.lang==l)].iterrows():
            a, b = r.pair.split('-')
            rate = (C.loc[a, b] + C.loc[b, a]) / (n[a] + n[b])
            if not ok(rate, r.rate, tol=6e-5):
                print(f"S MISMATCH {m} {l} {r.pair}: pairs.csv={r.rate} vs matrix={rate:.4f}"); fail+=1
    # text claims: 13 of 15 pairs rise EN->KY; Spearman .78--.97, all p<.001, min = Qwen3 EN-KY
    # (counted at the table's 3-decimal precision: Anger--Sadness .148/.148 is "flat")
    n_up = sum(ky > en for en, _, ky in paper_s.values())
    if n_up != 13:
        print(f"S MISMATCH: pairs more confused EN->KY = {n_up}, paper says 13"); fail+=1
    ra = pd.read_csv(RES/"tables/confusion_rank_agreement.csv")
    lo = ra.loc[ra.spearman.idxmin()]
    if not (len(ra)==15 and ok(ra.spearman.min(), .78, 6e-3) and ok(ra.spearman.max(), .97, 6e-3)
            and (ra.p < 1e-3).all() and (lo.model, lo.langs)==('Qwen3-8B','EN-KY')):
        print(f"S MISMATCH rank agreement: n={len(ra)} rho=[{ra.spearman.min():.3f},{ra.spearman.max():.3f}] "
              f"max p={ra.p.max():.2g} min at {lo.model} {lo.langs}"); fail+=1
else:
    print("S SKIP: confusion_pairs_summary.csv not yet generated")

print(f"\n{'ALL NUMBERS OK' if fail==0 else f'{fail} MISMATCHES'}")
