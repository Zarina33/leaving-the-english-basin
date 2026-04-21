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
bp = pd.read_csv(RES / "tables/bootstrap_probing.csv")
paper_t2 = {
    ('Gemma 4 E4B','EN'):(0,.596,.571,.621), ('Gemma 4 E4B','RU'):(6,.511,.486,.537), ('Gemma 4 E4B','KY'):(4,.458,.432,.484),
    ('Qwen3-8B','EN'):(9,.609,.584,.633),   ('Qwen3-8B','RU'):(11,.580,.555,.605),   ('Qwen3-8B','KY'):(27,.474,.449,.499),
    ('Llama-3.1-8B','EN'):(1,.634,.609,.659),('Llama-3.1-8B','RU'):(7,.570,.545,.595),('Llama-3.1-8B','KY'):(3,.484,.459,.510),
    ('Mistral-7B','EN'):(5,.609,.585,.634), ('Mistral-7B','RU'):(9,.559,.534,.584), ('Mistral-7B','KY'):(2,.440,.415,.466),
}
for (m,l),(lay,acc,lo,hi) in paper_t2.items():
    r = bp[(bp.model==m)&(bp.lang==l)].iloc[0]
    if not (r.layer==lay and ok(r.accuracy,acc) and ok(r.ci_lower,lo) and ok(r.ci_upper,hi,tol=5e-3)):
        print(f"T2 MISMATCH {m} {l}: paper={acc}@{lay}[{lo},{hi}] vs data={r.accuracy}@{r.layer}[{r.ci_lower},{r.ci_upper}]"); fail+=1

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

# --- Table 4: transfer ---
bt = pd.read_csv(RES/"tables/bootstrap_transfer.csv")
paper_t4 = {
    ('Gemma 4 E4B','EN→RU'):.564,('Gemma 4 E4B','EN→KY'):.328,('Gemma 4 E4B','RU→EN'):.561,
    ('Gemma 4 E4B','RU→KY'):.378,('Gemma 4 E4B','KY→EN'):.389,('Gemma 4 E4B','KY→RU'):.388,
    ('Qwen3-8B','EN→RU'):.740,('Qwen3-8B','EN→KY'):.366,('Qwen3-8B','RU→EN'):.718,
    ('Qwen3-8B','RU→KY'):.353,('Qwen3-8B','KY→EN'):.447,('Qwen3-8B','KY→RU'):.477,
    ('Llama-3.1-8B','EN→RU'):.737,('Llama-3.1-8B','EN→KY'):.393,('Llama-3.1-8B','RU→EN'):.765,
    ('Llama-3.1-8B','RU→KY'):.422,('Llama-3.1-8B','KY→EN'):.441,('Llama-3.1-8B','KY→RU'):.472,
    ('Mistral-7B','EN→RU'):.628,('Mistral-7B','EN→KY'):.238,('Mistral-7B','RU→EN'):.734,
    ('Mistral-7B','RU→KY'):.255,('Mistral-7B','KY→EN'):.297,('Mistral-7B','KY→RU'):.330,
}
for (m,d),v in paper_t4.items():
    r = bt[(bt.model==m)&(bt.direction==d)].iloc[0]
    if not ok(r.accuracy,v):
        print(f"T4 MISMATCH {m} {d}: paper={v} vs data={r.accuracy}"); fail+=1

# --- Appendix D: selectivity ---
ps = pd.read_csv(RES/"tables/probe_selectivity.csv")
paper_d = {
    ('Gemma 4 E4B','EN'):(.593,.610,.189,.404,.017),('Gemma 4 E4B','RU'):(.507,.553,.166,.341,.047),
    ('Gemma 4 E4B','KY'):(.455,.472,.157,.297,.018),('Qwen3-8B','EN'):(.601,.597,.157,.443,-.003),
    ('Qwen3-8B','RU'):(.598,.599,.164,.434,.001),('Qwen3-8B','KY'):(.466,.480,.174,.292,.014),
    ('Llama-3.1-8B','EN'):(.624,.645,.167,.457,.020),('Llama-3.1-8B','RU'):(.566,.592,.174,.392,.026),
    ('Llama-3.1-8B','KY'):(.472,.481,.167,.305,.009),('Mistral-7B','EN'):(.617,.618,.157,.459,.001),
    ('Mistral-7B','RU'):(.564,.576,.169,.395,.013),('Mistral-7B','KY'):(.445,.450,.171,.274,.005),
}
for (m,l),(lin,mlp,ctrl,sel,gap) in paper_d.items():
    r = ps[(ps.model==m)&(ps.lang==l)].iloc[0]
    vals = {'linear_acc':lin,'mlp_acc':mlp,'control_acc':ctrl,'selectivity':sel,'mlp_gap':gap}
    for f,v in vals.items():
        if not ok(getattr(r,f),v):
            print(f"D MISMATCH {m} {l} {f}: paper={v} vs data={getattr(r,f)}"); fail+=1

# --- Appendix E: Δacc = rounded subtractions of Table 4 ---
appe = [
    ('EN→RU','Qwen3','Gemma 4',.176),('EN→RU','Llama','Gemma 4',.173),('EN→RU','Qwen3','Mistral',.112),
    ('EN→RU','Llama','Mistral',.109),('EN→RU','Mistral','Gemma 4',.064),('EN→RU','Qwen3','Llama',.003),
    ('EN→KY','Llama','Mistral',.155),('EN→KY','Qwen3','Mistral',.128),('EN→KY','Gemma 4','Mistral',.090),
    ('EN→KY','Llama','Gemma 4',.065),('EN→KY','Qwen3','Gemma 4',.038),('EN→KY','Llama','Qwen3',.027),
    ('RU→EN','Llama','Gemma 4',.204),('RU→EN','Mistral','Gemma 4',.173),('RU→EN','Qwen3','Gemma 4',.157),
    ('RU→EN','Llama','Qwen3',.047),('RU→EN','Mistral','Qwen3',.016),('RU→EN','Llama','Mistral',.031),
    ('RU→KY','Llama','Mistral',.167),('RU→KY','Gemma 4','Mistral',.123),('RU→KY','Qwen3','Mistral',.098),
    ('RU→KY','Llama','Qwen3',.069),('RU→KY','Llama','Gemma 4',.044),('RU→KY','Gemma 4','Qwen3',.025),
    ('KY→EN','Qwen3','Mistral',.150),('KY→EN','Llama','Mistral',.144),('KY→EN','Gemma 4','Mistral',.092),
    ('KY→EN','Qwen3','Gemma 4',.058),('KY→EN','Llama','Gemma 4',.052),('KY→EN','Qwen3','Llama',.006),
    ('KY→RU','Qwen3','Mistral',.147),('KY→RU','Llama','Mistral',.142),('KY→RU','Qwen3','Gemma 4',.089),
    ('KY→RU','Llama','Gemma 4',.084),('KY→RU','Gemma 4','Mistral',.058),('KY→RU','Qwen3','Llama',.005),
]
short2full = {'Gemma 4':'Gemma 4 E4B','Qwen3':'Qwen3-8B','Llama':'Llama-3.1-8B','Mistral':'Mistral-7B'}
for d,a,b,expected in appe:
    av = paper_t4[(short2full[a],d)]; bv = paper_t4[(short2full[b],d)]
    diff = round(av - bv, 3)
    if abs(diff - expected) > 1e-6:
        print(f"E MISMATCH {d} {a} vs {b}: paper={expected} vs rounded-subtract={diff}"); fail+=1

print(f"\n{'ALL NUMBERS OK' if fail==0 else f'{fail} MISMATCHES'}")
