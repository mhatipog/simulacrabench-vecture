"""GSS 2022 -> benchmark-format proxy studies (IID respondent split).
usage: gss_build.py <GSS2022.dta> <out_prefix> <n_studies> <targets_per_study>"""
import sys,os,json,numpy as np,pandas as pd
src,prefix,ns,nt=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
d=pd.read_stata(src,convert_categoricals=False,convert_missing=True)
ms=lambda v:getattr(v,'string',None)
def clean(c):
    s=d[c];m=s.map(ms);v=pd.to_numeric(s.where(m.isna()),errors='coerce');return v,m
G={}
v,m=clean('age');G['age']=pd.cut(v,[0,24,34,44,54,64,200],labels=['18-24','25-34','35-44','45-54','55-64','65+']).astype(object).where(v.notna(),'Not answered')
v,m=clean('income16');G['income16']=pd.cut(v,[0,8,12,15,18,21,30],labels=['b1','b2','b3','b4','b5','b6']).astype(object).where(v.notna(),'Not answered')
for c in ['sex','race','degree','marital','region','wrkstat','childs','born','relig','ballot','form','mode']:
    v,m=clean(c);G[c]=v.map(lambda x:'v%d'%x if x==x else None).where(v.notna(),'Not answered')
given=list(G)
skip=set(given)|{'year','id','vpsu','vstrat','fileversion','samptype','age','cohort'}
cands=[]
for c in d.columns:
    if c in skip or c.startswith(('wt','vs','ad')):continue
    s=d[c];m=s.map(ms);v=s[m.isna()]
    try:v=pd.to_numeric(v)
    except Exception:continue
    if 2<=v.nunique()<=9 and len(v)>=400:cands.append(c)
rng=np.random.default_rng(2022);rng.shuffle(cands)
n=len(d)
for sidx in range(ns):
    tg=sorted(cands[sidx*nt:(sidx+1)*nt],key=list(d.columns).index)
    items={};cols={}
    for g in given:
        vals=sorted(set(G[g]));items[g]={'question':g,'class':'GIVEN','values':vals,'gate':None};cols[g]=np.asarray(G[g],object)
    for c in tg:
        v,m=clean(c);lab=v.map(lambda x:'c%g'%x if x==x else None)
        iap=(m=='.i').to_numpy();other=(v.isna()&~iap).to_numpy()
        out=np.asarray(lab,object);vals=['c%g'%x for x in sorted(v.dropna().unique())]
        if other.any():out[other]='No answer';vals.append('No answer')
        gate=None
        if iap.any():
            out[iap]='NA_GATED'
            b=pd.Series(~iap).groupby(cols['ballot']).mean()
            gate={'parent':'ballot','observed_if':[k for k,x in b.items() if x>.05] or list(b.index)}
        items[c]={'question':c,'class':'PREDICT','values':vals,'gate':gate};cols[c]=out
    perm=np.random.default_rng(100+sidx).permutation(n)
    fr=pd.DataFrame({k:cols[k][perm] for k in items});fr.insert(0,'respondent_id',['G%05d'%i for i in range(n)])
    ntr,ndev=3000,600;fr['role']=np.repeat(np.array(['TRAIN','DEV','TEST'],object),[ntr,ndev,n-ntr-ndev])
    S={'dataset':{'n_rows':n,'version':'gss2022-proxy-%d'%sidx,'description':'GSS 2022 proxy'},'items':items,'split':{'n_train':ntr,'n_dev':ndev,'n_test':n-ntr-ndev},'gated_value':'NA_GATED'}
    out=prefix+str(sidx);os.makedirs(out,exist_ok=True)
    fr.astype(object).to_parquet(out+'/respondents.parquet',index=False);json.dump(S,open(out+'/schema.json','w'))
    print(out,'targets',len(tg),'gated',sum(1 for c in tg if items[c]['gate']),'sumK',sum(len(items[c]['values'])+bool(items[c]['gate']) for c in tg))
