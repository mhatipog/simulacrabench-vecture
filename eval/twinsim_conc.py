"""Twin-panel mechanism simulator (NOT a predictor of leaderboard score).

Builds data in the shape of an official schema with three properties the
organizers document for the real files and their sandbox lacks:
  * weak-to-moderate demographic signal plus unobserved traits,
  * gates that the rows only partly obey (agreement .29-.98, median ~.83),
  * skip patterns that are determined by GIVEN fields (module fielding).
usage: realsim.py <schema.json> <out_dir> <seed> <n_train> <n_dev> <max_targets> <struct_frac> <gate_mode: noisy|strict>
"""
import sys,os,json,numpy as np,pandas as pd
src,out,seed,ntr,ndev,maxt,struct,mode=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),float(sys.argv[7]),sys.argv[8]
S=json.load(open(src,encoding='utf8'));S['gated_value']='NA_GATED';gv='NA_GATED'
it=S['items'];rng=np.random.default_rng(seed)
given=[k for k,r in it.items() if r['class']=='GIVEN']
pred=[k for k,r in it.items() if r['class']=='PREDICT']
# subset of targets, closed under gate parents, spread across the instrument
keep=set(pred if maxt>=len(pred) else [pred[i] for i in np.linspace(0,len(pred)-1,maxt).astype(int)])
ch=True
while ch:
    ch=False
    for k in list(keep):
        g=it[k].get('gate')
        if g and it[g['parent']]['class']=='PREDICT' and g['parent'] not in keep:keep.add(g['parent']);ch=True
pred=[k for k in pred if k in keep]
n=ntr+ndev+50
X={};code={}
for i,k in enumerate(given):
    L=len(it[k]['values'])
    if i>=2 and L>20:   # nested geography: level block determined by second GIVEN
        par=code[given[1]];Lp=len(it[given[1]]['values'])
        owner=np.arange(L)%Lp;c=np.empty(n,int)
        for p in range(Lp):
            m=np.flatnonzero(par==p);opts=np.flatnonzero(owner==p)
            c[m]=rng.choice(opts,len(m),p=rng.dirichlet(np.full(len(opts),.8)))
    else:
        c=rng.choice(L,n,p=rng.dirichlet(np.full(L,.35))*.97+.03/L)
    code[k]=c;X[k]=np.asarray(it[k]['values'],object)[c]
# three unobserved traits, partly explained by demographics
traits=rng.normal(size=(n,3))
# panel twins: second half of rows re-interview a random earlier household
ntw=int(.39*n);tw_src=rng.permutation(n-ntw)[:ntw];tw_dst=np.arange(n-ntw,n)
for k in given[1:]:
    code[k][tw_dst]=code[k][tw_src];X[k]=np.asarray(it[k]['values'],object)[code[k]]
code[given[0]][tw_dst]=(code[given[0]][tw_src]+1)%len(it[given[0]]['values']);X[given[0]]=np.asarray(it[given[0]]['values'],object)[code[given[0]]]
traits[tw_dst]=traits[tw_src]
U=rng.random((n,400));U[tw_dst]=np.where(rng.random((ntw,400))<.75,U[tw_src],U[tw_dst])
for k in given[:6]:
    traits+=rng.normal(scale=.35,size=(len(it[k]['values']),3))[code[k]]
col={}
info={}
for name in pred:
    r=it[name];K=len(r['values'])
    logit=np.tile(np.log(rng.dirichlet(np.full(K,.9))+1e-9),(n,1))
    for k in rng.choice(given,size=min(len(given),3),replace=False):
        logit+=rng.normal(scale=.45,size=(len(it[k]['values']),K))[code[k]]
    a,b=rng.choice(len(given),2,replace=False)   # one interaction
    inter=rng.normal(scale=.3,size=(len(it[given[a]]['values']),len(it[given[b]]['values']),K))
    logit+=inter[code[given[a]],code[given[b]]]
    logit+=traits@rng.normal(scale=.8,size=(3,K))
    p=np.exp(logit-logit.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    y=(p.cumsum(1)>U[:,[len(col)]]).argmax(1)
    vals=np.asarray(r['values'],object)[y]
    g=r.get('gate')
    if g:
        par=col[g['parent']] if g['parent'] in col else X[g['parent']]
        opened=np.isin(par,np.asarray(g['observed_if'],object))
        if rng.random()<struct:
            # module fielding: asked only for some levels of a structural field
            f=given[int(rng.integers(0,min(2,len(given))))];L=len(it[f]['values'])
            on=rng.random(L)<.55
            if not on.any():on[rng.integers(L)]=True
            asked=on[code[f]]^(rng.random(n)<.004)
            info[name]='struct:'+f
        elif mode=='noisy' and it[g['parent']]['class']=='PREDICT':
            agree=float(np.clip(rng.beta(5,1.3),.29,.98))
            # disagreement is concentrated in part of the sample, not uniform
            f=given[int(rng.integers(0,min(3,len(given))))];L=len(it[f]['values'])
            w=rng.dirichlet(np.ones(L))*L;rate=np.clip((1-agree)*w[code[f]],0,.95)
            other=(traits[:,0]+rng.normal(size=n))>0
            asked=np.where(rng.random(n)<rate,other,opened)
            info[name]='noisy:%.2f'%agree
        else:
            asked=opened;info[name]='strict'
        vals=np.where(asked,vals,gv)
    col[name]=vals
keepcols=given+pred
S['items']={k:v for k,v in it.items() if k in set(keepcols)}
S['split']={'n_train':ntr,'n_dev':ndev,'n_test':50};S['dataset']['n_rows']=n
fr=pd.DataFrame({k:(X[k] if k in X else col[k]) for k in S['items']})
fr.insert(0,'respondent_id',['R%06d'%i for i in range(n)])
_perm=rng.permutation(n);fr=fr.iloc[_perm].reset_index(drop=True);fr['respondent_id']=['R%06d'%i for i in range(n)]
fr['role']=np.repeat(np.array(['TRAIN','DEV','TEST'],object),[ntr,ndev,50])
os.makedirs(out,exist_ok=True)
fr.astype(object).to_parquet(os.path.join(out,'respondents.parquet'),index=False)
json.dump(S,open(os.path.join(out,'schema.json'),'w',encoding='utf8'),ensure_ascii=False)
json.dump(info,open(os.path.join(out,'gate_info.json'),'w'))
kinds=[v.split(':')[0] for v in info.values()]
print(out,'n',n,'targets',len(pred),'gated',len(info),{k:kinds.count(k) for k in set(kinds)})
