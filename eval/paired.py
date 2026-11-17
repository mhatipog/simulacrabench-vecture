"""Paired comparison of two runs on identical cells: respondent-cluster bootstrap.
usage: paired.py <a.npz> <b.npz>   (positive delta = b better than a)"""
import sys,numpy as np
a=np.load(sys.argv[1],allow_pickle=True);b=np.load(sys.argv[2],allow_pickle=True)
assert (a['rid']==b['rid']).all() and (a['items']==b['items']).all()
d=b['lp']-a['lp'];rid=a['rid'];u,inv=np.unique(rid,return_inverse=True)
per=np.bincount(inv,weights=d);cnt=np.bincount(inv)
rng=np.random.default_rng(0);bs=[]
for _ in range(2000):
    s=rng.integers(0,len(u),len(u));bs.append(per[s].sum()/cnt[s].sum())
lo,hi=np.percentile(bs,[2.5,97.5])
print('nll_a %.5f nll_b %.5f  delta(b-a logscore) %+.5f  95%%CI [%+.5f, %+.5f]  rel %.2f%%  respondents %d'%(-a['lp'].mean(),-b['lp'].mean(),d.mean(),lo,hi,100*d.mean()/(-a['lp'].mean()),len(u)))
