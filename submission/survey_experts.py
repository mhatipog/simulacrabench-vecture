"""Supervised multi-task direct+factorized survey expert and categorical GBDT.

Contrasts V39's masked-answer self-conditioning: this model only sees GIVEN
features, matching the information available for every hidden respondent.
The two probabilistic heads are trained using the exact full-response target.
"""
import copy
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

FLOOR=0.001


def prepare(frame,schema):
    given=[n for n,r in schema['items'].items() if r['class']=='GIVEN']
    targets=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    if not targets:raise ValueError('No target questions')
    hidden=np.flatnonzero(frame[targets[0]].isna().to_numpy())
    visible=np.flatnonzero(~frame[targets[0]].isna().to_numpy())
    codes=[];offset=1;L=[]
    for key in given:
        opts=[str(z) for z in schema['items'][key]['values']]
        val=frame[key].astype('string').fillna('__MISSING__')
        enc={z:i+1 for i,z in enumerate(opts)}
        col=val.map(enc).fillna(0).to_numpy(np.int32)
        codes.append(col+offset)
        offset+=len(opts)+2
        L.append(len(opts))
    X=np.column_stack(codes).astype(np.int32) if codes else np.ones((len(frame),1),dtype=np.int32)
    sizes=[];y=[];allopts=[]
    for key in targets:
        r=schema['items'][key]
        opts=[str(z) for z in r['values']]+([str(schema['gated_value'])] if r.get('gate') else [])
        if len(opts)>1000:raise ValueError('Excessive options')
        mp={z:i for i,z in enumerate(opts)}
        vals=frame.iloc[visible][key].astype('string').map(mp).fillna(-1).to_numpy(np.int32)
        if (vals<0).any():raise ValueError('Unexpected visible response for '+key)
        y.append(vals);sizes.append(len(opts));allopts.append(opts)
    Y=np.column_stack(y).astype(np.int64)
    return given,targets,allopts,hidden,visible,X,Y,np.array(sizes,dtype=int),offset


def scored_loss(probs, labels, width):
    p=np.asarray(probs,dtype=np.float64)
    return float(-np.log(FLOOR+(1.-width*FLOOR)*p[np.arange(len(labels)),labels]).mean())


def _prior(y,k):
    c=np.bincount(y,minlength=k).astype(float)+0.5
    return c/c.sum()


def fit_boost(X,Y,sizes,train,query,seed=0,query_is_features=False,active=None):
    """One categorical LightGBM per observed response question (no old code).

    Uses all categorical features; no label / respondent id is a feature.
    Each target is optimized for categorical log-loss, not accuracy.
    """
    from lightgbm import LGBMClassifier
    Xtr=np.asarray(X[train],dtype=np.int32)
    Xq=np.asarray(query if query_is_features else X[query],dtype=np.int32)
    outputs=[];fitted=0;degenerate=0
    categorical=list(range(Xtr.shape[1]))
    for j,k in enumerate(sizes):
        yy=Y[train,j];prior=_prior(yy,k)
        if (active is not None and not active[j]) or len(np.unique(yy))<2 or len(yy)<25:
            z=np.broadcast_to(prior,(len(Xq),k)).copy();degenerate+=1
        else:
            model=LGBMClassifier(n_estimators=96,learning_rate=.055,num_leaves=11,
                  max_depth=-1,min_child_samples=65,reg_lambda=15.,
                  reg_alpha=.18,min_split_gain=.001,subsample=1.,
                  colsample_bytree=.95,n_jobs=1,verbosity=-1,random_state=seed+j)
            model.fit(Xtr,yy,categorical_feature=categorical)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",message="X does not have valid feature names")
                raw=model.predict_proba(Xq)
            z=np.broadcast_to(prior*.035,(len(Xq),k)).copy()
            for i,c in enumerate(model.classes_):z[:,int(c)]+=.965*raw[:,i]
            z/=z.sum(1,keepdims=True)
            fitted+=1
        outputs.append(z)
    return outputs,{'fit_count':fitted,'degenerate':degenerate}


class SurveyNet(nn.Module):
    """Two shared multi-output heads: direct categorical and gated factorized."""
    def __init__(self,num_features,vocab,sizes,prior):
        super().__init__()
        self.sizes=list(map(int,sizes));self.T=len(sizes);self.K=max(sizes)
        self.feature_emb=nn.Embedding(vocab+5,14)
        d=max(1,num_features)*14
        self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.LayerNorm(256),nn.Dropout(.15),
                               nn.Linear(256,256),nn.GELU(),nn.Dropout(.12))
        self.head=nn.Linear(256,self.T*self.K)
        self.gate_head=nn.Linear(256,self.T)
        self.register_buffer('mask',torch.arange(self.K)[None,:]<torch.tensor(sizes)[:,None])
        self.register_buffer('gated',torch.tensor(prior['gated'],dtype=torch.bool))
        self.register_buffer('prior',torch.tensor(prior['logits'],dtype=torch.float32))
        with torch.no_grad():
            self.head.weight.mul_(.02);self.head.bias.zero_()
            self.gate_head.weight.mul_(.02);self.gate_head.bias.zero_()
            self.gate_head.bias.copy_(torch.tensor(prior['gatelogits'],dtype=torch.float32))
    def forward(self,X):
        x=self.net(self.feature_emb(X).reshape(X.shape[0],-1))
        logits=self.head(x).view(-1,self.T,self.K)+self.prior[None,:,:]
        logits=logits.masked_fill(~self.mask[None,:,:],-1.e4)
        gates=self.gate_head(x)
        return logits,gates


def _logprobs(logits,gates,sizes,gated):
    """Normalised outputs for full direct and separate gated-conditional head."""
    allp=torch.softmax(logits,dim=-1)
    gp=torch.sigmoid(gates)
    parts=[]
    for j,k in enumerate(sizes):
        q=allp[:,j,:k]
        if gated[j]:
            conditional=torch.softmax(logits[:,j,:k-1],dim=-1)
            route=torch.cat(((1.-gp[:,j:j+1])*conditional,gp[:,j:j+1]),dim=1)
            # A stable integration of unconditional and gate-factorized estimates;
            # both are learned directly from visible full responses.
            q=.4*q+.6*route
        parts.append(q)
    return parts


def _loss(net,X,Y,sizes,gated):
    logits,gates=net(X)
    # Vectorized exact categorical log-likelihood across every target.
    direct=F.cross_entropy(logits.transpose(1,2),Y,reduction='none')
    valid_g=torch.as_tensor(gated,device=Y.device,dtype=torch.bool)
    upper=torch.as_tensor(sizes,device=Y.device)-1
    skipped=(Y==upper[None,:]) & valid_g[None,:]
    asked=~skipped
    conditional_mask=net.mask.clone()
    conditional_mask[valid_g,upper[valid_g]]=False
    conditional=logits.masked_fill(~conditional_mask[None,:,:],-1.e4)
    conditioned_y=torch.where(skipped,torch.zeros_like(Y),Y)
    answer=F.cross_entropy(conditional.transpose(1,2),conditioned_y,reduction='none')
    gate_loss=F.binary_cross_entropy_with_logits(gates,skipped.float(),reduction='none')
    full=torch.where(valid_g[None,:],.40*direct+.60*(gate_loss+answer*asked.float()),direct)
    return full.mean()


def _prob(net,x,sizes,gated,bs=1024):
    allp=[[] for _ in sizes]
    net.eval()
    with torch.inference_mode():
        for start in range(0,len(x),bs):
            logits,g=net(x[start:start+bs])
            arrays=_logprobs(logits,g,sizes,gated)
            for j,p in enumerate(arrays):allp[j].append(p.cpu().numpy().astype(np.float64))
    return [np.concatenate(parts,axis=0) if parts else np.zeros((0,k),float) for k,parts in zip(sizes,allp)]


def train_net_with_gates(X,Y,sizes,gated,vocab,train,valid,Xquery,seed=20261103,max_epochs=20):
    pass
    torch.manual_seed(seed)
    X=np.asarray(X,dtype=np.int32);Y=np.asarray(Y,dtype=np.int64)
    tx=torch.as_tensor(X,dtype=torch.long)
    ty=torch.as_tensor(Y,dtype=torch.long)
    qx=torch.as_tensor(np.asarray(Xquery,dtype=np.int32),dtype=torch.long)
    gated=np.asarray(gated,dtype=bool)
    priorlogits=np.full((len(sizes),int(max(sizes))),-15.,dtype=np.float32)
    gatelogits=np.full(len(sizes),-2.,dtype=np.float32)
    for j,k in enumerate(sizes):
        counts=np.bincount(Y[train,j],minlength=k).astype(float)+1.5
        priorlogits[j,:k]=np.log(counts/counts.sum())
        if gated[j]:
            pg=np.clip(counts[-1]/counts.sum(),1.e-3,1.-1.e-3)
            gatelogits[j]=np.log(pg/(1.-pg))
    model=SurveyNet(tx.shape[1],vocab,sizes,{'logits':priorlogits,'gatelogits':gatelogits,'gated':gated})
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.065)
    rng=np.random.default_rng(seed)
    bs=384 if len(train)>1000 else max(32,len(train)//3)
    best=np.inf;best_epoch=1;best_state=None;patience=0
    for epoch in range(max_epochs):
        model.train()
        for p in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/bs)))):
            loss=_loss(model,tx[p],ty[p],sizes,gated)
            optim.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.0)
            optim.step()
        if valid is not None and len(valid):
            val_probs=_prob(model,tx[valid],sizes,gated)
            val_loss=float(np.mean([scored_loss(val_probs[j],Y[valid,j],k) for j,k in enumerate(sizes)]))
            if val_loss < best-0.0003:
                best=val_loss;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());patience=0
            else:patience+=1
            if patience>=5 and epoch>=7:break
        else:best_epoch=epoch+1
    if best_state is not None:model.load_state_dict(best_state)
    preds=_prob(model,qx,sizes,gated)
    diagnostics={'best_epoch':best_epoch,'best_validation_loss':best,'epochs_run':epoch+1}
    if valid is not None and len(valid):
        diagnostics['validation_probs']=_prob(model,tx[valid],sizes,gated)
    return preds,diagnostics
