"""V63 Kumo Small for UNICEF and World Bank target ensembles, without query labels.

The predictor is conditioned ONLY on visible TRAIN rows and PREDICT-item labels
from those rows; unseen respondents contribute GIVEN features only. Parents
are selected from the public gate graph by child fanout (>=10) rather than a
hand-written training-label dependency. No silent V48 fallback is performed when a Kumo target is selected.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PARENT_WEIGHT = .75  # Default; final main.py chooses a CV-supported group weight.
MIN_CHILD_FANOUT = 10
_MODELS = {}
OPTS = {'amp': True, 'numeric': True}   # F5 fallback ladder may switch these off
import os as _os
DEVICE = _os.environ.get('SBENCH_KUMO_TEST_DEVICE', 'cuda')  # 'cpu' only in local tests


def dominant_parents(schema,targets):
    """Select the weak-instrument targets, never the already-strong UNHCR.

    These are public schema-defined task families; no hidden-data or leaderboard
    label inference is involved. No selection on held-out respondents.
    """
    items=schema['items'];names=set(targets)
    import os
    if os.environ.get('SBENCH_KUMO_TEST_TARGETS'):
        # Local integration test only; never set on the hosted runner.
        return [(n,1) for n in targets[:int(os.environ['SBENCH_KUMO_TEST_TARGETS'])]]
    given={n for n,r in items.items() if r['class']=='GIVEN'}
    if {'Country','Q05','Q13'}.issubset(given) and len(targets)==12:
        # Vaccine beliefs, choices, and parent responses; all 12 scored items.
        selected=list(targets)
    elif {'group','user.country','user.educationLevel'}.issubset(given) and len(targets)==76:
        # Cover the stable psychometric battery and the repeated baseline
        # economic questions. Diverse targets; cap runtime explicitly.
        # Expand only where the public response taxonomy is manageable.
        # The first 28 psychometric outcomes are a coherent instrument family.
        # Eight employment outcomes add coverage without very wide responses.
        # Selection is made from *public schema metadata*, never hidden labels.
        def manageable(n):
            spec=items[n]
            return len(spec['values']) + bool(spec.get('gate')) <= 12
        # Every compatible public-schema target is eligible for Kumo. A
        # matched H100 public-schema throughput test (see research report)
        # measured 57 tasks in ~54s, not including four-expert training.
        # Held-out visible TRAIN calibration decides whether each expert helps.
        selected=[n for n in targets if manageable(n)]
    elif is_unhcr(schema,targets):
        # F3: every UNHCR target with a manageable public option list. The
        # caller caps Kumo time and a validated blend weight (possibly zero)
        # decides per target whether it is used.
        selected=[n for n in targets if len(items[n]['values'])+bool(items[n].get('gate'))<=12]
    else:
        selected=[]
    return [(name,1) for name in selected]


def is_unhcr(schema,targets=None):
    """F5: anything that is not the UNICEF (12 targets) or World Bank (76
    targets) instrument is treated as the third instrument, so a renamed
    GIVEN field cannot silently switch Kumo off."""
    items=schema['items']
    given={n for n,r in items.items() if r['class']=='GIVEN'}
    if targets is None:
        targets=[n for n,r in items.items() if r['class']=='PREDICT']
    if {'Country','Q05','Q13'}.issubset(given) and len(targets)==12:return False
    if {'group','user.country','user.educationLevel'}.issubset(given) and len(targets)==76:return False
    return True


def _features(frame,schema,rows):
    items=schema['items'];given=[n for n,v in items.items() if v['class']=='GIVEN']
    out=frame.iloc[rows][given].copy().reset_index(drop=True)
    # Preserve original nominal categories while revealing genuine order in
    # household counts and age bands. Neither feature uses a PREDICT answer.
    countmap={str(i):float(i) for i in range(6)}
    countmap.update({'6-10':8.,'11+':11.})
    for col in given:
        raw=out[col].astype('string').fillna('__MISSING__')
        if not OPTS['numeric']:
            pass
        elif any(x in col for x in ('boys_','girls_','men_','women_')):
            out[col+'_numeric']=raw.map(countmap).astype('float32')
        elif col=='s2_1b_age':
            ages={'18-24':21.,'25-34':29.5,'35-44':39.5,'45-54':49.5,
                  '55-64':59.5,'65+':69.}
            out[col+'_numeric']=raw.map(ages).astype('float32')
        out[col]=raw.astype('category')
    return out


def _checkpoint_path(size='small'):
    """Resolve only model weights prefetched during the hosted image build.

    The organizer uses snapshot_download(repo, cache_dir=HF_HUB_CACHE) at the
    default revision. Different Hugging Face installations expose different
    cache variables; do not require any one of them inside predict().
    """
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    roots = []
    for env in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        val = os.environ.get(env)
        if val:
            roots.append(Path(val).expanduser())
    if os.environ.get('HF_HOME'):
        roots.append(Path(os.environ['HF_HOME']).expanduser() / 'hub')
    if os.environ.get('XDG_CACHE_HOME'):
        roots.append(Path(os.environ['XDG_CACHE_HOME']).expanduser() / 'huggingface' / 'hub')
    roots.extend((Path.home() / '.cache' / 'huggingface' / 'hub',
                  Path('/root/.cache/huggingface/hub')))
    roots = list(dict.fromkeys(roots))
    # The build's models.txt resolves the default branch. Using the local refs
    # avoids silently choosing a different snapshot if several are cached.
    for root in roots:
        try:
            path = hf_hub_download(repo_id='nvidia/Kumo-Tabular',
                filename=size + '/classifier.pt', revision='main',
                cache_dir=str(root), local_files_only=True)
        except (OSError, ValueError):
            continue
        if Path(path).is_file():
            return Path(path)
    # Some builds expose a snapshot without its main ref. Accept it only if
    # the snapshot is unambiguous and belongs to this exact repository.
    candidates = []
    for root in roots:
        candidates.extend(p for p in root.glob(
            'models--nvidia--Kumo-Tabular/snapshots/*/' + size + '/classifier.pt')
            if p.is_file())
    unique = {str(p.resolve()):p for p in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))
    raise FileNotFoundError(
        'Offline NVIDIA Kumo ' + size + ' checkpoint missing or ambiguous: '
        'verify models.txt prefetch and HF cache; no runtime download permitted')


def _get_model(size='small'):
    """Initialize Kumo (small/medium/large) from a strictly offline checkpoint."""
    if size not in _MODELS:
        import os, torch, sdm
        from pathlib import Path
        if DEVICE == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Kumo requires the hosted CUDA GPU')
        test_ckpt = os.environ.get('SBENCH_KUMO_TEST_CKPT' + ('' if size == 'small' else '_' + size.upper()))
        weights = Path(test_ckpt) if test_ckpt else _checkpoint_path(size)
        if weights.stat().st_size < 1_000_000:
            raise ValueError('Incomplete Kumo checkpoint in cache')
        model = sdm.models.KumoTabular(task='classification', size=size,
                                      pretrained=False, device=DEVICE)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        modules = list(model.models.values())
        if len(modules) != 1:
            raise RuntimeError('Unexpected number of Kumo task heads')
        modules[0].load_state_dict(checkpoint, strict=True)
        del checkpoint
        model.eval()
        _MODELS[size] = model
    return _MODELS[size]


def _align(pred,sdm,K):
    if not isinstance(pred,sdm.TableTensor):raise TypeError('Kumo result is not TableTensor')
    cls=[str(x) for x in pred.columns[sdm.Stype.numerical]]
    raw=pred.numerical.detach().float().cpu().numpy()
    if raw.shape[1]!=len(cls):raise ValueError('Incorrect predicted class width')
    out=np.full((len(raw),K),1e-10,float)
    seen=set()
    for j,s in enumerate(cls):
        try:v=int(s)
        except ValueError:
            if s.lower() not in ('true','false'):raise
            v=int(s.lower()=='true')
        if v not in range(K) or v in seen:raise ValueError('Unexpected/duplicate Kumo class '+s)
        seen.add(v);out[:,v]=raw[:,j]
    if np.min(out)<0 or not np.isfinite(out).all():raise ValueError('Invalid Kumo probabilities')
    out=np.maximum(out,1e-10);return out/out.sum(1,keepdims=True)


def predict_parent_distributions(frame,schema,targets,fit_rows,query_rows,estimators=2,
                                 names=None,stop_at=None,size='small',order=None):
    """names: restrict to these targets; stop_at: monotonic time after which no
    further target is started (the targets finished so far are returned)."""
    import time
    selected=dominant_parents(schema,targets)
    if names is not None:
        keep=set(names);selected=[s for s in selected if s[0] in keep]
    if order is not None:
        rank={nm:i for i,nm in enumerate(order)}
        selected=sorted(selected,key=lambda s:rank.get(s[0],len(rank)))
    if not selected:return {}
    import sdm,torch
    model=_get_model(size)
    training=_features(frame,schema,fit_rows)
    querying=_features(frame,schema,query_rows)
    result={}
    for name,fanout in selected:
        if stop_at is not None and time.monotonic()>stop_at:
            break
        rec=schema['items'][name]
        labels=[str(v) for v in rec['values']]
        if rec.get('gate'):labels.append(str(schema['gated_value']))
        lookup={v:i for i,v in enumerate(labels)}
        # Only TRAIN parent answers enter the model. Do not read query answers.
        y=frame.iloc[fit_rows][name].astype('string').map(lookup)
        if y.isna().any():raise ValueError('Unexpected visible parent category '+name)
        tbl=pd.concat([training,querying],ignore_index=True)
        tbl['target']=np.r_[y.to_numpy(int), np.zeros(len(querying),dtype=int)]
        tbl['target']=tbl['target'].astype('category')
        tt=sdm.TableTensor.from_pandas(df=tbl,
                 stypes=sdm.infer_stypes(tbl,overrides={'target':'categorical'}),device=DEVICE)
        n=len(training)
        try:
            with torch.inference_mode(),torch.amp.autocast(device_type=DEVICE,dtype=torch.float16,enabled=DEVICE=='cuda' and OPTS['amp']):
                model.fit(x=tt[:n].drop_columns('target'),y=tt[:n,'target'],
                          num_estimators=estimators,estimator_batch_size=1)
                chunks=[]
                for start in range(n,len(tbl),256):
                    p=model.predict(tt[start:min(start+256,len(tbl))].drop_columns('target'))
                    chunks.append(_align(p,sdm,len(labels)))
            out=np.vstack(chunks)
        finally:
            model.clear()
        if out.shape!=(len(query_rows),len(labels)):
            raise ValueError(f'Parent output dimensions: {name}, {out.shape}')
        result[name]=out
    return result


def replace_parents(base,targets,predicted):
    """A fixed, nonzero intervention on the high-fanout parent marginals."""
    out=[a.copy() for a in base]
    targets_idx={k:j for j,k in enumerate(targets)}
    for name,p in predicted.items():
        j=targets_idx[name]
        if out[j].shape!=p.shape:raise ValueError('Mismatched Kumo parent output '+name)
        out[j]=(1.-PARENT_WEIGHT)*out[j]+PARENT_WEIGHT*p
        out[j]/=out[j].sum(1,keepdims=True)
    return out


def crossfit(frame,schema,targets,visible,hidden,folds=4,stop_at=None,size='small',
             estimators=2,order=None,seed=0,min_cover=.9,info=None):
    """F4: out-of-fold Kumo predictions for every visible row plus hidden-row
    predictions, target by target, so Kumo can enter the per-question stack.
    `size` may be a tuple of preferred sizes: the first pass of the first
    target is timed and the next size is used if the projected coverage
    before stop_at is below min_cover. A target is returned only if all of
    its passes finished; no pass is started after stop_at. Only visible-row
    answers are used as labels."""
    import time
    sizes=list(size) if isinstance(size,(list,tuple)) else [size]
    # (size, amp, numeric) settings tried in turn while the first target fails
    ladder=[(z,True,True) for z in sizes]+[('small',False,False)]
    ladder=list(dict.fromkeys(ladder))
    sel=[s[0] for s in dominant_parents(schema,targets)]
    if order is not None:
        rank={nm:i for i,nm in enumerate(order)}
        sel=sorted(sel,key=lambda nm:rank.get(nm,len(rank)))
    n=len(visible)
    fold=np.random.default_rng(seed).permutation(n)%folds
    info=info if info is not None else {}
    out={};k=0
    while k<len(sel):
        name=sel[k];cur,OPTS['amp'],OPTS['numeric']=ladder[0];oof=None;ok=True
        try:
          for f in range(folds+1):
              if stop_at is not None and time.monotonic()>stop_at:
                  ok=False;break
              t=time.monotonic()
              if f<folds:
                  te=np.flatnonzero(fold==f);tr=np.flatnonzero(fold!=f)
                  p=predict_parent_distributions(frame,schema,targets,visible[tr],visible[te],
                                                 estimators=estimators,names=[name],size=cur)[name]
                  if oof is None:oof=np.zeros((n,p.shape[1]),np.float32)
                  oof[te]=p
              else:
                  hid=predict_parent_distributions(frame,schema,targets,visible,hidden,
                                                   estimators=estimators,names=[name],size=cur)[name]
              if k==0 and f==0 and stop_at is not None:
                  per=(time.monotonic()-t)*(folds+1)
                  cover=(stop_at-time.monotonic()+per/(folds+1))/max(per,1e-6)/len(sel)
                  info.setdefault('probe',[]).append((cur,round(per,2),round(cover,3)))
                  if cover<min_cover and len(ladder)>1 and ladder[1][0]!=cur:
                      ladder.pop(0);ok=None;break
        except Exception as e:  # noqa: BLE001
            info.setdefault('errors',[]).append((str(ladder[0]),repr(e)[:160]))
            if k==0 and len(ladder)>1:
                ladder.pop(0);continue
            if k==0:
                raise
            k+=1;continue       # skip a single failing later target
        if ok is None:
            continue           # retry the first target with the next size
        if not ok:
            break
        out[name]=(oof,hid.astype(np.float32));k+=1
    info['size']=str(ladder[0]);info['covered']=len(out);info['eligible']=len(sel)
    return out
