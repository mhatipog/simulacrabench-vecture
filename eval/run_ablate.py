"""Fit the F5 experts once, re-stack under every ablation variant, score each
with the organizers' sampler and scorer.

usage: run_ablate.py <ablation_source_dir> <data_dir> <phase> <out_prefix>
env:   SB_OFFICIAL  path of the organizers' starter kit (score.py, config.yml)
       SBA_VARIANTS ';'-separated variants (see source_ablate/main.py)
Writes <out_prefix>.<variant>.npz (per-cell log probability) and prints one
RESULT line per variant.
"""
import sys, os, time, json
import numpy as np

sub, data, phase, prefix = (os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]),
                            int(sys.argv[3]), os.path.abspath(sys.argv[4]))
OFF = os.path.abspath(os.environ['SB_OFFICIAL'])
sys.path.insert(0, OFF); os.chdir(OFF)
from make_sandbox import load_config
import score as S

cfg = load_config(os.path.join(OFF, 'config.yml'))
schema = json.load(open(os.path.join(data, 'schema.json'), encoding='utf8'))
resp = S.load_frames(data, schema)
frame, cells, truth = S.sample_rows(schema, resp, phase)
sys.path.insert(0, sub); os.chdir(sub)
t0 = time.time(); import main; t1 = time.time()
main.predict(frame.copy(), schema); t2 = time.time()
ref = S.marginal_reference(schema, frame, cells, truth, cfg['scoring']['floor'])
items = np.array([c[2] for c in cells]); rid = np.array([c[1] for c in cells])
for v, r in main.ABLATIONS.items():
    vec = r['vectors']
    S.check(schema, vec, cells)
    fl = S.floored(vec, schema, cells, cfg['scoring']['floor'])
    res = S.score(schema, cfg, fl, truth, cells, ref); res.pop('by_item')
    lp = np.array([np.log(p[S.options_for(schema, c[2]).index(t)]) for p, t, c in zip(fl, truth, cells)])
    np.savez_compressed(f'{prefix}.{v}.npz', lp=lp, items=items, rid=rid)
    print('RESULT', json.dumps({'variant': v, 'data': os.path.basename(data), 'phase': phase,
                                'skill': round(res['skill'], 6), 'nll': round(-res['log_score'], 6),
                                'oof_skill': r['info'].get('oof_skill'), 'R': round(ref, 6),
                                'predict_s': round(t2 - t1, 1), 'cells': res['n_cells'],
                                'info': {k: r['info'][k] for k in r['info'] if k != 'oof_skill'}}), flush=True)
