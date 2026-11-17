"""Run a submission's predict() on one dataset with the organizers' sampler and scorer.

usage: run_eval.py <submission_dir> <data_dir> <phase 1|2> [out.npz]
env:   SB_OFFICIAL = starter-kit checkout (score.py, make_sandbox.py, config.yml)
       SBENCH_KUMO=off to run without a GPU; SBENCH_BUDGET=<s> to force the plan
"""
import sys, os, time, json
import numpy as np

sub, data, phase = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
out = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else None
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
vec = main.predict(frame.copy(), schema); t2 = time.time()
S.check(schema, vec, cells)
fl = S.floored(vec, schema, cells, cfg['scoring']['floor'])
ref = S.marginal_reference(schema, frame, cells, truth, cfg['scoring']['floor'])
r = S.score(schema, cfg, fl, truth, cells, ref); r.pop('by_item')
print('RESULT', json.dumps({'sub': os.path.basename(sub), 'data': os.path.basename(data), 'phase': phase,
                            'skill': round(r['skill'], 6), 'nll': round(-r['log_score'], 6), 'R': round(ref, 6),
                            'import_s': round(t1 - t0, 1), 'predict_s': round(t2 - t1, 1), 'cells': r['n_cells']}))
if out:
    lp = np.array([np.log(v[S.options_for(schema, c[2]).index(t)]) for v, t, c in zip(fl, truth, cells)])
    np.savez_compressed(out, lp=lp, items=np.array([c[2] for c in cells]), rid=np.array([c[1] for c in cells]))
