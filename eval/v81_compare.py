"""Paired comparison of the earlier V81 ensemble with the submitted stack (F5, Kumo off).

usage: v81_compare.py <v81_npz_dir> <ablation_out_dir> <out.tex> <out.json>
V81 runs: <dir>/V81_<name>.npz from run_eval.py; F5: <ablation_out>/<key>.full.npz.
"""
import sys, os, json
import numpy as np

v81dir, abl, tex, js = sys.argv[1:5]
SETS = [('gss0', 'real_gss0', 'GSS-1'), ('gss1', 'real_gss1', 'GSS-2'), ('gss2', 'real_gss2', 'GSS-3'),
        ('unhcr_A', 'sim_unhcr_A', 'Gate-A'), ('unhcr_B', 'sim_unhcr_B', 'Gate-B'), ('unhcr_C', 'sim_unhcr_C', 'Gate-C')]


def boot(d, rid, B=2000):
    u, inv = np.unique(rid, return_inverse=True)
    per = np.bincount(inv, weights=d); cnt = np.bincount(inv)
    s = np.random.default_rng(0).integers(0, len(u), (B, len(u)))
    bs = per[s].sum(1) / cnt[s].sum(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


rows, out = [], {}
for a, b, lab in SETS:
    pa, pb = os.path.join(v81dir, f'V81_{a}.npz'), os.path.join(abl, f'{b}.full.npz')
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    A, Bz = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    assert (A['rid'] == Bz['rid']).all() and (A['items'] == Bz['items']).all()
    m, lo, hi = boot(Bz['lp'] - A['lp'], A['rid'])
    out[lab] = dict(nll_v81=float(-A['lp'].mean()), nll_f5=float(-Bz['lp'].mean()), delta=m, lo=lo, hi=hi)
    rows.append(f"{lab} & {-A['lp'].mean():.4f} & {-Bz['lp'].mean():.4f} & ${m:+.4f}$ [${lo:+.4f}$, ${hi:+.4f}$] \\\\")
json.dump(out, open(js, 'w'), indent=1)
open(tex, 'w').write('\n'.join([
    r'\begin{table}[t]', r'\centering\small',
    r'\caption{Earlier ensemble (V81) against the submitted stack (F5, Kumo off, full plan), paired on identical cells. '
    r'Mean log loss per cell (nats) and gain of F5 with 95\% respondent-bootstrap intervals.}',
    r'\label{tab:v81f5}', r'\begin{tabular}{lrrl}', r'\toprule',
    r'Dataset & V81 loss & F5 loss & Gain (95\% CI) \\', r'\midrule', *rows,
    r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))
print(json.dumps(out, indent=1))
