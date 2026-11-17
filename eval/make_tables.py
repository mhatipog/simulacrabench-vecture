"""Summarise the ablation sweep: paired deltas against the submitted stack.

usage: make_tables.py <ablation_out_dir> <out_dir>
Reads <dataset>.<variant>.npz and <dataset>.log (RESULT lines) written by
run_ablate.py. Writes ablation_summary.json, ablation_full.csv and two LaTeX
tables (main text and appendix). Delta = variant minus full stack, mean log
score per cell in milli-nats; negative = removing the component hurts.
CI = 2,000 respondent-cluster bootstrap resamples (seed 0).
"""
import sys, os, json, glob
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
VARIANTS = ['drop_lgb', 'drop_cat', 'drop_net', 'drop_eb', 'drop_lin', 'drop_nn', 'drop_fact',
            'nofactstack', 'noiso', 'notemp', 'nopq', 'fit950', 'avg', 'best1']
LABEL = {
    'drop_lgb': r'$-$ LightGBM (direct)', 'drop_cat': r'$-$ CatBoost', 'drop_net': r'$-$ multitask net',
    'drop_eb': r'$-$ empirical-Bayes tables', 'drop_lin': r'$-$ multinomial logit (direct+cond.)',
    'drop_nn': r'$-$ kernel neighbours', 'drop_fact': r'$-$ gate/answer-if-asked LightGBM',
    'nofactstack': r'no gate factorisation (one stack per item)', 'noiso': r'no isotonic gate recalibration',
    'notemp': r'no temperature', 'nopq': r'pooled weights only (no per-item)',
    'fit950': r'stack fitted on 950 rows (V81-style)', 'avg': r'equal-weight average of experts',
    'best1': r'single best expert (selected out-of-fold)'}
DATA = [('real_gss0', 'GSS-1'), ('real_gss1', 'GSS-2'), ('real_gss2', 'GSS-3'),
        ('sim_unhcr_A', 'Gate-A'), ('sim_unhcr_B', 'Gate-B'), ('sim_unhcr_C', 'Gate-C'),
        ('sim_twin_C', 'Twin'), ('sim_twin_D', 'Twin-HH'),
        ('sandbox_unicef', 'SB-UNICEF'), ('sandbox_world_bank', 'SB-WB'), ('sandbox_unhcr', 'SB-UNHCR')]


def boot(d, rid, B=2000):
    u, inv = np.unique(rid, return_inverse=True)
    per = np.bincount(inv, weights=d); cnt = np.bincount(inv)
    rng = np.random.default_rng(0)
    s = rng.integers(0, len(u), (B, len(u)))
    bs = per[s].sum(1) / cnt[s].sum(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


res = {}
for key, lab in DATA:
    full = os.path.join(src, f'{key}.full.npz')
    if not os.path.exists(full):
        continue
    meta = {}
    for line in open(os.path.join(src, key + '.log'), encoding='utf8', errors='replace'):
        if line.startswith('RESULT '):
            r = json.loads(line[7:]); meta[r['variant']] = r
    a = np.load(full)
    ent = dict(label=lab, skill_full=meta['full']['skill'], oof_full=meta['full']['oof_skill'],
               nll_full=meta['full']['nll'], cells=meta['full']['cells'],
               respondents=int(len(np.unique(a['rid']))), predict_s=meta['full']['predict_s'],
               w_direct=meta['full']['info'].get('w_direct'), w_gate=meta['full']['info'].get('w_gate'),
               w_cond=meta['full']['info'].get('w_cond'), variants={})
    for v in VARIANTS:
        p = os.path.join(src, f'{key}.{v}.npz')
        if not os.path.exists(p):
            continue
        b = np.load(p)
        assert (a['rid'] == b['rid']).all() and (a['items'] == b['items']).all()
        m, lo, hi = boot(b['lp'] - a['lp'], a['rid'])
        ent['variants'][v] = dict(delta=m, lo=lo, hi=hi, skill=meta[v]['skill'], oof_skill=meta[v]['oof_skill'],
                                  info=meta[v]['info'])
    res[key] = ent
json.dump(res, open(os.path.join(dst, 'ablation_summary.json'), 'w'), indent=1)

with open(os.path.join(dst, 'ablation_full.csv'), 'w') as fh:
    fh.write('dataset,variant,delta_mnats,ci_lo,ci_hi,skill,oof_skill,skill_full\n')
    for key, e in res.items():
        for v, r in e['variants'].items():
            fh.write(f"{e['label']},{v},{1e3*r['delta']:.3f},{1e3*r['lo']:.3f},{1e3*r['hi']:.3f},"
                     f"{r['skill']:.6f},{r['oof_skill']},{e['skill_full']:.6f}\n")


def cell(r, ci=False):
    m, lo, hi = 1e3 * r['delta'], 1e3 * r['lo'], 1e3 * r['hi']
    sig = lo > 0 or hi < 0
    s = f'{m:+.1f}'
    s = r'\textbf{' + s + '}' if sig else s
    return s + (f' {{\\scriptsize[{lo:+.1f},{hi:+.1f}]}}' if ci else '')


def table(keys, ci, caption, label, size):
    keys = [k for k in keys if k in res]
    cols = 'l' + 'r' * len(keys)
    out = [r'\begin{table}[t]', r'\centering', size, r'\setlength{\tabcolsep}{3pt}',
           r'\caption{' + caption + '}', r'\label{' + label + '}', r'\begin{tabular}{' + cols + '}', r'\toprule',
           'Variant & ' + ' & '.join(res[k]['label'] for k in keys) + r' \\', r'\midrule',
           r'Full stack (submitted), skill & ' + ' & '.join(f"{res[k]['skill_full']:.4f}" for k in keys) + r' \\',
           r'\midrule']
    for v in VARIANTS:
        if not any(v in res[k]['variants'] for k in keys):
            continue
        out.append(LABEL[v] + ' & ' + ' & '.join(cell(res[k]['variants'][v], ci) if v in res[k]['variants']
                                                  else '--' for k in keys) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out) + '\n'


main_keys = ['real_gss0', 'real_gss1', 'real_gss2', 'sim_unhcr_A', 'sim_twin_D', 'sandbox_unhcr']
open(os.path.join(dst, 'ablation_main.tex'), 'w').write(table(
    main_keys, False,
    r'Paired ablations of the submitted stack (Kumo off, identical expert fits). Entries are the change in '
    r'mean log score per scored cell, in milli-nats, when the component is removed; negative means the '
    r'component helps. Bold: 95\% respondent-bootstrap interval excludes zero. GSS: real survey; Gate/Twin: '
    r'synthetic mechanism tests; SB: organizer sandbox. Not leaderboard numbers.',
    'tab:ablation', r'\scriptsize'))
open(os.path.join(dst, 'ablation_appendix.tex'), 'w').write(table(
    [k for k, _ in DATA], False,
    r'All ablations on all eleven evaluation sets (milli-nats per cell, variant minus full stack; bold: 95\% '
    r'interval excludes zero). Intervals are listed in \texttt{results/ablation\_full.csv}.',
    'tab:ablation-all', r'\tiny'))
print(json.dumps({k: {'skill': e['skill_full'], 'n_var': len(e['variants'])} for k, e in res.items()}, indent=0))
