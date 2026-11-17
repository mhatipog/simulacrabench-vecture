"""ABLATION BUILD of SimulacraBench F5 (not a submission). F5 KumoStack (robust): F2 plus cross-fitted Kumo (large) as a stacked UNHCR expert.

Cross-fitted, gate-factorised stacking of survey experts.

* Features are the GIVEN block only; labels come from visible rows only.
* Every expert is cross-fitted (K folds) so stacking weights, temperatures
  and gate calibration are estimated on all visible rows out-of-fold.
* For a gated item the gate probability and the answer distribution are
  stacked separately, then the gate is isotonically recalibrated.
* A budget controller times every expert on this machine before committing
  to a plan, so the run adapts to the phase limit instead of timing out.
* No network, no file writes. A selected expert that fails raises, except
  the optional UNHCR Kumo pass, whose failure is logged in SBF_ACTIVATION
  and leaves the stack prediction unchanged.
"""
import os
import time
import json

_T0 = time.monotonic()

import numpy as np
import torch

from sb_core import (prepare, onehot, prior_of, kfold, pmap, FLOOR, NTHREADS)
import sb_experts as E
import sb_stack as S
import sb_eb as EB
from survey_experts import train_net_with_gates

SEED = 20261111
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, 'sb_config.json'), encoding='utf-8') as _fh:
    CONFIG = json.load(_fh)
KUMO_MODE = os.environ.get('SBENCH_KUMO', CONFIG['kumo'])      # require | off
SAFETY = float(CONFIG.get('budget_safety', .80))
KUMO_SHARE = float(os.environ.get('SBENCH_KUMO_SHARE', CONFIG.get('kumo_unhcr_share', .22)))
KUMO_UNHCR_SIZE = os.environ.get('SBENCH_KUMO_SIZE', CONFIG.get('kumo_unhcr_size', 'small'))
KUMO_FOLDS = int(os.environ.get('SBENCH_KUMO_FOLDS', CONFIG.get('kumo_unhcr_folds', 4)))
KUMO_EST = int(os.environ.get('SBENCH_KUMO_EST', CONFIG.get('kumo_unhcr_estimators', 2)))
_SEEN = set()
_USED = [0.]
# Approximate relative work (rows x total option count) of the three official
# instruments, used only to share the wall-clock budget between predict calls.
_OFFICIAL = {12: 1.5, 76: 3.5, 213: 5.0}
LAST_DIAG = {}
ABLATIONS = {}


def _log(tag, obj):
    print(tag, json.dumps(obj, default=str), flush=True)


def _phase_budget(schema, n_visible):
    sp = schema.get('split') or {}
    forced = os.environ.get('SBENCH_BUDGET')
    if forced:
        return float(forced), 'forced'
    if n_visible == sp.get('n_train', -1) + sp.get('n_dev', -1):
        return 3600., 'final'
    return 900., 'development'


def _allocate(schema, n_visible, n_targets, work):
    total, phase = _phase_budget(schema, n_visible)
    elapsed = time.monotonic() - _T0
    remaining = total * SAFETY - elapsed
    _SEEN.add(n_targets)
    later = sum(w for k, w in _OFFICIAL.items() if k not in _SEEN)
    mine = _OFFICIAL.get(n_targets, work)
    share = mine / (mine + later) if (mine + later) > 0 else 1.
    share = share if later else 1.
    return max(20., remaining * share), phase, total


def tiny_sample(Xtr, Y, Xq, C):
    """Similarity-weighted Dirichlet estimate for a handful of visible rows."""
    out = []
    for j, k in enumerate(C):
        prior = prior_of(Y[:, j], int(k))
        sim = (Xtr[None, :, :] == Xq[:, None, :]).mean(axis=2)
        wg = np.exp(2. * sim) - 1.
        z = np.empty((len(Xq), int(k)))
        for i in range(len(Xq)):
            local = np.bincount(Y[:, j], weights=wg[i], minlength=int(k)) + 10. * prior
            z[i] = local / local.sum()
        out.append(z)
    return out


# ----------------------------------------------------------------- Kumo ----

def _temperature(p, tau):
    if tau == 1.0:
        return p
    q = np.power(np.maximum(p, 1e-10), 1.0 / tau)
    return q / q.sum(axis=1, keepdims=True)


def _item_loss(y, k, p):
    return float(-np.log(FLOOR + (1. - FLOOR * k) * p[np.arange(len(y)), y]).mean())


def select_kumo_blend(targets, Y, C, base, alt, by_section=False):
    """Validated mixture of the stack with Kumo (rule unchanged from V81;
    F3 pools UNHCR targets by questionnaire section)."""
    grouping = {}
    for name in alt:
        fam = ('unhcr_' + name.split('_')[0] if by_section else 'wb_psych'
               if name.startswith('sc_nstd_') else 'wb_economic'
               if name.startswith('user.') else 'wb_followup'
               if name.startswith('tr_') else 'unicef')
        grouping.setdefault(fam, []).append(targets.index(name))
    temps = (.85, 1.0, 1.15, 1.35, 1.65)
    alphas = (0., .10, .20, .35, .50, .65, .80, 1.)
    mapping, diag = {}, {}
    for fam, ids in grouping.items():
        baseline = np.mean([_item_loss(Y[:, j], C[j], base[j]) for j in ids])
        trials = []
        for t in temps:
            prep = {j: _temperature(alt[targets[j]], t) for j in ids}
            for a in alphas:
                l = np.mean([_item_loss(Y[:, j], C[j], (1. - a) * base[j] + a * prep[j])
                             for j in ids])
                trials.append((float(l + .0012 * a * a + .0005 * abs(np.log(t))), a, t))
        _, af, tf = min(trials)
        kw = {j: _temperature(alt[targets[j]], tf) for j in ids}
        fitted = []
        for j in ids:
            sc = [(_item_loss(Y[:, j], C[j], (1. - a) * base[j] + a * kw[j])
                   + .010 * (a - af) ** 2 + .0012 * a * a, a) for a in alphas]
            a = .75 * af + .25 * min(sc)[1]
            mapping[targets[j]] = (float(a), float(tf))
            fitted.append(_item_loss(Y[:, j], C[j], (1. - a) * base[j] + a * kw[j]))
        diag[fam] = dict(n_targets=len(ids), pooled_alpha=float(af), temperature=float(tf),
                         baseline_loss=round(float(baseline), 6),
                         blended_loss=round(float(np.mean(fitted)), 6))
    return mapping, diag


# ------------------------------------------------------------ pipeline ----

def _eb_run(keys, labels, Y, C, tr, query_rows, seed):
    """Choose tables on an inner split of `tr`, count on all of `tr`."""
    perm = np.random.default_rng(seed).permutation(tr)
    cut = max(1, int(.8 * len(perm)))
    fit, sel = np.sort(perm[:cut]), np.sort(perm[cut:])

    def one(j):
        y = Y[:, j]; k = int(C[j])
        spec = EB.select_spec(keys, labels, y, k, fit, sel)
        return EB.predict_spec(keys, labels, y, k, tr, query_rows, spec), spec[0]
    res = pmap(one, range(len(C)))
    kinds = [r[1] for r in res]
    return [r[0] for r in res], {k: kinds.count(k) for k in set(kinds)}


def predict(frame, schema):
    started = time.monotonic()
    D = prepare(frame, schema)
    targets, hidden, visible = D['targets'], D['hidden'], D['visible']
    Y, C, gated = D['Y'], D['C'], D['gated']
    J = len(targets)
    if not len(hidden):
        return []
    n = len(visible)
    if n < 1:
        raise ValueError('No visible TRAIN records')
    Xc_v, Xc_h = D['Xc'][visible], D['Xc'][hidden]
    if n < 90:
        mats = tiny_sample(Xc_v, Y, Xc_h, C)
        return [mats[j][i].tolist() for i in range(len(hidden)) for j in range(J)]

    budget, phase, total = _allocate(schema, n, J, n * float(C.sum()))
    deadline = started + budget
    G = Xc_v.shape[1]
    XL = np.concatenate([D['Xc'].astype(np.float32), D['Xnum']], axis=1)
    XLv, XLh = XL[visible], XL[hidden]
    Hall = onehot(D['Xc'], D['levels'])
    Hv, Hh = Hall[visible], Hall[hidden]
    Xo_v, Xo_h = D['Xoff'][visible], D['Xoff'][hidden]
    allj = list(range(J))
    gidx = [j for j in allj if gated[j] and C[j] >= 2]
    NAfull = np.column_stack([Y[:, j] == C[j] - 1 for j in gidx]) if gidx else np.zeros((n, 0), bool)
    if gidx:
        _, first, inv = np.unique(NAfull, axis=1, return_index=True, return_inverse=True)
        NApat = NAfull[:, first]; pat_of = dict(zip(gidx, np.ravel(inv)))
    else:
        NApat = NAfull; pat_of = {}

    # ---- Kumo first pass (GPU, hosted only) so its cost is known ----------
    kumo = None
    kerr = None
    kinfo = None
    if KUMO_MODE == 'require':
        import kumo_parents as KP
        sel = KP.dominant_parents(schema, targets)
        capped = KP.is_unhcr(schema) or bool(os.environ.get('SBENCH_KUMO_TEST_CAPPED'))
        if sel:
            rng = np.random.default_rng(SEED + 5 + J)
            mix = np.sort(rng.permutation(n)[:min(950, max(45, round(n * .165)))])
            ctx = np.setdiff1d(np.arange(n), mix)
            tk = time.monotonic()
            if capped:
                # F3/F4: UNHCR. Time-capped and allowed to fail; the stack alone
                # is the V48-certified path, and any failure is reported in the log.
                # F4 prefers the larger Kumo checkpoint (the repo default) and
                # falls back to small if it cannot be loaded.
                ksize = None
                # most uncertain targets first, so a time stop drops the easiest
                def _ent(j):
                    p = prior_of(Y[:, j], int(C[j])); return float(-(p * np.log(p + 1e-12)).sum())
                ent_order = [targets[j] for j in sorted(range(J), key=lambda j: -_ent(j))]
                ksize = []
                for size in dict.fromkeys((KUMO_UNHCR_SIZE, 'small')):
                    try:
                        KP._get_model(size); ksize.append(size)
                    except Exception as e:      # noqa: BLE001 - reported below
                        kerr = (kerr + ' | ' if kerr else '') + size + ': ' + repr(e)[:200]
                kinfo = {}
                try:
                    cf = KP.crossfit(frame, schema, targets, visible, hidden, folds=KUMO_FOLDS,
                                     stop_at=tk + KUMO_SHARE * budget, size=ksize,
                                     estimators=KUMO_EST, order=ent_order,
                                     seed=SEED + 9, info=kinfo) if ksize else {}
                except Exception as e:          # noqa: BLE001 - reported below
                    cf, kerr = {}, (kerr + ' | ' if kerr else '') + repr(e)[:300]
                for nm, (po_, ph_) in cf.items():
                    j = targets.index(nm)
                    if (po_.shape != (n, int(C[j])) or ph_.shape != (len(hidden), int(C[j]))
                            or not (np.isfinite(po_).all() and np.isfinite(ph_).all())):
                        raise RuntimeError('Invalid Kumo cross-fit output for ' + nm)
                if cf:
                    kumo = dict(capped=True, cf=cf, names=list(cf), size=kinfo.get('size'),
                                info=kinfo, seconds=time.monotonic() - tk)
            else:
                kcv = KP.predict_parent_distributions(frame, schema, targets, visible[ctx], visible[mix])
                if len(kcv) != len(sel):
                    raise RuntimeError('Kumo validation outputs incomplete')
                sec = time.monotonic() - tk
                kumo = dict(mix=mix, kcv=kcv, seconds=sec, capped=False)
                # second pass predicts len(hidden) rows from n rows of context
                ratio = max(1., len(hidden) / len(mix)) * n / max(1, len(ctx))
                deadline -= 1.25 * ratio * sec + 5.
    elif KUMO_MODE != 'off':
        raise RuntimeError('Unknown kumo mode ' + str(KUMO_MODE))

    # ---- probe expert cost on this machine, then choose a plan ------------
    rng = np.random.default_rng(SEED + J)
    probe_rows = np.sort(rng.permutation(n)[:max(60, int(.75 * n))])
    pj = list(rng.permutation(J)[:min(J, 2 * NTHREADS)])
    cost = {}
    t = time.monotonic(); E.lgb_direct(XLv, G, Y, C, probe_rows, XLh[:64], SEED, pj)
    cost['lgb'] = (time.monotonic() - t) * J / len(pj)
    cost['fact'] = cost['lgb'] * (.7 * len(gidx) / J + 1.7 * NApat.shape[1] / J)
    pc = pj[:min(len(pj), NTHREADS)]
    t = time.monotonic(); E.cat_direct(Xc_v, Y, C, probe_rows, Xc_h[:64], SEED, pc, iterations=30)
    cat_iters = 86 if J > 120 else 115
    cost['cat'] = (time.monotonic() - t) * J / len(pc) * cat_iters / 30.
    train_net_with_gates(Xo_v, Y, C, gated, D['vocab'], probe_rows[:256], None, Xo_h[:64],
                         seed=SEED, max_epochs=1)
    t = time.monotonic()
    train_net_with_gates(Xo_v, Y, C, gated, D['vocab'], probe_rows, None, Xo_h[:64],
                         seed=SEED, max_epochs=1)
    ep1 = time.monotonic() - t
    t = time.monotonic(); E.linear_experts(Hv, Y, C, gated, probe_rows, Hh[:64], SEED, steps=20)
    cost['lin'] = (time.monotonic() - t) * 7.
    ebkeys, eblabels = EB.build_keys(np.vstack([Xc_v, Xc_h]), np.arange(n))
    t = time.monotonic(); _eb_run(ebkeys, eblabels, Y[:, pj], C[pj], probe_rows, np.arange(64), SEED)
    cost['eb'] = (time.monotonic() - t) * J / len(pj) * 1.3
    cost['net'] = ep1 * 16.
    E.match_experts(Hv, Y, C, probe_rows[:256], Hv[:64])
    t = time.monotonic(); E.match_experts(Hv, Y, C, probe_rows, Hv[:512])
    cost['nn'] = (time.monotonic() - t) * (n + len(hidden)) / 512. / 5.
    avail = deadline - time.monotonic() - 8. - .02 * budget
    plan = None
    for use in (('lgb', 'fact', 'lin', 'eb', 'cat', 'net'), ('lgb', 'fact', 'lin', 'eb', 'net'),
                ('lgb', 'fact', 'eb', 'net'), ('lgb', 'fact', 'lin', 'eb'),
                ('lgb', 'lin', 'eb'), ('lin', 'eb')):
        per = sum(cost[u] for u in use) + cost['nn']
        for K in (5, 4, 3, 2):
            extra = cost['net'] * 1.4 if 'net' in use else 0.
            if (K + 1) * per + extra <= avail:
                plan = (use, K); break
        if plan:
            break
    if plan is None:
        plan = (('lin', 'eb'), 2)
    use, K = plan
    fold = kfold(n, K, SEED + J)
    names = [u for u in ('lgb', 'cat', 'net', 'eb', 'lin') if u in use] + ['nn1', 'nn2']
    kcf = kumo is not None and kumo['capped']
    if kcf:
        names.append('kumo')
    oof = {m: [np.zeros((n, int(k)), np.float32) for k in C] for m in names}
    hid = {}
    fact = 'fact' in use and bool(gidx)
    cnames = [c for c in (('clgb',) if fact else ()) + (('clin',) if 'lin' in use else ())]
    oofc = {m: {j: np.zeros((n, int(C[j]) - 1), np.float32) for j in gidx} for m in cnames}
    hidc = {}
    oofg = np.zeros((n, NApat.shape[1]), np.float32) if fact else None
    hidg = None
    net_epochs = None
    timing = {}

    def run_pass(tr, Xq_idx, final):
        """Fit every planned expert on rows `tr`; predict rows Xq (or hidden)."""
        nonlocal net_epochs
        q = (lambda a_v, a_h: a_h) if final else (lambda a_v, a_h: a_v[Xq_idx])
        out, outc, outg = {}, {}, None
        t = time.monotonic()
        if 'lgb' in use:
            out['lgb'] = E.lgb_direct(XLv, G, Y, C, tr, q(XLv, XLh), SEED + 17 * final, allj)
        timing['lgb'] = timing.get('lgb', 0.) + time.monotonic() - t; t = time.monotonic()
        if fact:
            outg = E.lgb_gate(XLv, G, NApat, tr, q(XLv, XLh), SEED)
            cl = E.lgb_cond(XLv, G, Y, C, tr, q(XLv, XLh), SEED, gidx)
            outc['clgb'] = dict(zip(gidx, cl))
        timing['fact'] = timing.get('fact', 0.) + time.monotonic() - t; t = time.monotonic()
        if 'cat' in use:
            out['cat'] = E.cat_direct(Xc_v, Y, C, tr, q(Xc_v, Xc_h), SEED + 17 * final, allj,
                                      iterations=cat_iters)
        timing['cat'] = timing.get('cat', 0.) + time.monotonic() - t; t = time.monotonic()
        if 'lin' in use:
            d, c = E.linear_experts(Hv, Y, C, gated, tr, q(Hv, Hh), SEED)
            out['lin'] = d
            outc['clin'] = {j: c[j] for j in gidx}
        timing['lin'] = timing.get('lin', 0.) + time.monotonic() - t; t = time.monotonic()
        if 'eb' in use:
            rows = np.arange(n, n + len(hidden)) if final else Xq_idx
            out['eb'], ebk = _eb_run(ebkeys, eblabels, Y, C, tr, rows, SEED + len(tr))
            if final:
                timing['eb_specs'] = ebk
        timing['eb'] = timing.get('eb', 0.) + time.monotonic() - t; t = time.monotonic()
        if 'net' in use:
            if net_epochs is None:
                r2 = np.random.default_rng(SEED + 3).permutation(tr)
                cut = int(.88 * len(r2))
                _, meta = train_net_with_gates(Xo_v, Y, C, gated, D['vocab'], np.sort(r2[:cut]),
                                               np.sort(r2[cut:]), Xo_h[:8], seed=SEED + J,
                                               max_epochs=20)
                net_epochs = int(meta['best_epoch'])
            p, _ = train_net_with_gates(Xo_v, Y, C, gated, D['vocab'], tr, None, q(Xo_v, Xo_h),
                                        seed=SEED + J + final, max_epochs=net_epochs)
            out['net'] = p
        timing['net'] = timing.get('net', 0.) + time.monotonic() - t; t = time.monotonic()
        out['nn1'], out['nn2'] = E.match_experts(Hv, Y, C, tr, q(Hv, Hh))
        timing['nn'] = timing.get('nn', 0.) + time.monotonic() - t
        return out, outc, outg

    for f in range(K):
        te = np.flatnonzero(fold == f); tr = np.flatnonzero(fold != f)
        out, outc, outg = run_pass(tr, te, 0)
        for m in names:
            if m == 'kumo':
                continue
            for j in allj:
                oof[m][j][te] = out[m][j]
        for m in cnames:
            for j in gidx:
                oofc[m][j][te] = outc[m][j]
        if fact:
            oofg[te] = np.column_stack(outg)
    hid, hidc, hg = run_pass(np.arange(n), None, 1)
    if kcf:
        # Kumo cross-fit (own folds) where covered, the visible marginal elsewhere
        hid['kumo'] = []
        for j in allj:
            nm = targets[j]
            if nm in kumo['cf']:
                oof['kumo'][j][:] = kumo['cf'][nm][0]; hid['kumo'].append(kumo['cf'][nm][1])
            else:
                pr = prior_of(Y[:, j], int(C[j])).astype(np.float32)
                oof['kumo'][j][:] = pr[None, :]; hid['kumo'].append(np.tile(pr, (len(hidden), 1)))
    if fact:
        hidg = np.column_stack(hg)


    # ---- stacking (ablation build: same algorithm as F5, re-run per variant) --
    # SBA_VARIANTS is a ';'-separated list of variants; each variant is a
    # '+'-separated set of flags ('full' = no flag = the submitted F5 stack).
    # Experts are fitted once above, so every variant is paired with F5 on
    # identical out-of-fold and hidden-row expert predictions.
    idx = np.arange(n)
    nh = len(hidden)
    ABLATIONS.clear()
    variants = [v.strip() for v in os.environ.get('SBA_VARIANTS', 'full').split(';') if v.strip()]

    def run_variant(flags):
        drop = {f[5:] for f in flags if f.startswith('drop_')}
        if 'nn' in drop:
            drop |= {'nn1', 'nn2'}
        if 'lin' in drop:
            drop.add('clin')
        fstack = fact and 'drop_fact' not in flags and 'nofactstack' not in flags
        vn = [m for m in names if m not in drop]
        vc = [m for m in cnames if m not in drop and (m != 'clgb' or fstack)]
        M = len(vn)
        if 'fit950' in flags:
            fr = np.sort(np.random.default_rng(SEED + 950).permutation(n)[:min(950, n)])
        else:
            fr = idx
        P_oof, P_hid = [None] * J, [None] * J
        temps, calib = {}, {}
        wlog = {'direct': np.zeros(M + 1)}
        gst = [j for j in gidx] if 'nofactstack' not in flags else []
        ung = [j for j in allj if j not in set(gst)]

        if 'avg' in flags or 'best1' in flags:
            if 'best1' in flags:
                tot = {m: sum(-np.log(FLOOR + (1 - int(C[j]) * FLOOR)
                                      * oof[m][j][fr, Y[fr, j]]).sum() for j in allj) for m in vn}
                pick = [min(tot, key=tot.get)]
            else:
                pick = vn
            for j in allj:
                P_oof[j] = sum(oof[m][j].astype(np.float64) for m in pick) / len(pick)
                P_hid[j] = sum(hid[m][j].astype(np.float64) for m in pick) / len(pick)
            info = {'picked': pick}
            return P_oof, P_hid, info

        def pooled(mats, ks, m):
            if not mats:
                return np.full(m, 1. / m)
            A = np.concatenate(mats); k = np.concatenate(ks)
            if len(A) > 60000:
                s = np.random.default_rng(SEED).permutation(len(A))[:60000]; A = A[s]; k = k[s]
            return S.fit_weights(A, k, np.full(m, 1. / m), .003)

        def fitw(A, k, prior, pen):
            return prior.copy() if 'nopq' in flags else S.fit_weights(A, k, prior, pen)

        def ptemp(p, y, k):
            return 1. if 'notemp' in flags else S.pick_temp(p, y, k)

        yf = Y[fr]
        Au = {}
        for j in ung:
            y = Y[:, j]; pr = prior_of(y, int(C[j]))
            Au[j] = np.column_stack([oof[m][j][fr, yf[:, j]] for m in vn] + [pr[yf[:, j]]])
        gw = pooled([Au[j] for j in ung], [np.full(len(fr), C[j]) for j in ung], M + 1)

        def stack_ung(j):
            k = int(C[j]); y = Y[:, j]; pr = prior_of(y, k)
            w = fitw(Au[j], k, gw, .004)
            po = sum(w[i] * oof[m][j] for i, m in enumerate(vn)) + w[M] * pr[None, :]
            ph = sum(w[i] * hid[m][j] for i, m in enumerate(vn)) + w[M] * pr[None, :]
            T = ptemp(po[fr], y[fr], k)
            return w, T, S.temper(po.astype(np.float64), T), S.temper(ph.astype(np.float64), T)
        for j, (w, T, po, ph) in zip(ung, pmap(stack_ung, ung)):
            P_oof[j], P_hid[j] = po, ph; temps[j] = T; wlog['direct'] += w / max(1, len(ung))

        gn = vn + (['gbin'] if fstack else []) + ['prior']
        cn = vn + vc + ['prior']
        Ag, Ac = {}, {}
        for j in gst:
            k = int(C[j]); y = Y[:, j]; na = y == k - 1
            cols = [oof[m][j][fr, k - 1] for m in vn]
            if fstack:
                cols.append(oofg[fr, pat_of[j]])
            cols.append(np.full(len(fr), (na.sum() + .5) / (n + 1.)))
            g = np.column_stack(cols).astype(np.float64)
            Ag[j] = np.where(na[fr][:, None], g, 1. - g)
            a = fr[~na[fr]]
            if k > 2 and len(a):
                cc = []
                for m in vn:
                    c = oof[m][j][a][:, :k - 1].astype(np.float64)
                    c = c / np.maximum(c.sum(1, keepdims=True), 1e-12)
                    cc.append(c[np.arange(len(a)), y[a]])
                for m in vc:
                    cc.append(oofc[m][j][a, y[a]])
                pr = prior_of(y[~na], k - 1)
                cc.append(pr[y[a]])
                Ac[j] = np.column_stack(cc)
        ggw = pooled([Ag[j] for j in gst], [np.full(len(fr), C[j]) for j in gst], len(gn))
        gcw = pooled([Ac[j] for j in Ac], [np.full(len(Ac[j]), C[j]) for j in Ac], len(cn))
        wlog['gate'] = np.zeros(len(gn)); wlog['cond'] = np.zeros(len(cn))

        def stack_gated(j):
            k = int(C[j]); y = Y[:, j]; na = y == k - 1
            wg = fitw(Ag[j], k, ggw, .004)
            pg = (na.sum() + .5) / (n + 1.)

            def gate(src, srcg):
                cols = [src[m][j][:, k - 1] for m in vn]
                if fstack:
                    cols.append(srcg[:, pat_of[j]])
                cols.append(np.full(len(cols[0]), pg))
                return np.clip(np.column_stack(cols).astype(np.float64) @ wg, 0., 1.)
            go, gh = gate(oof, oofg), gate(hid, hidg)
            if 'noiso' in flags:
                mp, gain = None, 0.
            else:
                mp, gain = S.calibrate_gate(go[fr], na[fr], k, SEED + j)
            if mp is not None:
                go, gh = S.iso_apply(mp, go), S.iso_apply(mp, gh)
            if k == 2:
                co, ch = np.ones((n, 1)), np.ones((nh, 1)); wc = None; T = 1.
            else:
                a = fr[~na[fr]]
                pr = prior_of(Y[~na, j], k - 1) if (~na).any() else np.full(k - 1, 1. / (k - 1))
                wc = fitw(Ac[j], k, gcw, .004 if len(a) > 400 else .02) if j in Ac else gcw

                def cond(src, srcc, rows):
                    tot = np.zeros((rows, k - 1))
                    for i, m in enumerate(vn):
                        c = src[m][j][:, :k - 1].astype(np.float64)
                        tot += wc[i] * c / np.maximum(c.sum(1, keepdims=True), 1e-12)
                    for i, m in enumerate(vc):
                        tot += wc[M + i] * srcc[m][j]
                    return tot + wc[-1] * pr[None, :]
                co, ch = cond(oof, oofc, n), cond(hid, hidc, nh)
                T = ptemp(co[a], y[a], k - 1) if len(a) > 50 else 1.
                co, ch = S.temper(co, T), S.temper(ch, T)
            po = np.column_stack([(1. - go)[:, None] * co, go])
            ph = np.column_stack([(1. - gh)[:, None] * ch, gh])
            return wg, wc, T, mp is not None, gain, po, ph
        for j, r in zip(gst, pmap(stack_gated, gst)):
            wg, wc, T, used, gain, po, ph = r
            P_oof[j], P_hid[j] = po, ph; temps[j] = T
            wlog['gate'] += wg / len(gst)
            if wc is not None:
                wlog['cond'] += wc / max(1, len(Ac))
            calib[targets[j]] = (bool(used), round(gain, 5))
        info = dict(w_direct=dict(zip(vn + ['prior'], np.round(wlog['direct'], 3).tolist())),
                    w_gate=dict(zip(gn, np.round(wlog['gate'], 3).tolist())) if gst else {},
                    w_cond=dict(zip(cn, np.round(wlog['cond'], 3).tolist())) if gst else {},
                    gate_calibrated=int(sum(u for u, _ in calib.values())),
                    temp_ne1=int(sum(t != 1. for t in temps.values())))
        return P_oof, P_hid, info

    out_main = None
    for v in variants:
        flags = set(f for f in v.split('+') if f and f != 'full')
        P_oof, P_hid, info = run_variant(flags)
        num = den = 0.
        for j in allj:
            q = P_hid[j]
            if q.shape != (nh, int(C[j])) or not np.isfinite(q).all() or q.min() < 0:
                raise RuntimeError('Invalid probability matrix: ' + targets[j])
            q = np.maximum(q, 1e-9); P_hid[j] = q / q.sum(1, keepdims=True)
            y = Y[:, j]; k = int(C[j]); pr = prior_of(y, k)
            num += -np.log(FLOOR + (1 - k * FLOOR) * P_oof[j][idx, y]).sum()
            den += -np.log(FLOOR + (1 - k * FLOOR) * pr[y]).sum()
        vec = [P_hid[j][i].tolist() for i in range(nh) for j in range(J)]
        info['oof_skill'] = round(1. - num / den, 5)
        ABLATIONS[v] = dict(vectors=vec, info=info)
        _log('SBA_VARIANT', dict(variant=v, **info))
        if out_main is None:
            out_main = vec
    diag = dict(n_visible=n, n_hidden=nh, n_targets=J, phase=phase, budget_s=round(budget, 1),
                experts=list(use), folds=K, net_epochs=net_epochs, threads=NTHREADS,
                probe_cost_s={k: round(v, 1) for k, v in cost.items()},
                time_by_expert_s={k: (round(v, 1) if isinstance(v, float) else v) for k, v in timing.items()},
                gate_patterns=int(NApat.shape[1]), gated=len(gidx), kumo_mode=KUMO_MODE,
                seconds=round(time.monotonic() - started, 1))
    LAST_DIAG.clear(); LAST_DIAG.update(diag)
    _log('SBF_ACTIVATION', diag)
    return out_main
