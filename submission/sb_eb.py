"""Empirical-Bayes table expert, vectorised across targets.

Same model family as the V48/V81 evidence expert: Dirichlet-smoothed
P(answer | one GIVEN field or a coarse pair of fields), or a tempered
naive-Bayes combination of the best single fields, or the plain marginal.
The choice is made on a selection split that is disjoint from the rows the
tables are counted on; the caller passes rows from inside one training fold.
"""
import numpy as np
from sb_core import FLOOR, prior_of

MASS1, MASS2 = 22., 42.


def build_keys(Xc, rows_for_levels):
    """Integer group ids (dense, 0..L-1 over all rows) for singles and pairs."""
    G = Xc.shape[1]
    keys, labels = [], []
    dense = []
    for i in range(G):
        _, inv = np.unique(Xc[:, i], return_inverse=True)
        dense.append(inv.astype(np.int64)); keys.append(dense[-1]); labels.append((i,))
    nl = [len(np.unique(d[rows_for_levels])) for d in dense]
    for i in range(G):
        for j in range(i + 1, G):
            if nl[i] * nl[j] > 360:
                continue
            _, inv = np.unique(dense[i] * (dense[j].max() + 1) + dense[j], return_inverse=True)
            keys.append(inv.astype(np.int64)); labels.append((i, j))
    return keys, labels


def _table(key, y, k, fit, query, mass, prior):
    L = int(key.max()) + 1
    raw = np.bincount(key[fit] * k + y[fit], minlength=L * k).reshape(L, k)
    p = (raw + mass * prior[None, :]) / (raw.sum(1, keepdims=True) + mass)
    return p[key[query]]


def _nll(p, y, k):
    return -np.log(FLOOR + (1. - k * FLOOR) * p[np.arange(len(y)), y])


def select_spec(keys, labels, y, k, fit, sel):
    prior = prior_of(y[fit], k)
    ys = y[sel]
    ref = _nll(np.broadcast_to(prior, (len(sel), k)), ys, k)
    results = []
    for q, (key, lab) in enumerate(zip(keys, labels)):
        mass = MASS1 if len(lab) == 1 else MASS2
        adv = ref - _nll(_table(key, y, k, fit, sel, mass, prior), ys, k)
        se = adv.std(ddof=1) / np.sqrt(len(adv)) if len(adv) > 1 else 1.
        results.append((adv.mean() - .65 * se - (.0015 if len(lab) == 2 else .0005), q))
    results.sort(reverse=True)
    cands = [(results[0][0], ('table', results[0][1], None))]
    singles = [r for r in results if len(labels[r[1]]) == 1][:5]
    if len(singles) > 1:
        cum = np.zeros((len(sel), k))
        lp = np.log(prior)[None, :]
        for rank, (_, q) in enumerate(singles, start=1):
            cum += np.log(np.maximum(_table(keys[q], y, k, fit, sel, MASS1, prior), 1e-9)) - lp
            if rank in (2, 3, 5):
                for temp in (.25, .5, .75):
                    z = lp + cum * temp
                    z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
                    adv = ref - _nll(z, ys, k)
                    se = adv.std(ddof=1) / np.sqrt(len(adv))
                    cands.append((adv.mean() - .65 * se - .001,
                                  ('evidence', tuple(r[1] for r in singles[:rank]), temp)))
    cands.append((0., ('prior', None, None)))
    return max(cands, key=lambda c: c[0])[1]


def predict_spec(keys, labels, y, k, fit, query, spec):
    typ, which, temp = spec
    prior = prior_of(y[fit], k)
    if typ == 'prior':
        return np.broadcast_to(prior, (len(query), k)).copy()
    if typ == 'table':
        return _table(keys[which], y, k, fit, query,
                      MASS1 if len(labels[which]) == 1 else MASS2, prior)
    lp = np.log(prior)[None, :]
    z = np.zeros((len(query), k))
    for q in which:
        z += np.log(np.maximum(_table(keys[q], y, k, fit, query, MASS1, prior), 1e-9)) - lp
    z = lp + temp * z
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)
