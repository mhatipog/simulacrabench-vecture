"""Factorised stacking on out-of-fold predictions.

For a gated item the scorer's log loss splits exactly into a gate term (was
the sentinel realised?) and, on asked rows only, an answer term. The two are
therefore stacked separately, each on the rows that inform it, and the gate
probability is finally recalibrated by smoothed isotonic regression so that
structurally determined cells can be predicted as sharply as the visible
data support -- and no more sharply.
"""
import numpy as np
from scipy.optimize import minimize
from sb_core import FLOOR

TEMPS = (.8, .9, 1., 1.1, 1.25, 1.5)


def fit_weights(A, k, prior, penalty):
    """Convex weights minimising the floored log loss of A @ w (A: n x M)."""
    M = A.shape[1]
    if len(A) == 0:
        return prior.copy()
    s = np.broadcast_to(1. - np.asarray(k, float) * FLOOR, (len(A),))

    def fn(w):
        q = FLOOR + s * (A @ w)
        return (-np.log(q).mean() + penalty * np.sum((w - prior) ** 2),
                -((s / q)[:, None] * A).mean(0) + 2. * penalty * (w - prior))
    r = minimize(fn, prior, method='SLSQP', jac=True, bounds=[(0., 1.)] * M,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.,
                              'jac': lambda w: np.ones(M)},
                 options={'maxiter': 80, 'ftol': 1e-10})
    w = r.x if r.success and np.isfinite(r.x).all() else prior.copy()
    w = np.maximum(w, 0.)
    return w / max(w.sum(), 1e-12)


def temper(p, T):
    if T == 1.:
        return p
    q = np.power(np.maximum(p, 1e-12), 1. / T)
    return q / q.sum(1, keepdims=True)


def pick_temp(p, y, k):
    idx = np.arange(len(y))
    best = (np.inf, 1.)
    for T in TEMPS:
        q = temper(p, T)
        l = -np.log(FLOOR + (1. - k * FLOOR) * q[idx, y]).mean() + 4e-4 * np.log(T) ** 2
        if l < best[0] - 1e-9:
            best = (l, T)
    return best[1]


def _pav(x, y, w):
    """Weighted isotonic regression; returns block (x mean, y mean, weight)."""
    bx, by, bw = [], [], []
    for xi, yi, wi in zip(x, y, w):
        bx.append(xi * wi); by.append(yi * wi); bw.append(wi)
        while len(bw) > 1 and by[-2] / bw[-2] >= by[-1] / bw[-1]:
            by[-2] += by[-1]; bx[-2] += bx[-1]; bw[-2] += bw[-1]
            by.pop(); bx.pop(); bw.pop()
    bw = np.array(bw)
    return np.array(bx) / bw, np.array(by) / bw, bw


def iso_fit(g, e, bins=300, m=1.):
    """Smoothed isotonic map for a gate probability. Returns (xs, ys)."""
    order = np.argsort(g, kind='stable')
    g = g[order]; e = e[order].astype(float)
    parts = np.array_split(np.arange(len(g)), min(bins, max(1, len(g) // 8)))
    x = np.array([g[p].mean() for p in parts])
    y = np.array([e[p].mean() for p in parts])
    w = np.array([len(p) for p in parts], float)
    bx, by, bw = _pav(x, y, w)
    # Jeffreys-style shrinkage: a pure block of n rows gives about .5/(n+1),
    # never zero, and small blocks stay close to the incoming probability.
    ys = (bw * by + .5 + m * bx) / (bw + 1. + m)
    ys = np.maximum.accumulate(ys)
    return bx, np.clip(ys, 1e-6, 1. - 1e-6)


def iso_apply(mp, g):
    return np.interp(g, mp[0], mp[1])


def gate_loss(g, e, k):
    s = 1. - k * FLOOR
    return -np.log(FLOOR + s * np.where(e, g, 1. - g))


def calibrate_gate(g, e, k, seed):
    """Accept the isotonic map only if it wins a two-way cross-check."""
    n = len(g)
    if n < 200 or e.all() or not e.any():
        return None, 0.
    half = np.random.default_rng(seed).permutation(n) < n // 2
    gain = 0.
    for a in (half, ~half):
        mp = iso_fit(g[a], e[a])
        b = ~a
        gain += (gate_loss(g[b], e[b], k) - gate_loss(iso_apply(mp, g[b]), e[b], k)).sum()
    gain /= n
    if gain <= 2e-4:
        return None, float(gain)
    return iso_fit(g, e), float(gain)
