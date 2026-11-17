"""Base experts. Each returns one (n_query, K_j) matrix per requested target.

direct  : full categorical answer including the gate sentinel
gate    : P(sentinel | GIVEN) from a dedicated binary model
cond    : P(answer | asked, GIVEN), fitted on asked rows only
"""
import warnings
import numpy as np
import torch
from sb_core import pmap, prior_of

LGB_DIRECT = dict(n_estimators=96, learning_rate=.055, num_leaves=11,
                  min_child_samples=65, reg_lambda=15., reg_alpha=.18,
                  min_split_gain=.001, colsample_bytree=.95)
LGB_GATE = dict(n_estimators=160, learning_rate=.07, num_leaves=15,
                min_child_samples=20, reg_lambda=2., colsample_bytree=.95)


def _lgb(Xtr, ytr, k, Xq, ncat, params, seed, eps=.01):
    from lightgbm import LGBMClassifier
    prior = prior_of(ytr, k)
    if len(np.unique(ytr)) < 2 or len(ytr) < 25:
        return np.broadcast_to(prior, (len(Xq), k)).copy()
    p = dict(params)
    p['min_child_samples'] = int(max(5, min(p['min_child_samples'], len(ytr) // 12)))
    model = LGBMClassifier(n_jobs=1, verbosity=-1, random_state=seed,
                           max_bin=63, **p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(Xtr, ytr, categorical_feature=list(range(ncat)))
        raw = model.predict_proba(Xq)
    z = np.broadcast_to(prior * eps, (len(Xq), k)).copy()
    z[:, model.classes_.astype(int)] += (1. - eps) * raw
    return z / z.sum(1, keepdims=True)


def lgb_direct(XL, ncat, Y, C, tr, Xq, seed, targets):
    Xtr = XL[tr]
    return pmap(lambda j: _lgb(Xtr, Y[tr, j], int(C[j]), Xq, ncat,
                               LGB_DIRECT, seed + j), targets)


def lgb_cond(XL, ncat, Y, C, tr, Xq, seed, targets):
    """Answer model on asked rows only; K-1 real options."""
    def one(j):
        k = int(C[j]) - 1
        rows = tr[Y[tr, j] < k]
        if k < 2:
            return np.ones((len(Xq), 1))
        if len(rows) < 25:
            return np.broadcast_to(prior_of(Y[rows, j], k), (len(Xq), k)).copy()
        return _lgb(XL[rows], Y[rows, j], k, Xq, ncat, LGB_DIRECT, seed + 7 + j)
    return pmap(one, targets)


def lgb_gate(XL, ncat, NA, tr, Xq, seed):
    """One sharp binary model per distinct skip pattern (columns of NA)."""
    def one(c):
        y = NA[tr, c].astype(int)
        return _lgb(XL[tr], y, 2, Xq, ncat, LGB_GATE, seed + 13 + c, eps=.002)[:, 1]
    return pmap(one, range(NA.shape[1]))


def cat_direct(Xc, Y, C, tr, Xq, seed, targets, iterations=86):
    from catboost import CatBoostClassifier
    xtr = Xc[tr].astype(str)
    xq = Xq.astype(str)
    feats = list(range(xtr.shape[1]))

    def one(j):
        k = int(C[j]); y = Y[tr, j]; prior = prior_of(y, k)
        if len(np.unique(y)) < 2 or len(y) < 30:
            return np.broadcast_to(prior, (len(xq), k)).copy()
        m = CatBoostClassifier(iterations=iterations, depth=5, learning_rate=.058,
                               l2_leaf_reg=7., random_strength=.8,
                               loss_function='MultiClass' if k > 2 else 'Logloss',
                               thread_count=1, verbose=False,
                               allow_writing_files=False, random_seed=seed + j,
                               one_hot_max_size=12, bootstrap_type='Bernoulli',
                               subsample=.85)
        m.fit(xtr, y, cat_features=feats)
        raw = m.predict_proba(xq)
        z = np.broadcast_to(prior * .01, (len(xq), k)).copy()
        z[:, np.asarray(m.classes_).astype(int)] += .99 * raw
        return z / z.sum(1, keepdims=True)
    return pmap(one, targets)


def linear_experts(H, Y, C, gated, tr, Hq, seed, ridge=.8, steps=140):
    """Multinomial logistic regression for every target in one linear map.

    Returns (direct, cond). cond for a gated target is fitted on asked rows
    and excludes the sentinel; for ungated targets cond is None. Heads are
    laid out grouped by option count so each group is one contiguous block.
    """
    torch.manual_seed(seed)
    C = [int(c) for c in C]
    J = len(C)
    heads = [('d', j, C[j]) for j in range(J)]
    heads += [('c', j, C[j] - 1) for j in range(J) if gated[j] and C[j] > 2]
    heads.sort(key=lambda h: h[2])
    blocks, start, pos = [], 0, 0
    while pos < len(heads):
        k = heads[pos][2]; end = pos
        while end < len(heads) and heads[end][2] == k:
            end += 1
        blocks.append((start, k, heads[pos:end])); start += k * (end - pos); pos = end
    tot = start
    b0 = np.zeros(tot, np.float32)
    ys = []
    for a, k, hs in blocks:
        yb = np.full((len(tr), len(hs)), -100, np.int64)
        for i, (kind, j, _) in enumerate(hs):
            yy = Y[tr, j]
            if kind == 'c':
                asked = yy < k
                yb[asked, i] = yy[asked]; pr = prior_of(yy[asked], k, 1.)
            else:
                yb[:, i] = yy; pr = prior_of(yy, k, 1.)
            b0[a + i * k:a + (i + 1) * k] = np.log(pr)
        ys.append(torch.as_tensor(yb))
    x = torch.as_tensor(H[tr]); xq = torch.as_tensor(Hq)
    W = torch.zeros(x.shape[1], tot, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=.07)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ce = torch.nn.functional.cross_entropy
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        z = x @ W + b
        loss = (ridge / len(tr)) * (W * W).sum()
        for (a, k, hs), yb in zip(blocks, ys):
            zb = z[:, a:a + k * len(hs)].reshape(len(tr), len(hs), k)
            loss = loss + ce(zb.transpose(1, 2), yb, ignore_index=-100,
                             reduction='sum') / len(tr)
        loss.backward()
        opt.step(); sched.step()
    direct, cond = [None] * J, [None] * J
    for j in range(J):
        if gated[j] and C[j] <= 2:
            cond[j] = np.ones((len(Hq), max(1, C[j] - 1)))
    with torch.no_grad():
        z = xq @ W + b
        for a, k, hs in blocks:
            p = torch.softmax(z[:, a:a + k * len(hs)].reshape(len(Hq), len(hs), k), 2)
            p = p.numpy().astype(np.float64)
            for i, (kind, j, _) in enumerate(hs):
                (direct if kind == 'd' else cond)[j] = p[:, i, :]
    return direct, cond


MATCH_BETAS = (.1, .3)


def match_experts(H, Y, C, tr, Hq, mass=.02, chunk=2048):
    """Kernel nearest-neighbour experts on the GIVEN block.

    Weight of a visible row is beta ** (number of GIVEN fields that differ),
    so a row with an identical or near-identical background dominates when
    one exists and the estimate falls back to the marginal when none does.
    Uses GIVEN fields and visible answers only. Returns one list per beta.
    """
    C = [int(c) for c in C]
    off = np.cumsum([0] + C)
    G = float(H[:1].sum()) if len(H) else 1.
    G = float(np.max(H.sum(1))) if len(H) else 1.
    Yoh = np.zeros((len(tr), off[-1]), np.float32)
    for j in range(len(C)):
        Yoh[np.arange(len(tr)), off[j] + Y[tr, j]] = 1.
    priors = [prior_of(Y[tr, j], C[j]) for j in range(len(C))]
    Ht = torch.as_tensor(H[tr]); Yt = torch.as_tensor(Yoh)
    outs = [[np.empty((len(Hq), c)) for c in C] for _ in MATCH_BETAS]
    with torch.no_grad():
        for a in range(0, len(Hq), chunk):
            d = G - torch.as_tensor(Hq[a:a + chunk]) @ Ht.T
            for b, beta in enumerate(MATCH_BETAS):
                W = torch.exp(d * float(np.log(beta)))
                num = (W @ Yt).numpy().astype(np.float64)
                den = W.sum(1).numpy().astype(np.float64)[:, None]
                for j in range(len(C)):
                    outs[b][j][a:a + chunk] = ((num[:, off[j]:off[j + 1]] + mass * priors[j][None, :])
                                               / (den + mass))
    return outs
