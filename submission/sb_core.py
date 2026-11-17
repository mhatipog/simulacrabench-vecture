"""Shared data preparation, proper-score helpers and a thread pool.

Only GIVEN columns are ever used as features. Labels are read from visible
rows only. Nothing here touches the network or the filesystem.
"""
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

FLOOR = 1e-3
NTHREADS = max(1, min(8, os.cpu_count() or 1))


def pmap(fn, items, threads=None):
    items = list(items)
    t = NTHREADS if threads is None else threads
    if t <= 1 or len(items) < 2:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(t) as ex:
        return list(ex.map(fn, items))


def _level_number(label):
    """Numeric meaning of a banded level ('6-10' -> 8, '11+' -> 11) or None."""
    s = str(label).strip()
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*\+?', s)
    if m:
        return float(m.group(1))
    return None


def prepare(frame, schema):
    items = schema['items']
    gv = str(schema['gated_value'])
    given = [n for n, r in items.items() if r['class'] == 'GIVEN']
    targets = [n for n, r in items.items() if r['class'] == 'PREDICT']
    if not targets:
        raise ValueError('No target questions')
    blank = frame[targets].isna().to_numpy().any(axis=1)
    hidden = np.flatnonzero(blank)
    visible = np.flatnonzero(~blank)
    n = len(frame)
    G = max(1, len(given))
    Xc = np.zeros((n, G), np.int32)          # per column: 0 unknown, 1..L
    levels = []
    numeric = {}
    for c, key in enumerate(given):
        opts = [str(z) for z in items[key]['values']]
        if items[key].get('gate'):
            opts.append(gv)
        enc = {z: i + 1 for i, z in enumerate(opts)}
        Xc[:, c] = frame[key].astype('string').map(enc).fillna(0).to_numpy(np.int32)
        levels.append(len(opts))
        vals = [_level_number(z) for z in opts]
        ok = [v is not None for v in vals]
        if sum(ok) >= max(3, int(.6 * len(opts))):
            table = np.array([np.nan] + [v if v is not None else np.nan for v in vals])
            numeric[c] = (table[Xc[:, c]], tuple(opts))
    if not given:
        levels = [1]
    # Offset codes for the embedding network.
    offs = np.cumsum([1] + [L + 2 for L in levels[:-1]])
    Xoff = Xc + offs[None, :].astype(np.int32)
    vocab = int(offs[-1] + levels[-1] + 2)
    # Derived numeric features: totals over groups of count-like columns that
    # share one level list (household roster), difference for exact pairs.
    extra = [v[0] for v in numeric.values()]
    groups = {}
    for c, (col, opts) in numeric.items():
        groups.setdefault(opts, []).append(col)
    for cols in groups.values():
        if len(cols) >= 3:
            extra.append(np.nansum(np.column_stack(cols), axis=1))
        elif len(cols) == 2:
            extra.append(cols[0] - cols[1])
    Xnum = (np.column_stack(extra).astype(np.float32) if extra
            else np.zeros((n, 0), np.float32))
    sizes, ys = [], []
    vis = frame.iloc[visible]
    for key in targets:
        r = items[key]
        opts = [str(z) for z in r['values']] + ([gv] if r.get('gate') else [])
        mp = {z: i for i, z in enumerate(opts)}
        v = vis[key].astype('string').map(mp).fillna(-1).to_numpy(np.int64)
        if (v < 0).any():
            raise ValueError('Unexpected visible response for ' + key)
        ys.append(v)
        sizes.append(len(opts))
    Y = np.column_stack(ys) if len(visible) else np.zeros((0, len(targets)), np.int64)
    gated = np.array([bool(items[k].get('gate')) for k in targets], bool)
    return dict(given=given, targets=targets, hidden=hidden, visible=visible,
                Xc=Xc, Xoff=Xoff, Xnum=Xnum, vocab=vocab, levels=levels,
                Y=Y, C=np.array(sizes, int), gated=gated)


def onehot(Xc, levels):
    """Dense one-hot of the GIVEN block (unknown level -> all zeros)."""
    cols = []
    for c, L in enumerate(levels):
        m = np.zeros((len(Xc), L), np.float32)
        ok = Xc[:, c] > 0
        m[np.flatnonzero(ok), Xc[ok, c] - 1] = 1.
        cols.append(m)
    return np.concatenate(cols, axis=1)


def prior_of(y, k, a=.5):
    c = np.bincount(y, minlength=k).astype(float) + a
    return c / c.sum()


def cell_loss(p_true, k):
    """Scorer's loss for the probability placed on the realised option."""
    return -np.log(FLOOR + (1. - k * FLOOR) * np.clip(p_true, 0., 1.))


def kfold(n, k, seed):
    rng = np.random.default_rng(seed)
    f = np.empty(n, np.int32)
    f[rng.permutation(n)] = np.arange(n) % k
    return f
