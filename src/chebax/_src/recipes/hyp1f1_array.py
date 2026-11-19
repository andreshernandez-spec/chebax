"""Per-element hyp1f1: M(a, b, x) and ln M with a and b per element in [0.1,
10] (nan outside) and x >= 0 (nan below), differentiable in all three. Each
element reads one cell of hyp1f1_array_gen's table:

    x <= X_S:  ln M = log1p((a/b) x e^L), L = ln R tabulated
    x >  X_S:  ln M = (x + U_0) + (U - U_0 - (a - b) ln t), t = X_S/x

X_S is a power of two per (ln a, ln b) base column and U_0 the cell's constant
coefficient (its low part added with the rest), so x and U cancelling next to
the seam where b > a costs one rounding. The cell is a leaf of the base
column's k-d tree, found by DEPTH compares. No lgamma per element; float32
reads the leading NKEEP32 coefficients of each cell and takes ln M from logs
where (a/b) x e^L would overflow. Measured accuracy and speed:
tests/test_hyp1f1_array.py, experiments/56 and 57.
"""

import functools
import math
from collections import defaultdict

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import hyp1f1_array_table as _t
from chebax._src.recipes._common import _basis_and_slope, canon_float, nan_outside
from jax.custom_derivatives import SymbolicZero

_LO, _HI = math.log(_t.ALO), math.log(_t.AHI)
_W0 = (_HI - _LO) / _t.NB0
_PACK_BYTES = 512
# float32 takes ln M from logs past this ln q: (a/b) x e^L overflows at ~88
_BIG32 = 30.0


@functools.lru_cache(maxsize=2)
def _table(dtype):
    """(table, constants' low parts, kept count, nesting, degrees) in the
    compute dtype: nesting lists, for each i, each j with its (column, k)
    pairs, every level in descending order, the constant left out."""
    dt = np.dtype(dtype)
    n = _t.TABLE.shape[1] if dt == np.float64 else _t.NKEEP32
    tab = np.ascontiguousarray(_t.TABLE[:, :n], dt)
    nest = defaultdict(lambda: defaultdict(list))
    for m, (i, j, k) in enumerate(_t.IDX[:n]):
        if m:  # IDX[0] = (0, 0, 0), added last with its low part
            nest[i][j].append((m, k))
    nest = tuple(
        (
            i,
            tuple(
                (j, tuple(sorted(nest[i][j], key=lambda mk: -mk[1])))
                for j in sorted(nest[i], reverse=True)
            ),
        )
        for i in sorted(nest, reverse=True)
    )
    deg = tuple(1 + max(v[a] for v in _t.IDX[:n]) for a in range(3))
    # the constant's low part in the compute dtype: in float32 what rounding
    # the float64 constant dropped (ln M ~ 6 from x ~ 32 near the seam)
    c0 = _t.TABLE[:, 0]
    lo = (c0 - c0.astype(dt).astype(np.float64)) + np.asarray(_t.C0LO)
    return tab, lo.astype(dt), n, nest, deg


@functools.lru_cache(maxsize=2)
def _tree(dtype):
    """Tree arrays (dim, split, left, right), roots [region, ia, ib], per cell
    the lower box corners and 2 / widths, and X_S per base column, in the
    compute dtype."""
    dt = np.dtype(dtype)
    tr = np.asarray(_t.TREE)
    box = np.asarray(_t.BOX).reshape(-1, 3, 2)
    return (
        tr[:, 0].astype(np.int32),
        tr[:, 1].astype(dt),
        tr[:, 2].astype(np.int32),
        tr[:, 3].astype(np.int32),
        np.asarray(_t.ROOT, np.int32),
        box[:, :, 0].astype(dt),
        (2.0 / (box[:, :, 1] - box[:, :, 0])).astype(dt),
        np.asarray(_t.XS_COL, dt).reshape(-1),
    )


def _column(la, lb):
    """Base column of (ln a, ln b) and its seam X_S."""
    nb0 = _t.NB0
    ia = jnp.clip(jnp.floor((la - _LO) / _W0), 0, nb0 - 1).astype(jnp.int32)
    ib = jnp.clip(jnp.floor((lb - _LO) / _W0), 0, nb0 - 1).astype(jnp.int32)
    col = ia * nb0 + ib
    return col, jnp.asarray(_tree(la.dtype.name)[7])[col]


def _locate(col, tail, la, lb, s):
    """Cell of (ln a, ln b, s) in column col's tail or inner tree, the local
    coordinates in [-1, 1] and their slopes in ln a, ln b and s."""
    dim, split, left, right, root, lo, scale, _ = (
        jnp.asarray(v) for v in _tree(la.dtype.name)
    )
    node = root.reshape(-1)[tail.astype(jnp.int32) * _t.NB0**2 + col]
    for _ in range(_t.DEPTH):
        d = dim[node]
        v = jnp.where(d == 0, la, jnp.where(d == 1, lb, s))
        nxt = jnp.where(v >= split[node], right[node], left[node])
        node = jnp.where(d < 0, node, nxt)
    cell = left[node]
    lo_, sc = lo[cell], scale[cell]
    z = [(u - lo_[..., q]) * sc[..., q] - 1.0 for q, u in enumerate((la, lb, s))]
    return cell, z, [sc[..., q] for q in range(3)]


def _patch(cell, za, zb, zs, grad):
    """The cell's series at za, zb, zs in [-1, 1], and with grad its za, zb, zs
    partials: one gather per kept coefficient, summed nested (k in j in i),
    each level from its smallest terms up. The value comes as (constant,
    rest), the constant's low part in rest."""
    tab, c0lo, n, nest, deg = _table(za.dtype.name)
    flat = jnp.asarray(tab).reshape(-1)
    base = cell * n
    c0 = flat[base]
    ta, dta = _basis_and_slope(za, deg[0])
    tb, dtb = _basis_and_slope(zb, deg[1])
    ts, dts = _basis_and_slope(zs, deg[2])
    f = fa = fb = fs = 0.0
    for i, rows in nest:
        a_i = b_i = s_i = 0.0
        for j, terms in rows:
            r = rs = 0.0
            for m, k in terms:
                c = flat[base + m]
                r = r + c * ts[k]
                if grad:
                    rs = rs + c * dts[k]
            a_i = a_i + tb[j] * r
            if grad:
                b_i = b_i + dtb[j] * r
                s_i = s_i + tb[j] * rs
        f = f + ta[i] * a_i
        if grad:
            fa = fa + dta[i] * a_i
            fb = fb + ta[i] * b_i
            fs = fs + ta[i] * s_i
    f = f + jnp.asarray(c0lo)[cell]
    if not grad:
        return c0, f, None, None, None
    # as four results XLA keeps every gathered coefficient for the reverse
    # pass; packed behind a barrier it keeps the four sums (betainc_array)
    if n * tab.dtype.itemsize >= _PACK_BYTES:
        parts = jax.lax.optimization_barrier(jnp.stack((f, fa, fb, fs), axis=-1))
        f, fa, fb, fs = (parts[..., m] for m in range(4))
    return c0, f, fa, fb, fs


def _parts(a, b, x, grad):
    """(ln M, d/da, d/db, d/dx) for broadcast a, b, x, the slopes None
    without grad. Invalid x lanes read x = 1."""
    ok = (x >= 0.0) & (x < jnp.inf)
    xi = jnp.where(ok, x, 1.0)
    la, lb = jnp.log(a), jnp.log(b)
    col, xs = _column(la, lb)
    tail = xi > xs
    t = xs / jnp.where(tail, xi, 2.0 * xs)
    cell, (za, zb, zs), (sa, sb, ss) = _locate(
        col, tail, la, lb, jnp.where(tail, t, xi)
    )
    c0, rest, va, vb, vs = _patch(cell, za, zb, zs, grad)
    v = c0 + rest

    # inner: q = (a/b) x e^L as a product (from logs it lost ~4 eps where L
    # and ln(a x/b) cancel); float32 takes logs where q would overflow
    r = a / b
    if v.dtype == jnp.float64:
        big = jnp.zeros(v.shape, bool)
        e = jnp.exp(v)
        q = r * xi * e
        lm_in = jnp.log1p(q)
    else:
        lq = v + jnp.log(r * jnp.maximum(xi, 1e-30))
        big = lq > _BIG32
        e = jnp.exp(jnp.where(big, 0.0, v))
        q = r * xi * e
        lm_in = jnp.where(big, lq + jnp.log1p(jnp.exp(-lq)), jnp.log1p(q))
    lt = jnp.log(t)
    lm_tl = (xi + c0) + (rest - (a - b) * lt)
    lm = jnp.where(tail, lm_tl, lm_in)
    if not grad:
        return lm, None, None, None
    sig = jnp.where(big, 1.0, q / (1.0 + q))
    # d ln M/dx = (a/b) e^L (1 + x L_x)/(1 + q): from sig/x where q >= 1 or
    # float32's q would overflow (x > 0 there; e is masked to 1 in those lanes)
    small = (q < 1.0) & ~big
    rate = jnp.where(small, r * e / (1.0 + q), sig / jnp.maximum(xi, 1e-30))
    da_in = sig / a * (1.0 + va * sa)
    db_in = sig / b * (-1.0 + vb * sb)
    dx_in = rate * (1.0 + xi * vs * ss)
    da_tl = -lt + va * sa / a
    db_tl = lt + vb * sb / b
    dx_tl = 1.0 + (a - b) / xi - vs * ss * t / xi
    sel = lambda u, w: jnp.where(tail, u, w)
    return lm, sel(da_tl, da_in), sel(db_tl, db_in), sel(dx_tl, dx_in)


def _finish(a, b, x, v, at_inf):
    v = jnp.where(x == jnp.inf, at_inf, v)
    bad = (x < 0.0) | jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b)
    return jnp.where(bad, jnp.nan, v)


def _log_parts(a, b, x, grad):
    lm, da, db, dx = _parts(a, b, x, grad)
    val = _finish(a, b, x, lm, jnp.inf)
    if not grad:
        return val, None, None, None
    return val, *(_finish(a, b, x, d, jnp.nan) for d in (da, db, dx))


def _lin_parts(a, b, x, grad):
    lm, da, db, dx = _parts(a, b, x, grad)
    m = jnp.exp(lm)
    val = _finish(a, b, x, m, jnp.inf)
    if not grad:
        return val, None, None, None
    return val, *(_finish(a, b, x, m * d, jnp.nan) for d in (da, db, dx))


def _rule(parts_fn):
    def rule(primals, tangents):
        on = tuple(not isinstance(t, SymbolicZero) for t in tangents)
        v, da, db, dx = parts_fn(*primals, any(on))
        out = jnp.zeros_like(v)
        for part, tan, use in zip((da, db, dx), tangents, on):
            if use:
                out = out + part * tan
        return v, out

    return rule


@jax.custom_jvp
def _lin(a, b, x):
    return _lin_parts(a, b, x, False)[0]


@jax.custom_jvp
def _log(a, b, x):
    return _log_parts(a, b, x, False)[0]


_lin.defjvp(_rule(_lin_parts), symbolic_zeros=True)
_log.defjvp(_rule(_log_parts), symbolic_zeros=True)


def _args(a, b, x):
    a, b, x = jnp.broadcast_arrays(canon_float(a), canon_float(b), canon_float(x))
    return nan_outside(a, _t.ALO, _t.AHI), nan_outside(b, _t.ALO, _t.AHI), x


def hyp1f1_array(a, b, x):
    """M(a, b, x) with a and b per element in [0.1, 10] (nan outside, with nan
    derivatives), broadcast with x >= 0 (nan below); differentiable in all
    three. Relative error over max(1, ln M): 6.0e-16 in float64, 3.9e-7 in
    float32 (experiments/57)."""
    return _lin(*_args(a, b, x))


def log_hyp1f1_array(a, b, x):
    """ln M(a, b, x) as hyp1f1_array, with no overflow at large x."""
    return _log(*_args(a, b, x))
