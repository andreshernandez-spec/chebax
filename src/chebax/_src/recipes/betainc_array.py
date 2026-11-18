"""Per-element betainc: I_x(a, b) and ln I_x(a, b) with a and b per element
in [0.1, 100] (nan outside), differentiable in a, b and x. Each element reads
one cell of betainc_array_gen's table in one orientation, (p, q, t) = (a, b,
x) up to s = S_SPLIT of (a, b)'s column (x_s = (a + 1)/(a + b + 2) or past it
where I(x_s) is small), else (b, a, 1 - x) with I = 1 - I_t(p, q):

    ln I_t(p, q) = -(p phi(u) + q phi(v)) + T(ln p, ln q, s)

nu = p + q, u = (t nu - p)/p, v = -(t nu - p)/q, phi(z) = z - log1p(z),
s = ln(1 - t)/ln(1 - x_s). No lgamma or digamma per element: the shape slopes
of the first term are log1p(u) = ln(t/x0) and log1p(v).

T moves up to ~7 per unit of s next to the seam at large a, b, so the inputs
to the table are formed to about an ulp: t nu - p = t q - (1 - t) p from
exact products of half-width parts, 1 - x and 1 - x_s from exact sums, s
near 1 from 1 - s, and the patch sum nested smallest-first. Each of these
was 1e-15 or more in ln I before.

Endpoints as in betainc_scalar (I = 0 at x <= 0, 1 at x >= 1, the endpoint
density as the x-slope there). One table read per element whichever the
orientation; float32 reads the leading NKEEP32 coefficients of each cell, and
XLA flushes subnormal float32 x to 0. float32 also turns orientation at
S_SPLIT32 (I ~ 0.5), not S_SPLIT. Measured accuracy and speed:
tests/test_betainc_array.py, experiments/54 and 55.
"""

import functools
import math
from collections import defaultdict

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import betainc_array_table as _t
from chebax._src.recipes._common import (
    _basis_and_slope,
    canon_float,
    edge_slope,
    locate_zones,
    log1p,
    nan_outside,
)
from jax.custom_derivatives import SymbolicZero

_LO, _HI = math.log(_t.ALO), math.log(_t.AHI)
_W = (_HI - _LO) / _t.NAB
# phi's series in w = z/(2 + z) to w^24: tail below 2e-17 for |w| <= 0.2
_W_SER = 0.2
_S = tuple(1.0 if k % 2 == 0 else (k + 1) / (k + 2) for k in range(25))
_PACK_BYTES = 512


@functools.lru_cache(maxsize=2)
def _table(dtype):
    """(table, kept count, nesting, degrees) in the compute dtype: nesting
    lists, for each i, each j with its (column, k) pairs, every level in
    descending order."""
    dt = np.dtype(dtype)
    n = _t.TABLE.shape[1] if dt == np.float64 else _t.NKEEP32
    tab = np.ascontiguousarray(_t.TABLE[:, :n], dt)
    nest = defaultdict(lambda: defaultdict(list))
    for m, (i, j, k) in enumerate(_t.IDX[:n]):
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
    return tab, n, nest, deg


@functools.lru_cache(maxsize=2)
def _s_cells(dtype):
    """Per cell: its s interval's lower cut and 2 / width (d zs/ds)."""
    lo, scale = [], []
    for col, n in enumerate(_t.NS):
        cuts = _t.SCUTS[col]
        lo += [cuts[l] for l in range(n)]
        scale += [2.0 / (cuts[l + 1] - cuts[l]) for l in range(n)]
    dt = np.dtype(dtype)
    return np.asarray(lo, dt), np.asarray(scale, dt)


def tv_dtype(v):
    return jnp.result_type(v).name


def _patch(cell, za, zb, zs, grad):
    """T on cell at local coordinates za, zb, zs in [-1, 1], and with grad its
    za, zb, zs partials: one gather per kept coefficient, summed nested (k in
    j in i), each level from its smallest terms up. Summed flat, the ~650 terms
    lost 5-7 eps to rounding in ln I."""
    tab, n, nest, deg = _table(za.dtype.name)
    flat = jnp.asarray(tab).reshape(-1)
    base = cell * n
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
    if not grad:
        return f, None, None, None
    # As four results XLA keeps every gathered coefficient for the reverse
    # pass; packed behind a barrier it keeps the four sums (as patch_sum)
    if n * tab.dtype.itemsize >= _PACK_BYTES:
        parts = jax.lax.optimization_barrier(jnp.stack((f, fa, fb, fs), axis=-1))
        f, fa, fb, fs = (parts[..., m] for m in range(4))
    return f, fa, fb, fs


def _phi(z):
    """(phi(z), log1p(z)) for z > -1/2: series in w = z/(2+z) near 0."""
    w = z / (2.0 + z)
    ser = jnp.abs(w) <= _W_SER
    ws = jnp.where(ser, w, 0.0)
    acc = _S[-1]
    for c in _S[-2::-1]:
        acc = acc * ws + c
    phi_s = 2.0 * ws * ws * acc
    zd = jnp.where(ser, 1.0, z)
    l1 = log1p(zd)
    return jnp.where(ser, phi_s, zd - l1), jnp.where(ser, z - phi_s, l1)


def _hi(a):
    """a with the low half of its significand cleared, by bits: Veltkamp's
    split c a - (c a - a) is undone when XLA contracts it into an fma (d was
    2e-14 off under jit, 2e-16 eagerly)."""
    wide = a.dtype == jnp.float64
    it, bits = (jnp.int64, 27) if wide else (jnp.int32, 12)
    mask = ~((1 << bits) - 1)
    i = jax.lax.bitcast_convert_type(a, it)
    return jax.lax.bitcast_convert_type(i & jnp.asarray(mask, it), a.dtype)


def _diff_prods(t, q, u, p):
    """t q - u p from products of half-width parts, each exact: no step relies
    on a product being rounded, so XLA fusing a product into a sum (which made
    the textbook two-product 2e-14 off under jit) changes nothing. Error
    ~eps |t q - u p| plus eps^2 |t q|."""
    th, qh, uh, ph = _hi(t), _hi(q), _hi(u), _hi(p)
    tl, ql, ul, pl = t - th, q - qh, u - uh, p - ph
    return (
        (th * qh - uh * ph)
        + ((th * ql + tl * qh) - (uh * pl + ul * ph))
        + (tl * ql - ul * pl)
    )


def _two_sum(a, b):
    """(s, e) with s + e = a + b exactly (Knuth)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _one_minus_xs(p, q):
    """1 - x_s = (q + 1)/(p + q + 2) as rh + rl, from exact sums and the
    division's exact residual."""
    nh, nl = _two_sum(q, jnp.ones_like(q))
    sh, sl = _two_sum(p, q)
    mh, ml = _two_sum(sh, 2.0 * jnp.ones_like(sh))
    ml = ml + sl
    rh = nh / mh
    res = _diff_prods(nh, jnp.ones_like(nh), rh, mh)  # nh - rh mh
    return rh, (res + nl - rh * ml) / mh


def _col(la, lb):
    """Column of (ln p, ln q), and its local coordinates in [-1, 1]."""
    fa = jnp.clip(jnp.floor((la - _LO) / _W), 0.0, _t.NAB - 1.0)
    fb = jnp.clip(jnp.floor((lb - _LO) / _W), 0.0, _t.NAB - 1.0)
    za = 2.0 * (la - (_LO + fa * _W)) / _W - 1.0
    zb = 2.0 * (lb - (_LO + fb * _W)) / _W - 1.0
    return fa.astype(jnp.int32) * _t.NAB + fb.astype(jnp.int32), za, zb


def _lane(a, b, x):
    """Per lane: the orientation, its (p, q, t), 1 - t (with the low part of
    its exact value), ln t, ln(1 - t), d = t q - (1 - t) p to a few ulps of d,
    1 - x_s as rh + rl and ln(1 - x_s), the column and its coordinates. The
    lower orientation runs to s = S_SPLIT of (a, b)'s column. Invalid lanes
    get x = 1/4. float32 lanes turn at S_SPLIT32 instead."""
    inside = (x > 0.0) & (x < 1.0)
    xi = jnp.where(inside, x, 0.25)
    # 1 - x = y + ey exactly (Fast2Sum, |1| >= |x|); the barrier stops XLA
    # folding (1 - x) - 1 into -x, which zeroed ey under jit (d 2e-14 off)
    y = jax.lax.optimization_barrier(1.0 - xi)
    ey = -xi - (y - 1.0)
    # ln(1 - x): XLA's float64 log1p is off on (-0.63, -0.25) before jax 0.12,
    # and log(y) misses ey; from the exact parts instead
    l1x = jnp.where(xi < 0.25, log1p(-xi), jnp.log(y) + ey / y)
    lx = jnp.log(xi)
    # ln(1 - x_s) to ~an ulp in both orientations: T moves up to ~7 per unit of
    # s next to the seam at large p, q, so plain rounding (5 ulps) was 2e-15
    rh_ab, rl_ab = _one_minus_xs(a, b)
    rh_ba, rl_ba = _one_minus_xs(b, a)
    ls_ab = jnp.log(rh_ab) + rl_ab / rh_ab
    la, lb = jnp.log(a), jnp.log(b)
    col_ab, za_ab, zb_ab = _col(la, lb)
    # float32 turns near I = 0.5: past it T cancels an O(1) first term
    turn = _t.S_SPLIT if xi.dtype == jnp.float64 else _t.S_SPLIT32
    split = jnp.asarray(turn, xi.dtype)[col_ab]
    lower = l1x >= split * ls_ab  # s <= S_SPLIT, ls < 0
    sel = lambda u, v: jnp.where(lower, u, v)
    p, q = sel(a, b), sel(b, a)
    t_hi, t_lo = sel(xi, y), sel(0.0, ey)
    u_hi, u_lo = sel(y, xi), sel(ey, 0.0)  # 1 - t
    d = _diff_prods(t_hi, q, u_hi, p) + (t_lo * q - u_lo * p)
    rh, rl = sel(rh_ab, rh_ba), sel(rl_ab, rl_ba)
    ls = sel(ls_ab, jnp.log(rh_ba) + rl_ba / rh_ba)
    col_ba, za_ba, zb_ba = _col(lb, la)
    col, za, zb = sel(col_ab, col_ba), sel(za_ab, za_ba), sel(zb_ab, zb_ba)
    lt, l1t = sel(lx, l1x), sel(l1x, lx)
    return dict(
        lower=lower,
        p=p,
        q=q,
        t=t_hi,
        t1=u_hi,
        t1_lo=u_lo,
        lt=lt,
        l1t=l1t,
        d=d,
        rh=rh,
        rl=rl,
        ls=ls,
        col=col,
        za=za,
        zb=zb,
        split=sel(split, 1.0),
    )


def _log1p_far(z, prod, l_fallback):
    """(phi(z), log1p(z)): below z = -1/2 log1p(z) is ln(prod), prod = 1 + z
    formed as a product (from z, 1 + z loses ~eps/(1 + z)), or l_fallback
    where prod underflows."""
    far = z <= -0.5
    phi, l1 = _phi(jnp.where(far, 0.0, z))
    tiny = jnp.finfo(prod.dtype).tiny
    lp = jnp.where(prod >= tiny, jnp.log(jnp.maximum(prod, tiny)), l_fallback)
    l1 = jnp.where(far, lp, l1)
    return jnp.where(far, z - l1, phi), l1


def _parts(a, b, x, need):
    """(ln I_t, its p, q, t slopes, lower, t, 1 - t, ln t) for broadcast a, b,
    x; slopes None where need = (need_p or need_q, need_t) says no."""
    L = _lane(a, b, x)
    p, q, t, t1, lt, l1t, d, ls = (
        L[k] for k in ("p", "q", "t", "t1", "lt", "l1t", "d", "ls")
    )
    nu = p + q
    s = jnp.clip(l1t / ls, 0.0, L["split"])
    # near s = 1, sigma = 1 - s = ln((1 - x_s)/(1 - t))/ln(1 - x_s) from an
    # exact difference: a rounded s carries 1/2 ulp (6e-17) times that slope
    gap = (L["rh"] - t1) + (L["rl"] - L["t1_lo"])
    sigma = jnp.clip(log1p(gap / t1) / ls, 1.0 - L["split"], 1.0)

    col = L["col"]
    l, zs = locate_zones(s, _t.SCUTS, _t.NS, col)
    cell = jnp.asarray(_t.BASE, jnp.int32)[col] + l
    c0, scale = (jnp.asarray(v)[cell] for v in _s_cells(tv_dtype(p)))
    zs = jnp.where(s > 0.5, ((1.0 - c0) - sigma) * scale - 1.0, zs)
    grad = need[0] or need[1]
    tv, ta_, tb_, ts_ = _patch(cell, L["za"], L["zb"], zs, grad)

    # log1p(u) = ln(t nu/p) and log1p(v) = ln((1 - t) nu/q) below -1/2, as one
    # log of the product (ln t + ln(nu/p) cancelled: 3e-7 in float32 at a = 1,
    # b = 68), from two logs only where the product underflows
    u, v = d / p, -d / q
    phi_u, lu = _log1p_far(u, t * (nu / p), lt + jnp.log(nu / p))
    phi_v, lv = _log1p_far(v, t1 * (nu / q), l1t + jnp.log(nu / q))
    lnit = -(p * phi_u + q * phi_v) + tv
    lower = L["lower"]
    if not grad:
        return lnit, None, None, None, lower, t, t1, lt
    t_s = ts_ * scale
    ds_p = s / (ls * (nu + 2.0))
    ds_q = -(s / ls) * (1.0 / (q + 1.0) - 1.0 / (nu + 2.0))
    ds_t = -1.0 / (t1 * ls)
    dp = lu + ta_ * (2.0 / _W) / p + t_s * ds_p
    dq = lv + tb_ * (2.0 / _W) / q + t_s * ds_q
    # t times d ln I_t/dt: positive, finite at t -> 0
    gt = p - q * t / t1 + t * t_s * ds_t
    return lnit, dp, dq, gt, lower, t, t1, lt


def _finish(a, b, x, v, lo, hi):
    v = jnp.where(x <= 0.0, lo, jnp.where(x >= 1.0, hi, v))
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, v)


def _cdf_parts(a, b, x, need):
    lnit, dp, dq, gt, lower, t, t1, lt = _parts(a, b, x, need)
    e = jnp.exp(lnit)
    val = _finish(a, b, x, jnp.where(lower, e, -jnp.expm1(lnit)), 0.0, 1.0)
    if not (need[0] or need[1]):
        return val, None, None, None
    sgn = jnp.where(lower, e, -e)
    da = sgn * jnp.where(lower, dp, dq)
    db = sgn * jnp.where(lower, dq, dp)
    # dI/dx = e d ln I_t/dt (sign: dt/dx = -1 above, and I = 1 - I_t), from logs:
    # e times p/t is 0 * inf where I_t underflows and t is tiny
    dens = jnp.exp(lnit + jnp.log(jnp.maximum(gt, 0.0)) - lt)
    z = (0.0, 0.0)
    return (
        val,
        _finish(a, b, x, da, *z),
        _finish(a, b, x, db, *z),
        _finish(a, b, x, dens, *z),
    )


def _log_parts(a, b, x, need):
    lnit, dp, dq, gt, lower, t, t1, lt = _parts(a, b, x, need)
    up = jnp.log(-jnp.expm1(lnit))  # ln(1 - I_t)
    val = _finish(a, b, x, jnp.where(lower, lnit, up), -jnp.inf, 0.0)
    if not (need[0] or need[1]):
        return val, None, None, None
    e = jnp.exp(lnit)
    r = -e / -jnp.expm1(lnit)  # d ln(1 - I_t)/d ln I_t
    f = jnp.where(lower, 1.0, r)
    da = f * jnp.where(lower, dp, dq)
    db = f * jnp.where(lower, dq, dp)
    # d ln I/dx: gt/t below; above, the density over 1 - I_t
    dlx = jnp.where(lower, gt / t, -r * gt / t)
    z = (0.0, 0.0)
    return (
        val,
        _finish(a, b, x, da, *z),
        _finish(a, b, x, db, *z),
        _finish(a, b, x, dlx, *z),
    )


def _rule(parts_fn):
    def rule(primals, tangents):
        a, b, x = primals
        on = tuple(not isinstance(t, SymbolicZero) for t in tangents)
        need = (on[0] or on[1] or on[2], on[2])
        v, da, db, dx = parts_fn(a, b, x, need)
        out = jnp.zeros_like(v)
        for part, tan, use in zip((da, db, dx), tangents, on):
            if use:
                out = out + part * tan
        return v, out

    return rule


@jax.custom_jvp
def _cdf(a, b, x):
    return _cdf_parts(a, b, x, (False, False))[0]


@jax.custom_jvp
def _log_cdf(a, b, x):
    return _log_parts(a, b, x, (False, False))[0]


_cdf.defjvp(_rule(_cdf_parts), symbolic_zeros=True)
_log_cdf.defjvp(_rule(_log_parts), symbolic_zeros=True)


def _edge_density(a, b, x):
    """The Beta density at an endpoint, 0 in between (as betainc's)."""
    d0 = jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, b))
    d1 = jnp.where(b < 1.0, jnp.inf, jnp.where(b > 1.0, 0.0, a))
    return jnp.where(x == 0.0, d0, jnp.where(x == 1.0, d1, 0.0))


def _args(a, b, x):
    a, b, x = jnp.broadcast_arrays(canon_float(a), canon_float(b), canon_float(x))
    return nan_outside(a, _t.ALO, _t.AHI), nan_outside(b, _t.ALO, _t.AHI), x


def betainc_array(a, b, x):
    """I_x(a, b) with a and b per element in [0.1, 100] (nan outside, with nan
    derivatives), broadcast with x; differentiable in all three. Absolute
    error 6.7e-16 in float64 (tests/test_betainc_array.py), 2.3e-7 in float32
    (experiments/55); for the upper tail to relative accuracy use
    betainc_array(b, a, 1 - x)."""
    a, b, x = _args(a, b, x)
    return _cdf(a, b, x) + edge_slope(_edge_density(a, b, x), x)


def log_betainc_array(a, b, x):
    """ln I_x(a, b) as betainc_array, no underflow floor: 7.2e-16 in float64
    and 4.1e-7 in float32 relative to max(1, |ln I|) (tests/test_betainc_array.py,
    experiments/55). On the reflected side, where I >= ~0.45, it is
    log1p(-I_t); for the upper tail use log_betainc_array(b, a, 1 - x)."""
    a, b, x = _args(a, b, x)
    # d ln I/dx = density/I is inf at x = 0 and the density at x = 1
    slope = jnp.where(
        x == 0.0, jnp.inf, jnp.where(x == 1.0, _edge_density(a, b, x), 0.0)
    )
    return _log_cdf(a, b, x) + edge_slope(slope, x)
