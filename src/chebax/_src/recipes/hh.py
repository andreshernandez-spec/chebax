"""The Hermite function Hh_nu(z) = (1/nu!) int_0^inf u^nu e^(-(u + z)^2/2) du =
e^(-z^2/4) D_{-nu-1}(z), for real nu >= 0.1 (nan below) and real z, broadcast
together: hh and log_hh. Each element reads one 8 x 8 patch of hh_gen's table:

    ln Hh = nu (1 - ln v) - v^2/2 - ln(v R)/2 + T(p, sigma),
    R = sqrt(z^2 + 4 nu),  v = (z + R)/2 (= 2 nu/(R - z) below z = 0),

sigma = z/sqrt(z^2 + 4(nu + 1)), p = ln nu up to nu = 10 and 10/nu above, so
nu reaches infinity. No lgamma per element. The first part is formed in
double-double: where ln Hh is near 0 at large nu, nu and nu ln v cancel and
plain float64 lost ~nu eps (1.9e-13 at nu = 1e3). Derivatives are closed
forms, d/dz = -v (1 + 1/R^2) + T_z and d/dnu = -ln v - 1/(2 v R) - 1/R^2 +
T_nu (dHh/dz = -Hh_{nu-1}(z)); float32 reads the leading K32 x K32 block.
Measured accuracy and speed: tests/test_hh.py, experiments/58.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import hh_table as _t
from chebax._src.recipes._common import (
    _PACK_BYTES,
    _packed,
    _patch_parts,
    _unpacked,
    canon_float,
    nan_outside,
)
from jax.custom_derivatives import SymbolicZero

_LNLO, _LN1 = math.log(_t.NU_LO), math.log(_t.NU1)
_WL = (_LN1 - _LNLO) / _t.NP_L
_NS = len(_t.SCUTS) - 1


@functools.lru_cache(maxsize=2)
def _arrays(dtype):
    """Patches (float32: the leading K32 x K32 block), tree, roots and each
    patch's lower corner and 2 / widths, in the compute dtype."""
    dt = np.dtype(dtype)
    tab = _t.TABLE if dt == np.float64 else _t.TABLE[:, : _t.K32, : _t.K32]
    tr = np.asarray(_t.TREE)
    box = np.asarray(_t.BOX).reshape(-1, 2, 2)
    root = np.concatenate([np.asarray(r, np.int32).reshape(-1) for r in _t.ROOT])
    return (
        np.ascontiguousarray(tab, dt),
        tr[:, 0].astype(np.int32),
        tr[:, 1].astype(dt),
        tr[:, 2].astype(np.int32),
        tr[:, 3].astype(np.int32),
        root,
        box[:, :, 0].astype(dt),
        (2.0 / (box[:, :, 1] - box[:, :, 0])).astype(dt),
    )


# Double-double pieces for the first term: near ln Hh = 0 at large nu (|z|
# ~ nu/e) its parts nu and nu ln v cancel, and the rounding of v and ln v came
# back times nu (1.9e-13 at nu = 1e3). Products come from exact half-width
# parts, so no step relies on a product being rounded (XLA contracts products
# into fmas under jit; betainc_array).


def _hi(a):
    """a with the low half of its significand cleared, by bits."""
    wide = a.dtype == jnp.float64
    it, bits = (jnp.int64, 27) if wide else (jnp.int32, 12)
    i = jax.lax.bitcast_convert_type(a, it)
    return jax.lax.bitcast_convert_type(
        i & jnp.asarray(~((1 << bits) - 1), it), a.dtype
    )


def _fast_two_sum(a, b):
    """(s, e), s + e = a + b exactly, for |a| >= |b|."""
    s = a + b
    return s, b - (s - a)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _prod(a, b):
    """a b as (hi, lo) from exact partial products."""
    ah, bh = _hi(a), _hi(b)
    al, bl = a - ah, b - bh
    s, e = _fast_two_sum(ah * bh, ah * bl + al * bh)
    return _fast_two_sum(s, e + al * bl)


def _dd_sqrt(sh, sl):
    r = jnp.sqrt(sh)
    ph, pl = _prod(r, r)
    return _fast_two_sum(r, ((sh - ph) - pl + sl) / (2.0 * r))


def _dd_div(n, dh, dl):
    """n / (dh + dl) for a double n, as (hi, lo)."""
    q = n / dh
    ph, pl = _prod(q, dh)
    return _fast_two_sum(q, ((n - ph) - pl - q * dl) / dh)


@functools.lru_cache(maxsize=2)
def _log_consts(dtype):
    """(ln 2 hi, lo, ln(j/64) hi, lo) in the compute dtype; float32's ln 2 hi
    keeps 16 bits, so k hi is exact."""
    dt = np.dtype(dtype)
    hi64, lo64 = _t.LN2
    if dt == np.float64:
        return hi64, lo64, np.asarray(_t.LNC_HI), np.asarray(_t.LNC_LO)
    b = np.array([hi64], np.float32).view(np.int32)[0] & ~np.int32((1 << 8) - 1)
    h32 = np.array([b]).view(np.float32)[0]
    l32 = np.float32((hi64 - np.float64(h32)) + lo64)
    c_hi = np.asarray(_t.LNC_HI).astype(np.float32)
    c_lo = (np.asarray(_t.LNC_HI) - c_hi.astype(np.float64)) + np.asarray(_t.LNC_LO)
    return np.float32(h32), l32, c_hi, c_lo.astype(np.float32)


# ln(1 + r) - r for |r| <= 1/90: -r^2/2 + r^3/3 - ... to r^11, below 1e-22
_LOG1P = tuple((-1.0) ** (n + 1) / n for n in range(2, 12))


def _dd_log(xh, xl):
    """ln(xh + xl) as (hi, lo) for xh > 0 normal: 2^k c (1 + r), c = j/64 the
    mantissa to 1/64, ln c from the table, ln(1 + r) by its series."""
    ln2_hi, ln2_lo, c_hi, c_lo = (jnp.asarray(v) for v in _log_consts(xh.dtype.name))
    m, k = jnp.frexp(xh)
    low = m < math.sqrt(0.5)
    m = jnp.where(low, 2.0 * m, m)
    k = (k - low).astype(xh.dtype)
    j = jnp.round(m * 64.0)
    c = j / 64.0
    d = m - c  # exact: c is within 1/128 of m
    r = d / c
    ph, pl = _prod(r, c)
    r_lo = ((d - ph) - pl) / c
    poly = _LOG1P[-1]
    for a in _LOG1P[-2::-1]:
        poly = poly * r + a
    poly = poly * r * r
    idx = j.astype(jnp.int32) - _t.LNC_J0
    s, e = _two_sum(k * ln2_hi, c_hi[idx])
    s, e2 = _two_sum(s, r)
    lo = e + e2 + (k * ln2_lo + c_lo[idx] + (r_lo * (1.0 - r) + poly) + xl / xh)
    return _fast_two_sum(s, lo)


def _first(n, x):
    """(nu (1 - ln v) - v^2/2 - ln(v R)/2, v, ln v, R) with the first value to
    a few ulps of the result, from v and ln v in double-double; past the range
    where z^2 is finite in the compute dtype, from plain arithmetic (|ln Hh| >=
    ~z^2/2 or nu ln|z| there, so the plain rounding is relatively small)."""
    big = 2.0**400 if x.dtype == jnp.float64 else 2.0**50
    tame = (jnp.abs(x) < big) & (n < big)
    xt, nt = jnp.where(tame, x, 1.0), jnp.where(tame, n, 1.0)
    zh, zl = _prod(xt, xt)
    sh, se = _two_sum(zh, 4.0 * nt)
    rh, rl = _dd_sqrt(sh, se + zl)
    # v = (z + R)/2 above z = 0, 2 nu/(R - z) below (no cancellation either way)
    a, ae = _two_sum(xt, rh)
    up_h, up_l = _fast_two_sum(0.5 * a, 0.5 * (ae + rl))
    dh, de = _two_sum(rh, -xt)
    dn_h, dn_l = _dd_div(2.0 * nt, dh, de + rl)
    pos = xt >= 0.0
    vh, vl = jnp.where(pos, up_h, dn_h), jnp.where(pos, up_l, dn_l)
    lh, ll = _dd_log(vh, vl)
    nlh, nll = _prod(lh, nt)
    nll = nll + ll * nt
    v2h, v2l = _prod(vh, vh)
    v2l = v2l + 2.0 * vh * vl
    s1, e1 = _two_sum(nt, -nlh)
    s2, e2 = _two_sum(s1, -0.5 * v2h)
    lo = (e1 + e2) - nll - 0.5 * v2l
    # ln(v R) = ln(nu + v^2), whose rounding grows with ln nu
    wh, we = _two_sum(nt, v2h)
    wh, we = _fast_two_sum(wh, we + v2l)
    gh, gl = _dd_log(wh, we)
    s2, e3 = _two_sum(s2, -0.5 * gh)
    lo = lo + e3 - 0.5 * gl
    dd = s2 + lo
    # plain path
    r = jnp.hypot(x, 2.0 * jnp.sqrt(n))
    v = jnp.where(x >= 0.0, 0.5 * (x + r), 2.0 * n / (r - x))
    lv = jnp.log(v)
    plain = n * (1.0 - lv) - 0.5 * v * v - 0.5 * (lv + jnp.log(r))
    sel = lambda p, q: jnp.where(tame, p, q)
    return (
        sel(dd, plain),
        sel(vh, v),
        sel(lh, lv),
        sel(rh, r),
        sel(s2, plain),
        sel(lo, 0.0),
    )


def _locate(high, p, sg):
    """Patch of (p, sigma) in zone H (high) or L, its local coordinates in
    [-1, 1] and their slopes in p and sigma."""
    _, dim, split, left, right, root, lo, scale = (
        jnp.asarray(v) for v in _arrays(p.dtype.name)
    )
    ip = jnp.clip(jnp.floor((p - _LNLO) / _WL), 0, _t.NP_L - 1).astype(jnp.int32)
    isg = jnp.zeros(sg.shape, jnp.int32)
    for cut in _t.SCUTS[1:-1]:
        isg = isg + (sg >= cut)
    node = root[jnp.where(high, _t.NP_L * _NS + isg, ip * _NS + isg)]
    for _ in range(_t.DEPTH):
        d = dim[node]
        u = jnp.where(d == 0, p, sg)
        nxt = jnp.where(u >= split[node], right[node], left[node])
        node = jnp.where(d < 0, node, nxt)
    cell = left[node]
    lo_, sc = lo[cell], scale[cell]
    return (
        cell,
        (p - lo_[..., 0]) * sc[..., 0] - 1.0,
        (sg - lo_[..., 1]) * sc[..., 1] - 1.0,
        sc,
    )


def t_parts(n, x, grad):
    """(T, dT/dnu, dT/dz) at finite nu = n >= 0.1 and z = x, the slopes None
    without grad: the table's part of ln Hh, bounded (|T| < 0.8)."""
    high = n > _t.NU1
    p = jnp.where(high, _t.NU1 / n, jnp.log(n))
    q = jnp.hypot(x, 2.0 * jnp.sqrt(n + 1.0))
    sg = x / q
    cell, yp, ys, sc = _locate(high, p, sg)
    tab = _arrays(p.dtype.name)[0]
    f, fp, fs = _patch_parts(tab, cell, yp, ys, grad)
    if not grad:
        return f, None, None
    # as three results XLA keeps the gathered coefficients for the reverse pass
    pack = _packed if tab[0].size * tab.dtype.itemsize >= _PACK_BYTES else _unpacked
    f, fp, fs = pack(f, fp, fs)
    t_p = fp * sc[..., 0] * jnp.where(high, -p / n, 1.0 / n)
    t_s = fs * sc[..., 1]
    return f, t_p - t_s * (2.0 * sg / (q * q)), t_s * (1.0 - sg * sg) / q


def _parts(nu, z, grad):
    """(ln Hh, d/dnu, d/dz), the slopes None without grad. Lanes with nu or z
    nonfinite read nu = 1, z = 0."""
    ok = jnp.isfinite(nu) & jnp.isfinite(z)
    n = jnp.where(ok, nu, 1.0)
    x = jnp.where(ok, z, 0.0)
    _, v, lv, r, f_hi, f_lo = _first(n, x)
    f, t_nu, t_z = t_parts(n, x, grad)
    val = f_hi + (f_lo + f)
    if not grad:
        return val, None, None
    rr = 1.0 / (r * r)
    d_nu = -lv - 0.5 / (v * r) - rr + t_nu
    d_z = -v * (1.0 + rr) + t_z
    return val, d_nu, d_z


def _finish(nu, z, v, at_minus_inf, at_inf, at_nu_inf):
    """Limits: z = -inf gives at_minus_inf (Hh = inf), z = inf at_inf (Hh = 0),
    nu = inf at_nu_inf (Hh = 0) at finite z; nan in either nan."""
    v = jnp.where(z == -jnp.inf, at_minus_inf, jnp.where(z == jnp.inf, at_inf, v))
    v = jnp.where((nu == jnp.inf) & jnp.isfinite(z), at_nu_inf, v)
    return jnp.where(jnp.isnan(nu) | jnp.isnan(z), jnp.nan, v)


def _log_parts(nu, z, grad):
    val, d_nu, d_z = _parts(nu, z, grad)
    out = _finish(nu, z, val, jnp.inf, -jnp.inf, -jnp.inf)
    if not grad:
        return out, None, None
    return out, *(_finish(nu, z, d, jnp.nan, jnp.nan, jnp.nan) for d in (d_nu, d_z))


def _lin_parts(nu, z, grad):
    val, d_nu, d_z = _parts(nu, z, grad)
    h = jnp.exp(val)
    out = _finish(nu, z, h, jnp.inf, 0.0, 0.0)
    if not grad:
        return out, None, None
    return out, *(_finish(nu, z, h * d, jnp.nan, jnp.nan, jnp.nan) for d in (d_nu, d_z))


def _rule(parts_fn):
    def rule(primals, tangents):
        on = tuple(not isinstance(t, SymbolicZero) for t in tangents)
        v, d_nu, d_z = parts_fn(*primals, any(on))
        out = jnp.zeros_like(v)
        for part, tan, use in zip((d_nu, d_z), tangents, on):
            if use:
                out = out + part * tan
        return v, out

    return rule


@jax.custom_jvp
def _log(nu, z):
    return _log_parts(nu, z, False)[0]


@jax.custom_jvp
def _lin(nu, z):
    return _lin_parts(nu, z, False)[0]


_log.defjvp(_rule(_log_parts), symbolic_zeros=True)
_lin.defjvp(_rule(_lin_parts), symbolic_zeros=True)


def _args(nu, z):
    nu, z = jnp.broadcast_arrays(canon_float(nu), canon_float(z))
    return nan_outside(nu, _t.NU_LO, jnp.inf), z


def log_hh(nu, z):
    """ln Hh_nu(z) for real nu >= 0.1 (nan below, with nan derivatives) and
    real z, broadcast together, differentiable in both; no underflow or
    overflow floor. Error relative to max(1, |ln Hh|): 5.0e-16 in float64 and
    2.0e-7 in float32 (experiments/58)."""
    return _log(*_args(nu, z))


def hh(nu, z):
    """Hh_nu(z) = (1/nu!) int_0^inf u^nu e^(-(u + z)^2/2) du, as log_hh."""
    return _lin(*_args(nu, z))
