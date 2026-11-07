"""Per-element gammainc: P, Q and their logs with a different a >= 0.1 on every
element (chebax.gammainc_array; inputs normalized in gammainc.py). Per element:
pick the zone, locate one 2-D Chebyshev patch of gammainc_array_gen's table
(zones and formulas in its docstring), sum its KY x KP block, add the zone's
prefactor. No lgamma, no loop. float32 uses the leading KY32 x KP32 block.

Each zone forms its small probability directly as a log and the other as
log1p(-exp(direct)). That loses only in A0, where Q falls to 0.024 (a = 0.1,
x = 1.1): ln Q 8.4e-15 there, 7.8e-16 elsewhere (tests/test_gammainc_array.py).

The Temme zone's phi = d - log1p(d), d = (x - a)/a, cancels near d = 0, so it is
2 z^2 S(z), z = d/(2 + d), S = 1 + 2/3 z + z^2 + 4/5 z^3 + ..., and eta = 2 z sqrt(S).

Endpoints as in gammainc; a < 0.1, inf or nan gives nan. Gradients in a and x are
AD with masked branches, relatively accurate in the tails (d ln Q/da ~ ln(x/a) at
x = 1e100), inf only where d ln P/dx ~ a/x overflows; the x = 0 slope is the density.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_array_table as _t
from chebax._src.recipes._common import cheb_basis, edge_slope, locate_patch
from chebax._src.recipes.erf_family import erfcx
from chebax._src.recipes.gammaincinv_kernel import log_xpdf

_A0, _B0, _L1, _U1, _L2, _U2, _T = range(len(_t.NAMES))
_LN2PI = math.log(2.0 * math.pi)
# S(z) to z^24: tail below 2e-17 on |z| <= 0.2; the direct form beyond, ~10 eps
_Z_SER = 0.2
_S = tuple(1.0 if k % 2 == 0 else (k + 1) / (k + 2) for k in range(25))


@functools.lru_cache(maxsize=2)
def _tables(dtype):
    """(patch blocks, parameter cuts, argument cuts) in the compute dtype."""
    dt = np.dtype(dtype)
    tab = _t.TABLE if dt == np.float64 else _t.TABLE[:, : _t.KY32, : _t.KP32]
    return np.ascontiguousarray(tab, dt), _t.PCUTS.astype(dt), _t.YCUTS.astype(dt)


_basis, _locate = cheb_basis, locate_patch


def _log1p(x):
    """log1p for x in (-1, inf). XLA's float64 log1p is up to 128 ulp off on
    (-0.63, -0.25) (jax 0.11 CPU, increment 34; 5.5e-14 in ln P); log(1 + x) there
    gives 3.7e-16. XLA simplifies the textbook x log(u)/(u - 1) back to log1p."""
    return jnp.where(x < -0.25, jnp.log(1.0 + x), jnp.log1p(x))


def _phi_eta(d):
    """phi = d - log1p(d) and eta = sign(d) sqrt(2 phi), smooth through 0."""
    z = d / (2.0 + d)
    ser = jnp.abs(z) <= _Z_SER
    zs = jnp.where(ser, z, 0.0)
    s = _S[-1]
    for c in _S[-2::-1]:
        s = s * zs + c
    # direct lanes only; the dummy keeps log1p and sqrt off 0
    dd = jnp.where(ser, 1.0, d)
    phi_d = dd - _log1p(dd)
    eta_d = jnp.where(dd < 0.0, -1.0, 1.0) * jnp.sqrt(2.0 * phi_d)
    return (
        jnp.where(ser, 2.0 * zs * zs * s, phi_d),
        jnp.where(ser, 2.0 * zs * jnp.sqrt(s), eta_d),
    )


@jax.custom_jvp
def _log_ratio(x, a):
    """ln(x/a) for x, a > 0; ln x - ln a where x/a is not normal (x = 1e-36,
    a = 1000 in float32) or overflows (x = 1.7e308, a = 0.1)."""
    r = x / a
    normal = (r >= np.finfo(r.dtype).tiny) & (r < jnp.inf)
    return jnp.where(
        normal, jnp.log(jnp.where(normal, r, 1.0)), jnp.log(x) - jnp.log(a)
    )


@_log_ratio.defjvp
def _log_ratio_jvp(primals, tangents):
    # AD forms x/a^2 and 1/r, which leave the float range at its ends (a-term
    # off by 1 at x = 1e-306, a = 10)
    x, a = primals
    tx, ta = tangents
    return _log_ratio(x, a), _over(tx, x) - _over(ta, a)


@jax.custom_jvp
def _over(t, v):
    """t / v, differentiated as dt/v - (t/v)(dv/v). jax 0.5's rule forms
    t dv / v^2, and v^2 underflows to 0 at v = 1e-300 (1e-20 in float32),
    so where t is 0 a second derivative read 0/0: d2x/dp2 of
    gammaincinv_array at a = 1, p = 1e-300 was nan on the jax floor."""
    return t / v


@_over.defjvp
def _over_jvp(primals, tangents):
    (t, v), (dt, dv) = primals, tangents
    out = t / v
    return out, dt / v - out * (dv / v)


@jax.custom_jvp
def _exp0(v):
    """exp(v) with slope exactly 0 where exp(v) is 0."""
    return jnp.exp(v)


@_exp0.defjvp
def _exp0_jvp(primals, tangents):
    # P = 0 with d ln P/dx = a/x overflowing gives 0 * inf in forward mode
    (v,), (t,) = primals, tangents
    e = jnp.exp(v)
    return e, jnp.where(e == 0.0, 0.0, e * t)


@jax.custom_jvp
def _exp_cdf(lv, a, x, sign):
    """exp(lv), lv = ln P (sign 1) or ln Q (sign -1) of Gamma(a) at x."""
    return jnp.exp(lv)


@_exp_cdf.defjvp
def _exp_cdf_jvp(primals, tangents):
    # Below the smallest normal, exp(lv) times d lv/dx loses the density
    # (0 for 1e-25 at a = 2, x = 1e-25 in float32), so it is taken from its
    # logs there; the a-part, exp(lv) times a moderate d lv/da, rounds away
    lv, a, x, sign = primals
    dlv, _, dx, _ = tangents
    e = jnp.exp(lv)
    deep = (e < jnp.finfo(e.dtype).tiny) & (x > 0.0) & (x < jnp.inf)

    def underflow():
        xs, as_ = jnp.where(deep, x, 1.0), jnp.where(deep, a, 1.0)
        lpdf = log_xpdf(as_, _log_ratio(xs, as_)) - jnp.log(xs)
        pdf = jnp.where(deep, jnp.exp(lpdf), 0.0)
        return jnp.where(deep, sign * pdf * dx, jnp.where(e == 0.0, 0.0, e * dlv))

    # Most arrays have no underflowing CDF; the log-density fallback is costly.
    return e, jax.lax.cond(
        jnp.any(deep), underflow, lambda: jnp.where(e == 0.0, 0.0, e * dlv)
    )


def _log_pq(a, x, d=None):
    """(ln P, ln Q) for a, x of one shape; no x = 0 slope. d, if given, is x/a - 1
    held exactly, read instead of x for a >= 1 (gammaincinv_array's derivative
    needs it where one float step of x spans the distribution, a ~ 1e35)."""
    tab, pcuts, ycuts = (jnp.asarray(v) for v in _tables(a.dtype.name))
    ky, kp = tab.shape[1:]
    npar, narg, offset = (jnp.asarray(v) for v in (_t.NPAR, _t.NARG, _t.OFFSET))

    good_a = (a >= _t.ALO) & (a < jnp.inf)
    a = jnp.where(good_a, a, 1.0)
    ok = (x > 0.0) & (x < jnp.inf)
    xx = jnp.where(ok, x, 1.0)
    lam = xx / a if d is None else 1.0 + d
    big = a > _t.ASPLIT
    zone = jnp.where(
        a < 1.0,
        jnp.where(xx <= a + 1.0, _A0, _B0),
        jnp.where(
            lam <= _t.LAM_LO,
            jnp.where(big, _L2, _L1),
            jnp.where(lam >= _t.LAM_HI, jnp.where(big, _U2, _U1), _T),
        ),
    )
    in_a0, in_b0, in_t = zone == _A0, zone == _B0, zone == _T
    is_low = (zone == _L1) | (zone == _L2)
    is_up = (zone == _U1) | (zone == _U2)
    uses_u = in_a0 | in_b0 | (zone == _L1) | (zone == _U1)

    # a harmless x outside each branch's lanes: a zero cotangent times an
    # overflowing AD term (a/x at x = 1e-20 in float32) is nan
    x_a0 = jnp.where(in_a0, xx, a + 1.0)
    x_b0 = jnp.where(in_b0, xx, a + 1.0)
    x_lo = jnp.where(is_low, xx, a)
    x_up = jnp.where(is_up, xx, a)
    x_lu = jnp.where(is_low | is_up, xx, a)
    x_t = jnp.where(in_t, xx, a)
    if d is None:
        # d = (x - a)/a keeps x - a exact
        d_t = (x_t - a) / a
        lam_lo, lam_up = x_lo / a, a / x_up
        ln_lam = _log_ratio(x_lu, a)
        x_less_a = x_lu - a
    else:
        d_t = jnp.where(in_t, d, 0.0)
        d_lu = jnp.where(is_low | is_up, d, 0.0)
        lam_lo = jnp.where(is_low, 1.0 + d_lu, 1.0)
        lam_up = jnp.where(is_up, 1.0 / (1.0 + d_lu), 1.0)
        ln_lam = _log1p(d_lu)
        x_less_a = a * d_lu
    phi_t, eta = _phi_eta(d_t)

    par = jnp.select([uses_u, in_t], [_log1p(a), 1.0 / a], _t.ASPLIT / a)
    arg = jnp.select(
        [in_a0, in_b0, is_low, is_up],
        # _over: plain division's AD poisons the second a-derivative at
        # a = 1e20 in float32 through the lanes not taken
        [_over(x_a0, a + 1.0), _over(a + 1.0, x_b0), lam_lo, lam_up],
        eta,
    )
    i, sp = _locate(par, pcuts[zone], npar[zone])
    j, sy = _locate(arg, ycuts[zone], narg[zone])
    block = tab[offset[zone] + i * narg[zone] + j]
    f = jnp.einsum("...kj,...k,...j->...", block, _basis(sy, ky), _basis(sp, kp))

    ln_a = jnp.log(a)
    half_ln_2pi_a = 0.5 * (ln_a + _LN2PI)
    # ln Q = -y^2 + ln(erfcx(y)/2 + T/rt) for y >= 0 (ln P with -y, -T), y^2 = a phi
    y = eta * jnp.sqrt(0.5 * a)
    sign = jnp.where(y < 0.0, -1.0, 1.0)
    # sqrt(2 pi) sqrt(a): 2 pi a overflows from a = 2.9e307
    rt = math.sqrt(2.0 * math.pi) * jnp.sqrt(a)
    yt = jnp.where(in_t, jnp.abs(y), 0.0)
    ft = jnp.where(in_t, f, 0.0)
    # with d exact, a phi as y^2: phi is subnormal for |d| < 2e-154 (a = 1e307);
    # the barrier stops XLA folding y * y into a eta^2 for a constant a
    if d is None:
        a_phi = a * phi_t
    else:
        yb = jax.lax.optimization_barrier(y)
        a_phi = yb * yb
    temme = -a_phi + jnp.log(0.5 * erfcx(yt) + sign * ft / rt)

    # -a phi as -(x - a) + a ln lambda: as a product its d/da cancels two terms
    # of size x/a (13.6 for 17.7 at x = 1e8 in float32)
    direct = jnp.select(
        [in_a0, in_b0, is_low, is_up],
        [
            a * jnp.log(x_a0) - x_a0 + f,
            (a - 1.0) * jnp.log(x_b0) - x_b0 + ln_a + f,
            a * ln_lam - x_less_a - half_ln_2pi_a + f,
            (a - 1.0) * ln_lam - x_less_a - half_ln_2pi_a + f,
        ],
        temme,
    )
    is_p = in_a0 | is_low | (in_t & (y < 0.0))
    if d is None:
        other = _log1p(-_exp_cdf(direct, a, xx, jnp.where(is_p, 1.0, -1.0)))
    else:
        # the quantile's derivative: its x tangent is not the whole one
        other = _log1p(-_exp0(direct))
    lp = jnp.where(is_p, direct, other)
    lq = jnp.where(is_p, other, direct)

    top = x == jnp.inf
    lp = jnp.where(ok, lp, jnp.where(top, 0.0, -jnp.inf))
    lq = jnp.where(ok, lq, jnp.where(top, -jnp.inf, 0.0))
    bad = jnp.isnan(x) | ~good_a
    return jnp.where(bad, jnp.nan, lp), jnp.where(bad, jnp.nan, lq)


def _density0(a):
    """Gamma density at x = 0: inf below a = 1, 1 at it, 0 above."""
    return jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, 1.0))


def p(a, x):
    lp, _ = _log_pq(a, x)
    return _exp_cdf(lp, a, x, 1.0) + edge_slope(
        jnp.where(x == 0.0, _density0(a), 0.0), x
    )


def q(a, x):
    _, lq = _log_pq(a, x)
    return _exp_cdf(lq, a, x, -1.0) + edge_slope(
        jnp.where(x == 0.0, -_density0(a), 0.0), x
    )


def log_p(a, x):
    lp, _ = _log_pq(a, x)
    # d ln P/dx = density/P ~ a/x, so the slope at x = 0 is inf for every a
    return lp + edge_slope(jnp.where(x == 0.0, jnp.inf, 0.0), x)


def log_q(a, x):
    _, lq = _log_pq(a, x)
    # Q = 1 at x = 0, so d ln Q/dx there is just -density
    return lq + edge_slope(jnp.where(x == 0.0, -_density0(a), 0.0), x)
