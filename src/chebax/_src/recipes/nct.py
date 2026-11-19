"""The noncentral Student t log-density, nct_logpdf(nu, mu, x): nu >= 0.1
degrees of freedom (nan below), noncentrality mu and x real, all broadcast
together and per element, differentiable in all three.

With T = (Z + mu)/sqrt(V/nu) and s = sqrt(V/nu),

    f(x) = K int_0^inf s^nu exp(-nu s^2/2 - (x s - mu)^2/2) ds,
    K = 2 (nu/2)^(nu/2) / (Gamma(nu/2) sqrt(2 pi)).

The exponent's saddle s* solves P s^2 - x mu s - nu = 0, P = nu + x^2, and
with u = s sqrt(P) the integral is Gamma(nu + 1) Hh_nu(-y) times elementary
factors, y = mu x/sqrt(P). hh's table holds T = ln Hh - (the saddle-point
form of ln Hh with Stirling's lnGamma(nu + 1)), so

    ln f = -ln(2 pi)/2 - r^2/2 + nu h(delta) + ln(2 nu/G2)/2 + B(nu) + T(nu, -y)

with delta = s* - 1, r = x s* - mu, h(delta) = ln(1 + delta) - delta -
delta^2/2, G2 = nu/s*^2 + P, and B(nu) = R(nu) - R(nu/2) the difference of
Binet's lnGamma remainders, nct_gen's table. -r^2/2 and nu h are both <= 0
and every other term is bounded, so nothing cancels. The textbook forms
cancel near the mode at large mu: two 1F1 terms for x opposite to mu, and
central t plus ln Hh(-y) - ln Hh(0) adds -x^2/2 - mu^2/2 + mu x, losing mu^2
eps. delta and r come from forms with no subtraction of like terms:

    delta = -2 x (x - mu)/(2P + E1),  r = (x - mu)(2 nu + E1)/(2P + E1),
    E1 = sqrt(D) - x mu >= 0,  D = x^2 mu^2 + 4 nu P,

E1 as 4 nu P/(sqrt(D) + x mu) where x mu > 0. x, mu and sqrt(nu) are first
scaled by a power of two to at most 1, so nothing overflows; nan where nu
then underflows, past |x| or |mu| ~ 2^511 sqrt(nu) (2^63 in float32).

Slopes: on the saddle part by the envelope theorem (d/dmu = r, d/dx = -r s*,
d/dnu = h), the rest through ds*/dtheta = -F_theta/sqrt(D), F the quadratic
above, and hh's T slopes. nu = inf gives the normal density. Measured
accuracy and speed: tests/test_nct.py, experiments/59.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import nct_table as _t
from chebax._src.recipes._common import canon_float, nan_outside, panel_series
from chebax._src.recipes.hh import t_parts
from jax.custom_derivatives import SymbolicZero

_HALF_LN2PI = 0.5 * math.log(2.0 * math.pi)
_LLO = math.log(_t.NU_LO)
_WB = (math.log(_t.NU_S) - _LLO) / _t.NPAN


@functools.lru_cache(maxsize=2)
def _b_arrays(dtype):
    """B's panels (float32: the leading K32 terms), their s-derivatives and
    the Stirling terms, in the compute dtype."""
    dt = np.dtype(dtype)
    pan = _t.B_PANELS if dt == np.float64 else _t.B_PANELS[:, : _t.K32]
    der = np.array([np.polynomial.chebyshev.chebder(row) for row in pan])
    return (
        np.ascontiguousarray(pan, dt),
        np.ascontiguousarray(der, dt),
        np.asarray(_t.B_STIRLING, dt),
    )


def _b(nu, grad):
    """(B(nu), B'(nu)), the slope None without grad."""
    pan, der, stir = (jnp.asarray(v) for v in _b_arrays(nu.dtype.name))
    big = nu >= _t.NU_S
    i = jnp.clip(jnp.floor((jnp.log(nu) - _LLO) / _WB), 0, _t.NPAN - 1)
    s = jnp.clip((jnp.log(nu) - _LLO - i * _WB) * (2.0 / _WB) - 1.0, -1.0, 1.0)
    i = i.astype(jnp.int32)
    inv = 1.0 / jnp.where(big, nu, _t.NU_S)
    i2 = inv * inv
    poly = stir[-1]
    for c in stir[-2::-1]:
        poly = poly * i2 + c
    val = jnp.where(big, poly * inv, panel_series(pan, i, s))
    if not grad:
        return val, None
    dpoly = (2 * len(stir) - 1) * stir[-1]
    for k in range(len(stir) - 2, -1, -1):
        dpoly = dpoly * i2 + (2 * k + 1) * stir[k]
    slope = jnp.where(big, -dpoly * i2, panel_series(der, i, s) * (2.0 / _WB) / nu)
    return val, slope


# ln(1 + d) - d - d^2/2 = -d u - d^2/2 + 2 u^3 sum_k u^(2k)/(2k + 3), u = d/(2 + d):
# no cancellation, |u| <= 1/3 on [-1/2, 1]; the sums dropped are under 1e-17
# (float64) and 1e-8 (float32) of h there
_A1 = tuple(1.0 / (2 * k + 3) for k in range(16))


def _h(delta, ls):
    """h(delta) = ln(1 + delta) - delta - delta^2/2 <= 0, ls = ln(1 + delta)
    from s* itself (used outside [-1/2, 1], where the plain form loses at most
    a few ulps)."""
    inner = (delta >= -0.5) & (delta <= 1.0)
    d = jnp.where(inner, delta, 0.0)
    u = d / (2.0 + d)
    w = u * u
    terms = _A1 if delta.dtype == jnp.float64 else _A1[:7]
    a = terms[-1]
    for c in terms[-2::-1]:
        a = a * w + c
    series = (-d * u - 0.5 * d * d) + 2.0 * u * w * a
    return jnp.where(inner, series, ls - delta - 0.5 * delta * delta)


def _pow2(k, dtype):
    """2^k for integer k in the dtype's normal range, by bits."""
    if dtype == jnp.float64:
        bits = (k.astype(jnp.int64) + 1023) << 52
        return jax.lax.bitcast_convert_type(bits, jnp.float64)
    bits = (k.astype(jnp.int32) + 127) << 23
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _exponent(v):
    """e with v = f 2^e, 0.5 <= f < 1, for normal v > 0."""
    if v.dtype == jnp.float64:
        bits = jax.lax.bitcast_convert_type(v, jnp.int64)
        return ((bits >> 52) & 0x7FF) - 1022
    bits = jax.lax.bitcast_convert_type(v, jnp.int32)
    return ((bits >> 23) & 0xFF) - 126


def _parts(nu, mu, x, grad):
    """(ln f, d/dnu, d/dmu, d/dx) at finite nu >= 0.1, mu and x, the slopes
    None without grad; nan where nu/S^2 is below the normal range."""
    dt = x.dtype
    top = 1022 if dt == jnp.float64 else 126
    e = jnp.minimum(
        _exponent(jnp.maximum(jnp.maximum(jnp.abs(x), jnp.abs(mu)), jnp.sqrt(nu))),
        top,
    )
    down = _pow2(-e, dt)
    a, b = x * down, mu * down
    n = nu * down * down
    lost = n < jnp.finfo(dt).tiny
    n = jnp.where(lost, 1.0, n)
    big_s = _pow2(e, dt)
    p = n + a * a
    m = a * b
    sd = jnp.sqrt(m * m + 4.0 * n * p)
    pos = m >= 0.0
    e1 = jnp.where(pos, 4.0 * n * p / (sd + m), sd - m)
    den = 2.0 * p + e1
    amb = a - b
    delta = -2.0 * a * amb / den
    ss = jnp.where(pos, (m + sd) / (2.0 * p), 2.0 * n / (sd - m))
    # r = (x - mu)(2n + E1)/den = (x - mu)(1 - 2a^2/den) = x s* - mu, whichever
    # rounds least: the second where 2a^2/den is small (near the normal limit),
    # the third where x and mu differ in sign (two terms of one sign);
    # increments 65 has the rounding measured.
    f = 2.0 * a * a / den
    r = big_s * jnp.where(
        f <= 0.25,
        amb * (1.0 - f),
        jnp.where(pos, amb * ((2.0 * n + e1) / den), a * ss - b),
    )
    inv = 1.0 / ss
    ls = jnp.log(ss)
    h = _h(delta, ls)
    # nu/s*^2 as n/s*^2: s*^2 underflows first (s* ~ 1e-200 at x = -mu = 1e100)
    ns2 = (n * inv) * inv
    g = ns2 + p
    rp = jax.lax.rsqrt(p)
    z = -big_s * m * rp
    t, t_nu, t_z = t_parts(nu, z, grad)
    bv, b1 = _b(nu, grad)
    k = 1.0 / (1.0 + p / ns2)
    # 2n/G = 2 s*^2 k, which underflows with s*^2 (x = -mu = 1e100)
    ratio = 2.0 * n / g
    lg = jnp.where(
        ratio > 4.0 * jnp.finfo(dt).tiny,
        0.5 * jnp.log(ratio),
        ls + 0.5 * jnp.log(2.0 * k),
    )
    val = (-0.5 * r * r + nu * h) + ((lg + bv) + (t - _HALF_LN2PI))
    val = jnp.where(lost, jnp.nan, val)
    if not grad:
        return val, None, None, None
    # the ln(2 nu/G2)/2 term through k = (nu/s*^2)/G2 in [0, 1]; its nu-slope
    # 1/(2 nu) - G2_nu/(2 G2) is a^2/(2 nu G) - delta (2 + delta) k/(s* sqrt(D))
    ks = k / (sd * big_s)
    rp3 = rp * rp * rp
    d_nu = (
        h
        + 0.5 * a * a / (nu * g)
        - delta * (2.0 + delta) * inv * ks / big_s
        + b1
        + t_nu
        + t_z * m * rp3 / (2.0 * big_s)
    )
    d_mu = r + ks * a - t_z * a * rp
    d_x = -r * ss - ks * (2.0 * a * ss - b) - a / (g * big_s) - t_z * b * n * rp3
    return val, *(jnp.where(lost, jnp.nan, d) for d in (d_nu, d_mu, d_x))


def _full(nu, mu, x, grad):
    """_parts with the limits: nu = inf the normal density, x or mu infinite
    -inf (nan slopes), nan in any nan."""
    ok = jnp.isfinite(nu) & jnp.isfinite(mu) & jnp.isfinite(x)
    n = jnp.where(ok, nu, 1.0)
    b = jnp.where(ok, mu, 0.0)
    a = jnp.where(ok, x, 0.0)
    val, d_nu, d_mu, d_x = _parts(n, b, a, grad)
    normal = (nu == jnp.inf) & jnp.isfinite(mu) & jnp.isfinite(x)
    dev = jnp.where(normal, x - mu, 0.0)
    nan = jnp.isnan(nu) | jnp.isnan(mu) | jnp.isnan(x)

    def pick(v, at_normal, else_):
        v = jnp.where(ok, v, jnp.where(normal, at_normal, else_))
        return jnp.where(nan, jnp.nan, v)

    out = pick(val, -0.5 * dev * dev - _HALF_LN2PI, -jnp.inf)
    if not grad:
        return out, None, None, None
    return (
        out,
        pick(d_nu, 0.0, jnp.nan),
        pick(d_mu, dev, jnp.nan),
        pick(d_x, -dev, jnp.nan),
    )


@jax.custom_jvp
def _logpdf(nu, mu, x):
    return _full(nu, mu, x, False)[0]


def _rule(primals, tangents):
    on = tuple(not isinstance(t, SymbolicZero) for t in tangents)
    v, *parts = _full(*primals, any(on))
    out = jnp.zeros_like(v)
    for part, tan, use in zip(parts, tangents, on):
        if use:
            out = out + part * tan
    return v, out


_logpdf.defjvp(_rule, symbolic_zeros=True)


def nct_logpdf(nu, mu, x):
    """ln of the noncentral Student t density with nu >= 0.1 degrees of
    freedom (nan below, with nan derivatives) and noncentrality mu, at x;
    all real, broadcast together, differentiable in each. nu = inf gives the
    normal density about mu. No underflow floor. Error relative to max(1,
    |ln f|): see experiments/59."""
    nu, mu, x = jnp.broadcast_arrays(*(canon_float(v) for v in (nu, mu, x)))
    return _logpdf(nan_outside(nu, _t.NU_LO, jnp.inf), mu, x)
