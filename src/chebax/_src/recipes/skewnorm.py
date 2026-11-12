"""Skew-normal log-CDF and log survival function at shape a and point x,
broadcast together: log_skewnorm_cdf, log_skewnorm_sf.

F(x; a) = 2 int_{-inf}^x phi(t) Phi(a t) dt = Phi(x) - 2 T(x, a), T Owen's T.
That difference cancels for x < 0 < a, where F falls like e^(-x^2 (1+a^2)/2)
and both terms like e^(-x^2/2): at a = 10 it is off by more than 1e-6 from
x = -0.68 on, with scipy's or jax's owens_t (experiments/49). Here every case
is built from K(s, b) = F(-s; b) and erf, erfc of y = s/sqrt(2), s = |x|,
b = |a|, without cancellation:

  x <= 0, a >= 0:  F = K               x > 0, a < 0:   1 - F = K
  x <= 0, a < 0:   F = erfc(y) - K     x > 0, a >= 0:  F = erf(y) + K

with 1 - F the complement of each. K <= erfc(y)/2, so erfc(y) - K loses at
most a bit; of F and 1 - F the one below 1/2 is formed and the other is
log1p(-it). ln K is one patch of skewnorm_gen's table per element (charts in
its docstring) plus closed-form factors; float32 reads KY32 x KP32 blocks.

Derivatives are the closed forms dF/dx = 2 phi(x) Phi(a x) and dF/da =
-c0^2 e^(-sigma^2)/pi, c0^2 = 1/(1 + a^2), divided by F as exponentials of
log differences. Where ln F is near -sigma^2 (x <= 0 <= a), or near -x^2/2,
that shift is left out of both logs, so the ratio does not lose sigma^2 eps.
Second derivatives are AD of that rule.

a = +-inf gives the half-normal limits, x = -inf gives -inf and x = inf 0,
nan in either nan. Derivatives are for finite a and x, and come back inf or
nan where |a x| overflows sigma (~1e154 in float64, ~1e19 in float32).
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import skewnorm_table as _t
from chebax._src.recipes._common import canon_float, locate_static, log1p, patch_sum
from chebax._src.recipes.erf_family import erfcx

_BULK, _TAIL = range(len(_t.NAMES))
_LN2 = math.log(2.0)
_LN_PI = math.log(math.pi)
_LN_SQRT_PI = 0.5 * math.log(math.pi)
_LN_SQRT_PI_2 = math.log(math.sqrt(math.pi) / 2.0)
_LN_RSQRT_2PI = -0.5 * math.log(2.0 * math.pi)
_RSQRT2 = 1.0 / math.sqrt(2.0)
# each chart's (parameter cuts, argument cuts), compared with as constants
_CUTS = tuple(
    (tuple(_t.PCUTS[z][: _t.NPAR[z] + 1]), tuple(_t.YCUTS[z][: _t.NARG[z] + 1]))
    for z in range(len(_t.NAMES))
)
# erf(y)/y = 2/sqrt(pi) sum (-y^2)^n / (n! (2n + 1)): 13 terms to 1.4e-17 at y = 1/2
_ERF_SERIES = tuple(
    2.0 / math.sqrt(math.pi) * (-1.0) ** n / (math.factorial(n) * (2 * n + 1))
    for n in range(13)
)


@functools.lru_cache(maxsize=2)
def _table(dtype):
    """The patch blocks in the compute dtype."""
    dt = np.dtype(dtype)
    tab = _t.TABLE if dt == np.float64 else _t.TABLE[:, : _t.KY32, : _t.KP32]
    return np.ascontiguousarray(tab, dt)


def _log1mexp(v, use):
    """ln(1 - e^v) where use (v <= -ln 2 there), a dummy elsewhere: the
    masked lanes can hold v = 0, whose -inf would reach AD of the rule."""
    return log1p(-jnp.exp(jnp.where(use, v, -1.0)))


def _log_k(s, b):
    """(ln K + sigma^2, ln hyp) at finite s, b >= 0, hyp = sqrt(1 + b^2) = 1/c0."""
    tab = jnp.asarray(_table(s.dtype.name))
    hyp = jnp.hypot(1.0, b)
    sig = s * hyp * _RSQRT2
    tail = sig > _t.S1
    # bulk lanes get a tail sigma, so 1/sigma^2 and ln sigma stay finite
    sig_t = jnp.where(tail, sig, 2.0 * _t.S1)
    mu = b * s * _RSQRT2
    t = 1.0 / (1.0 + mu)
    # one locate per axis on the chart's coordinate, its cuts picked by chart
    # and compared as constants: gathering a row of cuts per element made
    # float32 values 2.2x slower on the GPU and 1.2x on the CPU (experiments/50)
    par = jnp.where(tail, 1.0 / (sig_t * sig_t), b / hyp)
    arg = jnp.where(tail, t, sig)
    (bulk_p, bulk_y), (tail_p, tail_y) = _CUTS
    i, sp = locate_static(par, (bulk_p, tail_p), tail)
    j, sy = locate_static(arg, (bulk_y, tail_y), tail)
    offset = jnp.where(tail, _t.OFFSET[_TAIL], _t.OFFSET[_BULK])
    n = offset + i * jnp.where(tail, _t.NARG[_TAIL], _t.NARG[_BULK]) + j
    f = patch_sum(tab, n, sy, sp)
    ln_hyp = jnp.log(hyp)
    # tail: ln N = G + ln(sqrt(pi) t/(2 sigma)), erfcx(mu) inside G; t/sigma
    # underflows only where sigma^2 overflows
    extra = _LN_SQRT_PI_2 + jnp.log(t / sig_t)
    return -ln_hyp - _LN_PI + f + jnp.where(tail, extra, 0.0), ln_hyp


def _parts(a, x):
    """ln F, and for the derivatives ln F + shift, ln f + shift (f the
    density) and shift - sigma^2, shift being sigma^2 or y^2 where ln F is
    near minus it and 0 elsewhere."""
    neg_x, neg_a = x <= 0.0, a < 0.0
    # selects, not abs: the x = 0 lane belongs to x <= 0, d s/dx = -1
    s = jnp.where(neg_x, -x, x)
    b = jnp.where(neg_a, -a, a)
    # past s^2/2 = inf (|x| > 1.3e154, 1.8e19 in float32) F is 0 or 1 exactly
    fin_s, fin_b = 0.5 * s * s < jnp.inf, b < jnp.inf
    s = jnp.where(fin_s, s, 0.0)
    b = jnp.where(fin_b, b, 0.0)

    y, mu = s * _RSQRT2, b * s * _RSQRT2
    y2, mu2 = 0.5 * s * s, 0.5 * (b * s) * (b * s)
    sig2 = y2 + mu2
    # erfcx(mu) is only for the derivatives (the value's is in the table): it
    # is 1/(sqrt(pi) mu) past mu = 1e8, whose subnormal values flush to 0
    ex = erfcx(mu)
    big_mu = mu > 1e8
    ln_ex = jnp.where(
        big_mu,
        -jnp.log(jnp.where(big_mu, mu, 1.0)) - _LN_SQRT_PI,
        jnp.log(jnp.where(big_mu, 1.0, ex)),
    )
    lk_s, ln_hyp = _log_k(s, b)
    lk = jnp.where(fin_b, lk_s - sig2, -jnp.inf)

    # A = erf(y) + K, B = erfc(y) - K, A + B = 1
    ln_erfcx_y = jnp.log(erfcx(y))
    # erf from its series below y = 1/2 and as 1 - erfc above (within 2 ulp
    # there, erfc <= 0.48): lax.erf made float64 CPU values 1.3x slower
    # (experiments/50)
    near = y < 0.5
    u = jnp.where(near, y * y, 0.0)
    series = _ERF_SERIES[-1]
    for c in _ERF_SERIES[-2::-1]:
        series = series * u + c
    erf_y = jnp.where(near, y * series, -jnp.expm1(ln_erfcx_y - y2))
    # ln(K/erfc(y)) without y^2: as lk - ln erfc(y) it cancels y^2 eps, 1e-10
    # in the slopes at x = -1000. K <= erfc(y)/2, so the log1p reads >= -1/2.
    ln_k_over_erfc = jnp.where(fin_b, lk_s - mu2 - ln_erfcx_y, -jnp.inf)
    l_b_tail = log1p(-jnp.exp(ln_k_over_erfc))
    l_b = ln_erfcx_y - y2 + l_b_tail
    # as log(erf_y + exp(lk)) XLA's CPU value program reads each patch
    # coefficient in its own loop, through memory: 4.6x slower in float64
    # (experiments/50)
    l_a = jnp.logaddexp(jnp.log(erf_y), lk)

    case1 = neg_x & ~neg_a  # F = K
    case2 = neg_x & neg_a  # F = B
    # the smaller of F and 1 - F, formed directly, then one log1mexp (two logs
    # and an exp) for the other, not one per case
    mixed = neg_x == neg_a
    small = jnp.where(mixed, jnp.minimum(l_a, l_b), lk)
    f_small = jnp.where(mixed, (l_b <= l_a) == case2, case1)
    log_f = jnp.where(f_small, small, _log1mexp(small, ~f_small))
    b_small = l_b <= l_a

    # ln f = ln 2 - ln sqrt(2 pi) - y^2 + ln Phi(a x), a x = -+sqrt(2) mu
    opposite = case1 | (~neg_x & neg_a)
    ln_phi = jnp.where(opposite, ln_ex - _LN2 - mu2, log1p(-0.5 * ex * jnp.exp(-mu2)))
    lpdf = _LN2 + _LN_RSQRT_2PI - y2 + ln_phi
    scaled2 = case2 & b_small
    l_scaled = jnp.select([case1, scaled2], [lk_s, ln_erfcx_y + l_b_tail], log_f)
    lpdf_scaled = jnp.select(
        [case1, scaled2], [_LN_RSQRT_2PI + ln_ex, _LN2 + _LN_RSQRT_2PI + ln_phi], lpdf
    )
    shift_less_sig2 = jnp.select([case1, scaled2], [jnp.zeros_like(mu2), -mu2], -sig2)

    log_f = jnp.where(fin_s, log_f, jnp.where(neg_x, -jnp.inf, 0.0))
    log_f = jnp.where(jnp.isnan(a) | jnp.isnan(x), jnp.nan, log_f)
    return log_f, l_scaled, lpdf_scaled, shift_less_sig2, ln_hyp


@jax.custom_jvp
def _log_cdf(a, x):
    return _parts(a, x)[0]


@_log_cdf.defjvp
def _log_cdf_jvp(primals, tangents):
    a, x = primals
    ta, tx = tangents
    log_f, l_scaled, lpdf_scaled, shift_less_sig2, ln_hyp = _parts(a, x)
    d_x = jnp.exp(lpdf_scaled - l_scaled)
    d_a = -jnp.exp(-2.0 * ln_hyp - _LN_PI + shift_less_sig2 - l_scaled)
    return log_f, d_x * tx + d_a * ta


def log_skewnorm_cdf(a, x):
    """ln of the standard skew-normal CDF, F(x; a) = 2 int_{-inf}^x phi(t)
    Phi(a t) dt, with a and x broadcast: relative in F where |ln F| <= 1 and in
    ln F beyond, for every finite a and x. Near F = 1 that bounds 1 - F only
    to ~x^2 eps; log_skewnorm_sf has it in full. Differentiable in both."""
    a, x = jnp.broadcast_arrays(canon_float(a), canon_float(x))
    return _log_cdf(a, x)


def log_skewnorm_sf(a, x):
    """ln(1 - F(x; a)) = log_skewnorm_cdf(-a, -x)."""
    a, x = jnp.broadcast_arrays(canon_float(a), canon_float(x))
    return _log_cdf(-a, -x)
