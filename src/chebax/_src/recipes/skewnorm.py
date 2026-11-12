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
from chebax._src.recipes._common import canon_float, locate_patch, log1p, patch_sum
from chebax._src.recipes.erf_family import erfcx

_BULK, _TAIL = range(len(_t.NAMES))
_LN2 = math.log(2.0)
_LN_PI = math.log(math.pi)
_LN_SQRT_PI = 0.5 * math.log(math.pi)
_LN_SQRT_PI_2 = math.log(math.sqrt(math.pi) / 2.0)
_LN_RSQRT_2PI = -0.5 * math.log(2.0 * math.pi)
_RSQRT2 = 1.0 / math.sqrt(2.0)


@functools.lru_cache(maxsize=2)
def _tables(dtype):
    """(patch blocks, parameter cuts, argument cuts) in the compute dtype."""
    dt = np.dtype(dtype)
    tab = _t.TABLE if dt == np.float64 else _t.TABLE[:, : _t.KY32, : _t.KP32]
    return np.ascontiguousarray(tab, dt), _t.PCUTS.astype(dt), _t.YCUTS.astype(dt)


def _log1mexp(v, use):
    """ln(1 - e^v) where use (v <= -ln 2 there), a dummy elsewhere: the
    masked lanes can hold v = 0, whose -inf would reach AD of the rule."""
    return log1p(-jnp.exp(jnp.where(use, v, -1.0)))


def _log_k(s, b, ln_ex):
    """(ln K + sigma^2, ln hyp) at finite s, b >= 0, hyp = sqrt(1 + b^2) = 1/c0,
    ln_ex = ln erfcx(mu)."""
    tab, pcuts, ycuts = (jnp.asarray(v) for v in _tables(s.dtype.name))
    npar, narg, offset = (jnp.asarray(v) for v in (_t.NPAR, _t.NARG, _t.OFFSET))
    hyp = jnp.hypot(1.0, b)
    sig = s * hyp * _RSQRT2
    tail = sig > _t.S1
    # bulk lanes get a tail sigma, so 1/sigma^2 and ln sigma stay finite
    sig_t = jnp.where(tail, sig, 2.0 * _t.S1)
    mu = b * s * _RSQRT2
    zone = jnp.where(tail, _TAIL, _BULK)
    par = jnp.where(tail, 1.0 / (sig_t * sig_t), b / hyp)
    arg = jnp.where(tail, 1.0 / (1.0 + mu), sig)
    i, sp = locate_patch(par, pcuts[zone], npar[zone])
    j, sy = locate_patch(arg, ycuts[zone], narg[zone])
    # the barrier keeps XLA from fusing the patch's gathers into one loop with
    # the logs and exps after them
    f = jax.lax.optimization_barrier(
        patch_sum(tab, offset[zone] + i * narg[zone] + j, sy, sp)
    )
    ln_hyp = jnp.log(hyp)
    # tail: ln N = ln(sqrt(pi)/(2 sigma)) + ln erfcx(mu) + ln R
    extra = _LN_SQRT_PI_2 - jnp.log(sig_t) + ln_ex
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
    ex = erfcx(mu)
    # erfcx(mu) = 1/(sqrt(pi) mu) past mu = 1e8; its subnormal values flush to 0
    big_mu = mu > 1e8
    ln_ex = jnp.where(
        big_mu,
        -jnp.log(jnp.where(big_mu, mu, 1.0)) - _LN_SQRT_PI,
        jnp.log(jnp.where(big_mu, 1.0, ex)),
    )
    lk_s, ln_hyp = _log_k(s, b, ln_ex)
    lk = jnp.where(fin_b, lk_s - sig2, -jnp.inf)

    # A = erf(y) + K, B = erfc(y) - K, A + B = 1
    pos_y = y > 0.0
    l_erf = jnp.where(pos_y, jnp.log(jax.lax.erf(jnp.where(pos_y, y, 1.0))), -jnp.inf)
    ln_erfcx_y = jnp.log(erfcx(y))
    # ln(K/erfc(y)) without y^2: as lk - ln erfc(y) it cancels y^2 eps, 1e-10
    # in the slopes at x = -1000. K <= erfc(y)/2, so the log1p reads >= -1/2.
    ln_k_over_erfc = jnp.where(fin_b, lk_s - mu2 - ln_erfcx_y, -jnp.inf)
    l_b_tail = log1p(-jnp.exp(ln_k_over_erfc))
    l_b = ln_erfcx_y - y2 + l_b_tail
    l_a = jnp.logaddexp(l_erf, lk)

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
