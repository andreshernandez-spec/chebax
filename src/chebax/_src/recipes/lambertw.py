"""Lambert W, real branches k = 0 and k = -1 (jax#13680): eight Halley steps on
w e^w - x from the branch-point series in p = sqrt(2(1 + e x)) or log
asymptotics, with dW/dx = 1 / (e^W (1 + W)). Near the top of the range, and
on k = -1 near 0, the steps run on w + ln|w| - ln|x| instead. Near -1/e any
double routine loses sqrt(eps / (x + 1/e)).
"""

import math

import jax
import jax.numpy as jnp

_ITERS = 8

# -1/e lies between two doubles (1/math.e and math.exp(-1) round apart), so
# accept one ulp below
_BR = -math.exp(-1.0)
_BR_LO = math.nextafter(_BR, -math.inf)


def _halley(x, w):
    for _ in range(_ITERS):
        ew = jnp.exp(w)
        f = w * ew - x
        w1 = w + 1.0
        denom = ew * w1 - (w + 2.0) * f / (2.0 * w1)
        wn = w - f / denom
        w = jnp.where(jnp.isfinite(wn), wn, w)
    return w


def _log_form(dtype):
    """x above which k = 0 solves in logs: w e^w - x overflows near the top of
    the range (3e-8 off at max float). 1e300 as a float32 is inf, which left
    float32 on the overflowing residual (93 ulps off at its max)."""
    return 1e300 if jnp.dtype(dtype) == jnp.float64 else 1e30


def _halley_log(lx, w, s=1.0):
    """Halley on w + ln(s w) - ln(s x), s = 1 for k = 0 and -1 for k = -1, where
    w e^w - x overflows (k = 0 at the top of the range) or e^w is subnormal
    (k = -1 near 0). About one ulp worse, so only out there."""
    for _ in range(_ITERS):
        g = w + jnp.log(s * w) - lx
        g1 = 1.0 + 1.0 / w
        g2 = -1.0 / (w * w)
        wn = w - g / (g1 - g * g2 / (2.0 * g1))
        w = jnp.where(jnp.isfinite(wn) & (s * wn > 0.0), wn, w)
    return w


def _branch_series(p, sign):
    # W = -1 + s p - p^2/3 + 11 s p^3/72, s = +1 (k=0), -1 (k=-1)
    return -1.0 + sign * p - p * p / 3.0 + sign * 11.0 * p**3 / 72.0


@jax.custom_jvp
def _w0(x):
    x = jnp.asarray(x)
    xs = jnp.where(jnp.isfinite(x), x, 2.0)  # masked lanes get a safe dummy
    p = jnp.sqrt(jnp.maximum(2.0 * (1.0 + jnp.e * xs), 0.0))
    l1 = jnp.log(jnp.maximum(xs, 2.0))
    l2 = jnp.log(l1)
    init = jnp.where(xs < 1.0, _branch_series(p, 1.0), l1 - l2 + l2 / l1)
    big = xs > _log_form(xs.dtype)
    w = jnp.where(big, _halley_log(l1, jnp.where(big, init, 1.0)), _halley(xs, init))
    w = jnp.where(x == jnp.inf, jnp.inf, w)
    return jnp.where((x < _BR_LO) | jnp.isnan(x), jnp.nan, w)


@jax.custom_jvp
def _wm1(x):
    x = jnp.asarray(x)
    p = jnp.sqrt(jnp.maximum(2.0 * (1.0 + jnp.e * x), 0.0))
    xn = jnp.where(x < 0.0, x, -0.1)  # masked lanes get a safe dummy
    l1 = jnp.log(-xn)
    init = jnp.where(x > -0.25, l1 - jnp.log(-l1), _branch_series(p, -1.0))
    # Halley on w e^w - x fails from -x ~ 1e-301 down (e^w nears the bottom of
    # the range; -706.9 for -711.2 at x = -1e-306); the log residual holds
    # 2e-16 there. Subnormal x is 0 to XLA's CPU, so no answer below tiny
    deep = -xn < 1e10 * jnp.finfo(xn.dtype).tiny
    w = jnp.where(
        deep, _halley_log(l1, jnp.where(deep, init, -2.0), -1.0), _halley(xn, init)
    )
    return jnp.where((x < _BR_LO) | (x >= 0.0) | jnp.isnan(x), jnp.nan, w)


def _dw(w):
    return 1.0 / (jnp.exp(w) * (1.0 + w))


@_w0.defjvp
def _w0_jvp(primals, tangents):
    (x,), (xdot,) = primals, tangents
    w = _w0(x)
    return w, _dw(w) * xdot


@_wm1.defjvp
def _wm1_jvp(primals, tangents):
    (x,), (xdot,) = primals, tangents
    w = _wm1(x)
    # W <= -1 here, so -inf at W = -1, where the formula divides by +0
    d = _dw(w)
    d = jnp.where(w <= -1.0, -jnp.abs(d), d)
    return w, d * xdot


def lambertw(x, k=0):
    """Real Lambert W: branch k = 0 on [-1/e, inf), k = -1 on [-1/e, 0).

    k is a static python int. Differentiable via the implicit gradient."""
    if k == 0:
        return _w0(x)
    if k == -1:
        return _wm1(x)
    raise ValueError(f"real lambertw branches are k=0 and k=-1, got {k}")
