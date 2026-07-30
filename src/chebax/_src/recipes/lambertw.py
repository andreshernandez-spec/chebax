"""Lambert W, real branches k = 0 and k = -1 (jax#13680).

No tables: like gammaincinv, this is a fixed-count solver with an implicit
gradient. Eight Halley iterations on f(w) = w e^w - x from closed-form
initializers (the branch-point series in p = sqrt(2(1 + e x)) near x = -1/e,
log-based asymptotics elsewhere); Halley's cubic convergence turns a
0.5-accurate initializer into machine precision in three steps, and
non-finite updates are simply kept back, so the iteration is branchless and
jittable. The gradient never differentiates the loop: custom_jvp binds
dW/dx = 1 / (e^W (1 + W)), exact by the implicit function theorem (infinite
at the branch point x = -1/e, where W = -1, which is the true behavior).

Domains: k=0 on [-1/e, inf); k=-1 on [-1/e, 0), accurate down to
x ~ -1e-306 (where e^W leaves the normal range). Inputs outside the branch
domain return nan. Near the branch point the achievable accuracy of ANY
double routine degrades like sqrt(eps / (x + 1/e)) because 1 + e x cancels;
the tests carry that allowance.
"""

import math

import jax
import jax.numpy as jnp

_ITERS = 8

# One canonical branch point. The true -1/e lies BETWEEN two doubles, and
# 1/math.e (division) and math.exp(-1) (exp) round to different ones, so a
# domain check against one form rejected the other with nan. Accept one ulp
# of grace below the canonical rounding: any double that is the nearest
# representation of -1/e maps to W = -1.
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


def _branch_series(p, sign):
    # W = -1 + s p - p^2/3 + 11 s p^3/72, s = +1 (k=0), -1 (k=-1)
    return -1.0 + sign * p - p * p / 3.0 + sign * 11.0 * p ** 3 / 72.0


@jax.custom_jvp
def _w0(x):
    x = jnp.asarray(x)
    xs = jnp.where(jnp.isfinite(x), x, 2.0)  # masked lanes get a safe dummy
    p = jnp.sqrt(jnp.maximum(2.0 * (1.0 + jnp.e * xs), 0.0))
    l1 = jnp.log(jnp.maximum(xs, 2.0))
    l2 = jnp.log(l1)
    init = jnp.where(xs < 1.0, _branch_series(p, 1.0), l1 - l2 + l2 / l1)
    w = _halley(xs, init)
    w = jnp.where(x == jnp.inf, jnp.inf, w)
    return jnp.where((x < _BR_LO) | jnp.isnan(x), jnp.nan, w)


@jax.custom_jvp
def _wm1(x):
    x = jnp.asarray(x)
    p = jnp.sqrt(jnp.maximum(2.0 * (1.0 + jnp.e * x), 0.0))
    xn = jnp.where(x < 0.0, x, -0.1)  # masked lanes get a safe dummy
    l1 = jnp.log(-xn)
    init = jnp.where(x > -0.25, l1 - jnp.log(-l1), _branch_series(p, -1.0))
    w = _halley(xn, init)
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
    # on this branch W <= -1, so the boundary derivative is -inf; at exactly
    # W = -1 the shared formula divides by +0 and would give the wrong sign
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
