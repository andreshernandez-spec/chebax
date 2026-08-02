"""dawsn and erfcx (recent jax ships both; kept for the C++ bake path).

Parameter-free, so these are plain functions, not factories. Both use one
hard select at |x| = 6.

- dawsn(x): odd. Inner: x * E(x^2) with E a Chebyshev series on [0, 36] —
  writing it that way (raw x, even series) makes the oddness and the
  gradient at 0 exact. Tail: sign(x) * G((6/x)^2) / |x|.
- erfcx(x): fitted on [0, 6], tail H((6/x)^2) / x; negative x by the
  reflection erfcx(-x) = 2 e^{x^2} - erfcx(x), which correctly overflows
  to inf past x ~ -26.6. The |x| used for the positive-branch input is
  built from a select (not jnp.abs) so the derivative at exactly 0 is
  right; abs would zero it by its tie convention.

Test oracles: D' = 1 - 2 x D and erfcx' = 2 x erfcx - 2/sqrt(pi).
"""

import functools

import jax
import jax.numpy as jnp

from chebax._src.recipes import erf_table as _et
from chebax._src.series import ChebSeries


@functools.lru_cache(maxsize=None)
def _series():
    # built lazily: module-level ChebSeries would initialize the jax backend
    # (and touch the device) as an import side effect. Built OUTSIDE any
    # active trace: the first call can come from inside a lax.cond branch
    # (gammainc's large-a path), and a cached series whose arrays were
    # created under that trace leaks its tracers into every later call.
    with jax.ensure_compile_time_eval():
        return (ChebSeries(_et.DAWSN_E, (0.0, _et.XS * _et.XS)),
                ChebSeries(_et.DAWSN_G, (0.0, 1.0)),
                ChebSeries(_et.ERFCX_C, (0.0, _et.XS)),
                ChebSeries(_et.ERFCX_H, (0.0, 1.0)))


def dawsn(x):
    """Dawson's integral D(x) = e^{-x^2} int_0^x e^{t^2} dt, any real x."""
    E, G, _, _ = _series()
    x = jnp.asarray(x)
    ax = jnp.where(x >= 0, x, -x)
    zi = jnp.where(ax <= _et.XS, x * x, _et.XS * _et.XS)
    inner = x * E(zi)
    xo = jnp.where(ax <= _et.XS, _et.XS, ax)
    tail = jnp.sign(x) * G((_et.XS / xo) ** 2) / xo
    return jnp.where(ax <= _et.XS, inner, tail)


def erfcx(x):
    """Scaled complementary error function e^{x^2} erfc(x), any real x.

    Overflows to inf (correctly) for x below ~ -26.6."""
    _, _, C, H = _series()
    x = jnp.asarray(x)
    ax = jnp.where(x >= 0, x, -x)
    xi = jnp.where(ax <= _et.XS, ax, _et.XS)
    inner = C(xi)
    xo = jnp.where(ax <= _et.XS, _et.XS, ax)
    tail = H((_et.XS / xo) ** 2) / xo
    pos = jnp.where(ax <= _et.XS, inner, tail)
    # clamp the reflection's exp argument for the masked (x >= 0) lanes:
    # exp(x^2) = inf there would poison gradients via 0 * inf
    xn = jnp.where(x < 0, x, 0.0)
    return jnp.where(x >= 0, pos, 2.0 * jnp.exp(xn * xn) - pos)
