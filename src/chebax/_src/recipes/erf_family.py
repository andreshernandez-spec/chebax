"""dawsn and erfcx (jax ships both; kept for the C++ bake path), split at
|x| = 6. dawsn is x E(x^2) inside, so oddness and the gradient at 0 are exact.
erfcx uses erfcx(-x) = 2 e^{x^2} - erfcx(x) for x < 0, and |x| by select,
since jnp.abs zeroes the derivative at 0.
"""

import functools

import jax
import jax.numpy as jnp

from chebax._src.recipes import erf_table as _et
from chebax._src.recipes._common import canon_tag
from chebax._src.series import ChebSeries


@functools.lru_cache(maxsize=4)
def _series_cached(_tag):
    # lazily, so import does not start the backend, and outside any trace
    # (gammainc's lax.cond), or the cache leaks tracers
    with jax.ensure_compile_time_eval():
        return (
            ChebSeries(_et.DAWSN_E, (0.0, _et.XS * _et.XS)),
            ChebSeries(_et.DAWSN_G, (0.0, 1.0)),
            ChebSeries(_et.ERFCX_C, (0.0, _et.XS)),
            ChebSeries(_et.ERFCX_H, (0.0, 1.0)),
        )


def _series():
    """The four series, keyed per x64 mode."""
    return _series_cached(canon_tag())


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
    """e^{x^2} erfc(x), any real x; inf below x ~ -26.6."""
    _, _, C, H = _series()
    x = jnp.asarray(x)
    ax = jnp.where(x >= 0, x, -x)
    xi = jnp.where(ax <= _et.XS, ax, _et.XS)
    inner = C(xi)
    xo = jnp.where(ax <= _et.XS, _et.XS, ax)
    tail = H((_et.XS / xo) ** 2) / xo
    pos = jnp.where(ax <= _et.XS, inner, tail)
    # masked lanes: exp(x^2) = inf would make 0 * inf gradients
    xn = jnp.where(x < 0, x, 0.0)
    return jnp.where(x >= 0, pos, 2.0 * jnp.exp(xn * xn) - pos)
