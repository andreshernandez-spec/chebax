"""J_v for real v in [0, 10] on x in [0, 8], from the baked nu-table.

J_v(x) = (x/2)^v / Gamma(v+1) * g_v(x^2), where g_v(z) = 0F1(; v+1; -z/4) is
entire, so g_v is one short Chebyshev series in z (the ../../bessel evidence,
PROJECT.md section 2). The baked table stores each z-coefficient as a
Chebyshev series in v; instantiation is 25 numpy Clenshaws, once per order,
cached. dJ/dx comes from jax.grad through the series custom_jvp; dJ/dv comes
from chebder along the v axis of the table (besselj_dnu).

Out of range is the caller's problem in v1: x > 8 evaluates the polynomial
outside its interval and diverges (tails are milestone M3); v outside [0, 10]
raises at instantiation.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.recipes import besselj_table as _tab
from chebax._src.series import ChebSeries


def _digamma(z):
    """float64 psi(z) for z >= 1: recurrence up to z >= 15, then the Bernoulli
    series through z^-12 (A&S 6.3.5 / 6.3.18). Truncation < 1e-16 there."""
    acc = 0.0
    while z < 15.0:
        acc -= 1.0 / z
        z += 1.0
    zi = 1.0 / (z * z)
    tail = 1 / 12 - zi * (1 / 120 - zi * (1 / 252 - zi * (1 / 240 - zi * (1 / 132 - zi * (691 / 32760)))))
    return acc + math.log(z) - 0.5 / z - zi * tail


def _coef_at(v):
    """The z-coefficient vector at order v: Clenshaw in v across table rows."""
    t = 2.0 * v / _tab.VMAX - 1.0
    return np.array([algorithms.chebval(t, row) for row in _tab.TABLE])


def _check_order(v):
    v = float(v)
    if not 0.0 <= v <= _tab.VMAX:
        raise ValueError(f"besselj table covers v in [0, {_tab.VMAX}], got {v}")
    return v


@jax.tree_util.register_pytree_node_class
class BesselJ:
    """Callable J_v on [0, 8]. Build with besselj(v)."""

    def __init__(self, v, g):
        self.v = float(v)
        self.g = g
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)

    def __call__(self, x):
        x = jnp.asarray(x)
        return self._inv_gamma * jnp.power(x / 2, self.v) * self.g(x * x)

    def astype(self, dtype):
        return BesselJ(self.v, self.g.astype(dtype))

    def __repr__(self):
        return f"BesselJ(v={self.v}, dtype={self.g.coef.dtype})"

    def tree_flatten(self):
        return (self.g,), self.v

    @classmethod
    def tree_unflatten(cls, v, children):
        obj = object.__new__(cls)
        obj.v = v
        obj.g = children[0]
        obj._inv_gamma = 1.0 / math.gamma(v + 1.0)
        return obj


@jax.tree_util.register_pytree_node_class
class BesselJdnu:
    """Callable dJ_v/dv on (0, 8]. Build with besselj_dnu(v).

    dJ/dv = pref * [(log(x/2) - psi(v+1)) g(z) + g_v(z)], g_v from chebder
    along the v axis. The log makes x = 0 nan; for v = 0 that singularity
    is real."""

    def __init__(self, v, g, gnu):
        self.v = float(v)
        self.g = g
        self.gnu = gnu
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)
        self._psi = _digamma(self.v + 1.0)

    def __call__(self, x):
        x = jnp.asarray(x)
        z = x * x
        pref = self._inv_gamma * jnp.power(x / 2, self.v)
        return pref * ((jnp.log(x / 2) - self._psi) * self.g(z) + self.gnu(z))

    def astype(self, dtype):
        return BesselJdnu(self.v, self.g.astype(dtype), self.gnu.astype(dtype))

    def __repr__(self):
        return f"BesselJdnu(v={self.v}, dtype={self.g.coef.dtype})"

    def tree_flatten(self):
        return (self.g, self.gnu), self.v

    @classmethod
    def tree_unflatten(cls, v, children):
        obj = object.__new__(cls)
        obj.v = v
        obj.g, obj.gnu = children
        obj._inv_gamma = 1.0 / math.gamma(v + 1.0)
        obj._psi = _digamma(v + 1.0)
        return obj


@functools.lru_cache(maxsize=None)
def besselj(v):
    """J_v on [0, 8] for real v in [0, 10]. Cached per order; no mpmath."""
    v = _check_order(v)
    return BesselJ(v, ChebSeries(_coef_at(v), (0.0, _tab.ZMAX)))


@functools.lru_cache(maxsize=None)
def besselj_dnu(v):
    """dJ_v/dv on (0, 8] for real v in [0, 10] (the order gradient)."""
    v = _check_order(v)
    t = 2.0 * v / _tab.VMAX - 1.0
    gnu = np.array([algorithms.chebval(t, algorithms.chebder(row)) for row in _tab.TABLE])
    gnu *= 2.0 / _tab.VMAX
    return BesselJdnu(v, ChebSeries(_coef_at(v), (0.0, _tab.ZMAX)),
                      ChebSeries(gnu, (0.0, _tab.ZMAX)))
