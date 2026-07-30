"""I_v for real v in [0, 10] on x >= 0, from baked log-tables.

Two regions, one hard select at x = 8:

- x in [0, 8]:  I = exp(Lh(v, x^2)) * (x/2)^v / Gamma(v+1), with
                Lh = ln[Gamma(v+1) (x/2)^(-v) I_v(x)] tabulated in z = x^2
                (the factored entire part, logged: I > 0 everywhere).
- x > 8:        I = exp(Lt(v, 8/x)) * e^x / sqrt(2 pi x), with
                Lt = ln[sqrt(2 pi x) e^(-x) I_v(x)].

scaled=True returns e^(-x) I_v(x) (scipy's ive): the tail then never forms
e^x, so it stays finite past x ~ 709 where the unscaled value correctly
overflows to inf. The f64 floor is the familiar prefactor pow term,
~eps * v * |ln(x/2)|.

besseli(v)/besseli_dnu(v) mirror the besselj API; besseli_fn(nu, x)
additionally takes nu as a traced jax scalar (uniform per call, in [0, 10],
unchecked under trace), differentiable in both arguments.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.recipes import besseli_table as _it
from chebax._src.recipes.besselj import _digamma
from chebax._src.series import ChebSeries, _chebval


def _coefs(table, v):
    tn = 2.0 * v / _it.VMAX - 1.0
    return np.array([algorithms.chebval(tn, row) for row in table])


def _coefs_dnu(table, v):
    tn = 2.0 * v / _it.VMAX - 1.0
    d = np.array([algorithms.chebval(tn, algorithms.chebder(row)) for row in table])
    return d * (2.0 / _it.VMAX)


def _check_order(v):
    v = float(v)
    if not 0.0 <= v <= _it.VMAX:
        raise ValueError(f"besseli table covers v in [0, {_it.VMAX}], got {v}")
    return v


def _eval_i(v, inv_gamma, x, lh, lt, scaled):
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    inner = jnp.exp(lh(xi * xi)) * jnp.power(xi / 2, v) * inv_gamma
    if scaled:
        inner = inner * jnp.exp(-xi)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    tail = jnp.exp(lt(_it.XS / xo)) / jnp.sqrt(2 * jnp.pi * xo)
    if not scaled:
        tail = tail * jnp.exp(xo)
    return jnp.where(x <= _it.XS, inner, tail)


@jax.tree_util.register_pytree_node_class
class BesselI:
    """Callable I_v (or e^-x I_v if scaled) on x >= 0. Build with besseli()."""

    def __init__(self, v, scaled, lh, lt):
        self.v = float(v)
        self.scaled = bool(scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)
        self.lh, self.lt = lh, lt

    def __call__(self, x):
        return _eval_i(self.v, self._inv_gamma, jnp.asarray(x), self.lh, self.lt, self.scaled)

    def astype(self, dtype):
        return BesselI(self.v, self.scaled, self.lh.astype(dtype), self.lt.astype(dtype))

    def __repr__(self):
        return f"BesselI(v={self.v}, scaled={self.scaled}, dtype={self.lh.coef.dtype})"

    def tree_flatten(self):
        return (self.lh, self.lt), (self.v, self.scaled)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.v, obj.scaled = aux
        obj._inv_gamma = 1.0 / math.gamma(obj.v + 1.0)
        obj.lh, obj.lt = children
        return obj


@jax.tree_util.register_pytree_node_class
class BesselIdnu:
    """Callable dI_v/dv (or e^-x dI_v/dv if scaled). Build with besseli_dnu()."""

    def __init__(self, v, scaled, lh, lt, lh_nu, lt_nu):
        self.v = float(v)
        self.scaled = bool(scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)
        self._psi = _digamma(self.v + 1.0)
        self.lh, self.lt, self.lh_nu, self.lt_nu = lh, lt, lh_nu, lt_nu

    def __call__(self, x):
        x = jnp.asarray(x)
        val = _eval_i(self.v, self._inv_gamma, x, self.lh, self.lt, self.scaled)
        xi = jnp.where(x <= _it.XS, x, _it.XS)
        inner_term = self.lh_nu(xi * xi) + jnp.log(xi / 2) - self._psi
        xo = jnp.where(x <= _it.XS, _it.XS, x)
        tail_term = self.lt_nu(_it.XS / xo)
        return val * jnp.where(x <= _it.XS, inner_term, tail_term)

    def astype(self, dtype):
        return BesselIdnu(self.v, self.scaled, *(s.astype(dtype) for s in (
            self.lh, self.lt, self.lh_nu, self.lt_nu)))

    def __repr__(self):
        return f"BesselIdnu(v={self.v}, scaled={self.scaled}, dtype={self.lh.coef.dtype})"

    def tree_flatten(self):
        return (self.lh, self.lt, self.lh_nu, self.lt_nu), (self.v, self.scaled)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.v, obj.scaled = aux
        obj._inv_gamma = 1.0 / math.gamma(obj.v + 1.0)
        obj._psi = _digamma(obj.v + 1.0)
        obj.lh, obj.lt, obj.lh_nu, obj.lt_nu = children
        return obj


@functools.lru_cache(maxsize=None)
def besseli(v, scaled=False):
    """I_v on x >= 0 for real v in [0, 10] (e^-x I_v if scaled). Cached."""
    v = _check_order(v)
    return BesselI(v, scaled,
                   ChebSeries(_coefs(_it.TABLE_IN, v), (0.0, _it.ZMAX)),
                   ChebSeries(_coefs(_it.TABLE_TAIL, v), (0.0, 1.0)))


@functools.lru_cache(maxsize=None)
def besseli_dnu(v, scaled=False):
    """dI_v/dv on x > 0 for real v in [0, 10] (times e^-x if scaled)."""
    v = _check_order(v)
    return BesselIdnu(
        v, scaled,
        ChebSeries(_coefs(_it.TABLE_IN, v), (0.0, _it.ZMAX)),
        ChebSeries(_coefs(_it.TABLE_TAIL, v), (0.0, 1.0)),
        ChebSeries(_coefs_dnu(_it.TABLE_IN, v), (0.0, _it.ZMAX)),
        ChebSeries(_coefs_dnu(_it.TABLE_TAIL, v), (0.0, 1.0)),
    )


def _traced_coefs(table, nu):
    tn = 2.0 * nu / _it.VMAX - 1.0
    return jax.vmap(lambda row: _chebval(tn, row, (-1.0, 1.0)))(jnp.asarray(table))


def besseli_fn(nu, x, scaled=False):
    """I_nu(x) with nu a (traceable) scalar, differentiable in both arguments.

    nu must be uniform per call and inside [0, 10] (unchecked under trace).
    Uses exp(gammaln) for the traced 1/Gamma(nu+1).
    """
    nu = jnp.asarray(nu)
    inv_gamma = jnp.exp(-jax.scipy.special.gammaln(nu + 1.0))
    return _eval_i(nu, inv_gamma, jnp.asarray(x),
                   ChebSeries(_traced_coefs(_it.TABLE_IN, nu), (0.0, _it.ZMAX)),
                   ChebSeries(_traced_coefs(_it.TABLE_TAIL, nu), (0.0, 1.0)),
                   scaled)
