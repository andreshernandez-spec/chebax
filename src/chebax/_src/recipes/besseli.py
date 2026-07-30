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

from chebax._src.pytree import Recipe, register_recipe
from chebax._src.recipes import besseli_table as _it
from chebax._src.recipes._common import (check_range, digamma64, param_coefs,
                                         param_coefs_der, traced_coefs)
from chebax._src.series import ChebSeries


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


@register_recipe
class BesselI(Recipe):
    """Callable I_v (or e^-x I_v if scaled) on x >= 0. Build with besseli()."""

    _static_fields = ("v", "scaled")
    _series_fields = ("lh", "lt")

    def _post_init(self):
        self.v = float(self.v)
        self.scaled = bool(self.scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)

    def __call__(self, x):
        return _eval_i(self.v, self._inv_gamma, jnp.asarray(x), self.lh, self.lt, self.scaled)


@register_recipe
class BesselIdnu(Recipe):
    """Callable dI_v/dv (or e^-x dI_v/dv if scaled). Build with besseli_dnu()."""

    _static_fields = ("v", "scaled")
    _series_fields = ("lh", "lt", "lh_nu", "lt_nu")

    def _post_init(self):
        self.v = float(self.v)
        self.scaled = bool(self.scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)
        self._psi = digamma64(self.v + 1.0)

    def __call__(self, x):
        x = jnp.asarray(x)
        val = _eval_i(self.v, self._inv_gamma, x, self.lh, self.lt, self.scaled)
        # dI/dv = I * (dLh/dv + ln(x/2) - psi(v+1)) inner, I * dLt/dv tail
        xi = jnp.where(x <= _it.XS, x, _it.XS)
        inner_term = self.lh_nu(xi * xi) + jnp.log(xi / 2) - self._psi
        xo = jnp.where(x <= _it.XS, _it.XS, x)
        tail_term = self.lt_nu(_it.XS / xo)
        return val * jnp.where(x <= _it.XS, inner_term, tail_term)


@functools.lru_cache(maxsize=None)
def besseli(v, scaled=False):
    """I_v on x >= 0 for real v in [0, 10] (e^-x I_v if scaled). Cached."""
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    return BesselI(v, scaled,
                   ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
                   ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)))


@functools.lru_cache(maxsize=None)
def besseli_dnu(v, scaled=False):
    """dI_v/dv on x > 0 for real v in [0, 10] (times e^-x if scaled)."""
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    return BesselIdnu(
        v, scaled,
        ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs_der(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
    )


def besseli_fn(nu, x, scaled=False):
    """I_nu(x) with nu a (traceable) scalar, differentiable in both arguments.

    nu must be uniform per call and inside [0, 10] (unchecked under trace).
    Uses exp(gammaln) for the traced 1/Gamma(nu+1).
    """
    nu = jnp.asarray(nu)
    inv_gamma = jnp.exp(-jax.scipy.special.gammaln(nu + 1.0))
    return _eval_i(nu, inv_gamma, jnp.asarray(x),
                   ChebSeries(traced_coefs(_it.TABLE_IN, 0.0, _it.VMAX, nu), (0.0, _it.ZMAX)),
                   ChebSeries(traced_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, nu), (0.0, 1.0)),
                   scaled)
