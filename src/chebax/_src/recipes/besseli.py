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

from chebax._src.pytree import Recipe
from chebax._src.recipes import besseli_table as _it
from chebax._src.recipes._common import (canon_tag, check_range, digamma64, param_coefs,
                                         param_coefs_der, traced_coefs)
from chebax._src.recipes.besselk import besselk, besselk_fn
from chebax._src.series import ChebSeries

# above this the tail switches to a single combined exp: e^x alone overflows
# at x ~ 709.78 while I_v(x) stays representable to x ~ 713
_X_SPLIT_EXP = 700.0


def _eval_i(v, inv_gamma, x, lh, lt, scaled):
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    static_zero = isinstance(v, (int, float)) and v == 0.0
    if static_zero:
        # (x/2)^0 = 1, but the generic power rule is 0 * (x/2)^-1 at the
        # origin: NaN gradient for a constant prefactor
        pref = 1.0
    elif isinstance(v, (int, float)):
        pref = jnp.power(xi / 2, v)
    else:
        # traced v == 0: constant prefactor 1. The power must also see a
        # safe dummy input, or reverse mode transposes its 0 * inf origin
        # coefficient into NaN through the select.
        veq0 = jnp.asarray(v) == 0.0
        pref = jnp.where(veq0, 1.0, jnp.power(jnp.where(veq0, 1.0, xi) / 2, v))
    inner = jnp.exp(lh(xi * xi)) * pref * inv_gamma
    if scaled:
        inner = inner * jnp.exp(-xi)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    lt_val = lt(_it.XS / xo)
    if scaled:
        tail = jnp.exp(lt_val) / jnp.sqrt(2 * jnp.pi * xo)
    else:
        # keep e^x as its own correctly rounded factor while it fits (a
        # combined exp costs eps*x of relative error); past the exp
        # overflow point only the combined form can express the result
        xc = jnp.where(xo > _X_SPLIT_EXP, _X_SPLIT_EXP, xo)
        split = jnp.exp(lt_val) / jnp.sqrt(2 * jnp.pi * xc) * jnp.exp(xc)
        combined = jnp.exp(lt_val + xo - 0.5 * jnp.log(2 * jnp.pi * xo))
        tail = jnp.where(xo > _X_SPLIT_EXP, combined, split)
    out = jnp.where(x <= _it.XS, inner, tail)
    if not scaled:
        out = jnp.where(jnp.isinf(x), jnp.inf, out)
    return jnp.where(jnp.isnan(x), jnp.nan, out)


@jax.tree_util.register_pytree_node_class
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


@jax.tree_util.register_pytree_node_class
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


@functools.lru_cache(maxsize=128)
def _besseli_cached(v, scaled, _tag):
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    return BesselI(v, scaled,
                   ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
                   ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)))


@jax.tree_util.register_pytree_node_class
class _BesselIdnu0(Recipe):
    """dI_v/dv at v = 0: exactly -K_0(x) (DLMF 10.38.2).

    The generic I * dlog form is catastrophic here: the true value is
    exponentially SMALL (e^-x) while I_0 is exponentially LARGE, so table
    noise in the log derivative is amplified by e^2x (measured: +3.8e25
    at x = 100 for a true value of -4.7e-45)."""

    _static_fields = ("scaled",)
    _series_fields = ("k0",)

    def _post_init(self):
        self.scaled = bool(self.scaled)

    def __call__(self, x):
        x = jnp.asarray(x)
        k = self.k0(x)
        return -(k * jnp.exp(-x)) if self.scaled else -k


@functools.lru_cache(maxsize=128)
def _besseli_dnu_cached(v, scaled, _tag):
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    if v == 0.0:
        return _BesselIdnu0(scaled, besselk(0.0))
    return BesselIdnu(
        v, scaled,
        ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs_der(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
    )


def _besseli_fn_impl(nu, x, scaled):
    inv_gamma = jnp.exp(-jax.scipy.special.gammaln(nu + 1.0))
    return _eval_i(nu, inv_gamma, x,
                   ChebSeries(traced_coefs(_it.TABLE_IN, 0.0, _it.VMAX, nu), (0.0, _it.ZMAX)),
                   ChebSeries(traced_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, nu), (0.0, 1.0)),
                   scaled)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _besseli_fn(nu, x, scaled):
    return _besseli_fn_impl(nu, x, scaled)


@_besseli_fn.defjvp
def _besseli_fn_jvp(scaled, primals, tangents):
    nu, x = primals
    dnu, dx = tangents
    val = _besseli_fn_impl(nu, x, scaled)
    _, d_x = jax.jvp(lambda xx: _besseli_fn_impl(nu, xx, scaled), (x,), (dx,))
    # nu direction: AD of the table reconstruction, except at nu = 0 where
    # the true derivative is the exponentially small -K_0 (see _BesselIdnu0)
    _, d_nu_ad = jax.jvp(lambda n: _besseli_fn_impl(n, x, scaled), (nu,), (dnu,))
    exact0 = -besselk_fn(jnp.zeros_like(nu), x)
    if scaled:
        exact0 = exact0 * jnp.exp(-x)
    d_nu = jnp.where(nu == 0.0, exact0 * dnu, d_nu_ad)
    return val, d_x + d_nu


def besseli_fn(nu, x, scaled=False):
    """I_nu(x) with nu a (traceable) scalar, differentiable in both arguments.

    nu must be uniform per call and inside [0, 10] (unchecked under trace).
    Uses exp(gammaln) for the traced 1/Gamma(nu+1). The nu gradient at
    nu = 0 is the exact -K_0(x) (clamped below x = 1e-6 with the K table)."""
    return _besseli_fn(jnp.asarray(nu), jnp.asarray(x), bool(scaled))


def besseli(v, scaled=False):
    """I_v on x >= 0 for real v in [0, 10] (e^-x I_v if scaled). Cached.

    v may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode."""
    return _besseli_cached(float(v), bool(scaled), canon_tag())


def besseli_dnu(v, scaled=False):
    """dI_v/dv on x > 0 for real v in [0, 10] (times e^-x if scaled).

    v = 0 uses the exact identity dI_v/dv|_0 = -K_0(x) (clamped below
    x = 1e-6 with the K table)."""
    return _besseli_dnu_cached(float(v), bool(scaled), canon_tag())
