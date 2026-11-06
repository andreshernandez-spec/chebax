"""K_v for real v in [0, 10] on x in [1e-6, inf), from log-tables split at x = 8:

    x <= 8:  K = exp(Ltil(v, ln x)) (x/2)^-v,  Ltil = ln[(x/2)^v K_v(x)],
             v in two panels, [0, 1] and [1, 10], around the Gamma-pole feature
    x > 8:   K = exp(Lt(v, 8/x)) sqrt(pi/(2x)) e^-x,  Lt = ln[sqrt(2x/pi) e^x K_v]

Logs of K avoid the I_{+-v} connection formula's cancellation near integer v.
f64 floor ~eps v |ln(x/2)| from the prefactor pow, up to ~3e-14 at v = 10,
x = 1e-6. x is clamped below 1e-6 (values freeze, gradients vanish). Past
x ~ 600 (K < 1e-260), backends that flush subnormals (XLA CPU) lose small
gradient terms such as the 1/(2x) in dK/dx.
"""

import functools
import math

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import besselk_table as _kt
from chebax._src.recipes._common import (
    canon_param,
    canon_float,
    canon_tag,
    check_range,
    deprecated_factory,
    panel_coefs,
    param_coefs,
    param_coefs_der,
    traced_coefs,
)
from chebax._src.series import ChebSeries

# factored out: 2 x overflows past x ~ 9e307 (1.7e38 in f32)
_SQRT_HALF_PI = math.sqrt(math.pi / 2)
_LOG_SQRT_HALF_PI = 0.5 * math.log(math.pi / 2)


def _panel(v):
    if v <= _kt.VSPLIT:
        return _kt.TABLE_IN_LO, 0.0, _kt.VSPLIT
    return _kt.TABLE_IN_HI, _kt.VSPLIT, _kt.VMAX


def _xin(x):
    return jnp.where(x < _kt.XMIN, _kt.XMIN, jnp.where(x <= _kt.XS, x, _kt.XS))


def _xout(x):
    return jnp.where(x <= _kt.XS, _kt.XS, x)


def _eval_k(v, x, ltil, ltail, scaled=False):
    """K_v(x), or e^x K_v(x) when scaled: the tail's own e^x K, since
    exp(log K + x) loses the -1/2 ln x (1.0 for a true 1.25e-10 at x = 1e20)."""
    xi = _xin(x)
    inner = jnp.exp(ltil(jnp.log(xi))) * jnp.power(xi / 2, -v)
    if scaled:
        inner = inner * jnp.exp(xi)
    xo = _xout(x)
    tail = jnp.exp(ltail(_kt.XS / xo)) * (_SQRT_HALF_PI / jnp.sqrt(xo))
    if not scaled:
        tail = tail * jnp.exp(-xo)
    return jnp.where(x <= _kt.XS, inner, tail)


def _eval_k_dnu(v, x, ltil, ltail, ltil_nu, ltail_nu):
    # dK/dv = K dlnK/dv, inner ln K = Ltil - v ln(x/2)
    xi = _xin(x)
    inner_k = jnp.exp(ltil(jnp.log(xi))) * jnp.power(xi / 2, -v)
    inner = inner_k * (ltil_nu(jnp.log(xi)) - jnp.log(xi / 2))
    xo = _xout(x)
    t = _kt.XS / xo
    tail_k = jnp.exp(ltail(t)) * (_SQRT_HALF_PI / jnp.sqrt(xo)) * jnp.exp(-xo)
    tail = tail_k * ltail_nu(t)
    return jnp.where(x <= _kt.XS, inner, tail)


@jax.tree_util.register_pytree_node_class
class BesselK(Recipe):
    """Callable K_v on [1e-6, inf). Build with besselk_frozen(v)."""

    _static_fields = ("v",)
    _series_fields = ("ltil", "ltail")

    def _post_init(self):
        self.v = float(self.v)

    def __call__(self, x):
        return _eval_k(self.v, jnp.asarray(x), self.ltil, self.ltail)


@jax.tree_util.register_pytree_node_class
class BesselKdnu(Recipe):
    """Callable dK_v/dv on [1e-6, inf). Build with besselk_dnu_frozen(v)."""

    _static_fields = ("v",)
    _series_fields = ("ltil", "ltail", "ltil_nu", "ltail_nu")

    def _post_init(self):
        self.v = float(self.v)

    def __call__(self, x):
        return _eval_k_dnu(
            self.v, jnp.asarray(x), self.ltil, self.ltail, self.ltil_nu, self.ltail_nu
        )


@functools.lru_cache(maxsize=128)
def _besselk_cached(v, _tag):
    v = check_range("besselk", "v", v, 0.0, _kt.VMAX)
    table, lo, hi = _panel(v)
    return BesselK(
        v,
        ChebSeries(param_coefs(table, lo, hi, v), (_kt.U0, _kt.U1)),
        ChebSeries(param_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)),
    )


@functools.lru_cache(maxsize=128)
def _besselk_dnu_cached(v, _tag):
    v = check_range("besselk", "v", v, 0.0, _kt.VMAX)
    table, lo, hi = _panel(v)
    return BesselKdnu(
        v,
        ChebSeries(param_coefs(table, lo, hi, v), (_kt.U0, _kt.U1)),
        ChebSeries(param_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(table, lo, hi, v), (_kt.U0, _kt.U1)),
        ChebSeries(param_coefs_der(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)),
    )


def _traced_series(nu):
    c_in = panel_coefs(_kt.TABLE_IN_LO, _kt.TABLE_IN_HI, 0.0, _kt.VSPLIT, _kt.VMAX, nu)
    c_tl = traced_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, nu)
    return (ChebSeries(c_in, (_kt.U0, _kt.U1)), ChebSeries(c_tl, (0.0, 1.0)))


def besselk_scalar(nu, x, scaled=False):
    """K_nu(x), nu a traced scalar in [0, 10] (unchecked under trace),
    differentiable in both. scaled=True gives e^x K (kve), finite past
    x ~ 746 where K underflows."""
    nu = canon_param(nu, "besselk_scalar", "nu")
    ltil, ltail = _traced_series(nu)
    return _eval_k(nu, canon_float(x), ltil, ltail, bool(scaled))


def log_besselk_scalar(nu, x):
    """ln K_nu(x) from the log-tables, finite for every finite x (-inf at
    inf); otherwise as besselk_scalar."""
    nu = canon_param(nu, "log_besselk_scalar", "nu")
    x = jnp.asarray(x)
    ltil, ltail = _traced_series(nu)
    xi = _xin(x)
    inner = ltil(jnp.log(xi)) - nu * jnp.log(xi / 2)
    xo = _xout(x)
    tail = ltail(_kt.XS / xo) + _LOG_SQRT_HALF_PI - 0.5 * jnp.log(xo) - xo
    return jnp.where(x <= _kt.XS, inner, tail)


def besselk(nu, x=None, scaled=False):
    """K_nu(x) for one order nu in [0, 10] (besselk_scalar). Without x it is
    the deprecated 0.4 factory. chebax.pergroup gives one order per group."""
    if x is None:
        return deprecated_factory("besselk", besselk_frozen, nu)
    return besselk_scalar(nu, x, scaled)


def log_besselk(nu, x):
    """ln K_nu(x) for one order nu in [0, 10]: log_besselk_scalar."""
    return log_besselk_scalar(nu, x)


def besselk_frozen(v):
    """K_v on [1e-6, inf) for a concrete v in [0, 10]. Cached."""
    return _besselk_cached(float(v), canon_tag())


def besselk_dnu(v, x=None):
    """dK_v/dv at x for a concrete v; for a traced v, differentiate
    besselk_scalar. Without x it is the deprecated 0.4 factory."""
    if x is None:
        return deprecated_factory("besselk_dnu", besselk_dnu_frozen, v)
    return besselk_dnu_frozen(v)(x)


def besselk_dnu_frozen(v):
    """dK_v/dv on [1e-6, inf) for v in [0, 10]; 0 at v = 0 (K is even in v)."""
    return _besselk_dnu_cached(float(v), canon_tag())
