"""K_v for real v in [0, 10] on x in [1e-6, inf), from baked log-tables.

Two regions, one hard select at x = 8:

- x in [1e-6, 8]:  K = exp(Ltil(v, ln x)) * (x/2)^(-v), where
                   Ltil = ln[(x/2)^v K_v(x)] is tabulated in u = ln x, with
                   v split into two instantiation-time panels ([0,1], [1,10])
                   around the Gamma-pole cancellation feature near v = 0.
- x > 8:           K = exp(Lt(v, 8/x)) * sqrt(pi/(2x)) * exp(-x), where
                   Lt = ln[sqrt(2x/pi) e^x K_v(x)]. exp(-x) is computed as
                   its own correctly-rounded factor, so accuracy does not
                   decay with x; K underflows to 0 gracefully past x ~ 746.

Tabulating logs of K directly is what dodges risk S3 (the I_{+-v}
connection formula cancels catastrophically near integer v) and the ~15
decades of dynamic range. The f64 floor is set by the (x/2)^(-v) prefactor:
pow costs ~eps * v * |ln(x/2)|, up to ~3e-14 at (v=10, x=1e-6) — same
mechanism as besselj's prefactor floor, larger constant because of the
wider log range.

Below XMIN = 1e-6 the input is clamped (values freeze, gradients vanish);
extend the table via the generator if smaller x is ever needed. dK/dv is 0
at v = 0 by evenness (K_{-v} = K_v). Near the underflow edge (x beyond
~600, K below ~1e-260) relative accuracy of gradients becomes platform
dependent: backends that flush subnormals (XLA CPU) lose small correction
terms like the 1/(2x) piece of dK/dx once products dip under 1e-308.

besselk(v)/besselk_dnu(v) mirror the besselj API (v fixed at instantiation,
cached, no mpmath). besselk_fn(nu, x) additionally takes nu as a traced jax
scalar (uniform per call): the coefficient reconstruction happens inside
the computation, so jax.grad works with respect to nu directly — this is
what lets a Matern kernel learn its smoothness by gradient descent. For
traced nu the domain cannot be checked; keep nu in [0, 10] yourself.
"""

import functools

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import besselk_table as _kt
from chebax._src.recipes._common import (canon_tag, check_range, param_coefs,
                                         param_coefs_der, traced_coefs)
from chebax._src.series import ChebSeries


def _panel(v):
    if v <= _kt.VSPLIT:
        return _kt.TABLE_IN_LO, 0.0, _kt.VSPLIT
    return _kt.TABLE_IN_HI, _kt.VSPLIT, _kt.VMAX


def _xin(x):
    return jnp.where(x < _kt.XMIN, _kt.XMIN, jnp.where(x <= _kt.XS, x, _kt.XS))


def _xout(x):
    return jnp.where(x <= _kt.XS, _kt.XS, x)


def _eval_k(v, x, ltil, ltail):
    xi = _xin(x)
    inner = jnp.exp(ltil(jnp.log(xi))) * jnp.power(xi / 2, -v)
    xo = _xout(x)
    tail = jnp.exp(ltail(_kt.XS / xo)) * jnp.sqrt(jnp.pi / (2 * xo)) * jnp.exp(-xo)
    return jnp.where(x <= _kt.XS, inner, tail)


def _eval_k_dnu(v, x, ltil, ltail, ltil_nu, ltail_nu):
    # dK/dv = K * d(ln K)/dv; inner ln K = Ltil - v ln(x/2), tail ln K has
    # no explicit v beyond Lt
    xi = _xin(x)
    inner_k = jnp.exp(ltil(jnp.log(xi))) * jnp.power(xi / 2, -v)
    inner = inner_k * (ltil_nu(jnp.log(xi)) - jnp.log(xi / 2))
    xo = _xout(x)
    t = _kt.XS / xo
    tail_k = jnp.exp(ltail(t)) * jnp.sqrt(jnp.pi / (2 * xo)) * jnp.exp(-xo)
    tail = tail_k * ltail_nu(t)
    return jnp.where(x <= _kt.XS, inner, tail)


@jax.tree_util.register_pytree_node_class
class BesselK(Recipe):
    """Callable K_v on [1e-6, inf). Build with besselk(v)."""

    _static_fields = ("v",)
    _series_fields = ("ltil", "ltail")

    def _post_init(self):
        self.v = float(self.v)

    def __call__(self, x):
        return _eval_k(self.v, jnp.asarray(x), self.ltil, self.ltail)


@jax.tree_util.register_pytree_node_class
class BesselKdnu(Recipe):
    """Callable dK_v/dv on [1e-6, inf). Build with besselk_dnu(v)."""

    _static_fields = ("v",)
    _series_fields = ("ltil", "ltail", "ltil_nu", "ltail_nu")

    def _post_init(self):
        self.v = float(self.v)

    def __call__(self, x):
        return _eval_k_dnu(self.v, jnp.asarray(x), self.ltil, self.ltail,
                           self.ltil_nu, self.ltail_nu)


@functools.lru_cache(maxsize=128)
def _besselk_cached(v, _tag):
    v = check_range("besselk", "v", v, 0.0, _kt.VMAX)
    table, lo, hi = _panel(v)
    return BesselK(v,
                   ChebSeries(param_coefs(table, lo, hi, v), (_kt.U0, _kt.U1)),
                   ChebSeries(param_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)))


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


def besselk_fn(nu, x):
    """K_nu(x) with nu a (traceable) scalar, differentiable in both arguments.

    nu must be uniform per call and inside [0, 10] (unchecked under trace).
    Costs the coefficient reconstruction (~10k flops) once per call, not per
    point; jit constant-folds it when nu is static.
    """
    nu = jnp.asarray(nu)
    c_lo = traced_coefs(_kt.TABLE_IN_LO, 0.0, _kt.VSPLIT, nu)
    c_hi = traced_coefs(_kt.TABLE_IN_HI, _kt.VSPLIT, _kt.VMAX, nu)
    c_in = jnp.where(nu <= _kt.VSPLIT, c_lo, c_hi)
    c_tl = traced_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, nu)
    return _eval_k(nu, jnp.asarray(x),
                   ChebSeries(c_in, (_kt.U0, _kt.U1)),
                   ChebSeries(c_tl, (0.0, 1.0)))


def besselk(v):
    """K_v on [1e-6, inf) for real v in [0, 10]. Cached per order; no mpmath.

    v may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode."""
    return _besselk_cached(float(v), canon_tag())


def besselk_dnu(v):
    """dK_v/dv on [1e-6, inf) for real v in [0, 10] (the order gradient)."""
    return _besselk_dnu_cached(float(v), canon_tag())
