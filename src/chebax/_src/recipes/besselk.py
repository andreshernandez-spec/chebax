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
import math

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.recipes import besselk_table as _kt
from chebax._src.series import ChebSeries, _chebval


def _panel(v):
    if v <= _kt.VSPLIT:
        return _kt.TABLE_IN_LO, 0.0, _kt.VSPLIT
    return _kt.TABLE_IN_HI, _kt.VSPLIT, _kt.VMAX


def _coefs(table, lo, hi, v):
    tn = 2.0 * (v - lo) / (hi - lo) - 1.0
    return np.array([algorithms.chebval(tn, row) for row in table])


def _coefs_dnu(table, lo, hi, v):
    tn = 2.0 * (v - lo) / (hi - lo) - 1.0
    d = np.array([algorithms.chebval(tn, algorithms.chebder(row)) for row in table])
    return d * (2.0 / (hi - lo))


def _check_order(v):
    v = float(v)
    if not 0.0 <= v <= _kt.VMAX:
        raise ValueError(f"besselk table covers v in [0, {_kt.VMAX}], got {v}")
    return v


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
    xi = _xin(x)
    inner_k = jnp.exp(ltil(jnp.log(xi))) * jnp.power(xi / 2, -v)
    inner = inner_k * (ltil_nu(jnp.log(xi)) - jnp.log(xi / 2))
    xo = _xout(x)
    t = _kt.XS / xo
    tail_k = jnp.exp(ltail(t)) * jnp.sqrt(jnp.pi / (2 * xo)) * jnp.exp(-xo)
    tail = tail_k * ltail_nu(t)
    return jnp.where(x <= _kt.XS, inner, tail)


@jax.tree_util.register_pytree_node_class
class BesselK:
    """Callable K_v on [1e-6, inf). Build with besselk(v)."""

    def __init__(self, v, ltil, ltail):
        self.v = float(v)
        self.ltil = ltil
        self.ltail = ltail

    def __call__(self, x):
        return _eval_k(self.v, jnp.asarray(x), self.ltil, self.ltail)

    def astype(self, dtype):
        return BesselK(self.v, self.ltil.astype(dtype), self.ltail.astype(dtype))

    def __repr__(self):
        return f"BesselK(v={self.v}, dtype={self.ltil.coef.dtype})"

    def tree_flatten(self):
        return (self.ltil, self.ltail), self.v

    @classmethod
    def tree_unflatten(cls, v, children):
        obj = object.__new__(cls)
        obj.v = v
        obj.ltil, obj.ltail = children
        return obj


@jax.tree_util.register_pytree_node_class
class BesselKdnu:
    """Callable dK_v/dv on [1e-6, inf). Build with besselk_dnu(v)."""

    def __init__(self, v, ltil, ltail, ltil_nu, ltail_nu):
        self.v = float(v)
        self.ltil, self.ltail = ltil, ltail
        self.ltil_nu, self.ltail_nu = ltil_nu, ltail_nu

    def __call__(self, x):
        return _eval_k_dnu(self.v, jnp.asarray(x), self.ltil, self.ltail,
                           self.ltil_nu, self.ltail_nu)

    def astype(self, dtype):
        return BesselKdnu(self.v, *(s.astype(dtype) for s in (
            self.ltil, self.ltail, self.ltil_nu, self.ltail_nu)))

    def __repr__(self):
        return f"BesselKdnu(v={self.v}, dtype={self.ltil.coef.dtype})"

    def tree_flatten(self):
        return (self.ltil, self.ltail, self.ltil_nu, self.ltail_nu), self.v

    @classmethod
    def tree_unflatten(cls, v, children):
        obj = object.__new__(cls)
        obj.v = v
        obj.ltil, obj.ltail, obj.ltil_nu, obj.ltail_nu = children
        return obj


@functools.lru_cache(maxsize=None)
def besselk(v):
    """K_v on [1e-6, inf) for real v in [0, 10]. Cached per order; no mpmath."""
    v = _check_order(v)
    table, lo, hi = _panel(v)
    return BesselK(v,
                   ChebSeries(_coefs(table, lo, hi, v), (_kt.U0, _kt.U1)),
                   ChebSeries(_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)))


@functools.lru_cache(maxsize=None)
def besselk_dnu(v):
    """dK_v/dv on [1e-6, inf) for real v in [0, 10] (the order gradient)."""
    v = _check_order(v)
    table, lo, hi = _panel(v)
    return BesselKdnu(
        v,
        ChebSeries(_coefs(table, lo, hi, v), (_kt.U0, _kt.U1)),
        ChebSeries(_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)),
        ChebSeries(_coefs_dnu(table, lo, hi, v), (_kt.U0, _kt.U1)),
        ChebSeries(_coefs_dnu(_kt.TABLE_TAIL, 0.0, _kt.VMAX, v), (0.0, 1.0)),
    )


def _traced_coefs(table, lo, hi, nu):
    tn = 2.0 * (nu - lo) / (hi - lo) - 1.0
    return jax.vmap(lambda row: _chebval(tn, row, (-1.0, 1.0)))(jnp.asarray(table))


def besselk_fn(nu, x):
    """K_nu(x) with nu a (traceable) scalar, differentiable in both arguments.

    nu must be uniform per call and inside [0, 10] (unchecked under trace).
    Costs the coefficient reconstruction (~10k flops) once per call, not per
    point; jit constant-folds it when nu is static.
    """
    nu = jnp.asarray(nu)
    c_lo = _traced_coefs(_kt.TABLE_IN_LO, 0.0, _kt.VSPLIT, nu)
    c_hi = _traced_coefs(_kt.TABLE_IN_HI, _kt.VSPLIT, _kt.VMAX, nu)
    c_in = jnp.where(nu <= _kt.VSPLIT, c_lo, c_hi)
    c_tl = _traced_coefs(_kt.TABLE_TAIL, 0.0, _kt.VMAX, nu)
    return _eval_k(nu, jnp.asarray(x),
                   ChebSeries(c_in, (_kt.U0, _kt.U1)),
                   ChebSeries(c_tl, (0.0, 1.0)))
