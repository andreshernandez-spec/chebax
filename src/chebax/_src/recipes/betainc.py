"""Regularized incomplete beta I_x(a, b) for a, b in [0.1, 10], x in [0, 1].

The library's first two-parameter recipe. Structure (DLMF 8.17.7):

    I_x(a, b) = exp( a ln x + b ln(1-x) - ln a - ln B(a,b) + L(a,b,x) )

for x <= 1/2, where L = ln 2F1(a+b, 1; a+1; x) comes from a baked
(x, a, b) tensor (each x-coefficient is a 2-D Chebyshev series in (a, b));
for x > 1/2 the reflection I_x(a,b) = 1 - I_{1-x}(b,a) is used, which
needs the tensor reconstructed at BOTH parameter orders. 1 - x is exact in
f64 for x in [1/2, 1], so the reflection costs no accuracy; users who need
the upper tail 1 - I_x(a, b) to relative accuracy should evaluate
betainc(b, a)(1 - x) themselves.

betainc(a, b) is the eager instance (cached, numpy reconstruction).
betainc_fn(a, b, x) takes a and b as traced scalars (uniform per call,
inside [0.1, 10], unchecked under trace): jax.grad works with respect to
both shape parameters — dI/da and dI/db, which jax itself lacks
(jax#38610) — flowing through gammaln/digamma and chebder along the
parameter axes of the tensor. dI/dx is AD through the formula; its exact
value is the Beta density, which the tests use as an oracle.

Endpoints are handled by hard selects (I(0) = 0, I(1) = 1 exactly, interior
lanes see raw x); gradients at exactly 0 and 1 are masked to the interior
formula's limits and can be infinite in exact math anyway (a < 1 or b < 1).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.recipes import betainc_table as _bt
from chebax._src.series import ChebSeries, _chebval


def _smap(v):
    return 2.0 * (v - _bt.ALO) / (_bt.AHI - _bt.ALO) - 1.0


def _tensor_coefs(a, b):
    """x-coefficient vector of ln F at (a, b): Clenshaw in a, then b. numpy."""
    sa, sb = _smap(a), _smap(b)
    out = []
    for k in range(_bt.TENSOR.shape[0]):
        vb = [algorithms.chebval(sa, _bt.TENSOR[k, :, n]) for n in range(_bt.TENSOR.shape[2])]
        out.append(algorithms.chebval(sb, np.array(vb)))
    return np.array(out)


def _tensor_coefs_traced(a, b):
    sa, sb = _smap(a), _smap(b)
    t = jnp.asarray(_bt.TENSOR)
    va = jax.vmap(jax.vmap(lambda row: _chebval(sa, row, (-1.0, 1.0))))(
        jnp.swapaxes(t, 1, 2))                      # (NX, NB)
    return jax.vmap(lambda row: _chebval(sb, row, (-1.0, 1.0)))(va)  # (NX,)


def _check_param(name, v):
    v = float(v)
    if not _bt.ALO <= v <= _bt.AHI:
        raise ValueError(f"betainc table covers {name} in [{_bt.ALO}, {_bt.AHI}], got {v}")
    return v


def _log_direct(a, b, x, lser):
    """ln I_x(a,b) for x in (0, 1/2], from the direct series."""
    lnB = (jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)
           - jax.scipy.special.gammaln(a + b))
    return a * jnp.log(x) + b * jnp.log1p(-x) - jnp.log(a) - lnB + lser(x)


def _eval_betainc(a, b, x, cab, cba):
    x = jnp.asarray(x)
    interior = (x > 0.0) & (x < 1.0)
    xi = jnp.where(interior, x, 0.25)          # masked lanes get a safe dummy
    xd = jnp.where(xi <= _bt.XSPLIT, xi, 0.25)
    yd = jnp.where(xi <= _bt.XSPLIT, 0.25, 1.0 - xi)   # exact for x in [1/2, 1]
    direct = jnp.exp(_log_direct(a, b, xd, cab))
    reflected = 1.0 - jnp.exp(_log_direct(b, a, yd, cba))
    core = jnp.where(xi <= _bt.XSPLIT, direct, reflected)
    return jnp.where(x <= 0.0, 0.0, jnp.where(x >= 1.0, 1.0, core))


@jax.tree_util.register_pytree_node_class
class BetaInc:
    """Callable I_x(a, b) on x in [0, 1]. Build with betainc(a, b)."""

    def __init__(self, a, b, cab, cba):
        self.a = float(a)
        self.b = float(b)
        self.cab = cab
        self.cba = cba

    def __call__(self, x):
        return _eval_betainc(self.a, self.b, x, self.cab, self.cba)

    def astype(self, dtype):
        return BetaInc(self.a, self.b, self.cab.astype(dtype), self.cba.astype(dtype))

    def __repr__(self):
        return f"BetaInc(a={self.a}, b={self.b}, dtype={self.cab.coef.dtype})"

    def tree_flatten(self):
        return (self.cab, self.cba), (self.a, self.b)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.a, obj.b = aux
        obj.cab, obj.cba = children
        return obj


@functools.lru_cache(maxsize=None)
def betainc(a, b):
    """I_x(a, b) for a, b in [0.1, 10]. Cached per (a, b); no mpmath."""
    a = _check_param("a", a)
    b = _check_param("b", b)
    return BetaInc(a, b,
                   ChebSeries(_tensor_coefs(a, b), (0.0, _bt.XSPLIT)),
                   ChebSeries(_tensor_coefs(b, a), (0.0, _bt.XSPLIT)))


def betainc_fn(a, b, x):
    """I_x(a, b) with a, b as (traceable) scalars, differentiable in all three.

    a and b must be uniform per call and inside [0.1, 10] (unchecked under
    trace). Reconstruction costs two tensor contractions per call, not per
    point; jit constant-folds them when a, b are static."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return _eval_betainc(a, b, x,
                         ChebSeries(_tensor_coefs_traced(a, b), (0.0, _bt.XSPLIT)),
                         ChebSeries(_tensor_coefs_traced(b, a), (0.0, _bt.XSPLIT)))
