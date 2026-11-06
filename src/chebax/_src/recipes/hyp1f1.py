"""Kummer's M(a, b, x) = 1F1(a; b; x) for (a, b) in [0.1, 10]^2, x >= 0, from
log-tables split at x = 30:

    x <= 30:  ln M = log1p((a/b) x R), ln R tabulated (hyp1f1_gen says why
              not ln M), exact relative accuracy down to x = 0
    x > 30:   ln M = lgamma(b) - lgamma(a) + x + (a - b) ln x + T(30/x),
              DLMF 13.7.1 with its log-remainder T tabulated

Relative error grows as eps x. M overflows f64 past x ~ 700, where
log_hyp1f1_scalar still works. x < 0 gives nan (use the Kummer transform
e^x M(b-a, b, -x) when b - a is in the box). jax's hyp1f1 truncates at 1e-8
and is unstable (jax#21503).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from chebax import domains as _dom
from chebax._src import algorithms
from chebax._src.pytree import Recipe
from chebax._src.recipes import hyp1f1_table as _ht
from chebax._src.recipes._common import (
    canon_param,
    canon_tag,
    check_range,
    deprecated_factory,
)
from chebax._src.series import ChebSeries, _chebval

_LNLO, _LNHI = float(np.log(_ht.ALO)), float(np.log(_ht.AHI))


def _smap_log(v):
    return 2.0 * (jnp.log(v) - _LNLO) / (_LNHI - _LNLO) - 1.0


def _smap_raw(v):
    return 2.0 * (v - _ht.ALO) / (_ht.AHI - _ht.ALO) - 1.0


def _coefs_eager(tensor, sa, sb):
    """Argument coefficients at mapped (sa, sb), Clenshaw in a then b (numpy)."""
    out = []
    for k in range(tensor.shape[0]):
        vb = [algorithms.chebval(sa, tensor[k, :, n]) for n in range(tensor.shape[2])]
        out.append(algorithms.chebval(sb, np.array(vb)))
    return np.array(out)


def _coefs_traced(tensor, sa, sb):
    t = jnp.asarray(tensor)
    va = jax.vmap(jax.vmap(lambda row: _chebval(sa, row, (-1.0, 1.0))))(
        jnp.swapaxes(t, 1, 2)
    )  # (NX, NB)
    c = jax.vmap(lambda row: _chebval(sb, row, (-1.0, 1.0)))(va)
    # without the barrier, XLA fused this into the float32 GPU gradient in
    # (a, b) as one kernel that took 2.7 s per call whatever N (experiments/34)
    return jax.lax.optimization_barrier(c)


def _eval_logs(a, b, x, lnr, ltail):
    """(ln M inner, ln M tail), masked lanes fed finite dummies. x = 0 stays
    in the inner lane, so dM/dx there is a/b, not a constant branch's 0."""
    xd = jnp.where((x >= 0.0) & (x <= _ht.XS), x, 1.0)
    lm_in = jnp.log1p(a / b * xd * jnp.exp(lnr(xd)))
    xo = jnp.where(x > _ht.XS, x, 2.0 * _ht.XS)
    lm_tl = (
        jax.scipy.special.gammaln(b)
        - jax.scipy.special.gammaln(a)
        + xo
        + (a - b) * jnp.log(xo)
        + ltail(_ht.XS / xo)
    )
    return lm_in, lm_tl


def eval_hyp1f1(a, b, x, lnr, ltail, log):
    x = jnp.asarray(x)
    lm_in, lm_tl = _eval_logs(a, b, x, lnr, ltail)
    lm = jnp.where(x <= _ht.XS, lm_in, lm_tl)
    # at x = inf the tail is inf + (a-b) inf, nan when a = b; M -> inf
    big = x == jnp.inf
    out = lm if log else jnp.exp(jnp.where(big, 0.0, lm))
    out = jnp.where(big, jnp.inf, out)
    out = jnp.where(x < 0.0, jnp.nan, out)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)


@jax.tree_util.register_pytree_node_class
class Hyp1F1(Recipe):
    """Callable M(a, b, x) on x in [0, inf). Build with hyp1f1_frozen(a, b)."""

    _static_fields = ("a", "b")
    _series_fields = ("lnr", "ltail")

    def _post_init(self):
        self.a = float(self.a)
        self.b = float(self.b)

    def __call__(self, x):
        return eval_hyp1f1(self.a, self.b, x, self.lnr, self.ltail, log=False)


def _series_at(a, b):
    sa, sb = float(_smap_log(a)), float(_smap_log(b))
    st = float(_smap_raw(b))
    return (
        ChebSeries(_coefs_eager(_ht.TABLE_IN, sa, sb), (0.0, _ht.XS)),
        ChebSeries(_coefs_eager(_ht.TABLE_TAIL, sa, st), (0.0, 1.0)),
    )


@functools.lru_cache(maxsize=128)
def _hyp1f1_cached(a, b, _tag):
    a = check_range("hyp1f1", "a", a, *_dom.HYP1F1[:2])
    b = check_range("hyp1f1", "b", b, *_dom.HYP1F1[:2])
    return Hyp1F1(a, b, *_series_at(a, b))


def _traced_series(a, b):
    sa, sb = _smap_log(a), _smap_log(b)
    st = _smap_raw(b)
    return (
        ChebSeries(_coefs_traced(_ht.TABLE_IN, sa, sb), (0.0, _ht.XS)),
        ChebSeries(_coefs_traced(_ht.TABLE_TAIL, sa, st), (0.0, 1.0)),
    )


def hyp1f1_scalar(a, b, x):
    """M(a, b, x), a and b traced scalars in [0.1, 10] (unchecked under trace),
    differentiable in all three."""
    a = canon_param(a, "hyp1f1_scalar", "a")
    b = canon_param(b, "hyp1f1_scalar", "b")
    lnr, ltail = _traced_series(a, b)
    return eval_hyp1f1(a, b, jnp.asarray(x), lnr, ltail, log=False)


def log_hyp1f1_scalar(a, b, x):
    """ln M(a, b, x) as hyp1f1_scalar, with no overflow at large x."""
    a = canon_param(a, "log_hyp1f1_scalar", "a")
    b = canon_param(b, "log_hyp1f1_scalar", "b")
    lnr, ltail = _traced_series(a, b)
    return eval_hyp1f1(a, b, jnp.asarray(x), lnr, ltail, log=True)


def hyp1f1(a, b, x=None):
    """M(a, b, x) for one pair (a, b) in [0.1, 10]^2 (hyp1f1_scalar). Without
    x it is the deprecated 0.4 factory. chebax.pergroup gives one pair per group."""
    if x is None:
        return deprecated_factory("hyp1f1", hyp1f1_frozen, a, b)
    return hyp1f1_scalar(a, b, x)


def log_hyp1f1(a, b, x):
    """ln M(a, b, x) for one pair (a, b) in [0.1, 10]^2: log_hyp1f1_scalar."""
    return log_hyp1f1_scalar(a, b, x)


def hyp1f1_frozen(a, b):
    """M(a, b, x) on [0, inf) for concrete (a, b) in [0.1, 10]^2. Cached."""
    return _hyp1f1_cached(float(a), float(b), canon_tag())
