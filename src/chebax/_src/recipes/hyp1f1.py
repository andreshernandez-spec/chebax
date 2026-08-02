"""Kummer's confluent hypergeometric M(a, b, x) = 1F1(a; b; x) for
(a, b) in [0.1, 10]^2, x in [0, inf), from baked log-tables.

Two regions, one hard select at x = XS = 30:

- x in [0, 30]:  M = 1 + (a/b) x R, with ln R tabulated (R(a, b, 0) = 1;
                 see hyp1f1_gen for why ln M itself is the wrong kernel).
                 ln M = log1p((a/b) x exp(ln R)) keeps full relative
                 accuracy down to x = 0.
- x > 30:        ln M = lgamma(b) - lgamma(a) + x + (a - b) ln x + T(30/x),
                 the DLMF 13.7.1 asymptotic with its log-remainder T
                 tabulated. Accuracy does not decay with x; M itself
                 overflows f64 once ln M > ~709 (x ~ 700), where
                 log_hyp1f1_fn is the usable form.

M is positive on the box and the error contract is relative, degrading
as eps |ln M| ~ eps x for large x (the exact e^x factor carries the
dynamic range, the besselk/gammainc contract). x = 0 returns 1 exactly;
x < 0 returns nan (v1 covers x >= 0 only; for x < 0 use the Kummer
transform M(a, b, x) = e^x M(b-a, b, -x) yourself when b - a lands in
the box); nan propagates.

hyp1f1(a, b) is the eager cached instance (numpy reconstruction).
hyp1f1_fn(a, b, x) takes both parameters as traced scalars (uniform per
call, inside [0.1, 10], unchecked under trace): jax.grad works with
respect to a and b through the tables' log-mapped parameter axes and
gammaln. dM/dx is AD through the formula; its exact value is
(a/b) M(a+1, b+1, x), which the tests use as an oracle where a+1, b+1
stay in the box. The gradient at exactly x = 0 flows through the hard
select's masked lane and returns 0, not the true limit a/b.

jax.scipy.special.hyp1f1 truncates a term-recurrence at tolerance 1e-8
and is documented-unstable (jax#21503); this recipe is fixed-degree
polynomial evaluation against 40-digit tables.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.pytree import Recipe
from chebax._src.recipes import hyp1f1_table as _ht
from chebax._src.recipes._common import canon_tag, check_range
from chebax._src.series import ChebSeries, _chebval

_LNLO, _LNHI = float(np.log(_ht.ALO)), float(np.log(_ht.AHI))


def _smap_log(v):
    return 2.0 * (jnp.log(v) - _LNLO) / (_LNHI - _LNLO) - 1.0


def _smap_raw(v):
    return 2.0 * (v - _ht.ALO) / (_ht.AHI - _ht.ALO) - 1.0


def _coefs_eager(tensor, sa, sb):
    """Argument-coefficient vector at mapped (sa, sb): Clenshaw in a then
    b across the tensor's parameter axes. numpy."""
    out = []
    for k in range(tensor.shape[0]):
        vb = [algorithms.chebval(sa, tensor[k, :, n])
              for n in range(tensor.shape[2])]
        out.append(algorithms.chebval(sb, np.array(vb)))
    return np.array(out)


def _coefs_traced(tensor, sa, sb):
    t = jnp.asarray(tensor)
    va = jax.vmap(jax.vmap(lambda row: _chebval(sa, row, (-1.0, 1.0))))(
        jnp.swapaxes(t, 1, 2))                                   # (NX, NB)
    return jax.vmap(lambda row: _chebval(sb, row, (-1.0, 1.0)))(va)


def _eval_logs(a, b, x, lnr, ltail):
    """(ln M inner, ln M tail), masked lanes fed finite dummies.

    x = 0 belongs to the INNER lane, not to a hard select on the value:
    log1p there is exactly 0, so the value is unchanged, and its
    derivative is the exact a/b instead of the 0 a constant branch gives
    (review, 2026-08-02: grad of hyp1f1_fn(2, 3, .) at 0 was 0 for a true
    2/3)."""
    xd = jnp.where((x >= 0.0) & (x <= _ht.XS), x, 1.0)
    lm_in = jnp.log1p(a / b * xd * jnp.exp(lnr(xd)))
    xo = jnp.where(x > _ht.XS, x, 2.0 * _ht.XS)
    lm_tl = (jax.scipy.special.gammaln(b) - jax.scipy.special.gammaln(a)
             + xo + (a - b) * jnp.log(xo) + ltail(_ht.XS / xo))
    return lm_in, lm_tl


def eval_hyp1f1(a, b, x, lnr, ltail, log):
    x = jnp.asarray(x)
    lm_in, lm_tl = _eval_logs(a, b, x, lnr, ltail)
    lm = jnp.where(x <= _ht.XS, lm_in, lm_tl)
    # M ~ Gamma(b)/Gamma(a) e^x x^(a-b) grows without bound for every a > 0
    # in the box, but the tail bracket is inf + (a-b) inf at x = inf, which
    # is nan when a = b: mask it and set the limit by hand
    big = x == jnp.inf
    out = lm if log else jnp.exp(jnp.where(big, 0.0, lm))
    out = jnp.where(big, jnp.inf, out)
    out = jnp.where(x < 0.0, jnp.nan, out)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b),
                     jnp.nan, out)


@jax.tree_util.register_pytree_node_class
class Hyp1F1(Recipe):
    """Callable M(a, b, x) on x in [0, inf). Build with hyp1f1(a, b)."""

    _static_fields = ("a", "b")
    _series_fields = ("lnr", "ltail")

    def _post_init(self):
        self.a = float(self.a)
        self.b = float(self.b)

    def __call__(self, x):
        return eval_hyp1f1(self.a, self.b, x, self.lnr, self.ltail,
                           log=False)


def _series_at(a, b):
    sa, sb = float(_smap_log(a)), float(_smap_log(b))
    st = float(_smap_raw(b))
    return (ChebSeries(_coefs_eager(_ht.TABLE_IN, sa, sb), (0.0, _ht.XS)),
            ChebSeries(_coefs_eager(_ht.TABLE_TAIL, sa, st), (0.0, 1.0)))


@functools.lru_cache(maxsize=128)
def _hyp1f1_cached(a, b, _tag):
    a = check_range("hyp1f1", "a", a, _ht.ALO, _ht.AHI)
    b = check_range("hyp1f1", "b", b, _ht.ALO, _ht.AHI)
    return Hyp1F1(a, b, *_series_at(a, b))


def _traced_series(a, b):
    sa, sb = _smap_log(a), _smap_log(b)
    st = _smap_raw(b)
    return (ChebSeries(_coefs_traced(_ht.TABLE_IN, sa, sb), (0.0, _ht.XS)),
            ChebSeries(_coefs_traced(_ht.TABLE_TAIL, sa, st), (0.0, 1.0)))


def hyp1f1_fn(a, b, x):
    """M(a, b, x) with a, b (traceable) scalars, differentiable in all
    three arguments.

    a and b must be uniform per call and inside [0.1, 10] (unchecked
    under trace). Reconstruction costs two tensor contractions per call,
    not per point; jit constant-folds them when a and b are static."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    lnr, ltail = _traced_series(a, b)
    return eval_hyp1f1(a, b, jnp.asarray(x), lnr, ltail, log=False)


def log_hyp1f1_fn(a, b, x):
    """ln M(a, b, x), same contract as hyp1f1_fn but with no overflow
    ceiling: valid for arbitrarily large x (ln M ~ x), where M itself
    leaves f64 past x ~ 700. This is the form a likelihood wants."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    lnr, ltail = _traced_series(a, b)
    return eval_hyp1f1(a, b, jnp.asarray(x), lnr, ltail, log=True)


def hyp1f1(a, b):
    """M(a, b, x) on [0, inf) for (a, b) in [0.1, 10]^2. Cached per
    (a, b); no mpmath.

    a and b may be python numbers or concrete jax scalars; the bounded
    cache is keyed per x64 mode."""
    return _hyp1f1_cached(float(a), float(b), canon_tag())
