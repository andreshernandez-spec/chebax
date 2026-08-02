"""Regularized incomplete gamma P(a, x) and Q = 1 - P for a >= 0.1,
x in [0, inf), from baked log-tables.

a > 10 dispatches (one scalar cond) to the Temme-zone path in
gammainc_large; below, the original a <= 10 recipe. Two regions, one
hard select at x = XS = 8:

- x in [0, 8]:  P = exp(a ln x - x - lnGamma(a+1) + L(a, x)), where
                L = ln 1F1(1; a+1; x) is tabulated directly in x (entire,
                no transform needed); the x^a branch point and P's dynamic
                range live in the exactly-computed prefactor.
- x > 8:        Q = exp((a-1) ln x - x - lnGamma(a) + T(a, 8/x)), where
                T = ln[Gamma(a, x) x^(1-a) e^x]. exp(-x) is its own
                correctly-rounded factor, so accuracy does not decay with
                x; Q underflows to 0 gracefully past x ~ 700.

Each function is direct on the side where it is small (P below the
transition, Q in the tail) and 1-minus-the-other on the side where it is
~1: P and Q are absolutely accurate everywhere (the CDF contract) and
relatively accurate where they decay, except that Q loses relative (not
absolute) accuracy in the wedge x just below 8 with a small, where Q can
reach ~1e-6 before the tail region takes over. x <= 0 returns the CDF
limit (P = 0, Q = 1) and x = +inf the other one (P = 1, Q = 0); nan
propagates.

gammainc(a)/gammaincc(a) are eager cached instances (numpy
reconstruction). gammainc_fn(a, x)/gammaincc_fn(a, x) take a as a traced
scalar (uniform per call, inside [0.1, 10], unchecked under trace):
jax.grad works with respect to a through the table's parameter axis and
gammaln - one polynomial evaluation, where jax's own igamma_grad_a runs a
third lockstep looped series. dP/dx is AD through the formula; its exact
value is the Gamma density, which the tests use as an oracle. At x = 0
the formula's bracket is 0 * inf, so the density there (inf below a = 1,
1 at it, 0 above) is spliced in by hand; below a = 1 that slope is a real
infinity, so a zero cotangent on an x = 0 lane gives nan, the same way
jnp.sqrt does at 0. x < 0 and x = +inf are flat.

Unlike jax.scipy.special.gammainc (two whole-array while loops run to the
worst element's trip count, a fusion barrier), evaluation here is
branchless fixed-degree polynomials; experiments/05 measured the op
profile's headroom at 18-54x f64 on GPU before this recipe existed.
"""

import functools

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_large_table as _glt
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes._common import (canon_tag, check_range, edge_slope,
                                         param_coefs,
                                         traced_coefs)
from chebax._src.series import ChebSeries


def _eval_logs(a, x, lser, ltail):
    """(ln P inner, ln Q tail), masked lanes fed finite dummies."""
    pos = x > 0.0
    xd = jnp.where(pos & (x <= _gt.XS), x, 1.0)
    lp_in = (a * jnp.log(xd) - xd - jax.scipy.special.gammaln(a + 1.0)
             + lser(xd))
    # at x = inf the tail bracket is inf - inf, so mask it out and set the
    # limit ln Q = -inf by hand; all four entry points read it off that.
    big = x == jnp.inf
    xo = jnp.where((x > _gt.XS) & ~big, x, 2.0 * _gt.XS)
    lq_tl = ((a - 1.0) * jnp.log(xo) - xo - jax.scipy.special.gammaln(a)
             + ltail(_gt.XS / xo))
    return lp_in, jnp.where(big, -jnp.inf, lq_tl)


def _density0(a):
    """Gamma density at x = 0: inf below a = 1, 1 at it, 0 above."""
    return jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, 1.0))


def eval_gammainc(a, x, lser, ltail, lower):
    x = jnp.asarray(x)
    lp_in, lq_tl = _eval_logs(a, x, lser, ltail)
    d0 = _density0(a)
    if lower:
        core = jnp.where(x <= _gt.XS, jnp.exp(lp_in), 1.0 - jnp.exp(lq_tl))
        out = jnp.where(x > 0.0, core, 0.0)
    else:
        core = jnp.where(x <= _gt.XS, 1.0 - jnp.exp(lp_in), jnp.exp(lq_tl))
        out = jnp.where(x > 0.0, core, 1.0)
        d0 = -d0
    out = out + edge_slope(jnp.where(x == 0.0, d0, 0.0), x)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)


@jax.tree_util.register_pytree_node_class
class GammaIncP(Recipe):
    """Callable P(a, x) on x in [0, inf). Build with gammainc(a)."""

    _static_fields = ("a",)
    _series_fields = ("lser", "ltail")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return eval_gammainc(self.a, x, self.lser, self.ltail, lower=True)


@jax.tree_util.register_pytree_node_class
class GammaIncQ(Recipe):
    """Callable Q(a, x) on x in [0, inf). Build with gammaincc(a)."""

    _static_fields = ("a",)
    _series_fields = ("lser", "ltail")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return eval_gammainc(self.a, x, self.lser, self.ltail, lower=False)


def _series_at(a):
    return (ChebSeries(param_coefs(_gt.TABLE_IN, _gt.ALO, _gt.AHI, a),
                       (0.0, _gt.XS)),
            ChebSeries(param_coefs(_gt.TABLE_TAIL, _gt.ALO, _gt.AHI, a),
                       (0.0, 1.0)))


@functools.lru_cache(maxsize=128)
def _gammainc_cached(a, _tag):
    a = check_range("gammainc", "a", a, _gt.ALO, _gl.AHI)
    if a > _gt.AHI:
        return _gl.GammaIncLargeP(a, *_gl.series_eager(a))
    return GammaIncP(a, *_series_at(a))


@functools.lru_cache(maxsize=128)
def _gammaincc_cached(a, _tag):
    a = check_range("gammaincc", "a", a, _gt.ALO, _gl.AHI)
    if a > _gt.AHI:
        return _gl.GammaIncLargeQ(a, *_gl.series_eager(a))
    return GammaIncQ(a, *_series_at(a))


def _traced_series(a):
    return (ChebSeries(traced_coefs(_gt.TABLE_IN, _gt.ALO, _gt.AHI, a),
                       (0.0, _gt.XS)),
            ChebSeries(traced_coefs(_gt.TABLE_TAIL, _gt.ALO, _gt.AHI, a),
                       (0.0, 1.0)))


def _dispatch_fn(a, x, small, kind):
    """One scalar cond at a = 10: the small-a tables or the Temme-zone
    large-a path (a > 10, gammainc_large). Only the taken
    branch executes; each builds its own series."""

    def large():
        tm, dl, du = _gl.series_traced(a)
        return _gl.eval_large(a, x, tm, dl, du, kind)

    return jax.lax.cond(a <= _gt.AHI, small, large)


def gammainc_fn(a, x):
    """P(a, x) with a a (traceable) scalar, differentiable in both arguments.

    a must be uniform per call and at least 0.1 (unchecked under
    trace; above 10 it runs on the Temme-zone tables, whose v = 10/a
    axis reaches a = inf). Reconstruction
    costs two table contractions per call, not per point; jit
    constant-folds them when a is static."""
    a = jnp.asarray(a)
    x = jnp.asarray(x)

    def small():
        lser, ltail = _traced_series(a)
        return eval_gammainc(a, x, lser, ltail, lower=True)

    return _dispatch_fn(a, x, small, "p")


def gammaincc_fn(a, x):
    """Q(a, x) = 1 - P(a, x), same contract as gammainc_fn; relatively
    accurate in the upper tail (computed directly, not as 1 - P)."""
    a = jnp.asarray(a)
    x = jnp.asarray(x)

    def small():
        lser, ltail = _traced_series(a)
        return eval_gammainc(a, x, lser, ltail, lower=False)

    return _dispatch_fn(a, x, small, "q")


def log_gammainc_fn(a, x):
    """ln P(a, x), same contract as gammainc_fn but with no underflow
    floor in the lower tail.

    For x <= 8 the tables already hold the log; ln P evaluates directly
    (ln P ~ a ln x as x -> 0, arbitrarily negative). For x > 8 it is
    log1p(-exp(ln Q)): absolutely accurate in P, error in ln P is
    ~eps Q/P, small everywhere there since Q <= 0.3 in the box. x <= 0
    returns -inf, x = inf returns 0. For a > 10 the large path assembles
    in log space through erfcx, valid past the linear form's underflow."""
    a = jnp.asarray(a)
    x = jnp.asarray(x)

    def small():
        lser, ltail = _traced_series(a)
        lp_in, lq_tl = _eval_logs(a, x, lser, ltail)
        core = jnp.where(x <= _gt.XS, lp_in, jnp.log1p(-jnp.exp(lq_tl)))
        out = jnp.where(x > 0.0, core, -jnp.inf)
        return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)

    out = _dispatch_fn(a, x, small, "logp")
    # d ln P/dx = density/P ~ a/x, so the slope at x = 0 is inf for every a
    return out + edge_slope(jnp.where(x == 0.0, jnp.inf, 0.0), x)


def log_gammaincc_fn(a, x):
    """ln Q(a, x), same contract as gammaincc_fn but with no underflow
    ceiling in the tail.

    For x > 8 the tables already hold the log; ln Q evaluates directly
    (ln Q ~ -x, valid for arbitrarily large x where Q itself underflows
    past x ~ 700). For x <= 8 it is log1p(-exp(ln P)): absolutely
    accurate in Q, so the error in ln Q is ~eps/Q in the wedge x -> 8 at
    small a where Q reaches ~1e-6 (the same wedge gammaincc_fn's
    docstring notes). x <= 0 returns 0, x = inf returns -inf. This is the
    log survival function a deeply-censored likelihood wants."""
    a = jnp.asarray(a)
    x = jnp.asarray(x)

    def small():
        lser, ltail = _traced_series(a)
        lp_in, lq_tl = _eval_logs(a, x, lser, ltail)
        core = jnp.where(x <= _gt.XS, jnp.log1p(-jnp.exp(lp_in)), lq_tl)
        out = jnp.where(x > 0.0, core, 0.0)
        return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)

    out = _dispatch_fn(a, x, small, "logq")
    # Q = 1 at x = 0, so d ln Q/dx there is just -density
    return out + edge_slope(jnp.where(x == 0.0, -_density0(a), 0.0), x)


def gammainc(a):
    """P(a, x) on [0, inf) for a >= 0.1. Cached per a; no mpmath.

    a may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode. a > 10 runs on the Temme-zone
    tables (gammainc_large)."""
    return _gammainc_cached(float(a), canon_tag())


def gammaincc(a):
    """Q(a, x) = 1 - P(a, x) on [0, inf) for a >= 0.1. Cached per a."""
    return _gammaincc_cached(float(a), canon_tag())
