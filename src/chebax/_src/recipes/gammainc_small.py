"""Small-a numerics for the regularized incomplete gamma: P(a, x) and Q = 1 - P
for a in [0.1, 10], x in [0, inf), from baked log-tables. The frozen and
scalar paths both run on this (a > 10 is gammainc_large).

Two regions, one hard select at x = XS = 8:

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

jax.grad works with respect to a through the table's parameter axis and
gammaln: one polynomial evaluation, where jax's own igamma_grad_a runs a
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

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes._common import edge_slope, param_coefs, traced_coefs
from chebax._src.series import ChebSeries


def eval_logs(a, x, lser, ltail):
    """(ln P inner, ln Q tail), masked lanes fed finite dummies."""
    pos = x > 0.0
    xd = jnp.where(pos & (x <= _gt.XS), x, 1.0)
    lp_in = a * jnp.log(xd) - xd - jax.scipy.special.gammaln(a + 1.0) + lser(xd)
    # at x = inf the tail bracket is inf - inf, so mask it out and set the
    # limit ln Q = -inf by hand; all four entry points read it off that.
    big = x == jnp.inf
    xo = jnp.where((x > _gt.XS) & ~big, x, 2.0 * _gt.XS)
    lq_tl = (
        (a - 1.0) * jnp.log(xo) - xo - jax.scipy.special.gammaln(a) + ltail(_gt.XS / xo)
    )
    return lp_in, jnp.where(big, -jnp.inf, lq_tl)


def density0(a):
    """Gamma density at x = 0: inf below a = 1, 1 at it, 0 above."""
    return jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, 1.0))


def eval_gammainc(a, x, lser, ltail, lower):
    x = jnp.asarray(x)
    lp_in, lq_tl = eval_logs(a, x, lser, ltail)
    d0 = density0(a)
    if lower:
        core = jnp.where(x <= _gt.XS, jnp.exp(lp_in), 1.0 - jnp.exp(lq_tl))
        out = jnp.where(x > 0.0, core, 0.0)
    else:
        core = jnp.where(x <= _gt.XS, 1.0 - jnp.exp(lp_in), jnp.exp(lq_tl))
        out = jnp.where(x > 0.0, core, 1.0)
        d0 = -d0
    out = out + edge_slope(jnp.where(x == 0.0, d0, 0.0), x)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)


def series_eager(a):
    """(inner, tail) series at a concrete a, reconstructed in numpy."""
    return (
        ChebSeries(param_coefs(_gt.TABLE_IN, _gt.ALO, _gt.AHI, a), (0.0, _gt.XS)),
        ChebSeries(param_coefs(_gt.TABLE_TAIL, _gt.ALO, _gt.AHI, a), (0.0, 1.0)),
    )


def series_traced(a):
    """(inner, tail) series at a traced scalar a."""
    return (
        ChebSeries(traced_coefs(_gt.TABLE_IN, _gt.ALO, _gt.AHI, a), (0.0, _gt.XS)),
        ChebSeries(traced_coefs(_gt.TABLE_TAIL, _gt.ALO, _gt.AHI, a), (0.0, 1.0)),
    )
