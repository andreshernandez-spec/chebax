"""Small-a incomplete gamma, a in [0.1, 10], x in [0, inf), from log-tables, for
the frozen and scalar paths (a > 10 is gammainc_large). Split at x = XS = 8:

- x in [0, 8]:  P = exp(a ln x - x - lnGamma(a+1) + L(a, x)), L = ln 1F1(1; a+1; x)
- x > 8:        Q = exp((a-1) ln x - x - lnGamma(a) + T(a, 8/x)),
                T = ln[Gamma(a, x) x^(1-a) e^x], so accuracy holds as x grows.

The other side is 1 minus: absolutely accurate everywhere, relatively where
small, except Q just below x = 8 at small a, where it reaches ~1e-6. At x = 0
the slope is the density, spliced in (inf below a = 1, so a zero cotangent there
gives nan, as jnp.sqrt at 0). Branchless fixed-degree polynomials, unlike jax's
whole-array while loops (experiments/05).
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
    # inf - inf at x = inf: mask it and set ln Q = -inf
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
