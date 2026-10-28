"""The Beta quantile: betaincinv, the inverse of betainc in x (jax#2399).

a and b are traceable scalars in the betainc table domain [0.1, 100],
uniform per call, so jax.grad works with respect to all three arguments.
The solve is betainc_kernel.betaincinv_core, shared with stdtrit: fixed-
count safeguarded Newton in logit space, started in the deep lower tail
from the exact asymptotic I_x(a,b) ~ x^a / (a B(a,b)), on the half that
carries the tail mass (p > 1/2 is mirrored onto the reflected
orientation, where 1 - p is exact). Its gradient is not the iteration's
but the implicit function theorem's,

    dx*/dp = 1 / pdf(x*),    dx*/d(a, b) = -(dI/d(a, b))(x*) / pdf(x*),

with dI/da, dI/db from betainc_fn's traced gradients at the solution.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import float_params
from chebax._src.recipes.betainc import betainc_fn, with_panel_series
from chebax._src.recipes.betainc_kernel import betaincinv_core, ln_beta


@jax.custom_jvp
def _betaincinv_cj(a, b, p):
    """Inverse of betainc in x: the Beta(a, b) quantile function.

    a, b are (traceable) scalars in [0.1, 100], uniform per call; p any
    shape (the wide panels of increment 23 dispatch by one lax.switch).
    Quantiles below the smallest normal float of the canonical dtype return
    0.0 (and, mirrored, quantiles within eps of 1 return 1.0, the
    scipy-shared representation limit; use betaincinv(b, a, 1-p) for the
    distance from 1). A solve that fails its CDF-residual check returns nan
    instead of a wrong value."""
    return with_panel_series(a, b, lambda cab, cba: betaincinv_core(a, b, p, cab, cba))


@_betaincinv_cj.defjvp
def _betaincinv_jvp(primals, tangents):
    a, b, p = primals
    da, db, dp = tangents
    x = _betaincinv_cj(a, b, p)
    interior = (x > 0.0) & (x < 1.0)
    xs = jnp.where(interior, x, 0.5)
    pdf = jnp.exp((a - 1) * jnp.log(xs) + (b - 1) * jnp.log1p(-xs) - ln_beta(a, b))
    _, d_cdf = jax.jvp(lambda aa, bb: betainc_fn(aa, bb, xs), (a, b), (da, db))
    dx = (dp - d_cdf) / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


# The custom_jvp rule lives on the private name and float_params normalizes
# every argument first, so an integer, keyword or size-one parameter is a
# float scalar before any tangent exists (see its docstring).
betaincinv = float_params(_betaincinv_cj)
