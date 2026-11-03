"""betaincinv, the Beta quantile (jax#2399), by betainc_kernel.betaincinv_core
(shared with stdtrit): fixed-count safeguarded Newton in logit space from
I_x(a,b) ~ x^a / (a B(a,b)), p > 1/2 mirrored so 1 - p is exact. Gradient by
the implicit function theorem, dx/dp = 1/pdf(x), dx/d(a, b) = -dI/d(a, b) / pdf.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import float_params
from chebax._src.recipes.betainc import betainc_scalar, with_panel_series
from chebax._src.recipes.betainc_kernel import betaincinv_core, ln_beta


@jax.custom_jvp
def _betaincinv_cj(a, b, p):
    """The Beta(a, b) quantile for traceable a, b in [0.1, 100] and any p.
    Quantiles below the smallest normal give 0.0, within eps of 1 give 1.0 (use
    betaincinv(b, a, 1-p) for the distance from 1); a failed solve gives nan."""
    return with_panel_series(a, b, lambda cab, cba: betaincinv_core(a, b, p, cab, cba))


@_betaincinv_cj.defjvp
def _betaincinv_jvp(primals, tangents):
    a, b, p = primals
    da, db, dp = tangents
    x = _betaincinv_cj(a, b, p)
    interior = (x > 0.0) & (x < 1.0)
    xs = jnp.where(interior, x, 0.5)
    pdf = jnp.exp((a - 1) * jnp.log(xs) + (b - 1) * jnp.log1p(-xs) - ln_beta(a, b))
    _, d_cdf = jax.jvp(lambda aa, bb: betainc_scalar(aa, bb, xs), (a, b), (da, db))
    dx = (dp - d_cdf) / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


# float_params makes every argument a float scalar before any tangent exists
betaincinv = float_params(_betaincinv_cj)
