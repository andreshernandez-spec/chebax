"""The Matern correlation k(r) = 2^(1-nu)/Gamma(nu) z^nu K_nu(z),
z = sqrt(2 nu) r / l, differentiable in nu through besselk_scalar's order.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import besselk_table as _kt
from chebax._src.recipes._common import canon_param
from chebax._src.recipes.besselk import log_besselk_scalar


def matern(nu, r, lengthscale=1.0):
    """Matern correlation (unit variance) at r >= 0, nu in (0, 10] uniform per
    call, differentiable in nu, r and the lengthscale. k(0) = 1 exactly, with
    gradient 0. z below besselk's 1e-6 clamp clamps prefactor and K together
    (GPJax's distances give ~1e-18 for identical inputs); at small nu that k
    is below 1 (~0.75 at nu = 0.05), as the Matern is."""
    nu = canon_param(nu, "matern", "nu")
    r = jnp.asarray(r)
    pos = r > 0.0
    rs = jnp.where(pos, r, 1.0)
    z = jnp.sqrt(2.0 * nu) * rs / lengthscale
    z = jnp.maximum(z, _kt.XMIN)
    log_c = (1.0 - nu) * jnp.log(2.0) - jax.scipy.special.gammaln(nu)
    # one exp of the summed logs: apart, z^nu underflows and K overflows at
    # small z (nu = 6, z = 1e-6 in float32 gave 0 * inf = nan)
    k = jnp.exp(log_c + nu * jnp.log(z) + log_besselk_scalar(nu, z))
    # 0 at r = inf (the K tail gives nan), nan for r < 0
    out = jnp.where(pos, k, 1.0)
    out = jnp.where(r == jnp.inf, 0.0, out)
    out = jnp.where(r < 0.0, jnp.nan, out)
    return jnp.where(jnp.isnan(r) | jnp.isnan(nu), jnp.nan, out)
