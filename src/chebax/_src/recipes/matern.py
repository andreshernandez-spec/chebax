"""The Matern correlation with a differentiable smoothness parameter.

    k(r) = 2^(1-nu)/Gamma(nu) * z^nu * K_nu(z),   z = sqrt(2 nu) r / l

Unit variance (k(0) = 1 exactly); multiply by a signal variance for a
covariance. Learning nu by gradient needs dK_nu/dnu, a derivative in the
ORDER of the Bessel function, which is what besselk_fn's traced order
supplies; no mainstream GP stack offers it.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes.besselk import besselk_fn


def matern(nu, r, lengthscale=1.0):
    """Matern correlation at distances r, differentiable in everything.

    nu must be uniform per call (a scalar or one traced latent) and
    inside (0, 10]; r >= 0 and the lengthscale broadcast freely.
    jax.grad works with respect to nu (learnable smoothness), r and the
    lengthscale. Half-integer nu (1/2 = exponential, 3/2, 5/2) are
    ordinary points of the K table, and the tests use their closed
    forms as oracles.

    r = 0 (every covariance diagonal) returns exactly 1 by hard select,
    with masked lanes fed a safe dummy so log(0) cannot poison
    gradients; the gradient at exactly r = 0 is masked to 0 (the true
    one-sided slope is 0 for nu > 1/2 anyway). Distances below
    ~1e-6 lengthscale/sqrt(2 nu) sit on besselk's clamp and read as
    k(that distance), still ~1. A signal variance is deliberately not a
    parameter: multiply the result."""
    nu = jnp.asarray(nu)
    r = jnp.asarray(r)
    pos = r > 0.0
    rs = jnp.where(pos, r, 1.0)
    z = jnp.sqrt(2.0 * nu) * rs / lengthscale
    log_c = (1.0 - nu) * jnp.log(2.0) - jax.scipy.special.gammaln(nu)
    k = jnp.exp(log_c + nu * jnp.log(z)) * besselk_fn(nu, z)
    out = jnp.where(pos, k, 1.0)
    return jnp.where(jnp.isnan(r) | jnp.isnan(nu), jnp.nan, out)
