"""von Mises (mu = 0) CDF and quantile, kappa in [0, 50], theta in [-pi, pi].

The circular-distribution gap: numpyro has the sampler but no cdf/icdf,
which blocks truncated and reparameterized circular models. Structure:

    F(theta; kappa) = 1/2 + theta/(2 pi) + theta * H(theta^2; kappa)

with H from a baked (w = theta^2, r = sqrt(kappa)) table — the dawsn move
makes oddness and the theta = 0 gradient exact by construction, and puts
the large-kappa boundary layer at the endpoint where Chebyshev clusters.

vonmises_cdf(kappa, theta) takes kappa traced (uniform per call, in
[0, 50], unchecked under trace): jax.grad works with respect to kappa —
learnable concentration. dF/dtheta is AD through the series; its exact
value is the von Mises density (the test oracle). vonmises_icdf(kappa, p)
is the fixed-count safeguarded solver with implicit-function-theorem
gradients, same as the other quantiles; I_0(kappa) for the density comes
from chebax's own besseli. theta outside [-pi, pi] is the caller's problem
(wrap first); mu != 0 is a shift: F(theta - mu) with wrapping.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import vonmises_table as _vt
from chebax._src.recipes._common import newton_bisect as _newton_bisect
from chebax._src.recipes._common import traced_coefs
from chebax._src.recipes.besseli import besseli_fn
from chebax._src.series import ChebSeries


def _h_series(kappa):
    r = jnp.sqrt(jnp.asarray(kappa))
    return ChebSeries(traced_coefs(_vt.TABLE, 0.0, _vt.RMAX, r), (0.0, _vt.WMAX))


def vonmises_cdf(kappa, theta):
    """CDF of vonMises(mu=0, kappa) at theta in [-pi, pi]; kappa traceable."""
    theta = jnp.asarray(theta)
    h = _h_series(kappa)
    tc = jnp.clip(theta, -jnp.pi, jnp.pi)
    core = 0.5 + tc / (2 * jnp.pi) + tc * h(tc * tc)
    return jnp.where(theta <= -jnp.pi, 0.0, jnp.where(theta >= jnp.pi, 1.0, core))


def _log_pdf(kappa, theta):
    i0 = besseli_fn(jnp.asarray(0.0), jnp.asarray(kappa))
    return jnp.asarray(kappa) * jnp.cos(theta) - jnp.log(2 * jnp.pi * i0)


@jax.custom_jvp
def vonmises_icdf(kappa, p):
    """Quantile of vonMises(mu=0, kappa): inverse of vonmises_cdf in theta."""
    kappa = jnp.asarray(kappa)
    p = jnp.asarray(p)
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    h = _h_series(kappa)

    def f_and_df(th):
        f = 0.5 + th / (2 * jnp.pi) + th * h(th * th) - pc
        return f, jnp.exp(_log_pdf(kappa, th))

    th0 = 2 * jnp.pi * (pc - 0.5)
    th = _newton_bisect(f_and_df, th0, jnp.full_like(pc, -jnp.pi),
                        jnp.full_like(pc, jnp.pi), 30)
    return jnp.where(p <= 0.0, -jnp.pi, jnp.where(p >= 1.0, jnp.pi, th))


@vonmises_icdf.defjvp
def _vonmises_icdf_jvp(primals, tangents):
    kappa, p = primals
    dkappa, dp = tangents
    th = vonmises_icdf(kappa, p)
    interior = (th > -jnp.pi) & (th < jnp.pi)
    ths = jnp.where(interior, th, 0.0)
    pdf = jnp.exp(_log_pdf(kappa, ths))
    _, d_cdf = jax.jvp(lambda k: vonmises_cdf(k, ths), (kappa,), (dkappa,))
    dth = (jnp.asarray(dp) - d_cdf) / pdf
    return th, jnp.where(interior, dth, 0.0)
