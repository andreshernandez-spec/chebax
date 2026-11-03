"""von Mises (mu = 0) CDF and quantile, kappa in [0, 50], theta in [-pi, pi]:

    F(theta; kappa) = 1/2 + theta/(2 pi) + theta H(theta^2; kappa),

H tabulated in (theta^2, sqrt(kappa)), so oddness and the theta = 0 gradient
are exact. kappa is a traced scalar (unchecked under trace). F is accurate
absolutely, not relatively, so the quantile returns nan below
min(p, 1-p) = 4096 eps. Wrap theta first; mu != 0 is F(theta - mu).
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import vonmises_table as _vt
from chebax._src.recipes._common import float_params
from chebax._src.recipes._common import newton_bisect
from chebax._src.recipes._common import traced_coefs
from chebax._src.recipes.besseli import besseli_scalar
from chebax._src.series import ChebSeries

# accepted CDF residual relative to min(p, 1-p)
_RESID_RTOL = 1e-6


def _solver_consts():
    """Newton count, residual floor (4x the measured 2.8 eps f64, 8.5 eps f32)
    and probability floor, where 32 eps / min(p, 1-p) reaches ~1e-4."""
    eps = float(jnp.finfo(jnp.empty(()).dtype).eps)
    return (32 if eps < 1e-10 else 20), 32.0 * eps, 4096.0 * eps


def _h_series(kappa):
    r = jnp.sqrt(jnp.asarray(kappa))
    return ChebSeries(traced_coefs(_vt.TABLE, 0.0, _vt.RMAX, r), (0.0, _vt.WMAX))


def _cdf_impl(kappa, theta):
    h = _h_series(kappa)
    tc = jnp.clip(theta, -jnp.pi, jnp.pi)
    core = 0.5 + tc / (2 * jnp.pi) + tc * h(tc * tc)
    return jnp.where(theta <= -jnp.pi, 0.0, jnp.where(theta >= jnp.pi, 1.0, core))


@jax.custom_jvp
def _vonmises_cdf_cj(kappa, theta):
    """CDF of vonMises(0, kappa) at theta in [-pi, pi], differentiable in
    both; d/dkappa at kappa = 0 is the exact sin(theta)/(2 pi)."""
    return _cdf_impl(kappa, theta)


@_vonmises_cdf_cj.defjvp
def _vonmises_cdf_jvp(primals, tangents):
    k, th = primals
    dkappa, dtheta = tangents
    F = _cdf_impl(k, th)
    _, d_th = jax.jvp(lambda t: _cdf_impl(k, t), (th,), (dtheta,))
    # AD divides by 2 sqrt(kappa), so below 1e-8 use sin(theta)/(2 pi) + O(kappa)
    ks = jnp.maximum(k, 1e-8)
    _, d_k_ad = jax.jvp(lambda kk: _cdf_impl(kk, th), (ks,), (dkappa,))
    interior = (th > -jnp.pi) & (th < jnp.pi)
    lead = (
        jnp.where(interior, jnp.sin(jnp.clip(th, -jnp.pi, jnp.pi)) / (2 * jnp.pi), 0.0)
        * dkappa
    )
    dF = d_th + jnp.where(k < 1e-8, lead, d_k_ad)
    # scale, not select, so nan survives reverse mode
    return F, dF * jnp.where(jnp.isnan(F), jnp.nan, 1.0)


def _log_pdf(kappa, theta):
    i0 = besseli_scalar(jnp.asarray(0.0), jnp.asarray(kappa))
    return jnp.asarray(kappa) * jnp.cos(theta) - jnp.log(2 * jnp.pi * i0)


@jax.custom_jvp
def _vonmises_icdf_cj(kappa, p):
    """Quantile of vonMises(0, kappa). A solve whose CDF residual exceeds
    max(1e-6 min(p, 1-p), 32 eps) returns nan, and so does min(p, 1-p) below
    4096 eps (9.1e-13 f64, 4.9e-4 f32), where the error would reach 0.11 at
    1e-15. Measured worst relative error 1.2e-5 at 1e-11 and 1.0e-4 just above
    the floor, for kappa in {2, 10, 25, 50}. p = 0, 1 give -pi, pi exactly."""
    iters, resid_atol, pmin_floor = _solver_consts()
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    h = _h_series(kappa)

    def f_and_df(th):
        f = 0.5 + th / (2 * jnp.pi) + th * h(th * th) - pc
        return f, jnp.exp(_log_pdf(kappa, th))

    # start at the Gaussian limit N(0, 1/kappa), clipped: converged by 15
    # iterations where the uniform start 2 pi (p - 1/2) needs 25
    tiny = jnp.finfo(pc.dtype).tiny
    th0 = jnp.clip(
        jax.scipy.special.ndtri(pc) / jnp.sqrt(jnp.maximum(kappa, tiny)),
        -jnp.pi,
        jnp.pi,
    )
    th0 = th0 + jnp.zeros_like(pc)
    th = newton_bisect(
        f_and_df, th0, jnp.full_like(th0, -jnp.pi), jnp.full_like(th0, jnp.pi), iters
    )
    f_fin, _ = f_and_df(th)
    pmin = jnp.minimum(pc, 1.0 - pc)
    bad = (jnp.abs(f_fin) > jnp.maximum(_RESID_RTOL * pmin, resid_atol)) | (
        pmin < pmin_floor
    )
    th = jnp.where(bad, jnp.nan, th)
    th = jnp.where(p <= 0.0, -jnp.pi, jnp.where(p >= 1.0, jnp.pi, th))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(kappa)
    return jnp.where(oob, jnp.nan, th)


@_vonmises_icdf_cj.defjvp
def _vonmises_icdf_jvp(primals, tangents):
    kappa, p = primals
    dkappa, dp = tangents
    th = vonmises_icdf(kappa, p)
    interior = (th > -jnp.pi) & (th < jnp.pi)
    ths = jnp.where(interior, th, 0.0)
    pdf = jnp.exp(_log_pdf(kappa, ths))
    _, d_cdf = jax.jvp(lambda k: vonmises_cdf(k, ths), (kappa,), (dkappa,))
    dth = (dp - d_cdf) / pdf
    # scale, not select: zero at the endpoints, nan for a rejected solve
    return th, dth * jnp.where(interior, 1.0, jnp.where(jnp.isnan(th), jnp.nan, 0.0))


# rules on the private names, so kappa is a float scalar before any tangent
vonmises_cdf = float_params(_vonmises_cdf_cj)
vonmises_icdf = float_params(_vonmises_icdf_cj)
