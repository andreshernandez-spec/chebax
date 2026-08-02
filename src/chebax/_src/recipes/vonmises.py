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
gradients, same as the other quantiles, and like them it checks its final
CDF residual instead of assuming convergence. The structure above is
accurate absolutely, not relatively, so far-tail probabilities stop being
resolvable below min(p, 1-p) ~ 4096 eps, whatever kappa is: past that
floor the quantile returns nan rather than a wrong angle (see its
docstring). I_0(kappa) for the density comes
from chebax's own besseli. theta outside [-pi, pi] is the caller's problem
(wrap first); mu != 0 is a shift: F(theta - mu) with wrapping.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import vonmises_table as _vt
from chebax._src.recipes._common import canon_float as _canon
from chebax._src.recipes._common import float_params
from chebax._src.recipes._common import newton_bisect as _newton_bisect
from chebax._src.recipes._common import traced_coefs
from chebax._src.recipes.besseli import besseli_fn
from chebax._src.series import ChebSeries


# a solve is accepted when its CDF residual is this small relative to
# min(p, 1-p); healthy solves land at the eps floor below instead
_RESID_RTOL = 1e-6


def _solver_consts():
    """Newton count, residual floor and resolvable-probability floor for the
    canonical dtype.

    The residual floor is the measured worst converged |F(theta) - p|
    (2.8 eps in float64, 8.5 eps in float32) with the usual 4x bar. F is
    accurate ABSOLUTELY, so that residual buys only ~32 eps / min(p, 1-p)
    of RELATIVE accuracy: the probability floor is where that reaches
    ~1e-4 (measured worst 1.0e-4 just above it, see the icdf docstring)."""
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
    """CDF of vonMises(mu=0, kappa) at theta in [-pi, pi]; kappa traceable.

    The kappa gradient is finite on the whole documented domain including
    kappa = 0, where it is the exact Fourier limit sin(theta)/(2 pi)."""
    return _cdf_impl(_canon(kappa), _canon(theta))


@_vonmises_cdf_cj.defjvp
def _vonmises_cdf_jvp(primals, tangents):
    kappa, theta = primals
    dkappa, dtheta = tangents
    k, th = _canon(kappa), _canon(theta)
    F = _cdf_impl(k, th)
    _, d_th = jax.jvp(lambda t: _cdf_impl(k, t), (th,), (_canon(dtheta),))
    # kappa direction: the table lives on r = sqrt(kappa), so AD divides
    # the (noisy, unconstrained) slope at r = 0 by 2 sqrt(kappa) -> inf at
    # the documented boundary. Below 1e-8 use the exact Fourier limit
    # dF/dkappa|_0 = sin(theta)/(2 pi) (the neglected terms are O(kappa)).
    ks = jnp.maximum(k, 1e-8)
    _, d_k_ad = jax.jvp(lambda kk: _cdf_impl(kk, th), (ks,), (_canon(dkappa),))
    interior = (th > -jnp.pi) & (th < jnp.pi)
    lead = jnp.where(interior, jnp.sin(jnp.clip(th, -jnp.pi, jnp.pi))
                     / (2 * jnp.pi), 0.0) * _canon(dkappa)
    dF = d_th + jnp.where(k < 1e-8, lead, d_k_ad)
    # nan has to survive reverse mode too: a where whose nan branch is a
    # constant is linear in nothing and transposes to a zero cotangent.
    return F, dF * jnp.where(jnp.isnan(F), jnp.nan, 1.0)


def _log_pdf(kappa, theta):
    i0 = besseli_fn(jnp.asarray(0.0), jnp.asarray(kappa))
    return jnp.asarray(kappa) * jnp.cos(theta) - jnp.log(2 * jnp.pi * i0)


@jax.custom_jvp
def _vonmises_icdf_cj(kappa, p):
    """Quantile of vonMises(mu=0, kappa): inverse of vonmises_cdf in theta.

    Guarantee instead of silent saturation: the final CDF residual is
    checked against max(1e-6 min(p, 1-p), 32 eps) and a solve that misses
    it returns nan. The absolute half of that bar is the real limit. F is
    built as 1/2 + theta/(2 pi) + theta H(theta^2), which is accurate to a
    few eps ABSOLUTELY and carries no relative accuracy in the tails, so
    the best a converged solve can promise is ~32 eps / min(p, 1-p)
    relative. That bound is set by the CDF's representation, not by kappa:
    measured worst relative error is 1.2e-5 at min(p, 1-p) = 1e-11 and
    1.0e-4 just above the floor, across kappa in {2, 10, 25, 50} alike. Below
    min(p, 1-p) = 4096 eps (9.1e-13 in float64, 4.9e-4 in float32) the
    quantile returns nan rather than an angle whose CDF is off by a
    percent or more (at 1e-15 the relative error reaches 0.11).

    NaN inputs and p outside [0, 1] return nan; the endpoints return
    -pi and pi exactly."""
    kappa = _canon(kappa)
    p = _canon(p)
    iters, resid_atol, pmin_floor = _solver_consts()
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    h = _h_series(kappa)

    def f_and_df(th):
        f = 0.5 + th / (2 * jnp.pi) + th * h(th * th) - pc
        return f, jnp.exp(_log_pdf(kappa, th))

    # Start from the large-kappa Gaussian limit theta ~ N(0, 1/kappa),
    # clipped to the circle (kappa -> 0 clips to the endpoint, where the
    # uniform CDF is linear and one Newton step is exact). The uniform
    # start 2 pi (p - 1/2) sits at the far end of the interval once kappa
    # concentrates and Newton then creeps additively: measured, it needed
    # 25 iterations where this one is converged by 15.
    tiny = jnp.finfo(pc.dtype).tiny
    th0 = jnp.clip(jax.scipy.special.ndtri(pc)
                   / jnp.sqrt(jnp.maximum(kappa, tiny)), -jnp.pi, jnp.pi)
    th0 = th0 + jnp.zeros_like(pc)
    th = _newton_bisect(f_and_df, th0, jnp.full_like(th0, -jnp.pi),
                        jnp.full_like(th0, jnp.pi), iters)
    # Convergence is checked, never assumed, and so is resolvability: a
    # residual at the eps floor means nothing where the density is too
    # small to turn it into a theta (see the docstring).
    f_fin, _ = f_and_df(th)
    pmin = jnp.minimum(pc, 1.0 - pc)
    bad = ((jnp.abs(f_fin) > jnp.maximum(_RESID_RTOL * pmin, resid_atol))
           | (pmin < pmin_floor))
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
    pdf = jnp.exp(_log_pdf(_canon(kappa), ths))
    _, d_cdf = jax.jvp(lambda k: vonmises_cdf(k, ths),
                       (_canon(kappa),), (_canon(dkappa),))
    dth = (_canon(dp) - d_cdf) / pdf
    # scale, not select: the endpoints are a genuine zero, and a rejected
    # solve must hand back nan in reverse mode as well as forward
    return th, dth * jnp.where(interior, 1.0,
                               jnp.where(jnp.isnan(th), jnp.nan, 0.0))


# Public entry points: the rules sit on the private names so an INTEGER
# kappa (vonmises_cdf(5, theta)) is a float before any tangent exists.
# jax gives integer primals float0 tangents, which the rules cannot do
# arithmetic on (review, 2026-08-02).
vonmises_cdf = float_params(_vonmises_cdf_cj)
vonmises_icdf = float_params(_vonmises_icdf_cj)
