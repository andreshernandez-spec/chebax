"""von Mises (mu = 0) CDF and quantile, kappa in [0, 50], theta in [-pi, pi]:

    F(theta; kappa) = 1/2 + theta/(2 pi) + theta H(theta^2; kappa),

H tabulated in (theta^2, sqrt(kappa)), so oddness and the theta = 0 gradient
are exact. kappa is a traced scalar (unchecked under trace); its bucket sets
how many w-coefficients run (BUCKETS), picked in Python when kappa is
concrete (eager calls and their gradients) and by lax.switch otherwise,
which XLA folds for a constant under jit; a vmapped kappa runs every bucket.
F is accurate absolutely, not relatively, so the quantile returns nan below
min(p, 1-p) = 4096 eps. Wrap theta first; mu != 0 is F(theta - mu).
"""

import functools

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

# (kappa upper bound, w-coefficients kept): the cut at which the dropped tail
# over the bucket is under 1e-15 in H, dH/dw (sum k^2 |c_k|) and dH/dkappa
# (experiments/24's bounds on r in [0, sqrt(kappa)]); the last is the table
BUCKETS = ((2.0, 31), (10.0, 53), (_vt.KMAX, _vt.TABLE.shape[0]))


def _solver_consts():
    """Newton count, residual floor (4x the measured 2.8 eps f64, 8.5 eps f32)
    and probability floor, where 32 eps / min(p, 1-p) reaches ~1e-4."""
    eps = float(jnp.finfo(jnp.empty(()).dtype).eps)
    return (32 if eps < 1e-10 else 20), 32.0 * eps, 4096.0 * eps


def _h_series(kappa, n=None):
    r = jnp.sqrt(jnp.asarray(kappa))
    table = _vt.TABLE if n is None else _vt.TABLE[:n]
    return ChebSeries(traced_coefs(table, 0.0, _vt.RMAX, r), (0.0, _vt.WMAX))


def _concrete(x):
    """x as a float if it is known at trace time, else None. A forward-mode
    tracer's primal is concrete under an eager gradient (jax 0.11 has no
    public query), so eager grad, jacfwd and hessian stay on the static path."""
    while isinstance(x, jax.core.Tracer):
        x = getattr(x, "primal", None)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bucketed(core, kappa, *args):
    """core(kappa, *args, n) on kappa's bucket: picked here for a concrete
    kappa, by lax.switch for a traced one. Outside the custom_jvp rules, so
    the bucket is one choice per call."""
    branches = [functools.partial(core, n=n) for _, n in BUCKETS]
    edges = [edge for edge, _ in BUCKETS[:-1]]
    k = _concrete(kappa)
    if k is not None:
        return branches[sum(k > edge for edge in edges)](kappa, *args)
    index = sum(jnp.where(kappa > edge, 1, 0) for edge in edges)
    return jax.lax.switch(index, branches, kappa, *args)


def _cdf_impl(kappa, theta, n):
    h = _h_series(kappa, n)
    tc = jnp.clip(theta, -jnp.pi, jnp.pi)
    core = 0.5 + tc / (2 * jnp.pi) + tc * h(tc * tc)
    return jnp.where(theta <= -jnp.pi, 0.0, jnp.where(theta >= jnp.pi, 1.0, core))


def _cdf_dkappa(kappa, theta, n):
    """dF/dkappa as a value: the kappa-derivative of the coefficient vector
    (N-independent) through one series. AD divides by 2 sqrt(kappa), so below
    1e-8 use sin(theta)/(2 pi) + O(kappa)."""
    ks = jnp.maximum(kappa, 1e-8)
    _, dcoef = jax.jvp(lambda k: _h_series(k, n).coef, (ks,), (jnp.ones_like(ks),))
    tc = jnp.clip(theta, -jnp.pi, jnp.pi)
    interior = (theta > -jnp.pi) & (theta < jnp.pi)
    lead = jnp.sin(tc) / (2 * jnp.pi)
    ad = tc * ChebSeries(dcoef, (0.0, _vt.WMAX))(tc * tc)
    return jnp.where(interior, jnp.where(kappa < 1e-8, lead, ad), 0.0)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _cdf_cj(kappa, theta, n):
    return _cdf_impl(kappa, theta, n)


@_cdf_cj.defjvp
def _cdf_jvp(n, primals, tangents):
    k, th = primals
    dkappa, dtheta = tangents
    # derivatives as values, tangents only in the two products: reverse mode
    # then stores two vectors, and jvp-of-impl would run a series per tangent
    h = _h_series(k, n)
    tc = jnp.clip(th, -jnp.pi, jnp.pi)
    w = tc * tc
    hw = h(w)
    interior = (th > -jnp.pi) & (th < jnp.pi)
    core = 0.5 + tc / (2 * jnp.pi) + tc * hw
    F = jnp.where(th <= -jnp.pi, 0.0, jnp.where(th >= jnp.pi, 1.0, core))
    dF_dth = jnp.where(interior, 1 / (2 * jnp.pi) + hw + 2 * w * h.deriv()(w), 0.0)
    dF = dF_dth * dtheta + _cdf_dkappa(k, th, n) * dkappa
    # scale, not select, so nan survives reverse mode
    return F, dF * jnp.where(jnp.isnan(F), jnp.nan, 1.0)


def _vonmises_cdf_cj(kappa, theta):
    """CDF of vonMises(0, kappa) at theta in [-pi, pi], differentiable in
    both; d/dkappa at kappa = 0 is the exact sin(theta)/(2 pi)."""
    return _bucketed(_cdf_cj, kappa, theta)


def _log_pdf(kappa, theta):
    i0 = besseli_scalar(jnp.asarray(0.0), jnp.asarray(kappa))
    return jnp.asarray(kappa) * jnp.cos(theta) - jnp.log(2 * jnp.pi * i0)


def _icdf_impl(kappa, p, n):
    iters, resid_atol, pmin_floor = _solver_consts()
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    h = _h_series(kappa, n)

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


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _icdf_cj(kappa, p, n):
    return _icdf_impl(kappa, p, n)


@_icdf_cj.defjvp
def _icdf_jvp(n, primals, tangents):
    kappa, p = primals
    dkappa, dp = tangents
    th = _icdf_impl(kappa, p, n)
    interior = (th > -jnp.pi) & (th < jnp.pi)
    ths = jnp.where(interior, th, 0.0)
    pdf = jnp.exp(_log_pdf(kappa, ths))
    dth = (dp - _cdf_dkappa(kappa, ths, n) * dkappa) / pdf
    # scale, not select: zero at the endpoints, nan for a rejected solve
    return th, dth * jnp.where(interior, 1.0, jnp.where(jnp.isnan(th), jnp.nan, 0.0))


def _vonmises_icdf_cj(kappa, p):
    """Quantile of vonMises(0, kappa). A solve whose CDF residual exceeds
    max(1e-6 min(p, 1-p), 32 eps) returns nan, and so does min(p, 1-p) below
    4096 eps (9.1e-13 f64, 4.9e-4 f32), where the error would reach 0.11 at
    1e-15. Measured worst relative error 1.2e-5 at 1e-11 and 1.0e-4 just above
    the floor, for kappa in {2, 10, 25, 50}. p = 0, 1 give -pi, pi exactly."""
    return _bucketed(_icdf_cj, kappa, p)


# rules on the private names, so kappa is a float scalar before any tangent
vonmises_cdf = float_params(_vonmises_cdf_cj)
vonmises_icdf = float_params(_vonmises_icdf_cj)
