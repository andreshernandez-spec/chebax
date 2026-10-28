"""Tweedie densities for 1 < p < 2, through Wright's function at beta = 0.

    W(rho, z) = Phi(rho, 0; z) = sum_{n>=1} z^n / (n! Gamma(rho n))

The compound Poisson-gamma Tweedie law with mean mu, dispersion phi and
power p in (1, 2) puts mass exp(-lambda) at 0 and has, for y > 0,

    f(y) = exp(-lambda - y/theta) W(rho, z) / y,
    lambda = mu^(2-p) / (phi (2-p)),  theta = phi (p-1) mu^(p-1),
    rho = (2-p)/(p-1),  z = lambda (y/theta)^rho.

W runs on u = ln z, never on z: near p = 1 the argument overflows a double
on ordinary data (p = 1.05 is rho = 19, and (y/theta)^19 passes 1e308 at
y/theta ~ 1e16). Two regions in the saddle scale

    Z = (1+rho)/rho (rho z)^(1/(1+rho)) = y^(2-p) / (phi (2-p) (p-1)),

split at Z1(rho) = ZA + S1 (1+rho)^2 (tweedie_gen has the measurements
behind both constants and the reason the split is in Z):

- Z <= Z1: the series itself over n = 1..N_TERMS, as a two-pass logsumexp
  unrolled at trace time, so the program never builds a (points x terms)
  array. Only lgamma(rho n) depends on rho, and rho is uniform per call, so
  those are per call and each point pays N_TERMS exp.
- Z > Z1: ln W = Z + ln(t0)/2 - ln(2 pi (1+rho))/2 + T/Z, t0 = rho Z/(1+rho),
  with T(ln rho, Z1/Z) from the baked table. For the density the large
  terms cancel exactly, Z against lambda + y/theta, leaving
      ln f = -d(y, mu)/(2 phi) - ln(2 pi phi y^p)/2 + T/Z,
  the classical saddlepoint approximation (d the unit deviance) with its
  exact correction. Written that way the cancelling terms are never
  computed, so a concentrated density (small phi, Z up to 1e16 and beyond)
  keeps its digits, provided d itself does (see _half_deviance).

The f64 floor is lgamma's rounding in the direct sum: lgamma(rho n) reaches
~4000 near the split at p = 1.05, and eps |lgamma| survives into ln f
(2.4e-13 measured there). ln W elsewhere sits at ~1e-15.

Each branch sees the raw arguments in its own region and 1.0 elsewhere, so
neither poisons the other's gradient. p is traceable
and uniform per call; jax.grad works in p, mu, phi and y. The box is p in
[1.05, 1.95] (rho in [1/19, 19]); outside it, and for mu <= 0, phi <= 0 or
y < 0, the result is nan (not checked eagerly: p may be a tracer).
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import tweedie_table as _tt
from chebax._src.recipes._common import canon_float, canon_param, traced_coefs
from chebax._src.series import ChebSeries

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _log_series_parts(rho, u):
    """ln S for S = sum_{n=1..N} exp(n u - lgamma(n+1) - lgamma(rho n)), with
    its exact derivatives in u and rho, all from one unrolled pass.

    Both derivatives are averages over the same weights e_n / S, so the
    tangent costs one multiply-add per term. Reverse mode through the
    unrolled sum instead turned the scalar rho cotangent into one reduction
    over every point per term: 2.7 s against 20 ms for the value, N = 2^20
    on CPU."""
    n = jnp.arange(1, _tt.N_TERMS + 1, dtype=u.dtype)
    c = jax.lax.lgamma(n + 1.0) + jax.lax.lgamma(rho * n)
    g = n * jax.lax.digamma(rho * n)  # -d/drho of term n's exponent
    m = u - c[0]
    for k in range(1, _tt.N_TERMS):
        m = jnp.maximum(m, (k + 1) * u - c[k])
    s = su = sr = jnp.zeros_like(u)
    for k in range(_tt.N_TERMS):
        e = jnp.exp((k + 1) * u - c[k] - m)
        s = s + e
        su = su + (k + 1) * e
        sr = sr + g[k] * e
    return m + jnp.log(s), su / s, -sr / s


@jax.custom_jvp
def _log_series(rho, u):
    return _log_series_parts(rho, u)[0]


@_log_series.defjvp
def _log_series_jvp(primals, tangents):
    rho, u = primals
    drho, du = tangents
    val, d_u, d_rho = _log_series_parts(rho, u)
    return val, d_u * du + d_rho * drho


def _split(rho):
    """Z1(rho) and the tail series T(Z1/Z) at this rho."""
    z1 = _tt.ZA + _tt.S1 * (1.0 + rho) ** 2
    coefs = traced_coefs(_tt.TABLE_TAIL, -_tt.A1, _tt.A1, jnp.log(rho))
    return z1, ChebSeries(coefs, (0.0, 1.0))


def _in_box(rho):
    return (rho >= _tt.RHO_MIN) & (rho <= _tt.RHO_MAX)


def log_wright_bessel0(rho, z):
    """ln Phi(rho, 0; z), Wright's generalized Bessel function at beta = 0.

    rho is a (traceable) scalar in [1/19, 19], uniform per call; z >= 0 is
    any array. Differentiable in both. -inf at z = 0 (Phi(rho, 0; 0) = 0),
    nan for z < 0 or rho outside the box. This is the slice the Tweedie
    density needs (see tweedie_logpdf); other beta are not provided.
    """
    rho = canon_param(rho, "log_wright_bessel0", "rho")
    z = canon_float(z)
    ok = _in_box(rho)
    r = jnp.where(ok, rho, 1.0)
    z1, tail = _split(r)
    pos = z > 0
    zs = jnp.where(pos, z, 1.0)
    u = jnp.log(zs)
    lr = jnp.log(r)
    # pow, not exp((lr + u)/(1+rho)): that carries eps |ln z| (5e-14 at
    # z = 1e300) where pow stays at an ulp, and z^e cannot overflow for e < 1
    e = 1.0 / (1.0 + r)
    big_z = (1.0 + r) / r * jnp.power(r, e) * jnp.power(zs, e)
    inner = big_z <= z1
    # u at the split, for the dummy lanes of each branch
    u1 = (1.0 + r) * jnp.log(r * z1 / (1.0 + r)) - lr
    w_in = _log_series(r, jnp.where(inner, u, u1))
    zt = jnp.where(inner, z1, big_z)
    lt0 = jnp.log(r * zt / (1.0 + r))
    w_tail = zt + 0.5 * lt0 - _HALF_LOG_2PI - 0.5 * jnp.log1p(r) + tail(z1 / zt) / zt
    out = jnp.where(inner, w_in, w_tail)
    out = jnp.where(pos, out, jnp.where(z == 0, -jnp.inf, jnp.nan))
    return jnp.where(ok, out, jnp.nan)


# |y/mu - 1| up to which d/2 is summed as a series, and the term count:
# 26 terms leave under 3e-17 relative at |delta| = 1/4 for every a in (0, 1)
_NEAR = 0.25
_N_NEAR = 26


def _half_deviance_impl(a, mu, y):
    """d(y, mu)/2 for power p = 2 - a, a in (0, 1), y > 0, mu > 0.

    Near the mean, in delta = (y - mu)/mu, whose subtraction is exact there:
        d/2 = mu^a sum_{k>=2} c_k delta^k,  c_2 = 1/2,  c_{k+1} = c_k (a-k)/(k+1),
    the binomial series of ((1+delta)^a - 1 - a delta) / (a (a-1)). The
    closed forms lose this on concentrated data: ln y - ln mu carries
    eps |ln y| where only delta matters, and expm1(a x) - a expm1(x)
    cancels two first-order terms down to a quadratic one. In float32,
    (p, mu, phi, y) = (1.5, 1e6, 1e-12, 1e6 + 1/8) gave 2.53 for a log
    density of -5.28 (review, 2026-10-26).

    Away from it, y^a / (a (a-1)) - y mu^(a-1) / (a-1) + mu^a / a, which
    never forms y/mu: that ratio overflowed at y = 1e10, mu = 1e-300 under
    a finite density. Its cancellation costs eps a |delta| relative to Z,
    the same as one ulp of y."""
    near = jnp.abs(y - mu) <= _NEAR * mu
    d = (jnp.where(near, y, mu) - mu) / mu
    c = [0.5]
    for k in range(2, _N_NEAR + 1):
        c.append(c[-1] * (a - k) / (k + 1.0))
    acc = c[-1]
    for ck in c[-2::-1]:
        acc = acc * d + ck
    series = jnp.power(mu, a) * d * d * acc
    far = (jnp.power(y, a) / a - y * jnp.power(mu, a - 1.0)) / (a - 1.0) + jnp.power(
        mu, a
    ) / a
    return jnp.where(near, series, far)


@jax.custom_jvp
def _half_deviance(a, mu, y):
    return _half_deviance_impl(a, mu, y)


@_half_deviance.defjvp
def _half_deviance_jvp(primals, tangents):
    """The mu and y partials in closed form, -mu^(a-1) delta and
    (y^(a-1) - mu^(a-1))/(a-1), the latter through expm1 near the mean.
    AD of the forms above squares mu, in mu^(a-2) and in delta's own
    derivative, and that overflowed at mu = 1e-220: the mean gradient at
    y = mu, exactly 0, came out 0 * inf (review follow-up, 2026-10-26).
    The a partial is AD of the value with mu and y held."""
    a, mu, y = primals
    da, dmu, dy = tangents
    h, h_a = jax.jvp(lambda s: _half_deviance_impl(s, mu, y), (a,), (da,))
    near = jnp.abs(y - mu) <= _NEAR * mu
    delta = (y - mu) / mu
    m1 = jnp.power(mu, a - 1.0)
    h_mu = -m1 * delta
    lr = jnp.log1p(jnp.where(near, delta, 0.0))
    h_y = jnp.where(
        near, m1 * jnp.expm1((a - 1.0) * lr), jnp.power(y, a - 1.0) - m1
    ) / (a - 1.0)
    return h, h_a + h_mu * dmu + h_y * dy


@jax.custom_jvp
def _over(h, phi):
    """h / phi, whose phi-derivative is formed as -(h/phi)/phi. jax's own
    rule squares phi first, which underflows at phi = 1e-200 and turned
    a zero deviance into a nan dispersion gradient."""
    return h / phi


@_over.defjvp
def _over_jvp(primals, tangents):
    (h, phi), (dh, dphi) = primals, tangents
    q = h / phi
    return q, (dh - q * dphi) / phi


def tweedie_logpdf(p, mu, phi, y):
    """ln of the Tweedie density for power p in [1.05, 1.95].

    p is a (traceable) scalar, uniform per call; mu > 0, phi > 0 and y >= 0
    broadcast against each other (per-observation means are the GLM case).
    At y = 0 this is ln P(Y = 0) = -lambda (the point mass); for y > 0 the
    density. Differentiable in p, mu, phi and y (in y only where y > 0).
    nan outside the box or for mu <= 0, phi <= 0, y < 0.
    """
    p = canon_param(p, "tweedie_logpdf", "p")
    mu, phi, y = (canon_float(v) for v in (mu, phi, y))
    mu, phi, y = jnp.broadcast_arrays(mu, phi, y)
    # the box is checked on p as given: in float32, p = 1.05 rounds to
    # 1.04999995 and its rho to 19.000011, just past RHO_MAX
    ok = (p >= _tt.P_LO) & (p <= _tt.P_HI)
    pp = jnp.where(ok, p, 1.5)
    r = (2.0 - pp) / (pp - 1.0)
    a = 2.0 - pp
    valid = (mu > 0) & (phi > 0) & (y >= 0)
    mu_s = jnp.where(valid, mu, 1.0)
    phi_s = jnp.where(valid, phi, 1.0)
    pos = valid & (y > 0)
    y_s = jnp.where(pos, y, mu_s)
    z1, tail = _split(r)
    big_z = jnp.power(y_s, a) / (phi_s * a * (pp - 1.0))
    inner = pos & (big_z <= z1)
    outer = pos & ~inner

    # Each of the three forms sees its own lanes' arguments and 1.0 in the
    # others. An unselected form's partials meet a zero cotangent, so one
    # inf among them is a nan gradient: at phi = 1e-200, lambda / phi alone
    # overflows (review, 2026-10-26).
    def own(lanes, *vs):
        return [jnp.where(lanes, v, 1.0) for v in vs]

    m0, f0 = own(~pos, mu_s, phi_s)
    f_zero = -jnp.power(m0, a) / (f0 * a)
    # inner: the density as defined, ln W on u = ln z. mu enters only
    # through -lambda - y/theta = -d/(2 phi) + y^a / (a (a-1) phi), so the
    # mean gradient is the deviance's exact (y - mu) / (phi mu^p). As two
    # separate terms their mu-derivatives cancelled only to rounding at
    # y = mu, ~1e286 at mu = 1e-300, and y/theta underflowed at phi = 1e200
    # and took its share of the gradient with it. Likewise
    # z = y^rho / (a (p-1)^rho phi^(1+rho)) holds no mu: built from lambda
    # and theta, the residue of its mu terms had derivative eps/mu
    # (review follow-up, 2026-10-26).
    mi, fi, yi = own(inner, mu_s, phi_s, y_s)
    u = r * (jnp.log(yi) - jnp.log(pp - 1.0)) - (1.0 + r) * jnp.log(fi) - jnp.log(a)
    f_in = (
        -_over(_half_deviance(a, mi, yi), fi)
        + _over(jnp.power(yi, a) / (a * (a - 1.0)), fi)
        - jnp.log(yi)
        + _log_series(r, u)
    )
    # outer: saddlepoint form plus the tabulated correction T(Z1/Z)/Z, with
    # 1/Z built from phi directly rather than as a reciprocal of Z ~ 1/phi
    mt, ft, yt = own(outer, mu_s, phi_s, y_s)
    w = jnp.where(outer, ft * a * (pp - 1.0) / jnp.power(yt, a), 1.0 / z1)
    f_tail = (
        -_over(_half_deviance(a, mt, yt), ft)
        - 0.5 * (jnp.log(ft) + pp * jnp.log(yt))
        - _HALF_LOG_2PI
        + tail(z1 * w) * w
    )
    out = jnp.where(inner, f_in, jnp.where(outer, f_tail, f_zero))
    out = jnp.where(valid, out, jnp.nan)
    return jnp.where(ok, out, jnp.nan)
