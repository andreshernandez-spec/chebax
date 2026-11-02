"""Tweedie densities for 1 < p < 2, through Wright's function at beta = 0,
W(rho, z) = Phi(rho, 0; z) = sum_{n>=1} z^n / (n! Gamma(rho n)). For y > 0,

    f(y) = exp(-lambda - y/theta) W(rho, z) / y,
    lambda = mu^(2-p) / (phi (2-p)),  theta = phi (p-1) mu^(p-1),
    rho = (2-p)/(p-1),  z = lambda (y/theta)^rho,

and mass exp(-lambda) at 0. W runs on u = ln z, since z overflows near p = 1
((y/theta)^19 passes 1e308 at y/theta ~ 1e16). Split at Z1 = ZA + S1 (1+rho)^2
in Z = y^(2-p) / (phi (2-p) (p-1)) (tweedie_gen has the evidence):

- Z <= Z1: the series, a two-pass logsumexp unrolled at trace time, N_TERMS exp
  per point.
- Z > Z1: the saddlepoint form ln f = -d(y, mu)/(2 phi) - ln(2 pi phi y^p)/2
  plus the tabulated T/Z, which never forms the terms that cancel, so small phi
  (Z to 1e16 and beyond) keeps its digits if d does (_half_deviance).

The float64 floor is lgamma(rho n) ~ 4000 near the split at p = 1.05: 1.1e-12
in ln f just below it, ~1e-15 in ln W elsewhere. Each branch sees 1.0 outside
its lanes so neither poisons the other's gradient. p is traceable, uniform per
call, in [1.05, 1.95]; outside it, and for mu <= 0, phi <= 0 or y < 0, nan.
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import tweedie_table as _tt
from chebax._src.recipes._common import canon_float, canon_param, traced_coefs
from chebax._src.series import ChebSeries

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _log_series_parts(rho, u):
    """ln sum_{n=1..N} exp(n u - lgamma(n+1) - lgamma(rho n)) and its u and rho
    derivatives in one pass. Reverse mode through the sum took 2.7 s against
    20 ms for the value (N = 2^20, CPU)."""
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
    """ln Phi(rho, 0; z), Wright's generalized Bessel function at beta = 0, for
    traceable rho in [1/19, 19] and z >= 0 (-inf at 0, nan below or outside the
    box). Differentiable in both."""
    rho = canon_param(rho, "log_wright_bessel0", "rho")
    z = canon_float(z)
    ok = _in_box(rho)
    r = jnp.where(ok, rho, 1.0)
    z1, tail = _split(r)
    pos = z > 0
    zs = jnp.where(pos, z, 1.0)
    u = jnp.log(zs)
    lr = jnp.log(r)
    # pow, not exp of logs (5e-14 at z = 1e300); z^e cannot overflow for e < 1
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


# |y/mu - 1| up to which d/2 is a series; 26 terms leave 3e-17 at 1/4
_NEAR = 0.25
_N_NEAR = 26


def _deviance_lanes(mu, y):
    """|y/mu - 1| <= _NEAR, and |ln(y/mu)| <= 1 otherwise, without y/mu."""
    near = jnp.abs(y - mu) <= _NEAR * mu
    mid = ~near & (y <= math.e * mu) & (math.e * y >= mu)
    return near, mid


def _log_ratio(near, mid, mu, y):
    """ln(y/mu) in the near and mid lanes: log1p(delta), but log(y/mu) for y/mu
    in [1/e, 3/4], where XLA's float64 log1p is up to 128 ulp off (jax 0.11)."""
    low = mid & (y < mu)
    d = (jnp.where((near | mid) & ~low, y, mu) - mu) / mu
    return jnp.where(low, jnp.log(jnp.where(low, y, mu) / mu), jnp.log1p(d))


def _half_deviance_impl(a, mu, y):
    """d(y, mu)/2 for p = 2 - a, a in (0, 1), y, mu > 0, delta = (y - mu)/mu.

    near: mu^a sum_k c_k delta^k, c_2 = 1/2, c_{k+1} = c_k (a-k)/(k+1). The
    closed forms lose delta there (float32 gave 2.53 for -5.28 at p = 1.5,
    mu = 1e6, phi = 1e-12, y = mu + 1/8).
    mid (|ln(y/mu)| <= 1): mu^a (expm1(a x) - a delta) / (a (a-1)), x = ln(y/mu),
    ~9 eps / (1 - a) at |delta| = 1/4.
    far: y^a / (a (a-1)) - y mu^(a-1) / (a-1) + mu^a / a, no y/mu (it overflows
    at y = 1e10, mu = 1e-300), ~50 eps past |ln(y/mu)| = 1."""
    near, mid = _deviance_lanes(mu, y)
    d = (jnp.where(near | mid, y, mu) - mu) / mu
    dn = jnp.where(near, d, 0.0)
    c = [0.5]
    for k in range(2, _N_NEAR + 1):
        c.append(c[-1] * (a - k) / (k + 1.0))
    acc = c[-1]
    for ck in c[-2::-1]:
        acc = acc * dn + ck
    ma = jnp.power(mu, a)
    series = ma * dn * dn * acc
    x = _log_ratio(near, mid, mu, y)
    mid_form = ma * (jnp.expm1(a * x) - a * d) / (a * (a - 1.0))
    far = (jnp.power(y, a) / a - y * jnp.power(mu, a - 1.0)) / (a - 1.0) + ma / a
    return jnp.where(near, series, jnp.where(mid, mid_form, far))


@jax.custom_jvp
def _half_deviance(a, mu, y):
    return _half_deviance_impl(a, mu, y)


@_half_deviance.defjvp
def _half_deviance_jvp(primals, tangents):
    """Closed-form mu and y partials, -mu^(a-1) delta and
    (y^(a-1) - mu^(a-1))/(a-1): AD squares mu and overflows at mu = 1e-220.
    The a partial is AD of the value."""
    a, mu, y = primals
    da, dmu, dy = tangents
    h, h_a = jax.jvp(lambda s: _half_deviance_impl(s, mu, y), (a,), (da,))
    near, mid = _deviance_lanes(mu, y)
    delta = (y - mu) / mu
    m1 = jnp.power(mu, a - 1.0)
    h_mu = -m1 * delta
    lr = _log_ratio(near, mid, mu, y)
    h_y = jnp.where(
        near | mid, m1 * jnp.expm1((a - 1.0) * lr), jnp.power(y, a - 1.0) - m1
    ) / (a - 1.0)
    return h, h_a + h_mu * dmu + h_y * dy


@jax.custom_jvp
def _over(h, phi):
    """h / phi with d/dphi = -(h/phi)/phi: jax squares phi, which underflows
    at phi = 1e-200 and makes a zero deviance a nan gradient."""
    return h / phi


@_over.defjvp
def _over_jvp(primals, tangents):
    (h, phi), (dh, dphi) = primals, tangents
    q = h / phi
    return q, (dh - q * dphi) / phi


def tweedie_logpdf(p, mu, phi, y):
    """ln of the Tweedie density at y > 0, and ln P(Y = 0) at y = 0, for
    traceable p in [1.05, 1.95] and broadcasting mu > 0, phi > 0, y >= 0 (nan
    otherwise). Differentiable in p, mu, phi and y (where y > 0)."""
    p = canon_param(p, "tweedie_logpdf", "p")
    mu, phi, y = (canon_float(v) for v in (mu, phi, y))
    mu, phi, y = jnp.broadcast_arrays(mu, phi, y)
    # check p, not rho: float32 rounds p = 1.05 to a rho just past RHO_MAX
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

    # each form sees 1.0 outside its lanes: an inf in an unselected form's
    # partials is a nan gradient (lambda / phi overflows at phi = 1e-200)
    def own(lanes, *vs):
        return [jnp.where(lanes, v, 1.0) for v in vs]

    m0, f0 = own(~pos, mu_s, phi_s)
    f_zero = -jnp.power(m0, a) / (f0 * a)
    # inner: -lambda - y/theta as -d/(2 phi) + y^a / (a (a-1) phi) and z without
    # mu, so the mean gradient is exactly (y - mu) / (phi mu^p); as separate
    # terms it cancels to rounding, ~1e286 at mu = 1e-300
    mi, fi, yi = own(inner, mu_s, phi_s, y_s)
    u = r * (jnp.log(yi) - jnp.log(pp - 1.0)) - (1.0 + r) * jnp.log(fi) - jnp.log(a)
    f_in = (
        -_over(_half_deviance(a, mi, yi), fi)
        + _over(jnp.power(yi, a) / (a * (a - 1.0)), fi)
        - jnp.log(yi)
        + _log_series(r, u)
    )
    # outer: 1/Z from phi directly, not a reciprocal of Z ~ 1/phi
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
