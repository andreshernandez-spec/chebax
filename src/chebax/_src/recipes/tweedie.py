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

import functools
import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import tweedie_table as _tt
from chebax._src.recipes._common import canon_float, canon_param, traced_coefs
from chebax._src.series import ChebSeries

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _log_series_parts(rho, u, lead=1):
    """ln sum_{n=1..N} exp((n - lead) u - lgamma(n+1) - lgamma(rho n)) and its u
    and rho derivatives in one pass, lead = 1 being ln W - u. Reverse mode
    through the sum took 2.7 s against 20 ms for the value (N = 2^20, CPU)."""
    n = jnp.arange(1, _tt.N_TERMS + 1, dtype=u.dtype)
    c = jax.lax.lgamma(n + 1.0) + jax.lax.lgamma(rho * n)
    g = n * jax.lax.digamma(rho * n)  # -d/drho of term n's exponent
    m = (1 - lead) * u - c[0]
    for k in range(1, _tt.N_TERMS):
        m = jnp.maximum(m, (k + 1 - lead) * u - c[k])
    s = su = sr = jnp.zeros_like(u)
    for k in range(_tt.N_TERMS):
        e = jnp.exp((k + 1 - lead) * u - c[k] - m)
        s = s + e
        su = su + (k + 1 - lead) * e
        sr = sr + g[k] * e
    return m + jnp.log(s), su / s, -sr / s


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _log_series(rho, u, lead=0):
    """ln W(rho, e^u) - lead u. lead = 1 for the density, whose -ln y cancels
    the leading term's u: apart, their 1/y slopes cancel and lose d/dy (0 for
    1 at p = 1.5, y = 1e-7 in float32)."""
    return _log_series_parts(rho, u, lead)[0]


@_log_series.defjvp
def _log_series_jvp(lead, primals, tangents):
    rho, u = primals
    drho, du = tangents
    val, d_u, d_rho = _log_series_parts(rho, u, lead)
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
    h_mu = -jnp.power(mu, a - 1.0) * ((y - mu) / mu)
    return h, h_a + h_mu * dmu + _half_deviance_dy(a, mu, y) * dy


def _half_deviance_dy(a, mu, y):
    near, mid = _deviance_lanes(mu, y)
    m1 = jnp.power(mu, a - 1.0)
    lr = _log_ratio(near, mid, mu, y)
    return jnp.where(
        near | mid, m1 * jnp.expm1((a - 1.0) * lr), jnp.power(y, a - 1.0) - m1
    ) / (a - 1.0)


_barrier = jax.lax.optimization_barrier


def _numerator(a, mu, y, with_y):
    h = _half_deviance_impl(a, mu, y)
    return h - jnp.power(y, a) / (a * (a - 1.0)) if with_y else h


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _deviance_over(a, mu, y, phi, with_y=False):
    """d(y, mu) / (2 phi), less y^a / (a (a-1) phi) with_y (the series branch's
    lambda + y/theta, without mu^(1-p)/phi terms that cancel)."""
    return _numerator(a, mu, y, with_y) / phi


@_deviance_over.defjvp
def _deviance_over_jvp(with_y, primals, tangents):
    # The mean partial over phi, -mu^(a-1) delta / phi = -(y - mu) / (phi mu^p).
    # Where mu^(a-1) delta leaves the float range before phi brings it back
    # (1e320 for 1e120 at mu = 1e-200, phi = 1e200, y = 1e20) it is formed as
    # -(y - mu) h^2, h = mu^(-p/2) / sqrt(phi), in range when the result is
    a, mu, y, phi = primals
    da, dmu, dy, dphi = tangents
    h, h_a = jax.jvp(lambda s: _numerator(s, mu, y, with_y), (a,), (da,))
    q = h / phi
    t = jnp.power(mu, a - 1.0) * ((y - mu) / mu)
    far = ~jnp.isfinite(t) | ((jnp.abs(t) < jnp.finfo(t.dtype).tiny) & (y != mu))
    # the barriers stop XLA regrouping these into mu phi or h^2 (out of range)
    md, yd = jnp.where(far, 1.0, mu), jnp.where(far, 1.0, y)
    near_q = -_barrier(jnp.power(md, a - 1.0) * ((yd - md) / md)) / phi
    mf, yf, ff = (jnp.where(far, v, 1.0) for v in (mu, y, phi))
    hf = jnp.power(mf, -0.5 * (2.0 - a)) / jnp.sqrt(ff)
    q_mu = jnp.where(far, -_barrier((yf - mf) * hf) * hf, near_q)
    # with_y, d/dy is -mu^(a-1)/(a-1) exactly: formed as the difference of the
    # two y^(a-1)/(a-1) terms it kept y^(a-1) eps (4e-10 at y = 1e-12, jax 0.5.3)
    if with_y:
        h_y = _barrier(-jnp.power(mu, a - 1.0) / (a - 1.0))
    else:
        h_y = _half_deviance_dy(a, mu, y)
    # (dh - q dphi) / phi: jax's own d/dphi squares phi, and XLA turns
    # (h/phi)/phi into h/phi^2, both 0 at phi = 1e-200 and then nan
    return q, (h_a + h_y * dy - q * dphi) / phi + q_mu * dmu


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
    u0 = -r * jnp.log(pp - 1.0) - (1.0 + r) * jnp.log(fi) - jnp.log(a)
    u = r * jnp.log(yi) + u0
    # -ln y + u as (r - 1) ln y + u0, with the series' own u taken out
    f_in = (
        -_deviance_over(a, mi, yi, fi, True)
        + (r - 1.0) * jnp.log(yi)
        + u0
        + _log_series(r, u, 1)
    )
    # outer: 1/Z from phi directly, not a reciprocal of Z ~ 1/phi
    mt, ft, yt = own(outer, mu_s, phi_s, y_s)
    w = jnp.where(outer, ft * a * (pp - 1.0) / jnp.power(yt, a), 1.0 / z1)
    f_tail = (
        -_deviance_over(a, mt, yt, ft)
        - 0.5 * (jnp.log(ft) + pp * jnp.log(yt))
        - _HALF_LOG_2PI
        + tail(z1 * w) * w
    )
    out = jnp.where(inner, f_in, jnp.where(outer, f_tail, f_zero))
    out = jnp.where(valid, out, jnp.nan)
    return jnp.where(ok, out, jnp.nan)
