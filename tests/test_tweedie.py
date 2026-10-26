"""Tweedie densities, 1.05 <= p <= 1.95, and ln Phi(rho, 0; z) under them.

References are mpmath at 40 dps: Phi summed over a window around its peak
term (the terms are log-concave in n), the density from its definition.
Both are logarithms, so the error is ABSOLUTE in the log (relative in W and
f) and, where the log is large, relative to it: |err| / max(1, |ref|).
Large logs carry eps |ln W| from forming Z itself, which no method avoids.

Measured worst, float64 (bar about 4x in each test):
- ln W: values 7.7e-16; d/dz 1.0e-14 relative (oracle: dPhi/dz is
  Phi(rho, rho; z)); d/drho 1.6e-15; z to 1e300 against the leading saddle
  term, 3.2e-16. Both branches at the split agree to 4.4e-16 in value and
  3.5e-15 in d/du. The table's s -> 0 end matches
  c1(rho) = -(2 rho + 1)(rho + 2)/(24 rho) to 8.9e-16.
- density: values 2.4e-13; d/dmu 3.4e-15, d/dy 2.4e-13, d/dphi 1.5e-12,
  d/dp 3.7e-12. All worst near p = 1.05 in the direct sum, where
  lgamma(rho n) reaches ~4000 near the split and its rounding,
  eps |lgamma|, survives the cancellation down to an O(1) log-density;
  d/dp also carries drho/dp = -1/(p-1)^2 = -400 there.
- float32 (x64 off): ln W 7.8e-7, density 7.4e-6.
"""

import os
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import tweedie_gen
from chebax._src.recipes import tweedie_table as tt
from chebax._src.recipes.tweedie import _log_series, _split

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

RHOS = [1 / 19, 0.07, 0.111, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 14.0, 19.0]


def log_phi(rho, beta, z):
    """ln Phi(rho, beta; z) for z > 0, summed around its peak term."""
    rho, beta, z = mp.mpf(rho), mp.mpf(beta), mp.mpf(z)
    lz = mp.log(z)
    nmin = 1 if beta == 0 else 0

    def lt(n):
        return n * lz - mp.loggamma(n + 1) - mp.loggamma(rho * n + beta)

    n0 = max(nmin, int((rho * z) ** (1 / (1 + rho)) / rho))
    while n0 > nmin and lt(n0 - 1) > lt(n0):
        n0 -= 1
    while lt(n0 + 1) > lt(n0):
        n0 += 1
    peak, s = lt(n0), mp.mpf(0)
    for step, start in ((-1, n0 - 1), (1, n0)):
        n = start
        while n >= nmin:
            d = lt(n) - peak
            if d < -120:
                break
            s += mp.exp(d)
            n += step
    return peak + mp.log(s)


def z_of(rho, big_z):
    """z at saddle scale Z = (1+rho)/rho (rho z)^(1/(1+rho))."""
    rho, big_z = mp.mpf(rho), mp.mpf(big_z)
    return (rho * big_z / (1 + rho)) ** (1 + rho) / rho


def zgrid(rho):
    z1 = tt.ZA + tt.S1 * (1 + rho) ** 2
    return [1e-6, 0.01, 0.5, 3.0, 0.9 * z1, z1 * (1 - 1e-9), z1 * (1 + 1e-9),
            1.5 * z1, 10 * z1, 1e3, 1e4, 1e5]


def ref_logpdf(p, mu, phi, y):
    p, mu, phi, y = map(mp.mpf, (p, mu, phi, y))
    lam = mu ** (2 - p) / (phi * (2 - p))
    if y == 0:
        return -lam
    theta = phi * (p - 1) * mu ** (p - 1)
    rho = (2 - p) / (p - 1)
    u = mp.log(lam) + rho * (mp.log(y) - mp.log(theta))
    return -lam - y / theta - mp.log(y) + log_phi(rho, 0, mp.exp(u))


CASES = [(p, mu, phi, y)
         for p in (1.05, 1.08, 1.2, 1.5, 1.8, 1.95)
         for mu, phi in ((1.0, 1.0), (50.0, 0.05), (0.2, 3.0), (3.0, 0.3), (8.0, 1.7))
         for y in (0.0, 1e-6, 0.05, 0.9, 3.0, 20.0, 150.0)
         if y == 0 or y ** (2 - p) / (phi * (2 - p) * (p - 1)) <= 1e5]


def _rel(got, ref):
    return abs(got - ref) / max(1.0, abs(ref))


def test_log_wright_bessel0_values():
    worst = 0.0
    for rho in RHOS:
        for big_z in zgrid(rho):
            z = float(z_of(rho, big_z))
            worst = max(worst, _rel(float(chebax.log_wright_bessel0(rho, z)),
                                    float(log_phi(rho, 0, z))))
    assert worst <= 3e-15, worst


def test_log_wright_bessel0_dz():
    # d ln W/dz = Phi(rho, rho; z) / Phi(rho, 0; z)
    worst = 0.0
    for rho in RHOS:
        for big_z in zgrid(rho):
            z = float(z_of(rho, big_z))
            ref = float(mp.exp(log_phi(rho, rho, z) - log_phi(rho, 0, z)))
            got = float(jax.grad(lambda zz: chebax.log_wright_bessel0(rho, zz))(z))
            worst = max(worst, abs(got - ref) / abs(ref))
    assert worst <= 4e-14, worst


def test_log_wright_bessel0_drho():
    worst = 0.0
    for rho in (1 / 19, 0.25, 1.0, 4.0, 19.0 - 1e-9):
        for big_z in (0.5, 3.0, 1.5 * (tt.ZA + tt.S1 * (1 + rho) ** 2), 1e3, 1e4):
            z = float(z_of(rho, big_z))
            ref = float(mp.diff(lambda r: log_phi(r, 0, z), mp.mpf(rho)))
            got = float(jax.grad(lambda r: chebax.log_wright_bessel0(r, z))(rho))
            worst = max(worst, _rel(got, ref))
    assert worst <= 7e-15, worst


def _branches(rho):
    rho = jnp.asarray(rho)
    z1, tail = _split(rho)
    lr = jnp.log(rho)
    u1 = (1 + rho) * jnp.log(rho * z1 / (1 + rho)) - lr

    def inner(u):
        return _log_series(rho, u)

    def outer(u):
        big_z = (1 + rho) / rho * jnp.exp((lr + u) / (1 + rho))
        return (big_z + 0.5 * jnp.log(rho * big_z / (1 + rho))
                - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log1p(rho)
                + tail(z1 / big_z) / big_z)
    return u1, inner, outer


def test_branches_agree_at_the_split():
    # the series and the tabulated saddle form, both at the split point
    for rho in np.exp(np.linspace(-np.log(19), np.log(19), 23)):
        u1, inner, outer = _branches(rho)
        vi, vo = float(inner(u1)), float(outer(u1))
        di, do = float(jax.grad(inner)(u1)), float(jax.grad(outer)(u1))
        assert abs(vi - vo) / max(1.0, abs(vi)) <= 2e-15, rho
        assert abs(di - do) / abs(di) <= 2e-14, rho


def test_table_limit_is_c1():
    # T -> c1(rho) as Z -> inf, in closed form: the end mpmath cannot reach
    for rho in np.exp(np.linspace(-np.log(19), np.log(19), 41)):
        _, tail = _split(jnp.asarray(rho))
        c1 = -(2 * rho + 1) * (rho + 2) / (24 * rho)
        assert abs(float(tail(0.0)) - c1) <= 4e-15, rho


def test_huge_arguments_follow_the_saddle():
    # far past any reference: ln W tracks its leading term to eps relative
    for rho, z in ((1 / 19, 1e300), (1.0, 1e300), (19.0, 1e300), (0.3, 1e150)):
        got = float(chebax.log_wright_bessel0(rho, z))
        t0 = (rho * z) ** (1 / (1 + rho)) if rho * z < 1e308 else np.exp(
            (np.log(rho) + np.log(z)) / (1 + rho))
        big_z = (1 + rho) * t0 / rho
        lead = big_z + 0.5 * np.log(t0) - 0.5 * np.log(2 * np.pi * (1 + rho))
        assert np.isfinite(got) and abs(got - lead) / lead <= 1.5e-15, (rho, got, lead)


def test_log_wright_bessel0_edges():
    assert float(chebax.log_wright_bessel0(1.0, 0.0)) == -np.inf
    assert np.isnan(float(chebax.log_wright_bessel0(1.0, -1.0)))
    for rho in (1 / 19 * 0.999, 19.01, 0.0, -1.0):
        assert np.isnan(float(chebax.log_wright_bessel0(rho, 2.0))), rho
    # small z: Phi ~ z / Gamma(rho)
    z = 1e-250
    assert abs(float(chebax.log_wright_bessel0(2.5, z))
               - (np.log(z) - float(mp.loggamma(2.5)))) <= 1e-13


def test_tweedie_values():
    worst = 0.0
    for c in CASES:
        worst = max(worst, _rel(float(chebax.tweedie_logpdf(*c)), float(ref_logpdf(*c))))
    assert worst <= 1e-12, worst


@pytest.mark.parametrize("arg, bar", [(0, 1.5e-11), (1, 1.5e-14), (2, 6e-12), (3, 1e-12)])
def test_tweedie_gradients(arg, bar):
    worst = 0.0
    for c in CASES:
        if c[3] not in (0.0, 0.05, 3.0, 150.0) or c[1] not in (1.0, 50.0, 3.0):
            continue
        if arg == 3 and c[3] == 0.0:
            continue
        got = float(jax.grad(chebax.tweedie_logpdf, argnums=arg)(*map(float, c)))
        args = list(map(mp.mpf, c))
        ref = float(mp.diff(lambda v: ref_logpdf(*(args[:arg] + [v] + args[arg + 1:])), args[arg]))
        worst = max(worst, _rel(got, ref))
    assert worst <= bar, worst


def test_tweedie_normalizes_with_its_moments():
    # P(0) + int f = 1, E Y = mu, Var Y = phi mu^p: the whole density at once.
    # Integrated in v = ln y: near 0 the density is ~ y^(rho - 1), whose mass
    # spreads over hundreds of decades when rho is small (p near 2), and in v
    # that end is a plain exponential e^(rho v).
    from scipy import integrate
    for p, mu, phi in ((1.1, 1.0, 1.0), (1.5, 2.0, 0.5), (1.9, 0.5, 2.0), (1.3, 5.0, 0.2)):
        logf = jax.jit(lambda y: chebax.tweedie_logpdf(p, mu, phi, y))
        sd = np.sqrt(phi * mu ** p)
        vhi = np.log(mu + 60 * sd)

        def g(v, k):
            return float(np.exp(logf(np.exp(v)) + (k + 1) * v))
        p0 = float(np.exp(logf(0.0)))
        moments = [integrate.quad(g, -745.0, vhi, args=(k,), points=[np.log(mu)],
                                  epsabs=1e-14, epsrel=1e-13, limit=400)[0]
                   for k in (0, 1, 2)]
        assert abs(p0 + moments[0] - 1) <= 1e-9, (p, moments)
        assert abs(moments[1] - mu) / mu <= 1e-9, (p, moments)
        var = moments[2] - moments[1] ** 2
        assert abs(var - phi * mu ** p) / (phi * mu ** p) <= 1e-8, (p, moments)


def test_tweedie_zero_mass_and_invalid():
    p, mu, phi = 1.4, 2.0, 0.7
    lam = mu ** (2 - p) / (phi * (2 - p))
    assert float(chebax.tweedie_logpdf(p, mu, phi, 0.0)) == pytest.approx(-lam, rel=1e-15)
    for bad in ((1.0499, mu, phi, 1.0), (1.9501, mu, phi, 1.0), (p, 0.0, phi, 1.0),
                (p, mu, -1.0, 1.0), (p, mu, phi, -0.5)):
        assert np.isnan(float(chebax.tweedie_logpdf(*bad))), bad


def test_tweedie_broadcast_and_parameter_shape():
    mu = jnp.asarray([0.5, 1.0, 2.0, 4.0])
    y = jnp.asarray([0.0, 0.3, 2.0, 9.0])
    got = np.asarray(chebax.tweedie_logpdf(1.6, mu, 0.8, y))
    ref = [float(chebax.tweedie_logpdf(1.6, float(m), 0.8, float(v))) for m, v in zip(mu, y)]
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(chebax.tweedie_logpdf(jnp.asarray([1.6]), mu, 0.8, y)),
                               got, rtol=0, atol=0)
    with pytest.raises(ValueError, match="pergroup"):
        chebax.tweedie_logpdf(jnp.asarray([1.3, 1.6]), 1.0, 1.0, 1.0)


def test_tweedie_traced_and_mixed_batch_gradients():
    y = jnp.asarray([0.0, 1e-8, 0.4, 3.0, 40.0])
    mu = jnp.full_like(y, 2.0)

    def total(p, m, phi):
        return jnp.sum(chebax.tweedie_logpdf(p, m, phi, y))

    eager = float(total(1.3, mu, 0.9))
    jitted = float(jax.jit(total)(1.3, mu, 0.9))
    assert eager == pytest.approx(jitted, rel=1e-14)
    gp, gm, gphi = jax.grad(total, argnums=(0, 1, 2))(1.3, mu, 0.9)
    assert np.isfinite(float(gp)) and np.isfinite(float(gphi))
    assert np.all(np.isfinite(np.asarray(gm)))
    # the y = 0 lane is -lambda: d/dmu = -mu^(1-p)/phi exactly
    assert float(gm[0]) == pytest.approx(-(2.0 ** (1 - 1.3)) / 0.9, rel=1e-14)
    gy = np.asarray(jax.vmap(jax.grad(lambda v: chebax.tweedie_logpdf(1.3, 2.0, 0.9, v)))(y[1:]))
    assert np.all(np.isfinite(gy))


def test_x64_off():
    # float32 end to end, in a child process (conftest forces x64 on here)
    pts_w = [(r, z) for r in (1 / 19, 0.25, 1.0, 4.0, 19.0)
             for z in (1e-3, 0.7, 30.0, 1e4, 1e9, 1e30)]
    pts_f = [(p, mu, phi, y) for p in (1.05, 1.2, 1.5, 1.8, 1.95)
             for mu, phi in ((1.0, 1.0), (3.0, 0.3), (0.2, 3.0)) for y in (0.0, 0.05, 1.0, 6.0)]
    ref_w = [float(chebax.log_wright_bessel0(r, z)) for r, z in pts_w]
    ref_f = [float(chebax.tweedie_logpdf(*c)) for c in pts_f]
    code = (
        "import jax, jax.numpy as jnp, chebax\n"
        "assert jnp.empty(()).dtype == jnp.float32\n"
        f"ref_w, ref_f = {ref_w!r}, {ref_f!r}\n"
        f"for (r, z), ref in zip({pts_w!r}, ref_w):\n"
        "    got = float(chebax.log_wright_bessel0(r, jnp.float32(z)))\n"
        "    assert abs(got - ref) / max(1.0, abs(ref)) <= 3e-6, (r, z, got, ref)\n"
        f"for c, ref in zip({pts_f!r}, ref_f):\n"
        "    got = float(chebax.tweedie_logpdf(*c))\n"
        "    assert abs(got - ref) / max(1.0, abs(ref)) <= 3e-5, (c, got, ref)\n"
        "print('ok')\n")
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-1500:]


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    tweedie_gen.main(tmp_path)
    assert (tmp_path / "tweedie_table.py").read_text() == pathlib.Path(tt.__file__).read_text()
