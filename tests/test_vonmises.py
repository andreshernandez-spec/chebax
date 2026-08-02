"""von Mises CDF/quantile: values against mp.quad of the defining integral,
dF/dtheta against the closed-form density (the oracle), dF/dkappa against
mp.diff, and the quantile against the CDF (roundtrip) plus the implicit
identities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import vonmises_gen
from chebax._src.recipes import vonmises_table as vt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

KAPPAS = [0.5, 5.0, 20.0, 50.0]
TH = np.linspace(-np.pi, np.pi, 15)


def _cdf_ref(kappa, th):
    i0 = mp.besseli(0, mp.mpf(kappa))
    return float(mp.quad(lambda u: mp.exp(kappa * mp.cos(u)), [-mp.pi, mp.mpf(th)])
                 / (2 * mp.pi * i0))


def test_values():
    for k in KAPPAS:
        ref = np.array([_cdf_ref(k, t) for t in TH])
        got = np.asarray(chebax.vonmises_cdf(k, TH))
        assert np.max(np.abs(got - ref)) <= 2e-13, k


def test_kappa_zero_is_uniform():
    got = np.asarray(chebax.vonmises_cdf(0.0, TH))
    assert np.max(np.abs(got - (0.5 + TH / (2 * np.pi)))) <= 1e-14


def test_dtheta_is_density():
    ths = np.linspace(-3.0, 3.0, 21)
    for k in [0.5, 5.0, 50.0]:
        i0 = float(mp.besseli(0, mp.mpf(k)))
        pdf = np.exp(k * np.cos(ths)) / (2 * np.pi * i0)
        g = np.asarray(jax.vmap(jax.grad(chebax.vonmises_cdf, argnums=1),
                                in_axes=(None, 0))(jnp.asarray(k), jnp.asarray(ths)))
        assert np.max(np.abs(g - pdf) / pdf.max()) <= 1e-12, k


def test_dkappa():
    for k, t in [(0.5, 1.0), (5.0, -0.5), (30.0, 0.2)]:
        g = float(jax.grad(chebax.vonmises_cdf, argnums=0)(k, t))
        r = float(mp.diff(lambda kk: mp.quad(lambda u: mp.exp(kk * mp.cos(u)),
                                             [-mp.pi, mp.mpf(t)])
                          / (2 * mp.pi * mp.besseli(0, kk)), mp.mpf(k)))
        assert abs(g - r) <= 1e-11 * (1 + abs(r)), (k, t)


def test_icdf_roundtrip():
    ps = np.array([1e-4, 0.05, 0.3, 0.5, 0.7, 0.95, 1 - 1e-4])
    for k in KAPPAS:
        th = np.asarray(chebax.vonmises_icdf(k, ps))
        rt = np.asarray(chebax.vonmises_cdf(k, th))
        assert np.max(np.abs(rt - ps)) <= 5e-13, k
        assert (th > -np.pi).all() and (th < np.pi).all()


def test_icdf_gradient_identities():
    for k, p in [(5.0, 0.3), (30.0, 0.9)]:
        th = float(chebax.vonmises_icdf(k, p))
        i0 = float(mp.besseli(0, mp.mpf(k)))
        pdf = np.exp(k * np.cos(th)) / (2 * np.pi * i0)
        dp = float(jax.grad(chebax.vonmises_icdf, argnums=1)(k, p))
        assert abs(dp - 1.0 / pdf) / (1.0 / pdf) <= 1e-11, (k, p)


def test_endpoints_and_jit():
    assert float(chebax.vonmises_cdf(5.0, -np.pi)) == 0.0
    assert float(chebax.vonmises_cdf(5.0, np.pi)) == 1.0
    assert float(chebax.vonmises_icdf(5.0, 0.0)) == -np.pi
    th = jnp.asarray([-1.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.vonmises_cdf)(5.0, th)),
        np.asarray(chebax.vonmises_cdf(5.0, th)), rtol=0, atol=1e-15)


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    vonmises_gen.main(tmp_path)
    assert (tmp_path / "vonmises_table.py").read_text() == pathlib.Path(vt.__file__).read_text()


# ---- review 2026-07-30 regressions ------------------------------------------

def test_kappa_gradient_at_zero():
    # the sqrt(kappa) table axis made d/dkappa infinite at the documented
    # boundary; the exact Fourier limit is sin(theta)/(2 pi)
    import math
    g = float(jax.grad(chebax.vonmises_cdf)(0.0, 0.5))
    assert abs(g - math.sin(0.5) / (2 * math.pi)) <= 1e-12
    assert float(jax.grad(chebax.vonmises_cdf)(0.0, 0.0)) == 0.0
    # continuous across the series/AD threshold
    g1 = float(jax.grad(chebax.vonmises_cdf)(1e-8, 0.5))
    g2 = float(jax.grad(chebax.vonmises_cdf)(2e-8, 0.5))
    assert abs(g1 - g2) <= 1e-7
    assert np.isfinite(float(jax.grad(chebax.vonmises_icdf)(0.0, 0.6)))


def test_icdf_nan_and_oob():
    assert np.isnan(float(chebax.vonmises_icdf(5.0, jnp.nan)))
    assert np.isnan(float(chebax.vonmises_icdf(5.0, -1.0)))
    assert np.isnan(float(chebax.vonmises_icdf(5.0, 2.0)))


# ---- review 2026-08-01: icdf residual enforcement ---------------------------

def _icdf_ref(kappa, p, start=0.0):
    """mpmath Newton on the defining integral, to ~1e-35."""
    k = mp.mpf(kappa)
    i0 = mp.besseli(0, k)
    t = mp.mpf(float(start))
    for _ in range(300):
        f = (mp.quad(lambda u: mp.exp(k * mp.cos(u)), [-mp.pi, t])
             / (2 * mp.pi * i0) - mp.mpf(p))
        t = t - f / (mp.exp(k * mp.cos(t)) / (2 * mp.pi * i0))
        t = max(-mp.pi, min(mp.pi, t))
        if abs(f) < mp.mpf(10) ** -38:
            break
    return t


def test_icdf_rejects_unresolvable_tail():
    # F is 1/2 + theta/(2 pi) + theta H(theta^2), so it is accurate
    # absolutely, not relatively: a converged solve can only promise
    # ~32 eps / min(p, 1-p) RELATIVE. At p = 1e-15 that is 8e-2 and the old
    # fixed-count solve returned -1.11847 at kappa = 50, whose CDF is
    # 3.7e-14, 37x the target. The floor sits at 4096 eps (9.1e-13) and is
    # set by the CDF's representation, so it does not move with kappa.
    for k in [0.5, 2.0, 10.0, 50.0]:
        for p in [8e-13, 1e-13, 1e-15, 1e-16]:
            assert np.isnan(float(chebax.vonmises_icdf(k, p))), (k, p)
            assert np.isnan(float(chebax.vonmises_icdf(k, 1.0 - p))), (k, p)
    # above the floor the answer is real at every kappa, including the deep
    # tail at kappa = 50 that the density-based floor used to reject.
    # Measured worst relative CDF error over this set 1.0e-4, bar 1e-3.
    for k in [2.0, 10.0, 25.0, 50.0]:
        for p in [1e-9, 1e-10, 1e-11, 1e-12]:
            th = float(chebax.vonmises_icdf(k, p))
            assert np.isfinite(th), (k, p)
            rt = float(chebax.vonmises_cdf(k, th))
            assert abs(rt - p) <= 1e-3 * p, (k, p, rt)


def test_icdf_tail_accuracy_against_mpmath():
    # two-sided over the accepted set: worst |theta - mpmath| measured
    # 3.0e-9 (at the kappa = 50 floor, where eps/pdf is the limit), bar
    # 1.2e-8; worst |cdf(icdf(p)) - p| measured 4.4e-16, bar 2e-15.
    ps = np.array([1e-8, 1e-6, 1e-4, 0.02, 0.5, 0.98, 1 - 1e-4, 1 - 1e-6])
    for k in [0.5, 20.0, 50.0]:
        th = np.asarray(chebax.vonmises_icdf(k, ps))
        assert np.isfinite(th).all(), k
        rt = np.asarray(chebax.vonmises_cdf(k, th))
        assert np.max(np.abs(rt - ps)) <= 2e-15, k
        for t, p in zip(th, ps):
            assert abs(t - float(_icdf_ref(k, p, t))) <= 1.2e-8, (k, p)


def test_icdf_kappa_gradient_is_the_ift():
    # dtheta/dkappa = -(dF/dkappa)(theta) / pdf(theta), against mp.diff of
    # the mpmath inversion. Measured worst 2.0e-15 relative, bar 1e-14.
    for k, p in [(5.0, 0.3), (30.0, 0.9)]:
        g = float(jax.grad(chebax.vonmises_icdf, argnums=0)(k, p))
        r = float(mp.diff(lambda kk: _icdf_ref(kk, p), mp.mpf(k)))
        assert abs(g - r) <= 1e-14 * abs(r), (k, p)


def test_icdf_nan_survives_reverse_mode():
    # a rejected solve used to hand back a zero cotangent: the nan sat in a
    # constant where-branch, and a constant transposes to zero. 1e-14 is
    # below the 4096 eps resolvability floor, so the solve is rejected.
    for p in [1e-14, -1.0, jnp.nan]:
        assert np.isnan(float(jax.grad(chebax.vonmises_icdf, argnums=1)(50.0, p))), p
        assert np.isnan(float(jax.grad(chebax.vonmises_icdf, argnums=0)(50.0, p))), p
    assert np.isnan(float(jax.grad(chebax.vonmises_cdf, argnums=1)(5.0, jnp.nan)))
    # the endpoints keep their genuine zero
    assert float(jax.grad(chebax.vonmises_icdf, argnums=1)(5.0, 0.0)) == 0.0


def test_icdf_jit_and_vmap_match_eager():
    # bar 1e-10, measured 2.2e-12 at p = 1e-6: jit reassociates the Clenshaw
    # and the tail root moves by eps/pdf, which is 5e-11 there
    ps = jnp.asarray([1e-6, 0.25, 0.9])
    eager = np.asarray(chebax.vonmises_icdf(20.0, ps))
    assert np.isfinite(eager).all()
    np.testing.assert_allclose(np.asarray(jax.jit(chebax.vonmises_icdf)(20.0, ps)),
                               eager, rtol=0, atol=1e-10)
    vm = jax.vmap(chebax.vonmises_icdf, in_axes=(None, 0))(jnp.asarray(20.0), ps)
    np.testing.assert_allclose(np.asarray(vm), eager, rtol=0, atol=1e-10)


def test_icdf_x64_off():
    # x64 is process-global and conftest turns it on, so float32 needs its
    # own process. The canonical dtype drives the Newton count, the residual
    # floor and the resolvable-probability floor, which lands at
    # 4096 eps32 = 4.9e-4 (the same 32 eps / min(p, 1-p) relative bound as
    # float64, just at eps32). References are mpmath at 45 dps, taken at the
    # float32-rounded p; worst |theta - mpmath| measured 1.8e-5, bar 1.3e-4.
    # Worst relative CDF error over the accepted set measured 4.3e-4 at the
    # floor (kappa = 50), bar 2e-3.
    code = (
        "import numpy as np, jax.numpy as jnp, chebax\n"
        "assert jnp.empty(()).dtype == jnp.float32\n"
        "ref = {(2.0, 0.3): -0.4092592092053136,\n"
        "       (50.0, 1e-3): -0.4417406105070611,\n"
        "       (20.0, 0.99): 0.5298355123304800,\n"
        "       (10.0, 0.9): 0.4137189856796004}\n"
        "for (k, p), r in ref.items():\n"
        "    got = float(chebax.vonmises_icdf(k, p))\n"
        "    assert abs(got - r) <= 1.3e-4, (k, p, got, r)\n"
        "assert chebax.vonmises_icdf(2.0, 0.3).dtype == jnp.float32\n"
        "for k in [0.5, 2.0, 10.0, 50.0]:\n"
        "    for p in [1e-4, 1e-7, 1e-12]:\n"
        "        assert np.isnan(float(chebax.vonmises_icdf(k, p))), (k, p)\n"
        "    for p in [6e-4, 1e-3, 1e-2]:\n"
        "        th = float(chebax.vonmises_icdf(k, p))\n"
        "        assert np.isfinite(th), (k, p)\n"
        "        rt = float(chebax.vonmises_cdf(k, th))\n"
        "        assert abs(rt - p) <= 2e-3 * p, (k, p, rt)\n"
        "assert float(chebax.vonmises_icdf(5.0, 0.0)) == -np.float32(np.pi)\n"
        "assert np.isnan(float(chebax.vonmises_icdf(5.0, 2.0)))\n"
        "print('ok')\n")
    import os
    import subprocess
    import sys
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-1500:]
