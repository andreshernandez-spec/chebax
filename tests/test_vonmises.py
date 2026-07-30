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
