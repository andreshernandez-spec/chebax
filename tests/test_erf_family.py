"""M6 (third increment) acceptance: dawsn and erfcx on the real line.

References mpmath at 40 dps; errors pointwise relative (neither function
has zeros away from dawsn's x = 0, which is tested exactly). Derivative
tests use the exact ODE identities as oracles, so no mp.diff is needed:
D' = 1 - 2 x D and erfcx' = 2 x erfcx - 2/sqrt(pi).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import erf_gen
from chebax._src.recipes import erf_table as et

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

XPOS = np.sort(np.concatenate([
    np.logspace(-3, np.log10(6.0), 14), [6.0, 6.0001],
    np.logspace(np.log10(6.2), 4, 10),
]))
XD = np.concatenate([-XPOS[::-1], XPOS])          # dawsn: both signs
XE = np.concatenate([-np.linspace(0.1, 5.0, 8)[::-1], XPOS])  # erfcx: mild negatives


def test_dawsn_values():
    ref = np.array([float(mp.sqrt(mp.pi) / 2 * mp.exp(-mp.mpf(x) ** 2) * mp.erfi(mp.mpf(x)))
                    for x in XD])
    got = np.asarray(chebax.dawsn(XD))
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 5e-15
    assert float(chebax.dawsn(0.0)) == 0.0
    assert abs(float(jax.grad(chebax.dawsn)(0.0)) - 1.0) <= 1e-15


def test_dawsn_derivative_identity():
    g = np.asarray(jax.vmap(jax.grad(chebax.dawsn))(jnp.asarray(XD)))
    D = np.asarray(chebax.dawsn(XD))
    assert np.max(np.abs(g - (1.0 - 2.0 * XD * D))) <= 5e-14


def test_erfcx_values():
    ref = np.array([float(mp.erfc(mp.mpf(x)) * mp.exp(mp.mpf(x) ** 2)) for x in XE])
    got = np.asarray(chebax.erfcx(XE))
    assert np.max(np.abs(got - ref) / ref) <= 5e-15


def test_erfcx_derivative_identity():
    g = np.asarray(jax.vmap(jax.grad(chebax.erfcx))(jnp.asarray(XPOS)))
    C = np.asarray(chebax.erfcx(XPOS))
    resid = g - (2.0 * XPOS * C - 2.0 / np.sqrt(np.pi))
    assert np.max(np.abs(resid) / np.maximum(np.abs(g), 1.0)) <= 5e-14
    d0 = float(jax.grad(chebax.erfcx)(0.0))
    assert abs(d0 + 2.0 / np.sqrt(np.pi)) <= 1e-14


def test_erfcx_negative_overflow():
    assert float(chebax.erfcx(-27.0)) == np.inf
    ref = float(mp.erfc(mp.mpf(-20)) * mp.exp(mp.mpf(400)))
    got = float(chebax.erfcx(-20.0))
    assert abs(got - ref) / ref <= 1e-12  # dominated by exp(x^2) rounding, eps*x^2


def test_jit():
    xs = jnp.asarray(XPOS[:8])
    np.testing.assert_allclose(np.asarray(jax.jit(chebax.dawsn)(xs)),
                               np.asarray(chebax.dawsn(xs)), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(jax.jit(chebax.erfcx)(xs)),
                               np.asarray(chebax.erfcx(xs)), rtol=1e-14)


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    erf_gen.main(tmp_path)
    assert (tmp_path / "erf_table.py").read_text() == pathlib.Path(et.__file__).read_text()
