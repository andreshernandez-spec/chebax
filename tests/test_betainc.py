"""M6 (fourth increment) acceptance: betainc on (a, b) in [0.1, 10]^2.

References mpmath at 40 dps; errors absolute (I is a CDF in [0, 1], and
absolute error is the standard contract for CDFs; near the endpoints the
value itself decays, and users needing the small tail to relative accuracy
evaluate the complement parameter-swapped, as documented). dI/dx uses the
exact Beta density as its oracle; dI/da and dI/db are checked against
mp.diff, including through jax.grad on the traced betainc_fn -- the
gradients jax itself lacks (jax#38610).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import betainc_gen
from chebax._src.recipes import betainc_table as bt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

PAIRS = [(0.15, 0.15), (0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (0.15, 9.9),
         (9.9, 0.15), (5.5, 5.5), (9.9, 9.9), (1.0, 0.3)]
XB = np.sort(np.concatenate([
    [1e-4, 1e-2, 0.5, 0.5 - 1e-9, 0.5 + 1e-9, 1 - 1e-2, 1 - 1e-4],
    np.linspace(0.05, 0.95, 15),
]))


def _iref(a, b):
    return np.array([float(mp.betainc(mp.mpf(a), mp.mpf(b), 0, mp.mpf(x), regularized=True))
                     for x in XB])


def test_values_and_traced():
    for a, b in PAIRS:
        ref = _iref(a, b)
        got = np.asarray(chebax.betainc(a, b)(XB))
        assert np.max(np.abs(got - ref)) <= 2e-14, (a, b)
        gott = np.asarray(chebax.betainc_fn(a, b, XB))
        assert np.max(np.abs(gott - ref)) <= 2e-14, (a, b)


def test_endpoints_exact():
    f = chebax.betainc(2.5, 0.7)
    assert float(f(0.0)) == 0.0
    assert float(f(1.0)) == 1.0


def test_dx_against_beta_density():
    xs = np.linspace(0.02, 0.98, 25)
    for a, b in [(0.5, 0.5), (2.0, 3.0), (9.9, 0.15), (1.0, 1.0)]:
        pdf = (xs ** (a - 1) * (1 - xs) ** (b - 1)
               * math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)))
        g = np.asarray(jax.vmap(jax.grad(chebax.betainc(a, b)))(jnp.asarray(xs)))
        assert np.max(np.abs(g - pdf) / np.max(pdf)) <= 1e-13, (a, b)


def test_da_db_traced():
    for a, b, x in [(0.5, 0.5, 0.3), (2.0, 3.0, 0.6), (7.7, 1.3, 0.2), (1.0, 1.0, 0.7)]:
        da_ref = float(mp.diff(lambda t: mp.betainc(t, mp.mpf(b), 0, mp.mpf(x),
                                                    regularized=True), mp.mpf(a)))
        db_ref = float(mp.diff(lambda t: mp.betainc(mp.mpf(a), t, 0, mp.mpf(x),
                                                    regularized=True), mp.mpf(b)))
        da, db = jax.grad(chebax.betainc_fn, argnums=(0, 1))(
            jnp.asarray(a), jnp.asarray(b), jnp.asarray(x))
        scale = max(abs(da_ref), abs(db_ref))
        assert abs(float(da) - da_ref) / scale <= 1e-12, (a, b, x)
        assert abs(float(db) - db_ref) / scale <= 1e-12, (a, b, x)


def test_complement_identity():
    for a, b in [(0.3, 4.4), (6.6, 1.5)]:
        s = np.asarray(chebax.betainc(a, b)(XB)) + np.asarray(chebax.betainc(b, a)(1.0 - XB))
        assert np.max(np.abs(s - 1.0)) <= 5e-14


def test_jit_and_pytree():
    f = chebax.betainc(2.0, 3.0)
    xs = jnp.asarray(XB[:8])
    y = jax.jit(lambda q, x: q(x))(f, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(f(xs)), rtol=0, atol=1e-15)


def test_param_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.betainc(0.05, 1.0)
    with pytest.raises(ValueError, match="table covers"):
        chebax.betainc(1.0, 11.0)


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    betainc_gen.main(tmp_path)
    assert (tmp_path / "betainc_table.py").read_text() == pathlib.Path(bt.__file__).read_text()
    assert bt.META["dps"] == betainc_gen.DPS
