"""M6 (second increment) acceptance: bessely on [1e-6, 1e4], nu in [0, 10].

References mpmath at 40 dps. Y oscillates and is singular at 0, so errors
are relative to the modulus M = hypot(J, Y) (well-defined at Y's zeros; the
standard metric for oscillatory Bessel). Measured worst over the orders
below: values 2.7e-14, dY/dx 2.0e-14, dY/dnu 2.7e-14. The order set
includes exact integers (ordinary points here: the tables never touch the
connection formula) and the half-integers that exposed the two design traps
recorded in the generator docstring (floor-vs-round decomposition, and the
recurrence seam at x = 5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import bessely_gen
from chebax._src.recipes import bessely_table as yt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

NUS = [0.03, 0.5, 1.0, 1.5, 2.0, 3.0, 5.5, 7.77, 9.5, 9.97]
XY = np.sort(np.concatenate([
    np.logspace(-6, np.log10(5.0), 16), [5.0, 5.0001, 30.0, 30.0001],
    np.linspace(5.2, 30.0, 12), np.logspace(np.log10(30.5), 4, 10),
]))


def _refs(v):
    Y = np.array([float(mp.bessely(mp.mpf(v), mp.mpf(x))) for x in XY])
    J = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x))) for x in XY])
    return Y, np.hypot(J, Y)


def test_values():
    for v in NUS:
        Y, M = _refs(v)
        e = np.abs(np.asarray(chebax.bessely(v)(XY)) - Y) / M
        assert e.max() <= 1e-13, v


def test_dx():
    for v in [0.03, 1.0, 1.5, 5.5, 9.97]:
        Yp = np.array([float(mp.diff(lambda t: mp.bessely(mp.mpf(v), t), mp.mpf(x)))
                       for x in XY])
        Jp = np.array([float(mp.diff(lambda t: mp.besselj(mp.mpf(v), t), mp.mpf(x)))
                       for x in XY])
        g = np.asarray(jax.vmap(jax.grad(chebax.bessely(v)))(jnp.asarray(XY)))
        assert np.max(np.abs(g - Yp) / np.hypot(Jp, Yp)) <= 1e-13, v


def test_dnu():
    for v in [0.5, 1.0, 1.7, 9.5]:
        Yn = np.array([float(mp.diff(lambda t: mp.bessely(t, mp.mpf(x)), mp.mpf(v)))
                       for x in XY])
        _, M = _refs(v)
        got = np.asarray(chebax.bessely_dnu(v)(XY))
        assert np.max(np.abs(got - Yn) / (np.abs(Yn) + M)) <= 1e-13, v


def test_clamp_below_xmin():
    yv = chebax.bessely(1.5)
    assert float(yv(1e-8)) == float(yv(yt.XMIN))


def test_jit_and_pytree():
    yv = chebax.bessely(2.5)
    xs = jnp.asarray(XY[:10])
    y = jax.jit(lambda f, x: f(x))(yv, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(yv(xs)), rtol=1e-14)


def test_order_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.bessely(10.5)


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    bessely_gen.main(tmp_path)
    assert (tmp_path / "bessely_table.py").read_text() == pathlib.Path(yt.__file__).read_text()
