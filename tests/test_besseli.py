"""M6 (first increment) acceptance: besseli on x >= 0, nu in [0, 10].

References mpmath at 40 dps; errors pointwise relative (I > 0, no zeros).
Measured worst over the orders below: values 4.2e-15 (incl. the traced-nu
path), dI/dx 1.3e-15, dI/dnu 6.5e-13, scaled-vs-unscaled 1.2e-15.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import besseli_gen
from chebax._src.recipes import besseli_table as it

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

NUS = [0.03, 0.5, 1.0, 1.7, 2.0, 3.0, 5.5, 7.77, 9.97]
XI = np.sort(np.concatenate([
    np.logspace(-3, np.log10(8.0), 16), [8.0, 8.0001],
    np.logspace(np.log10(8.2), np.log10(600.0), 10),
]))


def _iref(v):
    return np.array([float(mp.besseli(mp.mpf(v), mp.mpf(x))) for x in XI])


def test_values_and_traced():
    for v in NUS:
        I = _iref(v)
        assert np.max(np.abs(np.asarray(chebax.besseli(v)(XI)) - I) / I) <= 2e-14, v
        assert np.max(np.abs(np.asarray(chebax.besseli_fn(v, XI)) - I) / I) <= 2e-14, v


def test_dx():
    for v in [0.03, 1.0, 1.7, 5.5, 9.97]:
        Ip = np.array([float(mp.diff(lambda t: mp.besseli(mp.mpf(v), t), mp.mpf(x)))
                       for x in XI])
        g = np.asarray(jax.vmap(jax.grad(chebax.besseli(v)))(jnp.asarray(XI)))
        assert np.max(np.abs(g - Ip) / np.abs(Ip)) <= 2e-14, v


def test_dnu():
    for v in [0.5, 1.0, 1.7, 7.77]:
        In = np.array([float(mp.diff(lambda t: mp.besseli(t, mp.mpf(x)), mp.mpf(v)))
                       for x in XI])
        got = np.asarray(chebax.besseli_dnu(v)(XI))
        assert np.max(np.abs(got - In) / np.abs(In)) <= 5e-12, v


def test_scaled():
    for v in [0.5, 2.5]:
        I = _iref(v)
        sc = np.asarray(chebax.besseli(v, scaled=True)(XI))
        ref = I * np.exp(-XI)
        assert np.max(np.abs(sc - ref) / ref) <= 1e-14, v
    # past the overflow point the scaled form keeps working
    ref720 = float(mp.besseli(mp.mpf(2.5), mp.mpf(720)) * mp.exp(-720))
    assert float(chebax.besseli(2.5)(720.0)) == np.inf
    got720 = float(chebax.besseli(2.5, scaled=True)(720.0))
    assert abs(got720 - ref720) / ref720 <= 1e-13


def test_x_zero():
    assert abs(float(chebax.besseli(0.0)(0.0)) - 1.0) <= 5e-15
    assert float(chebax.besseli(2.5)(0.0)) == 0.0


def test_jit_and_pytree():
    iv = chebax.besseli(2.5)
    xs = jnp.asarray(XI[:10])
    y = jax.jit(lambda f, x: f(x))(iv, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(iv(xs)), rtol=1e-15)


def test_order_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.besseli(-0.5)


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    besseli_gen.main(tmp_path)
    assert (tmp_path / "besseli_table.py").read_text() == pathlib.Path(it.__file__).read_text()


# ---- review 2026-07-30 regressions ------------------------------------------

def test_dnu_zero_order_is_neg_k0():
    # dI_v/dv|_{v=0} = -K_0(x): exponentially small, while the generic
    # I * dlog form amplified table noise by e^2x (returned +2.2e242 at
    # x=600 for a true value of -1.4e-262)
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    d = chebax.besseli_dnu(0.0)
    for x in [0.5, 5.0, 30.0, 100.0, 600.0]:
        ref = float(-mp.besselk(0, x))
        assert abs(float(d(x)) - ref) / abs(ref) <= 1e-13, x
    g = float(jax.grad(chebax.besseli_fn, argnums=0)(0.0, 30.0))
    assert abs(g - float(-mp.besselk(0, 30.0))) / 2.13e-14 <= 1e-12
    gs = float(jax.grad(lambda n: chebax.besseli_fn(n, 30.0, scaled=True))(0.0))
    ref_s = float(-mp.besselk(0, 30.0) * mp.exp(-30.0))
    assert abs(gs - ref_s) / abs(ref_s) <= 1e-12


def test_overflow_window_and_inf():
    # e^x alone overflows at 709.78 but I_v is representable to ~713
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    for v, x in [(0.0, 713.0), (10.0, 714.0)]:
        ref = float(mp.besseli(v, x))
        got = float(chebax.besseli(v)(x))
        assert np.isfinite(got) and abs(got - ref) / ref <= 1e-12, (v, x)
    assert float(chebax.besseli(0.0)(jnp.inf)) == np.inf
    assert float(chebax.besseli(0.0, scaled=True)(jnp.inf)) == 0.0
    assert np.isnan(float(chebax.besseli(2.5)(jnp.nan)))


def test_origin_gradients():
    # the generic power rule is 0 * (x/2)^-1 at v = 0: NaN for a constant
    assert float(jax.grad(chebax.besseli(0.0))(0.0)) == 0.0
    assert abs(float(jax.grad(chebax.besseli(0.0, scaled=True))(0.0)) + 1.0) <= 1e-14
    assert abs(float(jax.grad(chebax.besseli(1.0))(0.0)) - 0.5) <= 1e-14
    assert float(jax.grad(chebax.besseli_fn, argnums=1)(0.0, 0.0)) == 0.0
