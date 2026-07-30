"""Spherical Bessel wrappers: verified against scipy (an independent
implementation, unlike mpmath-via-the-same-relation). Oscillatory, so errors
are modulus-relative: |err| / hypot(j_n, y_n)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes.spherical import spherical_jn, spherical_yn

sc = pytest.importorskip("scipy.special")

XS = np.concatenate([np.logspace(-3, 0, 6), np.linspace(1.5, 50.0, 20)])


def test_values_vs_scipy():
    for n in [0, 1, 5, 9]:
        jr = sc.spherical_jn(n, XS)
        yr = sc.spherical_yn(n, XS)
        m = np.hypot(jr, yr)
        assert np.max(np.abs(np.asarray(spherical_jn(n)(XS)) - jr) / m) <= 1e-12, n
        assert np.max(np.abs(np.asarray(spherical_yn(n)(XS)) - yr) / m) <= 1e-12, n


def test_x_zero_limits():
    # j_0(0) = 1 to the Clenshaw floor (the inner region evaluates the real
    # series; the old exact literal came from a mask that also zeroed the
    # origin gradients); x^n makes j_n(0) = 0 exact for n >= 1
    assert abs(float(spherical_jn(0)(0.0)) - 1.0) <= 1e-15
    assert float(spherical_jn(3)(0.0)) == 0.0
    assert float(spherical_yn(2)(0.0)) == -np.inf
    assert np.isnan(float(spherical_jn(0)(jnp.nan)))


def test_origin_gradients_exact():
    # j_1'(0) = 1/3 through the integer power, not masked to 0 (review find)
    assert float(jax.grad(spherical_jn(0))(0.0)) == 0.0
    assert abs(float(jax.grad(spherical_jn(1))(0.0)) - 1.0 / 3.0) <= 1e-15
    assert float(jax.grad(spherical_jn(2))(0.0)) == 0.0


def test_yn_prefactor_consistent_with_clamp():
    # the wrapper used raw x in sqrt(pi/2x) while bessely clamps at 1e-6,
    # returning -1e13 where y_1(1e-8) should saturate at the clamp value
    assert float(spherical_yn(1)(1e-8)) == float(spherical_yn(1)(1e-6))


def test_derivative_identity():
    # j_0' = -j_1
    g = np.asarray(jax.vmap(jax.grad(spherical_jn(0)))(jnp.asarray(XS)))
    ref = -np.asarray(spherical_jn(1)(XS))
    assert np.max(np.abs(g - ref)) <= 1e-13


def test_n_out_of_range():
    with pytest.raises(ValueError, match="integer n"):
        spherical_jn(10)
