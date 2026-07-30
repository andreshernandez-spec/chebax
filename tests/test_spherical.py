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
    assert float(spherical_jn(0)(0.0)) == 1.0
    assert float(spherical_jn(3)(0.0)) == 0.0
    assert float(spherical_yn(2)(0.0)) == -np.inf


def test_derivative_identity():
    # j_0' = -j_1
    g = np.asarray(jax.vmap(jax.grad(spherical_jn(0)))(jnp.asarray(XS)))
    ref = -np.asarray(spherical_jn(1)(XS))
    assert np.max(np.abs(g - ref)) <= 1e-13


def test_n_out_of_range():
    with pytest.raises(ValueError, match="integer n"):
        spherical_jn(10)
