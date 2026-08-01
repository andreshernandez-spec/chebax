"""chebax.matern: the Matern correlation with traced smoothness.

References mpmath at 40 dps; errors pointwise relative (the correlation
is positive). Measured worst: values 1.3e-14 over nu in [0.05, 9.97],
r in [1e-5, 8]; d/dnu vs mp.diff (explicit h = 1e-7 step; the default
adaptive step gives garbage references near the Gamma pole at nu = 0)
8e-13 absolute-over-(1+|ref|). Half-integer closed forms (1/2
exponential, 3/2, 5/2) are independent oracles, and the lengthscale
gradient is checked against the exact identity dk/dl = -(r/l) dk/dr.
Bars ~4x.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

R = np.concatenate([np.logspace(-5, 0, 10), np.linspace(0.2, 8.0, 12)])


def _ref(nu, r, ell=1.0):
    z = mp.sqrt(2 * mp.mpf(nu)) * mp.mpf(r) / mp.mpf(ell)
    return float(2 ** (1 - mp.mpf(nu)) / mp.gamma(nu)
                 * z ** mp.mpf(nu) * mp.besselk(nu, z))


def test_values_and_half_integer_closed_forms():
    for nu in [0.05, 0.5, 1.0, 1.5, 1.7, 2.5, 5.5, 9.97]:
        ref = np.array([_ref(nu, r) for r in R])
        got = np.asarray(chebax.matern(nu, jnp.asarray(R)))
        assert np.max(np.abs(got - ref) / ref) <= 5e-14, nu
    ell = 1.3
    s = R / ell
    closed = {
        0.5: np.exp(-s),
        1.5: (1.0 + np.sqrt(3.0) * s) * np.exp(-np.sqrt(3.0) * s),
        2.5: (1.0 + np.sqrt(5.0) * s + 5.0 * s * s / 3.0)
             * np.exp(-np.sqrt(5.0) * s),
    }
    for nu, ref in closed.items():
        got = np.asarray(chebax.matern(nu, jnp.asarray(R), ell))
        assert np.max(np.abs(got - ref) / ref) <= 5e-14, nu


def test_diagonal_and_nan():
    assert float(chebax.matern(1.7, 0.0)) == 1.0
    assert float(jax.grad(chebax.matern, argnums=1)(1.7, 0.0)) == 0.0
    assert np.isnan(float(chebax.matern(1.7, np.nan)))
    assert np.isnan(float(chebax.matern(np.nan, 1.0)))


def _ref_mp(nu, r):
    # full-precision variant for differentiation: a float64-truncating
    # reference under mp.diff produces eps/h quantization noise
    z = mp.sqrt(2 * nu) * mp.mpf(r)
    return 2 ** (1 - nu) / mp.gamma(nu) * z ** nu * mp.besselk(nu, z)


def test_dnu_against_mp_diff():
    rg = np.array([0.05, 0.5, 2.0])
    for nu in [0.05, 0.5, 1.7, 5.5, 9.97]:
        ref = np.array([float(mp.diff(lambda v: _ref_mp(v, r), mp.mpf(nu),
                                      h=1e-7, method="step")) for r in rg])
        g = np.asarray(jax.vmap(jax.grad(chebax.matern, argnums=0),
                                in_axes=(None, 0))(jnp.asarray(nu),
                                                   jnp.asarray(rg)))
        assert np.max(np.abs(g - ref) / (1.0 + np.abs(ref))) <= 4e-12, nu


def test_lengthscale_gradient_identity():
    # dk/dl = -(r/l) dk/dr exactly (k depends on r and l only via r/l)
    for nu, r, ell in [(0.5, 0.8, 1.3), (1.7, 2.0, 0.7), (9.0, 0.3, 2.0)]:
        gl = float(jax.grad(chebax.matern, argnums=2)(nu, r, ell))
        gr = float(jax.grad(chebax.matern, argnums=1)(nu, r, ell))
        assert abs(gl + (r / ell) * gr) <= 1e-14 * max(1.0, abs(gl))


def test_pergroup_and_jit():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 3, 20)
    nus = rng.uniform(0.5, 9.5, 3)
    r = rng.uniform(0.01, 4.0, 20)
    f = chebax.pergroup(chebax.matern, idx)
    got = np.asarray(f(jnp.asarray(nus), jnp.asarray(r)))
    ref = np.array([float(chebax.matern(nus[g], r[i]))
                    for i, g in enumerate(idx)])
    np.testing.assert_allclose(got, ref, rtol=5e-15)
    jitted = jax.jit(chebax.matern)
    np.testing.assert_allclose(float(jitted(1.7, 0.8)),
                               float(chebax.matern(1.7, 0.8)), rtol=1e-14)
