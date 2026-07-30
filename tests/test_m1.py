"""M1 acceptance tests (PROJECT.md section 4) plus convention locks.

Errors are sup-normalized: max|err| / max|reference| over the grid.
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as npch

import chebax
from chebax._src import algorithms

XS = np.linspace(-1.0, 1.0, 1001)
RUNGE_BREAKS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def runge_deriv(x):
    return -50.0 * x / (1.0 + 25.0 * x**2) ** 2


def sup_err(approx, exact):
    return np.max(np.abs(np.asarray(approx) - exact)) / np.max(np.abs(exact))


# ---- acceptance: fit to <= 1e-15 sup-normalized ----------------------------
# The evaluation floor is Clenshaw's eps*sum|c|; measured 5-8e-16 here.

def test_fit_exp():
    p = chebax.fit(np.exp)
    assert sup_err(p(XS), np.exp(XS)) <= 1e-15
    assert p.degree <= 20


def test_fit_cos():
    p = chebax.fit(np.cos)
    assert sup_err(p(XS), np.cos(XS)) <= 1e-15


def test_fit_runge_segmented():
    p = chebax.fit(runge, breaks=RUNGE_BREAKS)
    assert sup_err(p(XS), runge(XS)) <= 1e-15


# ---- acceptance: derivatives ----------------------------------------------
# Differentiation amplifies the eps-level sample noise of the float64 build
# path by Markov's ~deg^2 factor (measured: 4e-14 at deg 14, 5e-13 at deg 38).
# The dps build path samples exactly and restores ~eps derivatives; that test
# carries the tight bar.

def test_deriv_exp():
    p = chebax.fit(np.exp)
    assert sup_err(p.deriv()(XS), np.exp(XS)) <= 1e-13


def test_deriv_cos():
    p = chebax.fit(np.cos)
    assert sup_err(p.deriv()(XS), -np.sin(XS)) <= 1e-13


def test_deriv_runge_segmented():
    p = chebax.fit(runge, breaks=RUNGE_BREAKS)
    assert sup_err(p.deriv()(XS), runge_deriv(XS)) <= 1e-12


def test_deriv_unequal_segments():
    # per-segment width scaling: sin on deliberately uneven breaks
    p = chebax.fit(np.sin, breaks=(-1.0, 0.2, 1.0))
    assert sup_err(p.deriv()(XS), np.cos(XS)) <= 1e-13


def test_deriv_mpmath_build():
    mp = pytest.importorskip("mpmath")
    p = chebax.fit(mp.exp, dps=30)
    assert sup_err(p.deriv()(XS), np.exp(XS)) <= 1e-14


# ---- acceptance: jit(vmap(grad(p))) equals the derivative series -----------

def test_grad_equals_deriv_series_single():
    p = chebax.fit(np.exp)
    d = p.deriv()(XS)
    g = jax.jit(jax.vmap(jax.grad(p)))(jnp.asarray(XS))
    np.testing.assert_allclose(np.asarray(g), np.asarray(d),
                               rtol=0, atol=1e-14 * np.max(np.abs(d)))


def test_grad_equals_deriv_series_piecewise():
    p = chebax.fit(runge, breaks=RUNGE_BREAKS)
    d = p.deriv()(XS)
    g = jax.jit(jax.vmap(jax.grad(p)))(jnp.asarray(XS))
    np.testing.assert_allclose(np.asarray(g), np.asarray(d),
                               rtol=0, atol=1e-14 * np.max(np.abs(d)))


# ---- gradients with respect to coefficients --------------------------------

def test_grad_wrt_coefficients():
    p = chebax.fit(np.exp)
    x0 = 0.3
    g = jax.grad(lambda q: q(x0))(p)
    expect = npch.chebvander(np.array([x0]), p.degree)[0]  # dp/dc_k = T_k(x0)
    np.testing.assert_allclose(np.asarray(g.coef), expect, rtol=0, atol=1e-14)


def test_grad_wrt_coefficients_piecewise():
    p = chebax.fit(runge, breaks=RUNGE_BREAKS)
    x0 = 0.7  # inside segment 3, local t = 2*(x-0.5)/0.5 - 1
    g = jax.grad(lambda q: q(x0))(p)
    gc = np.asarray(g.coef)
    t0 = 2.0 * (x0 - 0.5) / 0.5 - 1.0
    expect = npch.chebvander(np.array([t0]), p.degree)[0]
    np.testing.assert_allclose(gc[3], expect, rtol=0, atol=1e-14)
    assert np.all(gc[[0, 1, 2]] == 0.0)


# ---- convention locks against numpy.polynomial -----------------------------

def test_chebval_matches_numpy():
    # numpy arranges the recurrence differently, so equality is to rounding
    rng = np.random.default_rng(0)
    c = rng.standard_normal(8)
    p = chebax.ChebSeries(c)
    np.testing.assert_allclose(np.asarray(p(XS)), npch.chebval(XS, c),
                               rtol=0, atol=5e-15)


def test_chebval_domain_mapping():
    c = np.array([0.3, -1.2, 0.5, 0.7])
    p = chebax.ChebSeries(c, domain=(2.0, 5.0))
    xs = np.linspace(2.0, 5.0, 301)
    t = (2 * xs - 7.0) / 3.0
    np.testing.assert_allclose(np.asarray(p(xs)), npch.chebval(t, c),
                               rtol=0, atol=5e-15)


def test_chebder_chebint_match_numpy():
    rng = np.random.default_rng(1)
    c = rng.standard_normal(9)
    np.testing.assert_allclose(algorithms.chebder(c), npch.chebder(c), rtol=1e-15)
    np.testing.assert_allclose(algorithms.chebint(c), npch.chebint(c),
                               rtol=1e-15, atol=1e-16)


def test_integ_roundtrip():
    p = chebax.fit(np.exp, domain=(0.0, 3.0))
    q = p.integ().deriv()
    xs = np.linspace(0.0, 3.0, 301)
    np.testing.assert_allclose(np.asarray(q(xs)), np.asarray(p(xs)),
                               rtol=0, atol=1e-14 * np.e**3)


# ---- f32 pipeline ----------------------------------------------------------

def test_astype_f32():
    p = chebax.fit(np.exp).astype(jnp.float32)
    y = p(jnp.asarray(XS, jnp.float32))
    assert y.dtype == jnp.float32
    assert sup_err(y, np.exp(XS)) <= 1e-6
    d = p.deriv()(jnp.asarray(XS, jnp.float32))
    assert d.dtype == jnp.float32
    assert sup_err(d, np.exp(XS)) <= 1e-5


# ---- pytree behaviour ------------------------------------------------------

def test_pytree_through_jit():
    # jit may fuse to fma, so equality is to rounding, not bitwise
    p = chebax.fit(np.exp)
    y = jax.jit(lambda q, x: q(x))(p, jnp.asarray(XS))
    np.testing.assert_allclose(np.asarray(y), np.asarray(p(XS)), rtol=0, atol=1e-15 * np.e)
    pw = chebax.fit(runge, breaks=RUNGE_BREAKS)
    y = jax.jit(lambda q, x: q(x))(pw, jnp.asarray(XS))
    np.testing.assert_allclose(np.asarray(y), np.asarray(pw(XS)), rtol=0, atol=1e-15)


def test_tree_map_preserves_structure():
    p = chebax.fit(np.exp)
    q = jax.tree_util.tree_map(lambda a: a, p)
    assert isinstance(q, chebax.ChebSeries)
    assert q.domain == p.domain


# ---- generator path --------------------------------------------------------

def test_fit_with_mpmath():
    mp = pytest.importorskip("mpmath")
    p = chebax.fit(mp.exp, dps=30)
    assert sup_err(p(XS), np.exp(XS)) <= 5e-16


def test_runtime_does_not_import_mpmath():
    code = "import chebax, sys; assert 'mpmath' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_nonsmooth_raises():
    with pytest.raises(ValueError, match="no validated convergence"):
        chebax.fit(np.abs)


def test_aliasing_rejected():
    # T_34 samples exactly as -T_0 on the 17 initial nodes; the old
    # last-two-coefficients criterion certified the constant -1 (max err 2)
    f = chebax.fit(lambda x: np.cos(34 * np.arccos(np.clip(x, -1.0, 1.0))))
    xs = np.linspace(-1.0, 1.0, 2001)
    assert f.degree == 34
    assert np.max(np.abs(np.asarray(f(xs)) - np.cos(34 * np.arccos(xs)))) <= 1e-12


def test_oscillatory_noise_floor():
    # float64 sampling noise sits above tol here; the chop must settle at
    # the measured plateau instead of keeping hundreds of noise terms
    # (sin(20x) used to return degree 230) or failing outright (sin(50x))
    for k, dmax in [(20, 60), (50, 110)]:
        f = chebax.fit(lambda x, k=k: np.sin(k * x))
        xs = np.linspace(-1.0, 1.0, 2001)
        assert f.degree <= dmax
        assert np.max(np.abs(np.asarray(f(xs)) - np.sin(k * xs))) <= 1e-13


def test_max_deg_is_a_cap():
    # used to return degree 23 from a max_deg=17 call
    with pytest.raises(ValueError, match="no validated convergence"):
        chebax.fit(lambda x: np.exp(5 * x), max_deg=17)
    p = chebax.fit(lambda x: np.exp(5 * x))
    assert p.degree <= 256


def test_fixed_degree():
    p = chebax.fit(np.exp, deg=12)
    assert p.degree == 12
