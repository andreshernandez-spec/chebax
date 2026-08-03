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


# ---- review 2026-07-30: constructor validation and mixed dtype --------------

def test_series_constructor_validation():
    with pytest.raises(ValueError):
        chebax.ChebSeries(np.array([]))
    with pytest.raises(ValueError):
        chebax.ChebSeries(np.ones((2, 3)))
    with pytest.raises(ValueError):
        chebax.ChebSeries(np.ones(3), (1.0, 1.0))
    with pytest.raises(ValueError):
        chebax.ChebSeries(np.ones(3), (0.0, np.inf))
    with pytest.raises(ValueError):
        chebax.PiecewiseCheb(np.ones((2, 0)), (0.0, 1.0, 2.0))
    with pytest.raises(ValueError):
        chebax.PiecewiseCheb(np.ones((1, 3)), (0.0,))
    # integer coefficients promote (they truncated under integ and broke
    # coefficient AD with float0 tangents)
    s = chebax.ChebSeries(np.array([1, 2, 3]))
    assert jnp.issubdtype(s.coef.dtype, jnp.floating)
    assert np.asarray(s.integ().coef)[1] == -0.5


def test_mixed_dtype_matches_piecewise():
    # a float32 x against float64 coefficients must promote before the
    # domain map, like the segmented path; the residual difference is the
    # float32 input's own representation error
    p = chebax.fit(np.exp, domain=(-1.0, 1.0))
    x32 = jnp.asarray(0.7, jnp.float32)
    assert abs(float(p(x32)) - float(p(jnp.asarray(np.float64(x32))))) <= 1e-15


# ---- review 2026-08-01: aliasing over the Chebyshev modes -------------------
# _validated used to size its grid from the candidate, so it could resample
# the fitting grid itself and certify an alias: fit(T_29) returned -T_5 (sup
# error 2.0) because 6 + 11 is again 17. It now validates on 2n and 4n, both
# fixed by the fitting resolution n. T_k for k = 1..70 is the sweep that
# catches it: aliasing shows up as a returned degree below k.


def cheb_mode(k):
    """T_k = cos(k arccos x), sampled in numpy or (dps builds) in mpmath."""
    def f(x):
        if hasattr(x, "_mpf_"):
            import mpmath as mp
            return mp.cos(k * mp.acos(x))
        return np.cos(k * np.arccos(np.clip(np.asarray(x, dtype=float), -1.0, 1.0)))
    return f


def cheb_mode_refs(ks, m=401):
    """T_k on a fine grid at 50 dps; float64 arccos loses digits near +-1."""
    mp = pytest.importorskip("mpmath")
    xs = np.linspace(-1.0, 1.0, m)
    with mp.workdps(50):
        th = [mp.acos(mp.mpf(float(x))) for x in xs]
        return xs, {k: np.array([float(mp.cos(k * t)) for t in th]) for k in ks}


def mode_sup_err(p, ref, xs):
    return np.max(np.abs(algorithms.chebval(xs, np.asarray(p.coef, dtype=float)) - ref))


def test_no_aliasing_float64_modes():
    # float64 sampling, every mode up to 48 (fit(T_45) used to return degree
    # 21, sup error 2.0). Degrees run above k here because cos(k*arccos x) is
    # ill-conditioned near the endpoints and the chop settles at that noise
    # plateau; worst measured sup error 2.0e-13, bar 8e-13.
    ks = range(1, 49)
    xs, ref = cheb_mode_refs(ks)
    for k in ks:
        p = chebax.fit(cheb_mode(k))
        assert p.degree >= k, (k, p.degree)
        assert mode_sup_err(p, ref[k], xs) <= 8e-13, k


def test_no_aliasing_dps_modes():
    # the modes that fooled the old grid choice, sampled exactly. Worst
    # measured sup error 6.4e-15 (the float64 Clenshaw floor), bar 3e-14.
    ks = [17, 29, 39, 45, 63]
    xs, ref = cheb_mode_refs(ks)
    for k in ks:
        p = chebax.fit(cheb_mode(k), dps=60)
        assert p.degree == k, (k, p.degree)
        assert mode_sup_err(p, ref[k], xs) <= 3e-14, k


def test_no_aliasing_harmonic_family():
    # Found in review 2026-08-02. The validation grids were n, 2n and 4n,
    # all first-kind, and T_{16n} is exactly +1 on every one of them: with
    # the initial n = 17, fit(T272) returned the constant 1 with sup error
    # 2.0. The 15n+1 argument in review-2026-08-01 bounded the SMALLEST
    # common alias and said nothing about the harmonic family above it.
    # Validation now runs on points that are not the nodes of any
    # Chebyshev grid (see _offgrid), where an alias needs 2 q PHI in Z.
    # Worst measured sup error 2.2e-13 at degree 544, bar 1e-12.
    ks = [16 * 17, 8 * 17, 32 * 17, 4 * 17]
    xs, ref = cheb_mode_refs(ks, m=1201)
    for k in ks:
        p = chebax.fit(cheb_mode(k), max_deg=1024, dps=40)
        assert p.degree == k, (k, p.degree)
        assert mode_sup_err(p, ref[k], xs) <= 1e-12, k


def test_aliased_family_is_not_silently_certified():
    # the same modes under the DEFAULT cap, which they exceed: the honest
    # answer is the loud one, never a low-degree fit that validated
    for k in (272, 544):
        with pytest.raises(ValueError, match="no validated convergence"):
            chebax.fit(cheb_mode(k))


def test_exact_fit_at_its_own_max_deg():
    # The noise floor reads the last sixteenth of the coefficients, which
    # holds the TOP MODE when the fit is exact at that resolution: fitting
    # T_20 on 21 points put c[20] = 1 in the window, read a floor of 2, and
    # chopped the exact fit to one coefficient. With max_deg = 20 there is
    # no doubling left, so this raised. A plateau sits decades under the
    # function scale; a window near it is signal. Worst measured sup error
    # 1.4e-14 (the float64 Clenshaw floor), bar 6e-14.
    ks = [1, 5, 12, 20, 29, 33]
    xs, ref = cheb_mode_refs(ks)
    for k in ks:
        p = chebax.fit(cheb_mode(k), max_deg=k, dps=40)
        assert p.degree == k, (k, p.degree)
        assert mode_sup_err(p, ref[k], xs) <= 6e-14, k


@pytest.mark.slow
def test_no_aliasing_dps_modes_sweep():
    # the full sweep, ~15 s. Exact sampling, so the degree must come back as
    # k itself. Worst measured sup error 7.2e-15, bar 3e-14.
    ks = range(1, 71)
    xs, ref = cheb_mode_refs(ks)
    for k in ks:
        p = chebax.fit(cheb_mode(k), dps=60)
        assert p.degree == k, (k, p.degree)
        assert mode_sup_err(p, ref[k], xs) <= 3e-14, k


def test_validation_grids_cannot_repeat_the_fit_aliasing():
    # the construction _validated rests on. On m first-kind points T_K
    # samples as +-T_j with (j, sign) set by K mod 4m; below K = 15n + 1 no
    # mode folds the same way on n, 2n and 4n onto a j the candidate can hold
    # (j < n, the candidate comes from an n-point fit), and the three grids
    # share no point. 15n + 1 itself folds onto j = n - 1, which only an
    # untruncated candidate holds.
    def fold(K, m):
        r = K % (4 * m)
        if r > 2 * m:
            r = 4 * m - r
        return (2 * m - r, -1) if r > m else (r, 1)

    for n in (5, 17, 18, 33, 65, 129, 257):
        pts = [set(algorithms.chebpts(m)) for m in (n, 2 * n, 4 * n)]
        assert not (pts[0] & pts[1] or pts[0] & pts[2] or pts[1] & pts[2])
        for K in range(1, 15 * n + 1):
            j, s = fold(K, n)
            if j == K or j >= n:
                continue
            assert fold(K, 2 * n) != (j, s) or fold(K, 4 * n) != (j, s), (n, K)


def test_zero_samples_are_validated():
    # the node polynomial of chebpts(17) is exactly zero on all 17 initial
    # nodes, and 2^-16 * T_17 everywhere else. The zero-scale shortcut used
    # to return the zero series from those samples without ever leaving the
    # grid. Measured sup error 3.5e-19 against a 1.5e-5 scale, bar 1.4e-18.
    nodes = algorithms.chebpts(17)

    def node_poly(x):
        return np.prod(np.asarray(x, dtype=float)[..., None] - nodes, axis=-1)

    p = chebax.fit(node_poly, max_deg=64)
    assert p.degree == 17
    xs = np.linspace(-1.0, 1.0, 501)
    assert np.max(np.abs(np.asarray(p(xs)) - node_poly(xs))) <= 1.4e-18
    # a function that really is zero still fits as the zero series
    q = chebax.fit(lambda x: np.zeros_like(np.asarray(x, dtype=float)))
    assert q.degree == 0 and float(q(0.3)) == 0.0


def test_fixed_degree_is_an_instruction():
    # deg= skips validation on purpose: interpolating T_29 at 6 points is
    # T_5, and at a fixed degree an unresolved f is a legitimate request, so
    # aliasing cannot be told from under-resolution
    p = chebax.fit(cheb_mode(29), deg=5)
    assert p.degree == 5
    np.testing.assert_allclose(np.asarray(p.coef), np.eye(6)[5], rtol=0, atol=1e-14)


def test_fit_argument_validation():
    # all of these used to be a ZeroDivisionError, a silent reinterpretation
    # (deg=2.5 gave degree 3) or a silently ignored argument
    def boom(x):
        raise AssertionError("f must not be sampled before the arguments are checked")

    cases = [
        (dict(deg=-1), "deg must be a nonnegative integer"),
        (dict(deg=2.5), "deg must be a nonnegative integer"),
        (dict(deg=500), "above the max_deg=256"),
        (dict(deg=5, max_deg=4), "above the max_deg=4"),
        (dict(tol=0.0), "tol must be positive and finite"),
        (dict(tol=-1.0), "tol must be positive and finite"),
        (dict(tol=np.inf), "tol must be positive and finite"),
        (dict(max_deg=0), "max_deg must be a positive integer"),
        (dict(max_deg=-3), "max_deg must be a positive integer"),
        (dict(max_deg=8.5), "max_deg must be a positive integer"),
        (dict(dps=0), "dps must be a positive integer"),
        (dict(dps=-5), "dps must be a positive integer"),
        (dict(dps=3.5), "dps must be a positive integer"),
        (dict(breaks=(0.0,)), "at least two knots"),
        (dict(breaks=(0.0, 0.0)), "strictly increasing"),
        (dict(breaks=(1.0, 0.0, 2.0)), "strictly increasing"),
        (dict(breaks=(0.0, np.inf)), "breaks must be finite"),
        (dict(domain=(1.0, 1.0)), "domain must be finite"),
        (dict(domain=(0.0, np.inf)), "domain must be finite"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            chebax.fit(boom, **kw)
    # the edges of what is legal still fit
    assert chebax.fit(np.exp, deg=0).degree == 0
    assert chebax.fit(np.exp, deg=1, max_deg=1).degree == 1
