"""Erf-family acceptance: Dawson panels and scaled complementary error.

References are mpmath at 40 dps for values and 60 dps for the analytic JVPs.
The derivative oracles form D' = 1 - 2 x D and erfcx' = 2 x erfcx - 2/sqrt(pi)
from high-precision values, independently of the runtime. Errors are relative
away from Dawson's derivative zeros, with absolute checks near those zeros.

Experiment 37 sweeps every panel seam, both tails and normal-range limits.
Float64 worst: Dawson value 4.5e-16, first-derivative relative 2.3e-14,
second-derivative absolute in the inner region 4.2e-15; erfcx value 5.6e-16,
first derivative 9e-16 and second derivative 2.3e-14. The derivative bars
below are 2e-14 absolute for inner Dawson, 1e-13 relative in its tail, and
4e-15 / 1e-13 relative for erfcx's first / second derivatives. The earlier
value datasets retain their existing bars. No bars were relaxed.

The negative erfcx reflection splits x^2 exactly: rounding it first cost
5.7e-14 at x = -22.75. Float32 uses five terms for the split exponential's
correction; its own child-process check uses dtype-rounded 60-dps references.
The later sections retain the series-cache and coefficient-calculus
regressions from the 2026-08-01 review.
"""

import ast
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as npch

import chebax
from chebax._src import series
from chebax._src.recipes import erf_gen
from chebax._src.recipes import erf_table as et

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

XPOS = np.sort(
    np.concatenate(
        [
            np.logspace(-3, np.log10(6.0), 14),
            [6.0, 6.0001],
            np.logspace(np.log10(6.2), 4, 10),
        ]
    )
)
XD = np.concatenate([-XPOS[::-1], XPOS])  # dawsn: both signs
XE = np.concatenate([-np.linspace(0.1, 5.0, 8)[::-1], XPOS])  # erfcx: mild negatives
# deep negatives, mostly full mantissas: x^2 is not exact there
XN = -np.concatenate(
    [np.linspace(5.0, 26.5, 44), np.random.default_rng(0).uniform(0.0, 26.6, 400)]
)
XBIG = np.array([1e6, 1e10, 1e100, 1e300])  # past mpmath's erfc


def _erfcx_mp(x):
    x = mp.mpf(x)
    if x > 1e5:
        # asymptotic series, exact far past f64 here
        y = 1 / (2 * x * x)
        return (1 - y + 3 * y * y - 15 * y**3) / (x * mp.sqrt(mp.pi))
    return mp.erfc(x) * mp.exp(x * x)


def test_dawsn_values():
    ref = np.array(
        [
            float(mp.sqrt(mp.pi) / 2 * mp.exp(-mp.mpf(x) ** 2) * mp.erfi(mp.mpf(x)))
            for x in XD
        ]
    )
    got = np.asarray(chebax.dawsn(XD))
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 5e-15
    assert float(chebax.dawsn(0.0)) == 0.0
    assert abs(float(jax.grad(chebax.dawsn)(0.0)) - 1.0) <= 1e-15


def test_dawsn_derivative_identity():
    g = np.asarray(jax.vmap(jax.grad(chebax.dawsn))(jnp.asarray(XD)))
    D = np.asarray(chebax.dawsn(XD))
    assert np.max(np.abs(g - (1.0 - 2.0 * XD * D))) <= 5e-14


# jitted too: XLA folds e^a e^b into e^(a + b), which undid the x^2 split
# under jit only (5.7e-14 jitted against 4e-16 eager, 2026-10-31)
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("xs, bar", [(XE, 1.5e-15), (XN, 1.5e-15), (XBIG, 1e-15)])
def test_erfcx_values(xs, bar, jit):
    ref = np.array([float(_erfcx_mp(x)) for x in xs])
    f = jax.jit(chebax.erfcx) if jit else chebax.erfcx
    got = np.asarray(f(jnp.asarray(xs)))
    assert np.max(np.abs(got - ref) / ref) <= bar


def test_erfcx_derivative_identity():
    g = np.asarray(jax.vmap(jax.grad(chebax.erfcx))(jnp.asarray(XPOS)))
    C = np.asarray(chebax.erfcx(XPOS))
    resid = g - (2.0 * XPOS * C - 2.0 / np.sqrt(np.pi))
    assert np.max(np.abs(resid) / np.maximum(np.abs(g), 1.0)) <= 2e-15
    d0 = float(jax.grad(chebax.erfcx)(0.0))
    assert abs(d0 + 2.0 / np.sqrt(np.pi)) <= 1e-14


def test_erfcx_derivative_deep_negatives():
    g = np.asarray(jax.jit(jax.vmap(jax.grad(chebax.erfcx)))(jnp.asarray(XN)))
    ref = np.array(
        [float(2 * mp.mpf(x) * _erfcx_mp(x) - 2 / mp.sqrt(mp.pi)) for x in XN]
    )
    assert np.max(np.abs(g - ref) / np.abs(ref)) <= 1.5e-15


def test_erfcx_negative_overflow():
    assert float(chebax.erfcx(-27.0)) == np.inf
    assert float(chebax.erfcx(-np.inf)) == np.inf
    # the derivative overflows with the value: -inf, not the 0 * inf nan a
    # derivative through the x^2 split gives where its low part is 0
    for x in (-27.0, -28.0, -1e300, -np.inf):
        assert float(jax.grad(chebax.erfcx)(x)) == -np.inf


def test_jit():
    xs = jnp.asarray(XPOS[:8])
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.dawsn)(xs)), np.asarray(chebax.dawsn(xs)), rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.erfcx)(xs)), np.asarray(chebax.erfcx(xs)), rtol=1e-14
    )


# ---- review 2026-08-01: the series cache must be keyed per x64 mode --------

_FLIP = """
import jax
import chebax
from chebax._src.recipes import erf_family
before = (float(chebax.dawsn(1.0)), str(erf_family._tables().dawsn.dtype))
jax.config.update("jax_enable_x64", True)
after = (float(chebax.dawsn(1.0)), str(erf_family._tables().dawsn.dtype))
print(repr((before, after)))
"""


def test_tables_cache_keyed_per_x64_mode():
    # x64 is process-global and conftest turns it on, so the flip needs a
    # subprocess. A cache filled under x64 off used to pin float32
    # coefficients, and dawsn kept returning the float32 value after.
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run(
        [sys.executable, "-c", _FLIP],
        capture_output=True,
        text=True,
        env={**env, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    (v32, d32), (v64, d64) = ast.literal_eval(out.stdout.strip())
    assert (d32, d64) == ("float32", "float64")
    ref = float(mp.sqrt(mp.pi) / 2 * mp.exp(-1) * mp.erfi(1))
    assert abs(v64 - ref) / ref <= 5e-15
    assert abs(v32 - ref) / ref > 1e-8  # the float32 value really was different


# ---- review 2026-08-01: coefficient conversion loses nothing silently ------


def test_wide_integer_coefficients_rejected():
    # jnp.asarray narrows int64 to int32 first, so 2**53 + 1 used to arrive
    # as 1.0 under x64 off
    with pytest.raises(ValueError, match="not exactly representable"):
        chebax.ChebSeries(np.array([2**53 + 1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="not exactly representable"):
        chebax.PiecewiseCheb(np.array([[2**53 + 1, 1]], dtype=np.int64), (0.0, 1.0))
    # integers that survive the promotion still promote
    s = chebax.ChebSeries(np.array([2**40, 2, 3], dtype=np.int64))
    assert jnp.issubdtype(s.coef.dtype, jnp.floating)
    assert float(s.coef[0]) == float(2**40)


def test_complex_coefficients_rejected():
    for bad in (np.array([1 + 2j, 0.5]), np.array([1 + 2j, 0.5], dtype=np.complex64)):
        with pytest.raises(ValueError, match="complex"):
            chebax.ChebSeries(bad)
    with pytest.raises(ValueError, match="complex"):
        chebax.PiecewiseCheb(np.array([[1 + 2j, 0.5]]), (0.0, 1.0))


def test_astype_needs_a_floating_dtype():
    p = chebax.ChebSeries([1.0, 2.0])
    pw = chebax.fit(np.sin, breaks=(-1.0, 0.0, 1.0))
    for bad in (jnp.int32, np.int64, jnp.complex64, bool):
        with pytest.raises(ValueError, match="floating dtype"):
            p.astype(bad)
        with pytest.raises(ValueError, match="floating dtype"):
            pw.astype(bad)
    assert p.astype(jnp.float32).coef.dtype == jnp.float32


def test_constructor_still_takes_tracers():
    f = lambda c: chebax.ChebSeries(c, (0.0, 1.0))(0.3)
    c = jnp.asarray([1.0, 2.0, 3.0])
    assert float(jax.jit(f)(c)) == pytest.approx(float(f(c)), rel=1e-15)
    np.testing.assert_allclose(
        np.asarray(jax.grad(f)(c)), npch.chebvander(np.array([-0.4]), 2)[0], atol=1e-15
    )


# ---- review 2026-08-01: O(n) deriv/integ, no cached dense matrices ---------


def _mp_chebder(c):
    """Exact-arithmetic derivative coefficients, plain-c0 convention."""
    c = [mp.mpf(v) for v in c]
    n = len(c) - 1
    if n == 0:
        return [mp.mpf(0)]
    d = [mp.mpf(0)] * (n + 2)
    for k in range(n, 0, -1):
        d[k - 1] = d[k + 1] + 2 * k * c[k]
    d[0] /= 2
    return d[:n]


def _mp_chebint(c):
    """Exact-arithmetic antiderivative coefficients, vanishing at t = 0."""
    c = [mp.mpf(v) for v in c]
    n = len(c)
    out = [mp.mpf(0)] * (n + 1)
    for k in range(1, n + 1):
        lo = c[k - 1]
        hi = c[k + 1] if k + 1 < n else mp.mpf(0)
        out[k] = (lo - hi / 2) if k == 1 else (lo - hi) / (2 * k)
    out[0] = -sum(out[k] * mp.chebyt(k, 0) for k in range(1, n + 1))
    return out


def test_deriv_integ_coefficients_against_mpmath():
    """Rounding only: the reference is the same closed form in exact
    arithmetic, so the convention is locked by the two tests below, not
    here. Worst measured 3.7e-16 (deriv) and 2.0e-16 (integ), relative to
    max|coef|, over n in 1..1024; bars 1.5e-15 and 8e-16. The dense-matrix
    path this replaced measured 1.6e-15 (deriv) at n = 1024."""
    rng = np.random.default_rng(7)
    for n in (1, 2, 3, 4, 5, 9, 17, 64, 128, 257, 1024):
        c = rng.standard_normal(n)
        s = chebax.ChebSeries(c)
        for got, ref, bar in (
            (s.deriv().coef, _mp_chebder(c), 1.5e-15),
            (s.integ().coef, _mp_chebint(c), 8e-16),
        ):
            assert np.all(np.isfinite(np.asarray(got))), n
            scale = max(float(max(abs(v) for v in ref)), 1e-300)
            err = max(abs(float(a) - float(b)) for a, b in zip(np.asarray(got), ref))
            assert err / scale <= bar, (n, err / scale)


def test_deriv_integ_of_exp_against_mpmath():
    """Independent oracle, no coefficient recurrence in the reference:
    d/dx e^x and int_1^x e^t dt. Measured 5.7e-14 relative (deriv, the fit's
    own Markov amplification) and 8.9e-16 absolute (integ); bars 2.5e-13
    and 4e-15."""
    p = chebax.fit(np.exp, domain=(0.0, 2.0))
    xs = np.linspace(0.0, 2.0, 401)
    ref = np.array([float(mp.e ** mp.mpf(x)) for x in xs])
    assert np.max(np.abs(np.asarray(p.deriv()(xs)) - ref) / ref) <= 2.5e-13
    # integ() vanishes at the domain midpoint
    ref_i = np.array([float(mp.e ** mp.mpf(x) - mp.e) for x in xs])
    assert np.max(np.abs(np.asarray(p.integ()(xs)) - ref_i)) <= 4e-15


def test_deriv_integ_match_numpy_polynomial():
    # convention lock: same plain-c0, lowest-first arrangement, to rounding
    rng = np.random.default_rng(11)
    c = rng.standard_normal(33)
    s = chebax.ChebSeries(c)
    np.testing.assert_allclose(
        np.asarray(s.deriv().coef),
        npch.chebder(c),
        rtol=0,
        atol=1e-14 * np.max(np.abs(npch.chebder(c))),
    )
    np.testing.assert_allclose(
        np.asarray(s.integ().coef),
        npch.chebint(c),
        rtol=0,
        atol=1e-15 * np.max(np.abs(npch.chebint(c))),
    )


def test_series_module_retains_no_caches():
    # the (n, n) chebder/chebint matrices used to be lru_cache(maxsize=None):
    # 16 MiB per degree at n = 1024, never released
    assert [n for n, o in vars(series).items() if hasattr(o, "cache_info")] == []


def test_deriv_carries_no_dense_matrix():
    n = 256
    jx = jax.make_jaxpr(lambda c: chebax.ChebSeries(c).deriv().coef)(jnp.ones(n))
    assert not any(str(e.primitive) == "dot_general" for e in jx.jaxpr.eqns)
    assert all(np.size(v) <= 4 * n for v in jx.consts)


def test_deriv_degree_one_is_the_zero_map():
    s = chebax.ChebSeries(np.array([3.0]), (0.0, 4.0))
    assert np.asarray(s.deriv().coef).tolist() == [0.0]
    assert float(jax.grad(s)(1.0)) == 0.0
    pw = chebax.PiecewiseCheb(np.array([[3.0], [4.0]]), (0.0, 1.0, 2.0))
    assert np.asarray(pw.deriv().coef).ravel().tolist() == [0.0, 0.0]


def test_piecewise_deriv_matches_per_segment():
    pw = chebax.fit(np.sin, breaks=(-1.0, 0.2, 1.0))
    d = np.asarray(pw.deriv().coef)
    for i, (lo, hi) in enumerate(zip(pw.breaks, pw.breaks[1:])):
        seg = chebax.ChebSeries(pw.coef[i], (lo, hi))
        np.testing.assert_allclose(
            d[i], np.asarray(seg.deriv().coef), rtol=0, atol=1e-15
        )


def test_deriv_integ_on_traced_coefficients():
    rng = np.random.default_rng(3)
    c = jnp.asarray(rng.standard_normal(24))
    for f in (lambda q: q.deriv(), lambda q: q.integ()):
        eager = np.asarray(f(chebax.ChebSeries(c, (0.0, 2.0))).coef)
        traced = jax.jit(lambda c: f(chebax.ChebSeries(c, (0.0, 2.0))).coef)(c)
        np.testing.assert_allclose(
            np.asarray(traced), eager, rtol=0, atol=1e-14 * np.max(np.abs(eager))
        )
    # gradient wrt the coefficients is sum(chebder(e_k)) = T_k'(1) = k^2,
    # exactly (domain width 2 makes the chain factor 1); measured exact,
    # where npch.chebder's divisions are already 1 ulp off at k = 21
    n = 24
    g = jax.grad(lambda c: jnp.sum(chebax.ChebSeries(c, (0.0, 2.0)).deriv().coef))(c)
    np.testing.assert_allclose(np.asarray(g), np.arange(n) ** 2.0, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib

    erf_gen.main(tmp_path)
    assert (tmp_path / "erf_table.py").read_text() == pathlib.Path(
        et.__file__
    ).read_text()


@pytest.mark.parametrize("name", ["dawsn", "erfcx"])
def test_analytic_jvps_against_mpmath(name):
    """Pointwise first derivatives and second derivatives, including every seam.
    Dawson's inner errors are absolute (derivatives have zeros), bar 2e-14;
    tail relative bar 1e-13. erfcx relative bars 4e-15 and 1e-13. References
    use 60 dps; the broad sweep and measured floors are in experiment 37.
    """
    if name == "dawsn":
        seams = np.arange(et.NPD + 1) * et.XS / et.NPD
        xs = np.unique(
            np.r_[
                seams,
                -seams,
                [-1e4, -1e-20, 1e-20, 1e4],
                seams * (1 - 1e-7),
                seams * (1 + 1e-7),
            ]
        )
    else:
        seams = et.K * (et.NP / np.arange(1, et.NP + 1) - 1)
        xs = np.unique(
            np.r_[
                seams,
                seams * (1 - 1e-7),
                seams * (1 + 1e-7),
                [-26.0, -5.0, -1e-20, 1e-20, 1e4],
            ]
        )
    with mp.workdps(60):

        def ref(x):
            x = mp.mpf(float(x))
            if name == "dawsn":
                y = mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)
                d1 = 1 - 2 * x * y
                return d1, -2 * y - 2 * x * d1
            y = mp.exp(x * x) * mp.erfc(x)
            d1 = 2 * x * y - 2 / mp.sqrt(mp.pi)
            return d1, 2 * y + 2 * x * d1

        expected = np.array([[float(v) for v in ref(x)] for x in xs]).T
    f = getattr(chebax, name)
    x = jnp.asarray(xs)
    first = jax.grad(lambda x: f(x).sum())
    rev = jax.jit(first)(x)
    fwd = jax.jit(lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),))[1])(x)
    batched = jax.jit(jax.vmap(jax.grad(f)))(x)
    second = jax.jit(jax.grad(lambda x: first(x).sum()))(x)
    for k, got in ((0, rev), (0, fwd), (0, batched), (1, second)):
        assert np.all(np.isfinite(got))
        if name == "dawsn":
            inner = abs(xs) <= 6
            np.testing.assert_allclose(
                np.asarray(got)[inner], expected[k, inner], rtol=0, atol=2e-14
            )
            np.testing.assert_allclose(
                np.asarray(got)[~inner], expected[k, ~inner], rtol=1e-13, atol=0
            )
        else:
            np.testing.assert_allclose(
                got, expected[k], rtol=4e-15 if k == 0 else 1e-13, atol=0
            )


def test_erf_endpoints_and_signed_zero():
    tiny = jnp.finfo(jnp.float64).tiny
    small = jnp.asarray([-tiny, tiny, 2 * tiny])
    np.testing.assert_array_equal(jax.jit(chebax.dawsn)(small), small)
    first = jax.grad(lambda x: chebax.dawsn(x).sum())
    np.testing.assert_array_equal(jax.jit(first)(small), jnp.ones_like(small))
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: first(x).sum()))(small), -4 * small
    )
    for f in (chebax.dawsn, chebax.erfcx):
        vals = jax.jit(f)(jnp.asarray([-jnp.inf, 0.0, jnp.inf, jnp.nan]))
        grad = jax.jit(jax.vmap(jax.grad(f)))(
            jnp.asarray([-jnp.inf, 0.0, jnp.inf, jnp.nan])
        )
        assert np.isnan(vals[-1]) and np.isnan(grad[-1])
        assert vals[-2] == 0.0 and grad[-2] == 0.0
        if f is chebax.dawsn:
            assert vals[0] == 0.0 and grad[0] == 0.0
            assert np.signbit(float(f(-0.0)))
            assert grad[1] == 1.0
        else:
            assert np.isposinf(vals[0]) and np.isneginf(grad[0])
    # Normalization happens before the JVP sees integer or half-precision inputs.
    for f in (chebax.dawsn, chebax.erfcx):
        np.testing.assert_allclose(
            f(x=jnp.asarray([0, 1, 2])), f(jnp.asarray([0.0, 1.0, 2.0])), rtol=0, atol=0
        )
        assert f(jnp.float16(1.0)).dtype == jnp.float64


def test_erf_gradients_float32():
    """Independent 60-dps references at float32 inputs; tails stay relative.
    Value/erfcx-slope bars 2e-6, Dawson slope bar 5e-5 away from its zeros.
    """
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
import mpmath as mp
import chebax
mp.mp.dps = 60
tiny = jnp.finfo(jnp.float32).tiny
small = jnp.asarray([-tiny, tiny, 2*tiny])
np.testing.assert_array_equal(jax.jit(chebax.dawsn)(small), small)
first = jax.grad(lambda x:chebax.dawsn(x).sum())
np.testing.assert_array_equal(jax.jit(first)(small), jnp.ones_like(small))
np.testing.assert_array_equal(jax.jit(jax.grad(lambda x:first(x).sum()))(small), -4*small)
for name in ('dawsn','erfcx'):
    f = getattr(chebax,name)
    xs = np.asarray([-8., -3., -.5, 0., .2, .7, 1., 2., 5.5, 6., 6.0001, 10., 1e4],np.float32)
    values, slopes = [], []
    for t in xs:
        x = mp.mpf(float(t))
        if name == 'dawsn':
            y = mp.sqrt(mp.pi)/2*mp.exp(-x*x)*mp.erfi(x)
            d = 1-2*x*y
        else:
            y = mp.exp(x*x)*mp.erfc(x)
            d = 2*x*y-2/mp.sqrt(mp.pi)
        values.append(float(y)); slopes.append(float(d))
    rev = jax.jit(jax.grad(lambda x:f(x).sum()))(jnp.asarray(xs))
    fwd = jax.jit(lambda x:jax.jvp(f,(x,),(jnp.ones_like(x),))[1])(jnp.asarray(xs))
    np.testing.assert_allclose(f(xs), values, rtol=2e-6, atol=0)
    for got in (rev,fwd):
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, slopes, rtol=5e-5 if name=='dawsn' else 2e-6, atol=0)
"""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    run = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "JAX_ENABLE_X64": "false", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
