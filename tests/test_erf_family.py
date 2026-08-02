"""M6 (third increment) acceptance: dawsn and erfcx on the real line.

References mpmath at 40 dps; errors pointwise relative (neither function
has zeros away from dawsn's x = 0, which is tested exactly). Derivative
tests use the exact ODE identities as oracles, so no mp.diff is needed:
D' = 1 - 2 x D and erfcx' = 2 x erfcx - 2/sqrt(pi).

The sections after the acceptance tests cover the 2026-08-01 review fixes:
the x64 keying of this module's series cache, and, since series.py's own
file (test_m1.py) was not part of that change, the O(n) coefficient
recurrences that replaced its cached dense matrices and the coefficient
conversion that no longer loses values silently.
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


# ---- review 2026-08-01: the series cache must be keyed per x64 mode --------

_FLIP = """
import jax
import chebax
from chebax._src.recipes import erf_family
before = (float(chebax.dawsn(1.0)), str(erf_family._series()[0].coef.dtype))
jax.config.update("jax_enable_x64", True)
after = (float(chebax.dawsn(1.0)), str(erf_family._series()[0].coef.dtype))
print(repr((before, after)))
"""


def test_series_cache_keyed_per_x64_mode():
    # x64 is process-global and conftest turns it on, so the flip needs a
    # subprocess. A cache filled under x64 off used to pin float32
    # coefficients, and dawsn kept returning the float32 value after.
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run([sys.executable, "-c", _FLIP], capture_output=True, text=True,
                         env={**env, "JAX_PLATFORMS": "cpu"})
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
    np.testing.assert_allclose(np.asarray(jax.grad(f)(c)),
                               npch.chebvander(np.array([-0.4]), 2)[0], atol=1e-15)


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
        for got, ref, bar in ((s.deriv().coef, _mp_chebder(c), 1.5e-15),
                              (s.integ().coef, _mp_chebint(c), 8e-16)):
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
    np.testing.assert_allclose(np.asarray(s.deriv().coef), npch.chebder(c), rtol=0,
                               atol=1e-14 * np.max(np.abs(npch.chebder(c))))
    np.testing.assert_allclose(np.asarray(s.integ().coef), npch.chebint(c), rtol=0,
                               atol=1e-15 * np.max(np.abs(npch.chebint(c))))


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
        np.testing.assert_allclose(d[i], np.asarray(seg.deriv().coef), rtol=0, atol=1e-15)


def test_deriv_integ_on_traced_coefficients():
    rng = np.random.default_rng(3)
    c = jnp.asarray(rng.standard_normal(24))
    for f in (lambda q: q.deriv(), lambda q: q.integ()):
        eager = np.asarray(f(chebax.ChebSeries(c, (0.0, 2.0))).coef)
        traced = jax.jit(lambda c: f(chebax.ChebSeries(c, (0.0, 2.0))).coef)(c)
        np.testing.assert_allclose(np.asarray(traced), eager, rtol=0,
                                   atol=1e-14 * np.max(np.abs(eager)))
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
    assert (tmp_path / "erf_table.py").read_text() == pathlib.Path(et.__file__).read_text()
