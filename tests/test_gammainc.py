"""M6 acceptance: gammainc P/Q on a in [0.1, 10], x in [0, inf).

References mpmath at 40 dps. Errors are ABSOLUTE for values (the CDF
contract), plus RELATIVE in each function's direct zone (P for x <= 8,
Q for x >= 8), where the log-table representation keeps the decaying tail
resolvable. Measured worst cases: values 4.8e-15 absolute (both, at the
x = 8 seam), P 3.6e-14 relative in its direct zone, Q 7.7e-14 relative in
the tail (down to Q ~ 1e-290). dP/dx uses the exact Gamma density as
oracle with a split metric: relative 3.9e-13 where the density is not
negligible against P (>= 1e-4 P), absolute 1.5e-16 in the saturated
wedge (x -> 8, small a, P ~ 1), where the AD bracket a/x - 1 + L'
cancels to density/P and relative accuracy necessarily degrades as
eps P/density. dP/da and dQ/da vs mp.diff: 3.1e-15 / 4.4e-15 absolute,
including through jax.grad on the traced *_fn path (jax routes this
through a third looped series, igamma_grad_a). Bars at ~4x worst.

The IEEE endpoints (x = 0, x < 0, x = +inf, x = nan) and the density
slope at x = 0 have exact answers, so those tests assert equality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import gammainc_gen
from chebax._src.recipes import gammainc_table as gt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

AS = [0.1, 0.11, 0.5, 1.0, 1.001, 2.5, 5.5, 7.77, 9.9, 10.0]
# the three density-at-zero regimes, a = 1 exactly included
AEDGE = [0.1, 0.5, 0.999, 1.0, 1.001, 2.5, 10.0]
ALLFN = [chebax.gammainc_fn, chebax.gammaincc_fn,
         chebax.log_gammainc_fn, chebax.log_gammaincc_fn]


def _over_a(f):
    """One trace for the whole AEDGE list instead of one per a."""
    return jax.vmap(f, in_axes=(0, None))
# grid stops at 600, the same platform-subnormal caveat as besselk's
XG = np.sort(np.concatenate([
    np.logspace(-8, np.log10(8.0), 20), [8.0, 8.0001],
    np.logspace(np.log10(8.2), np.log10(600.0), 12),
]))


def _pref(a):
    return np.array([float(mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True))
                     for x in XG])


def _qref(a):
    return np.array([float(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True))
                     for x in XG])


def _dens(a):
    return np.array([float(mp.exp((mp.mpf(a) - 1) * mp.log(x) - x - mp.loggamma(a)))
                     for x in XG])


def test_values_absolute_and_traced():
    for a in AS:
        P, Q = _pref(a), _qref(a)
        for got in (np.asarray(chebax.gammainc(a)(XG)),
                    np.asarray(chebax.gammainc_fn(a, XG))):
            assert np.max(np.abs(got - P)) <= 2e-14, a
        for got in (np.asarray(chebax.gammaincc(a)(XG)),
                    np.asarray(chebax.gammaincc_fn(a, XG))):
            assert np.max(np.abs(got - Q)) <= 2e-14, a


def test_relative_in_direct_zones():
    for a in [0.1, 1.0, 2.5, 9.9]:
        P, Q = _pref(a), _qref(a)
        gp = np.asarray(chebax.gammainc(a)(XG))
        lo = (XG <= gt.XS) & (P > 0)
        assert np.max(np.abs(gp[lo] - P[lo]) / P[lo]) <= 1.5e-13, a
        # strictly beyond the seam: at x = 8 exactly the select routes to
        # the inner branch, where Q is 1 - P (absolutely, not relatively,
        # accurate)
        gq = np.asarray(chebax.gammaincc(a)(XG))
        hi = (XG > gt.XS) & (Q > 1e-290)
        assert np.max(np.abs(gq[hi] - Q[hi]) / Q[hi]) <= 3e-13, a


def test_dx_against_density():
    for a in [0.11, 1.0, 2.5, 7.77, 9.9]:
        d = _dens(a)
        P = _pref(a)
        g = np.asarray(jax.vmap(jax.grad(chebax.gammainc(a)))(jnp.asarray(XG)))
        unsat = d >= 1e-4 * np.maximum(P, 1e-300)
        assert np.max(np.abs(g[unsat] - d[unsat]) / d[unsat]) <= 1.5e-12, a
        assert np.max(np.abs(g[~unsat] - d[~unsat]), initial=0.0) <= 1e-15, a


def test_da_and_traced_grad():
    xs = XG[XG > 1e-6]
    for a in [0.11, 1.0, 2.5, 9.9]:
        da = np.array([float(mp.diff(
            lambda t: mp.gammainc(t, 0, mp.mpf(x), regularized=True), mp.mpf(a)))
            for x in xs])
        gp = np.asarray(jax.vmap(jax.grad(chebax.gammainc_fn, argnums=0),
                                 in_axes=(None, 0))(jnp.asarray(a), jnp.asarray(xs)))
        assert np.max(np.abs(gp - da)) <= 2e-14, a
        gq = np.asarray(jax.vmap(jax.grad(chebax.gammaincc_fn, argnums=0),
                                 in_axes=(None, 0))(jnp.asarray(a), jnp.asarray(xs)))
        assert np.max(np.abs(gq + da)) <= 2e-14, a


def test_endpoints_and_complement():
    for a in [0.1, 2.5, 10.0]:
        p, q = chebax.gammainc(a), chebax.gammaincc(a)
        assert float(p(0.0)) == 0.0 and float(q(0.0)) == 1.0
        assert float(p(-3.0)) == 0.0 and float(q(-3.0)) == 1.0
        assert np.isnan(float(p(np.nan))) and np.isnan(float(q(np.nan)))
        s = np.asarray(p(XG)) + np.asarray(q(XG))
        assert np.max(np.abs(s - 1.0)) <= 5e-15
    # a = 1 is exactly exponential
    p1 = np.asarray(chebax.gammainc(1.0)(XG))
    assert np.max(np.abs(p1 - (-np.expm1(-XG)))) <= 2e-14


def test_x_at_infinity():
    """x = +inf: P = 1, Q = 0, ln P = 0, ln Q = -inf for every a, exact,
    so the bar is equality. The tail bracket (a-1) ln x - x is inf - inf
    there, which made all four nan for a >= 1 (and 0 * inf at a = 1).
    Large finite x heads for the same limits: ln Q holds 1.1e-16 relative
    against mpmath out to x = 1e300 (measured, half an ulp), bar 1e-15
    rather than 4x because a backend's exp/log can move the last ulps.
    Both gradients are flat at the endpoint, dP/da included (it was nan
    for every a, even the ones whose value came out right)."""
    aa = jnp.asarray(AEDGE)
    p, q, lp, lq = [np.asarray(_over_a(f)(aa, np.inf)) for f in ALLFN]
    assert np.all(p == 1.0) and np.all(q == 0.0)
    assert np.all(lp == 0.0) and np.all(np.isneginf(lq))
    for f in ALLFN:
        gx = np.asarray(_over_a(jax.grad(f, 1))(aa, np.inf))
        ga = np.asarray(_over_a(jax.grad(f, 0))(aa, np.inf))
        assert np.all(gx == 0.0) and np.all(ga == 0.0), f.__name__
    assert np.all(np.asarray(_over_a(chebax.gammainc_fn)(aa, 1e300)) == 1.0)
    assert np.all(np.asarray(_over_a(chebax.gammaincc_fn)(aa, 1e300)) == 0.0)
    for a in AEDGE:
        assert float(chebax.gammainc(a)(np.inf)) == 1.0, a
        assert float(chebax.gammaincc(a)(np.inf)) == 0.0, a
    for x in [1e3, 1e30, 1e300]:
        r = np.array([float(mp.log(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf,
                                               regularized=True)))
                      for a in AEDGE])
        g = np.asarray(_over_a(chebax.log_gammaincc_fn)(aa, x))
        assert np.max(np.abs(g - r) / np.abs(r)) <= 1e-15, x


def test_x_zero_slope_is_the_density():
    """dP/dx at x = 0 is the Gamma density there, +inf below a = 1,
    exactly 1 at a = 1, exactly 0 above (it was masked to 0 for all
    three), and dQ/dx its negative. ln P ~ a ln x gives d ln P/dx = +inf
    for every a; Q = 1 at x = 0 so d ln Q/dx is just -density. All exact,
    bar is equality. The interior gradient converges to them: on
    x in [1e-14, 1e-6] it matches the mpmath density to 1.2e-14 relative
    (measured), bar 5e-14."""
    # below a = 1 the slope is a real infinity, so a zero cotangent on an
    # x = 0 lane gives nan, the way jnp.sqrt does at 0
    aa = jnp.asarray(AEDGE)
    an = np.asarray(AEDGE)
    d0 = np.where(an < 1.0, np.inf, np.where(an > 1.0, 0.0, 1.0))
    gp, gq, glp, glq = [np.asarray(_over_a(jax.grad(f, 1))(aa, 0.0))
                        for f in ALLFN]
    assert np.array_equal(gp, d0) and np.array_equal(gq, -d0)
    assert np.all(np.isposinf(glp)) and np.array_equal(glq, -d0)
    # the splice must not disturb the values
    assert np.all(np.asarray(_over_a(chebax.gammainc_fn)(aa, 0.0)) == 0.0)
    assert np.all(np.asarray(_over_a(chebax.gammaincc_fn)(aa, 0.0)) == 1.0)
    for a, d in zip(AEDGE, d0):  # cached instances take the same path
        assert float(jax.grad(chebax.gammainc(a))(0.0)) == d, a
        assert float(jax.grad(chebax.gammaincc(a))(0.0)) == -d, a
    xs = np.array([1e-14, 1e-10, 1e-6])
    for a in [0.5, 1.0, 1.001, 2.0]:
        d = np.array([float(mp.exp((mp.mpf(a) - 1) * mp.log(x) - x
                                   - mp.loggamma(a))) for x in xs])
        g = np.asarray(jax.vmap(jax.grad(chebax.gammainc_fn, 1),
                                in_axes=(None, 0))(jnp.asarray(a),
                                                   jnp.asarray(xs)))
        assert np.max(np.abs(g - d) / d) <= 5e-14, a


def test_masked_lanes_stay_finite():
    """x < 0 and x = nan: the CDF limits and nan out, with a flat gradient
    on both (the module feeds its masked branches finite dummies, so a
    lane that is switched off cannot poison the batch). Mixed batches,
    under jit and vmap, hold no nan at all; the only infinity is the true
    density slope at x = 0 for a < 1."""
    xs = jnp.asarray([-3.0, 0.0, 1e-8, 1.0, 8.0, 20.0, 1e300, np.inf, np.nan])
    aa = jnp.asarray(AEDGE)
    def grid(f):
        return jax.jit(jax.vmap(jax.vmap(f, in_axes=(None, 0)),
                                in_axes=(0, None)))

    for f, neg in [(chebax.gammainc_fn, 0.0), (chebax.gammaincc_fn, 1.0),
                   (chebax.log_gammainc_fn, -np.inf),
                   (chebax.log_gammaincc_fn, 0.0)]:
        v = np.asarray(grid(f)(aa, xs))
        assert np.all(v[:, 0] == neg) and np.all(np.isnan(v[:, -1])), f.__name__
        g = np.asarray(grid(jax.grad(f, 1))(aa, xs))
        assert np.all(g[:, 0] == 0.0) and np.all(g[:, -1] == 0.0), f.__name__
        assert not np.isnan(g).any(), (f.__name__, g)
    for a in AEDGE:
        assert float(chebax.gammainc(a)(-3.0)) == 0.0, a
        assert np.isnan(float(chebax.gammaincc(a)(np.nan))), a


def test_jit_pytree_and_out_of_range():
    p = chebax.gammainc(2.5)
    x = jnp.asarray([0.5, 3.0, 20.0])
    np.testing.assert_allclose(np.asarray(jax.jit(lambda d, v: d(v))(p, x)),
                               np.asarray(p(x)), rtol=1e-14)
    # only the lower end bites now: the Temme tables run to a = inf
    with pytest.raises(ValueError, match="gammainc"):
        chebax.gammainc(0.05)
    with pytest.raises(ValueError, match="gammaincc"):
        chebax.gammaincc(0.0)
    assert float(chebax.gammaincc(1000.5)(1000.5)) > 0.0


def test_matches_jax_gammainc():
    # cross-check against the incumbent away from its own hard regions
    for a in [0.5, 2.5, 9.9]:
        ref = np.asarray(jax.scipy.special.gammainc(a, jnp.asarray(XG)))
        got = np.asarray(chebax.gammainc(a)(XG))
        assert np.max(np.abs(got - ref)) <= 5e-13, a


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    import pathlib
    gammainc_gen.main(tmp_path)
    assert ((tmp_path / "gammainc_table.py").read_text()
            == pathlib.Path(gt.__file__).read_text())


def test_log_gammainc_pair():
    # ln P: direct for x <= 8, measured worst 1.1e-14 of max(1, |ln P|)
    # down to x = 1e-100 (ln P ~ -2300). ln Q: direct for x > 8 with no
    # underflow ceiling (checked at x = 1e4, ln Q ~ -1e4); for x <= 8 it
    # is log1p(-exp(ln P)), error is the value path's absolute
    # floor over Q (measured 12 eps / Q worst, the gammaincc_fn wedge);
    # the bar carries the value bar 2e-14 divided by Q. dQ/da vs
    # mp.diff measured 7e-13 of max(1, |.|). Bars ~4x.
    xg = np.concatenate([[1e-100, 1e-10, 1e-3], np.linspace(0.5, 8.0, 8),
                         [8.0001, 20.0, 100.0, 700.0, 1e4]])
    for a in [0.1, 1.0, 2.5, 9.9]:
        refp = np.array([float(mp.log(mp.gammainc(mp.mpf(a), 0, mp.mpf(x),
                                                  regularized=True)))
                         for x in xg])
        gotp = np.asarray(chebax.log_gammainc_fn(a, jnp.asarray(xg)))
        assert np.max(np.abs(gotp - refp)
                      / np.maximum(1.0, np.abs(refp))) <= 5e-14, a
        refq = np.array([float(mp.log(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf,
                                                  regularized=True)))
                         for x in xg])
        gotq = np.asarray(chebax.log_gammaincc_fn(a, jnp.asarray(xg)))
        bar = 5e-14 * np.maximum(1.0, np.abs(refq))
        bar[xg <= gt.XS] += 2e-14 / np.exp(refq[xg <= gt.XS])
        assert np.all(np.abs(gotq - refq) <= bar), a
    assert np.isneginf(float(chebax.log_gammainc_fn(2.0, 0.0)))
    assert float(chebax.log_gammaincc_fn(2.0, 0.0)) == 0.0
    assert np.isnan(float(chebax.log_gammaincc_fn(2.0, np.nan)))
    xs = np.linspace(0.5, 20.0, 9)
    np.testing.assert_allclose(
        np.exp(np.asarray(chebax.log_gammaincc_fn(2.5, jnp.asarray(xs)))),
        np.asarray(chebax.gammaincc(2.5)(xs)), rtol=1e-12)
    xa = np.array([0.5, 3.0, 20.0, 100.0])
    for a in [0.1, 2.5, 9.9]:
        da = np.array([float(mp.diff(lambda t: mp.log(mp.gammainc(
            t, mp.mpf(x), mp.inf, regularized=True)), mp.mpf(a))) for x in xa])
        gda = np.asarray(jax.vmap(jax.grad(chebax.log_gammaincc_fn, argnums=0),
                                  in_axes=(None, 0))(jnp.asarray(a),
                                                     jnp.asarray(xa)))
        assert np.max(np.abs(gda - da) / np.maximum(1.0, np.abs(da))) <= 3e-12, a
