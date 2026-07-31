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


def test_jit_pytree_and_out_of_range():
    p = chebax.gammainc(2.5)
    x = jnp.asarray([0.5, 3.0, 20.0])
    np.testing.assert_allclose(np.asarray(jax.jit(lambda d, v: d(v))(p, x)),
                               np.asarray(p(x)), rtol=1e-14)
    with pytest.raises(ValueError, match="gammainc"):
        chebax.gammainc(0.05)
    with pytest.raises(ValueError, match="gammaincc"):
        chebax.gammaincc(10.5)


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
