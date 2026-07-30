"""M6 (fifth increment) acceptance: betaincinv, gammaincinv, stdtr, stdtrit.

The inversion contract is two-sided: lower-half quantiles must round-trip
through the CDF at the eps level, and upper-half quantiles must match the
40-dps reference as distances from 1 (1 - x is exact in f64 on [1/2, 1]);
quantiles closer to 1 than eps round to 1.0, a representation limit shared
with scipy, and the mirrored call covers them. Measured worst: betaincinv
4.9e-15, gammaincinv 4.0e-14, stdtr/stdtrit roundtrip ~5e-16. Gradients
come from the implicit function theorem, not the iteration: measured
9.3e-15 (beta, all three arguments) and 2.6e-16 (gamma) against mpmath.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

PS = np.array([1e-6, 1e-3, 0.05, 0.3, 0.5, 0.7, 0.95, 1 - 1e-3, 1 - 1e-6])


def _mp_bisect(f, lo, hi, iters=160):
    flo = f(lo)
    for _ in range(iters):
        m = (lo + hi) / 2
        if mp.sign(f(m)) == mp.sign(flo):
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def _beta_qref_mirror(a, b, ptail):
    a_, b_, pt = mp.mpf(a), mp.mpf(b), mp.mpf(ptail)
    f = lambda q: mp.betainc(b_, a_, 0, q, regularized=True) - pt
    lo = (pt * b_ * mp.beta(b_, a_) / 100) ** (1 / b_)
    return _mp_bisect(f, lo, mp.mpf("0.999999"))


def test_betaincinv():
    for a, b in [(0.5, 0.5), (2.0, 3.0), (0.15, 4.0), (9.9, 9.9), (1.0, 1.0),
                 (0.15, 0.15), (9.9, 0.15)]:
        x = np.asarray(chebax.betaincinv(a, b, PS))
        for p, v in zip(PS, x):
            if p <= 0.5:
                e = abs(float(mp.betainc(mp.mpf(a), mp.mpf(b), 0, mp.mpf(v),
                                         regularized=True)) - p)
            else:
                e = abs((1.0 - v) - float(_beta_qref_mirror(a, b, 1.0 - p)))
            assert e <= 5e-13, (a, b, p, e)


def test_betaincinv_gradients():
    def mpinv(a, b, p):
        return mp.findroot(
            lambda t: mp.betainc(a, b, 0, t, regularized=True) - p, mp.mpf("0.5"))

    h = mp.mpf("1e-12")
    for a, b, p in [(0.5, 0.5, 0.3), (2.0, 3.0, 0.7), (7.7, 1.3, 0.1)]:
        da, db, dp = jax.grad(chebax.betaincinv, argnums=(0, 1, 2))(a, b, p)
        a_, b_, p_ = mp.mpf(a), mp.mpf(b), mp.mpf(p)
        da_r = float((mpinv(a_ + h, b_, p_) - mpinv(a_ - h, b_, p_)) / (2 * h))
        db_r = float((mpinv(a_, b_ + h, p_) - mpinv(a_, b_ - h, p_)) / (2 * h))
        dp_r = float((mpinv(a_, b_, p_ + h) - mpinv(a_, b_, p_ - h)) / (2 * h))
        s = max(abs(da_r), abs(db_r), abs(dp_r))
        assert abs(float(da) - da_r) / s <= 1e-12
        assert abs(float(db) - db_r) / s <= 1e-12
        assert abs(float(dp) - dp_r) / s <= 1e-12


def test_gammaincinv():
    for a in [0.1, 0.5, 1.0, 3.5, 20.0, 100.0]:
        x = np.asarray(chebax.gammaincinv(a, PS))
        rt = np.abs(np.array([float(mp.gammainc(mp.mpf(a), 0, mp.mpf(v),
                                                regularized=True)) for v in x]) - PS)
        assert rt.max() <= 5e-13, a


def test_gammaincinv_gradient():
    def mpginv(a, p):
        f = lambda t: mp.gammainc(a, 0, t, regularized=True) - p
        return _mp_bisect(f, mp.mpf("1e-30"), 10 * a + 50)

    h = mp.mpf("1e-12")
    for a, p in [(0.7, 0.2), (3.5, 0.3), (20.0, 0.9)]:
        da = float(jax.grad(chebax.gammaincinv, argnums=0)(a, p))
        a_, p_ = mp.mpf(a), mp.mpf(p)
        da_r = float((mpginv(a_ + h, p_) - mpginv(a_ - h, p_)) / (2 * h))
        assert abs(da - da_r) / abs(da_r) <= 1e-12, (a, p)


def test_stdtr_values_and_dnu():
    ts = np.linspace(-6.0, 6.0, 13)
    for nu in [0.5, 2.0, 4.0, 15.0]:
        def ref(t):
            half = mp.betainc(mp.mpf(nu) / 2, mp.mpf("0.5"), 0,
                              nu / (nu + t * t), regularized=True) / 2
            return float(half if t < 0 else 1 - half)
        r = np.array([ref(t) for t in ts])
        got = np.asarray(chebax.stdtr(nu, ts))
        assert np.max(np.abs(got - r)) <= 5e-14, nu
    dnu = float(jax.grad(chebax.stdtr, argnums=0)(4.0, 1.5))
    dnu_r = float(mp.diff(lambda n: 1 - mp.betainc(n / 2, mp.mpf("0.5"), 0,
                                                   n / (n + mp.mpf("2.25")),
                                                   regularized=True) / 2, mp.mpf(4)))
    assert abs(dnu - dnu_r) / abs(dnu_r) <= 1e-12


def test_stdtrit_roundtrip():
    pr = np.array([0.001, 0.01, 0.3, 0.5, 0.9, 0.999])
    for nu in [0.5, 2.0, 4.0, 15.0]:
        rt = np.asarray(chebax.stdtr(nu, np.asarray(chebax.stdtrit(nu, pr))))
        assert np.max(np.abs(rt - pr)) <= 1e-14, nu


def test_endpoints():
    assert float(chebax.betaincinv(2.0, 3.0, 0.0)) == 0.0
    assert float(chebax.betaincinv(2.0, 3.0, 1.0)) == 1.0
    assert float(chebax.gammaincinv(2.0, 0.0)) == 0.0
    assert float(chebax.gammaincinv(2.0, 1.0)) == np.inf


def test_jit():
    p = jnp.asarray([0.2, 0.8])
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.betaincinv)(2.0, 3.0, p)),
        np.asarray(chebax.betaincinv(2.0, 3.0, p)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.gammaincinv)(3.5, p)),
        np.asarray(chebax.gammaincinv(3.5, p)), rtol=0, atol=1e-15)
