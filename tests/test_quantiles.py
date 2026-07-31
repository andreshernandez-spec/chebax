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


def test_truncated_beta_reparameterization():
    # the numpyro#1365-style use case: inverse-CDF sampling of a truncation,
    # differentiable in the shape parameter. Midpoint-rule u makes the
    # "Monte Carlo" mean a deterministic quadrature, so it can be compared
    # tightly against the analytic truncated mean and its finite difference.
    b, lo, hi = 3.0, 0.2, 0.7
    u = jnp.asarray((np.arange(2000) + 0.5) / 2000)

    def tmean(aa):
        flo = chebax.betainc_fn(aa, b, lo)
        fhi = chebax.betainc_fn(aa, b, hi)
        return jnp.mean(chebax.betaincinv(aa, b, flo + u * (fhi - flo)))

    def mp_tmean(aa):
        aa = mp.mpf(aa)
        pdf = lambda t: t ** (aa - 1) * (1 - t) ** (mp.mpf(b) - 1) / mp.beta(aa, b)
        z = mp.quad(pdf, [lo, hi])
        return mp.quad(lambda t: t * pdf(t), [lo, hi]) / z

    m = float(tmean(2.0))
    assert abs(m - float(mp_tmean(2.0))) <= 5e-7  # midpoint-rule bias O(1/N^2)
    g = float(jax.grad(tmean)(2.0))
    h = 1e-6
    g_ref = (float(mp_tmean(2.0 + h)) - float(mp_tmean(2.0 - h))) / (2 * h)
    assert abs(g - g_ref) / abs(g_ref) <= 1e-4
    x = np.asarray(chebax.betaincinv(2.0, b, np.asarray(
        chebax.betainc_fn(2.0, b, lo) + u * (chebax.betainc_fn(2.0, b, hi)
                                             - chebax.betainc_fn(2.0, b, lo)))))
    assert (x > lo).all() and (x < hi).all()


def test_truncated_gamma_reparameterization():
    # same construction as the beta test, for the truncated-Gamma use case
    # (numpyro's stalled truncated-Gamma support is blocked on exactly this
    # quantile): CDF from jax's gammainc, quantile from chebax, midpoint u
    # makes the mean a quadrature against mp.quad references.
    lo, hi = 1.0, 4.0
    u = jnp.asarray((np.arange(2000) + 0.5) / 2000)

    def tmean(aa):
        flo = jax.scipy.special.gammainc(aa, lo)
        fhi = jax.scipy.special.gammainc(aa, hi)
        return jnp.mean(chebax.gammaincinv(aa, flo + u * (fhi - flo)))

    def mp_tmean(aa):
        aa = mp.mpf(aa)
        pdf = lambda t: t ** (aa - 1) * mp.e ** (-t) / mp.gamma(aa)
        z = mp.quad(pdf, [lo, hi])
        return mp.quad(lambda t: t * pdf(t), [lo, hi]) / z

    m = float(tmean(3.5))
    assert abs(m - float(mp_tmean(3.5))) <= 5e-7  # midpoint-rule bias O(1/N^2)
    g = float(jax.grad(tmean)(3.5))
    h = 1e-6
    g_ref = (float(mp_tmean(3.5 + h)) - float(mp_tmean(3.5 - h))) / (2 * h)
    assert abs(g - g_ref) / abs(g_ref) <= 1e-4
    x = np.asarray(chebax.gammaincinv(3.5, np.asarray(
        jax.scipy.special.gammainc(3.5, lo) + u * (
            jax.scipy.special.gammainc(3.5, hi)
            - jax.scipy.special.gammainc(3.5, lo)))))
    assert (x > lo).all() and (x < hi).all()


def test_jit():
    p = jnp.asarray([0.2, 0.8])
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.betaincinv)(2.0, 3.0, p)),
        np.asarray(chebax.betaincinv(2.0, 3.0, p)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(jax.jit(chebax.gammaincinv)(3.5, p)),
        np.asarray(chebax.gammaincinv(3.5, p)), rtol=0, atol=1e-15)


# ---- deep tails, median behavior, and failure semantics (review 2026-07-30)
# The pre-fix solver saturated silently: betaincinv(1,1,1e-50) returned
# 4.8e-29 (the e^-64 depth 64 additive Newton steps can reach) and
# gammaincinv(20,1e-20) returned 1.53 for a root at 0.865. stdtr used
# x = nu/(nu+t^2), which rounds to 1 near t = 0 and killed all first-order
# behavior at the median.

def test_deep_tail_betaincinv():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    # the contract is CDF accuracy: push the returned quantile back through
    # the exact CDF and require the target probability to eps-ish level
    for a, b, p in [(1.0, 1.0, 1e-50), (2.0, 3.0, 1e-30), (2.0, 3.0, 1e-100),
                    (0.5, 0.5, 1e-100), (0.1, 10.0, 1e-20), (10.0, 0.1, 1e-12)]:
        x = float(chebax.betaincinv(a, b, p))
        assert x > 0.0, (a, b, p)
        roundtrip = float(mp.betainc(a, b, 0, mp.mpf(x), regularized=True))
        assert abs(roundtrip - p) / p <= 1e-12, (a, b, p, x, roundtrip)


def test_deep_tail_gammaincinv():
    sps = pytest.importorskip("scipy.special")
    for a, p in [(20.0, 1e-20), (5.0, 1e-100), (0.5, 1e-30), (2.0, 1e-300)]:
        x = float(chebax.gammaincinv(a, p))
        ref = float(sps.gammaincinv(a, p))
        assert abs(x - ref) / ref <= 1e-12, (a, p, x, ref)
    # a quantile below the smallest positive float64 returns 0, not garbage
    assert float(chebax.gammaincinv(0.5, 1e-300)) == 0.0
    assert float(chebax.betaincinv(0.1, 0.1, 1e-300)) == 0.0


def test_stdtr_median_first_order():
    # x = nu/(nu+t^2) rounds to 1 near t = 0 (the old form returned exactly
    # 0.5); the complementary orientation resolves F - 1/2 = t*pdf(0) down
    # to the representation limit, half an ulp of 0.5
    assert abs(float(chebax.stdtr(4.0, 1e-8)) - (0.5 + 1e-8 * 0.375)) <= 6e-17
    assert abs(float(jax.grad(lambda t: chebax.stdtr(4.0, t))(0.0)) - 0.375) <= 1e-15
    assert float(chebax.stdtrit(4.0, 0.5)) == 0.0
    g = float(jax.grad(lambda p: chebax.stdtrit(4.0, p))(0.5))
    assert abs(g - 8.0 / 3.0) <= 1e-13


def test_quantile_endpoints_and_nan():
    assert float(chebax.stdtrit(4.0, 0.0)) == -np.inf
    assert float(chebax.stdtrit(4.0, 1.0)) == np.inf
    for v in [chebax.betaincinv(2.0, 3.0, jnp.nan),
              chebax.betaincinv(2.0, 3.0, -0.5),
              chebax.betaincinv(2.0, 3.0, 1.5),
              chebax.gammaincinv(2.0, jnp.nan),
              chebax.stdtr(4.0, jnp.nan),
              chebax.stdtrit(4.0, jnp.nan),
              chebax.betainc_fn(2.0, 3.0, jnp.nan)]:
        assert np.isnan(float(v))


def test_float32_inputs_promote():
    # x64-on plus explicit float32 arguments used to crash the solver with
    # a loop-carry dtype mismatch; now they promote to the canonical dtype
    x = chebax.betaincinv(2.0, 3.0, jnp.float32(0.3))
    assert abs(float(x) - float(chebax.betaincinv(2.0, 3.0, 0.3))) <= 1e-6
    t = chebax.stdtrit(4.0, jnp.float32(0.9))
    assert abs(float(t) - float(chebax.stdtrit(4.0, 0.9))) <= 1e-5


def test_gammaincinv_second_derivatives():
    # jax's igamma_grad_a has no JVP; the _dPda wrapper supplies the exact
    # x-derivative, so hessians in p and mixed a-p derivatives work. The
    # pure a-a second derivative needs d2P/da2 and raises with a clear
    # message instead of an internal jax error.
    d2 = float(jax.hessian(lambda p: chebax.gammaincinv(2.0, p))(0.3))
    assert abs(d2 - 0.661368) <= 1e-4
    dm = float(jax.grad(jax.grad(chebax.gammaincinv, argnums=1), argnums=0)(2.0, 0.3))
    assert abs(dm - 1.092278) <= 1e-4
    with pytest.raises(NotImplementedError, match="d2P/da2"):
        jax.hessian(lambda a: chebax.gammaincinv(a, 0.3))(2.0)
