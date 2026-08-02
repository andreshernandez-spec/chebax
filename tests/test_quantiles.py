"""M6 (fifth increment) acceptance: betaincinv, gammaincinv, stdtr, stdtrit.

The inversion contract is two-sided: lower-half quantiles must round-trip
through the CDF at the eps level, and upper-half quantiles must match the
40-dps reference as distances from 1 (1 - x is exact in f64 on [1/2, 1]);
quantiles closer to 1 than eps round to 1.0, a representation limit shared
with scipy, and the mirrored call covers them. Measured worst: betaincinv
4.9e-15, gammaincinv 4.0e-14, stdtr/stdtrit roundtrip ~5e-16. Gradients
come from the implicit function theorem, not the iteration: measured
9.3e-15 (beta, all three arguments) and 2.6e-16 (gamma) against mpmath.

The last block covers the 2026-08-01 review's blockers, whose measured
numbers sit in their own docstrings: the tails (gammaincinv against Q and
against ln P, stdtr past the t^2 overflow, stdtrit's exact 2 min(p, 1-p)),
the log-CDF log_stdtr, the shape domain, and one child process with x64
OFF, which is jax's default and which conftest hides from every other test
here.
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


def test_chi2inv():
    # a 2x-rescaled gammaincinv (which owns the accuracy contract);
    # scipy here is a wiring oracle at its own ~1e-10 accuracy
    from scipy import stats
    ps = np.array([1e-8, 0.05, 0.5, 0.95, 1 - 1e-8])
    for k in [0.7, 3.0, 11.5]:
        got = np.asarray(chebax.chi2inv(k, jnp.asarray(ps)))
        ref = stats.chi2.ppf(ps, k)
        assert np.max(np.abs(got - ref) / ref) <= 1e-9, k
    gk = float(jax.grad(chebax.chi2inv, argnums=0)(3.0, 0.5))
    h = 1e-6
    fd = (stats.chi2.ppf(0.5, 3.0 + h) - stats.chi2.ppf(0.5, 3.0 - h)) / (2 * h)
    assert abs(gk - fd) <= 1e-7
    assert float(chebax.chi2inv(3.0, 0.0)) == 0.0
    assert np.isposinf(float(chebax.chi2inv(3.0, 1.0)))


def test_endpoint_contract():
    # Exact values at p = 0 and p = 1, nan propagation, and zero masked
    # gradients at the endpoints are the CONTRACT (audited 2026-07-31:
    # already true; locked here so it cannot regress). Downstream code
    # need not clip p away from the endpoints for the inverses
    # themselves; clip only to protect downstream log-densities.
    assert float(chebax.betaincinv(2.0, 3.0, 0.0)) == 0.0
    assert float(chebax.betaincinv(2.0, 3.0, 1.0)) == 1.0
    assert float(chebax.gammaincinv(2.5, 0.0)) == 0.0
    assert np.isposinf(float(chebax.gammaincinv(2.5, 1.0)))
    assert np.isneginf(float(chebax.stdtrit(4.0, 0.0)))
    assert np.isposinf(float(chebax.stdtrit(4.0, 1.0)))
    for f in (lambda p: chebax.betaincinv(2.0, 3.0, p),
              lambda p: chebax.gammaincinv(2.5, p),
              lambda p: chebax.stdtrit(4.0, p)):
        assert np.isnan(float(f(np.nan)))
        assert float(jax.grad(f)(0.0)) == 0.0
    # shape gradients at the endpoints are masked to zero, not nan
    assert float(jax.grad(lambda a: chebax.betaincinv(a, 3.0, 0.0))(2.0)) == 0.0
    assert float(jax.grad(lambda a: chebax.gammaincinv(a, 1.0))(2.5)) == 0.0


def test_stdtr_extended_nu_range():
    # the slice tables (increment 21) extend nu from [0.2, 20] to
    # [0.2, 200]. Measured absolute worst 4.6e-14 (at nu = 199.9, the
    # panel edge grid below), roundtrip 2.8e-15, d/dnu at nu = 150
    # matching mp.diff to 7 digits; bars ~4x. The old range's accuracy
    # is covered by the existing stdtr tests, which now also run on the
    # slice path.
    def tcdf(nu, t):
        x = mp.mpf(nu) / (nu + mp.mpf(t) ** 2)
        p = mp.betainc(mp.mpf(nu) / 2, mp.mpf(1) / 2, 0, x,
                       regularized=True) / 2
        return p if t < 0 else 1 - p

    ts = np.concatenate([-np.logspace(2, -3, 9), [0.0], np.logspace(-3, 2, 9)])
    for nu in [20.1, 60.0, 150.0, 199.9]:
        ref = np.array([float(tcdf(nu, t)) for t in ts])
        got = np.asarray(chebax.stdtr(nu, jnp.asarray(ts)))
        assert np.max(np.abs(got - ref)) <= 2e-13, nu
        ps = np.linspace(0.01, 0.99, 21)
        tq = np.asarray(chebax.stdtrit(nu, jnp.asarray(ps)))
        rp = np.asarray(chebax.stdtr(nu, jnp.asarray(tq)))
        assert np.max(np.abs(rp - ps)) <= 2e-14, nu
    g = float(jax.grad(chebax.stdtr, argnums=0)(150.0, 1.5))
    ref_g = float(mp.diff(lambda n: tcdf(n, 1.5), mp.mpf(150)))
    assert abs(g - ref_g) <= 1e-11


@pytest.mark.slow
def test_stdtr_tables_regenerate_bit_for_bit(tmp_path):
    import pathlib
    from chebax._src.recipes import stdtr_gen
    from chebax._src.recipes import stdtr_table as st
    stdtr_gen.main(tmp_path)
    assert ((tmp_path / "stdtr_table.py").read_text()
            == pathlib.Path(st.__file__).read_text())


# ---- release blockers (review 2026-08-01): the float32 default, which
# conftest's x64 hides; gammaincinv's upper tail and shape domain; and the
# Student-t tails, lost to t*t and to 1 - |2p - 1|.


def _t_cdf_mp(nu, t):
    """F(t; nu) at the current mp precision; nu and t may be mpf, so
    mp.diff can differentiate through it."""
    nu, t = mp.convert(nu), mp.convert(t)
    x = nu / (nu + t * t)
    half = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return half if t < 0 else 1 - half


def test_gammaincinv_upper_tail():
    # P(a, x) - p has no digits left once 1 - p is at the rounding of 1
    # (gammaincinv(1, nextafter(1, 0)) returned 35.75 for an exact
    # 36.7368005696771), so the upper half solves Q(a, x) = 1 - p, which is
    # exact for p >= 1/2. Contract is the round trip through Q: measured
    # worst 7.5e-12 relative, at a = 0.5 where the root sits below the
    # gammainc tables' x = 8 seam and Q comes back as 1 - P; bar 3e-11.
    for a in [0.5, 2.0, 9.9, 20.0, 100.0]:
        for q in [1e-16, 1e-12, 1e-8, 1e-4]:
            p = 1.0 - q
            x = float(chebax.gammaincinv(a, p))
            got = mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True)
            assert abs(float(got) / (1.0 - p) - 1) <= 3e-11, (a, q, x)
    p = float(np.nextafter(1.0, 0.0))
    exact = -float(np.log1p(-p))          # a = 1 is the exponential quantile
    assert abs(float(chebax.gammaincinv(1.0, p)) - exact) <= 4e-15 * exact


def test_gammaincinv_deep_tail_large_shape():
    # Newton on P - p crawls one e-fold per step where P is exponentially
    # small, so a = 100, p = 1e-20 was still 20 e-folds out after 40 steps
    # and came back nan. The iteration is on ln P, which is nearly linear in
    # ln x there. Measured worst 1.7e-13 relative, bar 7e-13.
    for a in [20.0, 100.0, 500.0]:
        for p in [1e-30, 1e-20, 1e-10]:
            x = float(chebax.gammaincinv(a, p))
            got = mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True)
            assert abs(float(got) / p - 1) <= 7e-13, (a, p, x)


def test_gammaincinv_shape_domain():
    # a <= 0 and non-finite a are nan for every p, endpoints included; they
    # used to return 0.0 for interior p and inf at p = 1
    for a in [0.0, -1.0, -0.5, np.inf, np.nan]:
        for p in [0.0, 1e-8, 0.5, 1.0]:
            assert np.isnan(float(chebax.gammaincinv(a, p))), (a, p)
        assert np.isnan(float(chebax.chi2inv(2.0 * a, 0.5))), a
    assert float(chebax.gammaincinv(2.5, 0.0)) == 0.0
    assert np.isposinf(float(chebax.gammaincinv(2.5, 1.0)))
    assert float(chebax.chi2inv(3.0, 0.0)) == 0.0
    assert np.isposinf(float(chebax.chi2inv(3.0, 1.0)))


def test_stdtr_extreme_t():
    # t*t is inf past |t| ~ 1.3e154 and nu/(nu+t^2) has lost everything well
    # before that (stdtr(0.2, -1e154) returned 0.0 for 5.96e-32); the tail
    # runs on log|t|. Measured worst 6.0e-14 relative, bar 2.5e-13.
    for nu, t in [(0.2, -1e154), (0.2, -1e155), (0.2, -1e300), (0.5, -1e200),
                  (1.0, -1e290), (2.0, -1e150), (4.0, -1e70)]:
        got = float(chebax.stdtr(nu, t))
        assert abs(got / float(_t_cdf_mp(nu, t)) - 1) <= 2.5e-13, (nu, t, got)
        assert float(chebax.stdtr(nu, -t)) == 1.0
    assert float(chebax.stdtr(4.0, -np.inf)) == 0.0
    assert float(chebax.stdtr(4.0, np.inf)) == 1.0
    # the density behind the JVP is computed the same way, so the gradient
    # survives where t*t does not
    for nu, t in [(4.0, -1e154), (0.5, -1e200)]:
        assert np.isfinite(float(jax.grad(chebax.stdtr, argnums=0)(nu, t)))
        assert np.isfinite(float(jax.grad(chebax.stdtr, argnums=1)(nu, t)))


def test_stdtrit_deep_tail():
    # 1 - |2p - 1| rounds to 0 for small p, which made stdtrit(4, 1e-20)
    # return -inf for -131607.40128; the tail target is 2 min(p, 1-p), exact
    # in binary. Contract is the round trip through the mpmath CDF: measured
    # worst 1.4e-13 relative, bar 6e-13.
    for nu, p in [(4.0, 1e-20), (20.0, 1e-30), (2.0, 1e-50), (15.0, 1e-15),
                  (4.0, 1e-300)]:
        t = float(chebax.stdtrit(nu, p))
        assert np.isfinite(t), (nu, p)
        assert abs(float(_t_cdf_mp(nu, t)) / p - 1) <= 6e-13, (nu, p, t)
    pu = 1 - 1e-16          # mirrored: 2 (1 - p) is exact for p >= 1/2
    t = float(chebax.stdtrit(4.0, pu))
    assert abs(float(1 - _t_cdf_mp(4.0, t)) / (1.0 - pu) - 1) <= 6e-13
    assert float(chebax.stdtrit(4.0, 0.5)) == 0.0
    assert np.isneginf(float(chebax.stdtrit(4.0, 0.0)))
    assert np.isposinf(float(chebax.stdtrit(4.0, 1.0)))


def test_log_stdtr():
    # the log-CDF a truncated Student-t normalizer needs: accurate where the
    # probability itself underflows (stdtr(4, -1e80) is 0.0 while ln F is
    # -735.73). Measured worst 1.1e-15 relative to max(1, |ln F|), bar
    # 5e-15; gradients 1.9e-12 relative, bar 8e-12.
    from chebax._src.recipes.quantiles import log_stdtr, log_stdtr_sf
    for nu in [0.5, 2.0, 4.0, 20.0]:
        for t in [-1e300, -1e150, -1e40, -1e10, -100.0, -6.0, -1.0, 0.0,
                  1.0, 6.0, 100.0]:
            got = float(log_stdtr(nu, t))
            ref = float(mp.log(_t_cdf_mp(nu, t)))
            assert abs(got - ref) <= 5e-15 * max(1.0, abs(ref)), (nu, t, got)
            # symmetry, so the survival side needs no 1 - F anywhere
            assert float(log_stdtr_sf(nu, -t)) == got, (nu, t)
    assert float(chebax.stdtr(4.0, -1e80)) == 0.0
    assert abs(float(log_stdtr(4.0, -1e80)) + 735.7286174694265) <= 1e-11
    assert float(log_stdtr(4.0, 0.0)) == -float(np.log(2.0))
    assert float(log_stdtr(4.0, -np.inf)) == -np.inf
    assert float(log_stdtr(4.0, np.inf)) == 0.0
    assert np.isnan(float(log_stdtr(4.0, np.nan)))
    for nu, t in [(4.0, -30.0), (2.0, -6.0), (0.5, -1.5), (15.0, -3.0), (4.0, 2.0)]:
        gn = float(jax.grad(log_stdtr, argnums=0)(nu, t))
        gt = float(jax.grad(log_stdtr, argnums=1)(nu, t))
        rn = float(mp.diff(lambda n: mp.log(_t_cdf_mp(n, t)), mp.mpf(nu)))
        rt = float(mp.diff(lambda x: mp.log(_t_cdf_mp(nu, x)), mp.mpf(t)))
        assert abs(gn - rn) <= 8e-12 * abs(rn), (nu, t, gn, rn)
        assert abs(gt - rt) <= 8e-12 * abs(rt), (nu, t, gt, rt)
    # deep tail: d/dt ln F is pdf/F, which tends to nu/|t| instead of 0/0
    for nu, t in [(4.0, -1e10), (2.0, -1e40), (0.5, -1e100)]:
        g = float(jax.grad(log_stdtr, argnums=1)(nu, t))
        assert abs(g / (nu / abs(t)) - 1) <= 1e-12, (nu, t, g)


# jax's default is x64 OFF, which conftest turns on for the whole suite, so
# the default runtime only gets exercised in a child process. Every case
# below returned nan or a wrong value before the solver constants became
# dtype-specific (float32's exp saturates at +-88, not +-745).
X64_OFF_CASES = [
    ("betaincinv", (10.0, 10.0, 0.5)),
    ("betaincinv", (2.0, 3.0, 0.5)),
    ("betaincinv", (0.5, 0.5, 0.3)),
    ("betaincinv", (9.9, 9.9, 0.05)),
    ("betaincinv", (3.0, 2.0, 0.95)),
    ("gammaincinv", (20.0, 0.1)),
    ("gammaincinv", (100.0, 0.5)),
    ("gammaincinv", (2.0, 0.5)),
    ("gammaincinv", (0.5, 0.9)),
    ("gammaincinv", (3.5, 1e-8)),
    ("gammaincinv", (9.9, 0.999)),
    ("stdtrit", (20.0, 0.1)),
    ("stdtrit", (4.0, 0.25)),
    ("stdtrit", (2.0, 1e-6)),
    ("stdtrit", (0.5, 0.8)),
    ("chi2inv", (3.0, 0.5)),
    ("chi2inv", (0.7, 0.05)),
]

X64_OFF_RUNNER = """
import json
import sys

import numpy as np
import jax.numpy as jnp
import chebax

assert jnp.empty(()).dtype == jnp.float32, "x64 must be off in this process"
worst = 0.0
for name, args, ref in json.load(open(sys.argv[1])):
    got = float(getattr(chebax, name)(*args))
    assert np.isfinite(got), (name, args, got)
    worst = max(worst, abs(got - ref) / abs(ref))
assert worst <= 7e-6, worst

# the endpoint and failure semantics are the same contract as under x64,
# read at float32's own floor: 1.76e-41 is below the smallest normal
assert float(chebax.betaincinv(0.15, 4.0, 1e-6)) == 0.0
assert float(chebax.gammaincinv(2.0, 0.0)) == 0.0
assert np.isposinf(float(chebax.gammaincinv(2.0, 1.0)))
assert np.isnan(float(chebax.gammaincinv(0.0, 0.5)))
assert np.isnan(float(chebax.gammaincinv(2.0, 1.5)))
assert float(chebax.stdtrit(4.0, 0.5)) == 0.0
assert np.isneginf(float(chebax.stdtrit(4.0, 0.0)))
assert float(chebax.stdtr(4.0, -np.inf)) == 0.0
print(worst)
"""


def _mp_quantile(name, args, iters=60):
    """The reference for one X64_OFF_CASES entry, bisected in log space so
    one bracket covers every case."""
    e = mp.e
    if name == "betaincinv":
        a, b, p = args
        f = lambda v: mp.betainc(mp.mpf(a), mp.mpf(b), 0, e ** v / (1 + e ** v),
                                 regularized=True) - mp.mpf(p)
        u = _mp_bisect(f, mp.mpf(-200), mp.mpf(200), iters)
        return e ** u / (1 + e ** u)
    if name in ("gammaincinv", "chi2inv"):
        a, p = args
        a = mp.mpf(a) / 2 if name == "chi2inv" else mp.mpf(a)
        f = lambda v: mp.gammainc(a, 0, e ** v, regularized=True) - mp.mpf(p)
        x = e ** _mp_bisect(f, mp.mpf(-200), mp.mpf(300), iters)
        return 2 * x if name == "chi2inv" else x
    nu, p = args
    sgn = -1 if p < 0.5 else 1
    f = lambda v: _t_cdf_mp(nu, sgn * e ** v) - mp.mpf(p)
    return sgn * e ** _mp_bisect(f, mp.mpf(-200), mp.mpf(200), iters)


def test_x64_off_quantiles(tmp_path):
    # measured worst 1.6e-6 relative against 40-dps references (betaincinv
    # at a = b = 1/2, where the log-space kernel sums ~10-magnitude terms in
    # float32), bar 7e-6. Under the old float64 constants betaincinv(10, 10,
    # 0.5), gammaincinv(20, 0.1), gammaincinv(100, 0.5) and most of this
    # grid came back nan.
    import json
    import os
    import subprocess
    import sys
    cases = [[n, list(a), float(_mp_quantile(n, a))] for n, a in X64_OFF_CASES]
    ref = tmp_path / "cases.json"
    ref.write_text(json.dumps(cases))
    runner = tmp_path / "runner32.py"
    runner.write_text(X64_OFF_RUNNER)
    src = os.path.dirname(os.path.dirname(os.path.abspath(chebax.__file__)))
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": src}
    out = subprocess.run([sys.executable, str(runner), str(ref)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout) <= 7e-6
