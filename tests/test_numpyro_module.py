"""chebax.numpyro: wiring tests for the truncated distribution classes.

These test the numpyro Distribution contract (sampling stays in bounds and
follows the truncated law, log_prob is normalized and -inf outside,
pathwise and log_prob gradients flow into every parameter, NUTS runs with
a latent shape), and then the tail behaviour the 2026-08-01 review is
about: a truncation whose two endpoints sit in the SAME deep tail, where
F(hi) - F(lo) is exactly zero and every quantity built on it was inf or
nan.

The KS oracle is scipy's untruncated law renormalized by hand, never the
distribution's own cdf: a self-consistent normalizer passes against
itself. Accuracy references are mpmath at 40 dps, the metric relative
(absolute for the cdf round trip, the CDF convention); measured worst in
float64 is 1.7e-13 over the deep-tail cases and 1.3e-14 over the ordinary
ones, 4.7e-5 in float32 (the child process at the end, x64 being off by
default and conftest hiding that here), and the bars sit at ~4x. Accuracy
contracts for the quantiles underneath live in tests/test_quantiles.py.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import random

pytest.importorskip("numpyro")
scipy = pytest.importorskip("scipy")
mp = pytest.importorskip("mpmath")
from scipy import stats  # noqa: E402

import chebax  # noqa: E402
from chebax.numpyro import (TruncatedBeta, TruncatedGamma,  # noqa: E402
                            TruncatedStudentT)

mp.mp.dps = 40


def _scipy_trunc_cdf(frozen, lo, hi):
    """Truncated CDF from scipy's untruncated law, renormalized by hand.

    Independent of the distribution under test on purpose: the KS test used
    to call d.cdf, which a broken normalizer satisfies against itself. The
    upper branch normalizes on scipy's sf, since its cdf is 1.0 at both
    ends of a deep upper-tail interval, the same wall the module works
    around."""
    if frozen.cdf(lo) >= 0.5:
        slo, shi = frozen.sf(lo), frozen.sf(hi)
        return lambda v: np.clip((slo - frozen.sf(v)) / (slo - shi), 0.0, 1.0)
    flo, fhi = frozen.cdf(lo), frozen.cdf(hi)
    return lambda v: np.clip((frozen.cdf(v) - flo) / (fhi - flo), 0.0, 1.0)


def _ks(d, key, ref_cdf, n=20000):
    s = np.asarray(d.sample(key, (n,)))
    return s, stats.kstest(s, ref_cdf)


# ---------------------------------------------------------------- mpmath

def _mp_bisect(f, lo, hi, iters=200):
    flo = f(lo)
    for _ in range(iters):
        m = (lo + hi) / 2
        if mp.sign(f(m)) == mp.sign(flo):
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def _law(kind, p):
    """(cdf, logpdf) of one base law at 40 dps."""
    if kind == "gamma":
        c, r = mp.mpf(p[0]), mp.mpf(p[1])
        return (lambda x: mp.gammainc(c, 0, r * x, regularized=True),
                lambda x: (c * mp.log(r) + (c - 1) * mp.log(x) - r * x
                           - mp.loggamma(c)))
    if kind == "beta":
        a, b = mp.mpf(p[0]), mp.mpf(p[1])
        lnb = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
        return (lambda x: mp.betainc(a, b, 0, x, regularized=True),
                lambda x: (a - 1) * mp.log(x) + (b - 1) * mp.log(1 - x) - lnb)
    nu = mp.mpf(p[0])

    def cdf(t):
        half = mp.betainc(nu / 2, mp.mpf(0.5), 0, nu / (nu + t * t),
                          regularized=True) / 2
        return 1 - half if t > 0 else half

    return (cdf,
            lambda t: (mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2)
                       - mp.log(nu * mp.pi) / 2
                       - (nu + 1) / 2 * mp.log1p(t * t / nu)))


def _trunc_ref(kind, p, lo, hi, v, q, span):
    """(log_prob(v), cdf(v), icdf(q)) of the truncation to [lo, hi]."""
    cdf, lpdf = _law(kind, p)
    lo, hi, v = mp.mpf(lo), mp.mpf(hi), mp.mpf(v)
    z = cdf(hi) - cdf(lo)
    target = cdf(lo) + mp.mpf(q) * z
    x = _mp_bisect(lambda t: cdf(t) - target, mp.mpf(span[0]), mp.mpf(span[1]))
    return lpdf(v) - mp.log(z), (cdf(v) - cdf(lo)) / z, x


def _rel(got, ref):
    return float(abs(mp.mpf(float(got)) - ref) / max(abs(ref), mp.mpf(1e-300)))


# The blockers: both endpoints in one tail, plus the same intervals with an
# infinite far bound. (kind, params, low, high, value, q, bisection bracket)
DEEP = [
    ("gamma", (3.0, 1.0), 50.0, 51.0, 50.5, 0.5, (50.0, 51.0)),
    ("gamma", (2.0, 1.0), 1e-30, 2e-30, 1.5e-30, 0.25, (1e-30, 2e-30)),
    ("gamma", (3.0, 1.0), 50.0, np.inf, 55.0, 0.5, (50.0, 1e3)),
    ("beta", (2.0, 3.0), 0.999999, 0.9999999, 0.99999945, 0.5,
     (0.999999, 0.9999999)),
    ("beta", (0.5, 3.0), 1e-8, 2e-8, 1.5e-8, 0.5, (1e-8, 2e-8)),
    ("t", (4.0,), 1e5, 2e5, 1.5e5, 0.5, (1e5, 2e5)),
    ("t", (4.0,), -2e5, -1e5, -1.5e5, 0.75, (-2e5, -1e5)),
    ("t", (4.0,), 1e5, np.inf, 1.5e5, 0.5, (1e5, 1e9)),
]

ORDINARY = [
    ("gamma", (3.0, 2.0), 1.0, 6.0, 2.3, 0.37, (1.0, 6.0)),
    ("gamma", (3.0, 2.0), 1.0, np.inf, 2.3, 0.37, (1.0, 1e3)),
    ("gamma", (20.0, 1.0), 1.0, np.inf, 18.0, 0.8, (1.0, 1e3)),
    ("beta", (2.5, 3.5), 0.2, 0.8, 0.42, 0.63, (0.2, 0.8)),
    ("beta", (2.5, 3.5), 0.0, 1.0, 0.42, 0.63, (1e-40, 1 - 1e-30)),
    ("t", (4.0,), -1.0, 2.5, 0.8, 0.4, (-1.0, 2.5)),
    ("t", (150.0,), -3.0, 3.0, 1.1, 0.7, (-3.0, 3.0)),
]


def _build(kind, p, lo, hi):
    if kind == "gamma":
        return TruncatedGamma(p[0], p[1], low=lo, high=hi)
    if kind == "beta":
        return TruncatedBeta(p[0], p[1], low=lo, high=hi)
    return TruncatedStudentT(p[0], low=lo, high=hi)


# ------------------------------------------------------------- contract

def test_truncated_gamma_log_prob_and_sampling():
    c, r, lo, hi = 3.0, 2.0, 1.0, 6.0
    d = TruncatedGamma(c, r, low=lo, high=hi)
    xs = np.linspace(1.05, 5.95, 9)
    g = stats.gamma(c, scale=1.0 / r)
    ref = g.logpdf(xs) - np.log(g.cdf(hi) - g.cdf(lo))
    np.testing.assert_allclose(np.asarray(d.log_prob(jnp.asarray(xs))), ref,
                               rtol=1e-10)
    assert np.isneginf(float(d.log_prob(0.5)))
    assert np.isneginf(float(d.log_prob(7.0)))
    assert float(d.cdf(0.5)) == 0.0 and float(d.cdf(7.0)) == 1.0
    s, ks = _ks(d, random.PRNGKey(0), _scipy_trunc_cdf(g, lo, hi))
    assert ks.pvalue > 0.01
    assert s.min() >= lo and s.max() <= hi
    # one-sided truncation with high = inf
    d1 = TruncatedGamma(c, r, low=1.0)
    ref1 = g.logpdf(xs) - np.log(g.sf(1.0))
    np.testing.assert_allclose(np.asarray(d1.log_prob(jnp.asarray(xs))), ref1,
                               rtol=1e-10)


def test_truncated_beta_log_prob_and_sampling():
    a, b, lo, hi = 2.5, 3.5, 0.2, 0.8
    d = TruncatedBeta(a, b, low=lo, high=hi)
    xs = np.linspace(0.25, 0.75, 9)
    be = stats.beta(a, b)
    ref = be.logpdf(xs) - np.log(be.cdf(hi) - be.cdf(lo))
    np.testing.assert_allclose(np.asarray(d.log_prob(jnp.asarray(xs))), ref,
                               rtol=1e-10)
    s, ks = _ks(d, random.PRNGKey(1), _scipy_trunc_cdf(be, lo, hi))
    assert ks.pvalue > 0.01
    assert s.min() >= lo and s.max() <= hi


def test_truncated_studentt_log_prob_and_sampling():
    df, lo, hi = 4.0, -1.0, 2.5
    d = TruncatedStudentT(df, low=lo, high=hi)
    xs = np.linspace(-0.9, 2.4, 9)
    t = stats.t(df)
    ref = t.logpdf(xs) - np.log(t.cdf(hi) - t.cdf(lo))
    np.testing.assert_allclose(np.asarray(d.log_prob(jnp.asarray(xs))), ref,
                               rtol=1e-9)
    s, ks = _ks(d, random.PRNGKey(2), _scipy_trunc_cdf(t, lo, hi))
    assert ks.pvalue > 0.01
    assert s.min() >= lo and s.max() <= hi
    # untruncated default degrades to plain t
    d0 = TruncatedStudentT(df)
    np.testing.assert_allclose(np.asarray(d0.log_prob(jnp.asarray(xs))),
                               t.logpdf(xs), rtol=1e-9)


def test_deep_tail_sampling_follows_the_truncated_law():
    # the oracle renormalizes scipy's gamma on its survival function, the
    # same tail the class works in; scipy's cdf is 1.0 at both ends here
    c, lo, hi = 3.0, 50.0, 51.0
    d = TruncatedGamma(c, 1.0, low=lo, high=hi)
    s, ks = _ks(d, random.PRNGKey(6), _scipy_trunc_cdf(stats.gamma(c), lo, hi),
                n=5000)
    assert ks.pvalue > 0.01
    assert s.min() >= lo and s.max() <= hi


def test_pathwise_gradients_match_finite_differences():
    u = random.uniform(random.PRNGKey(3), (20000,))

    def gamma_mean(c):
        return jnp.mean(TruncatedGamma(c, 2.0, low=1.0, high=6.0).icdf(u))

    g = float(jax.grad(gamma_mean)(3.0))
    h = 1e-5
    fd = float((gamma_mean(3.0 + h) - gamma_mean(3.0 - h)) / (2 * h))
    assert abs(g - fd) < 1e-6

    def t_mean(df):
        return jnp.mean(TruncatedStudentT(df, low=-1.0, high=2.5).icdf(u))

    g = float(jax.grad(t_mean)(4.0))
    fd = float((t_mean(4.0 + h) - t_mean(4.0 - h)) / (2 * h))
    assert abs(g - fd) < 1e-6


def test_log_prob_gradient_in_shape():
    xs = jnp.asarray([0.3, 0.5, 0.7])

    def nll(a):
        return -jnp.sum(TruncatedBeta(a, 3.5, low=0.2, high=0.8).log_prob(xs))

    g = float(jax.grad(nll)(2.5))
    assert np.isfinite(g)


def test_bounds_validated_eagerly():
    with pytest.raises(ValueError, match="low < high"):
        TruncatedGamma(2.0, low=5.0, high=1.0)
    with pytest.raises(ValueError, match="low < high"):
        TruncatedStudentT(4.0, low=2.0, high=-2.0)


def test_nuts_with_latent_shape():
    import numpyro
    import numpyro.distributions as ndist
    from numpyro.infer import MCMC, NUTS

    df_true, lo, hi = 4.0, -1.0, 2.5
    y = TruncatedStudentT(df_true, low=lo, high=hi).sample(
        random.PRNGKey(4), (300,))

    def model(y):
        df = numpyro.sample("df", ndist.Uniform(0.5, 15.0))
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("y", TruncatedStudentT(df, low=lo, high=hi), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300, num_chains=1,
                progress_bar=False)
    mcmc.run(random.PRNGKey(5), np.asarray(y))
    post = mcmc.get_samples()["df"]
    assert np.isfinite(np.asarray(post)).all()


# ------------------------------------------------- the review's blockers

def test_deep_tail_truncations_against_mpmath():
    # Before the log-space normalizer every one of these had (F(lo), F(hi))
    # equal to the same float, so log_prob was +inf, cdf nan and icdf
    # outside the interval. Measured worst 1.7e-13 relative, bar 7e-13.
    worst = 0.0
    for kind, p, lo, hi, v, q, span in DEEP:
        d = _build(kind, p, lo, hi)
        rlp, rc, ri = _trunc_ref(kind, p, lo, hi, v, q, span)
        got = [float(d.log_prob(v)), float(d.cdf(v)), float(d.icdf(q))]
        assert np.isfinite(got).all(), (kind, p, lo, hi, got)
        assert lo <= got[2] <= hi, (kind, p, lo, hi, got[2])
        worst = max(worst, _rel(got[0], rlp), _rel(got[1], rc),
                    _rel(got[2], ri))
    assert worst <= 7e-13, worst


def test_ordinary_truncations_unchanged_against_mpmath():
    # the non-tail path has to stay at eps: measured worst 1.3e-14, bar 5e-14
    worst = 0.0
    for kind, p, lo, hi, v, q, span in ORDINARY:
        d = _build(kind, p, lo, hi)
        rlp, rc, ri = _trunc_ref(kind, p, lo, hi, v, q, span)
        worst = max(worst, _rel(float(d.log_prob(v)), rlp),
                    _rel(float(d.cdf(v)), rc), _rel(float(d.icdf(q)), ri))
    assert worst <= 5e-14, worst


def test_icdf_round_trips_through_cdf_in_the_tail():
    # The CDF metric is absolute. Measured worst 1.3e-10, all of it the Beta
    # whose bounds sit 1e-6 below 1: x is quantized at 1.1e-16 there, which
    # is 1.6e-10 of that interval's mass, and no recomputation from a stored
    # x beats it. Every other case is at 3e-14. Bar 5e-10.
    qs = jnp.asarray([1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-6])
    worst = 0.0
    for kind, p, lo, hi, _, _, _ in DEEP:
        d = _build(kind, p, lo, hi)
        x = np.asarray(d.icdf(qs))
        assert np.all((x >= lo) & (x <= hi)), (kind, p, x)
        worst = max(worst,
                    np.max(np.abs(np.asarray(d.cdf(x)) - np.asarray(qs))))
    assert worst <= 5e-10, worst


def test_gamma_qinv_survives_a_collapsed_asymptotic_init():
    # The upper-tail solver's asymptotic init needs x >> a; once
    # ln(1/s) < lnGamma(a) its fixed point collapses to the floor, and
    # from there Newton on the recipe's everywhere-finite logs crawls one
    # e-fold per step (the safeguard never bisects), so 40 steps stopped
    # 5.8e-10 short at (a, s) = (50, 1e-8), inside the residual bar. The
    # init now starts from max(WH, asymptotic). Metric: relative error of
    # Q(a, x) at the returned x vs s, mpmath the judge. Measured worst
    # 2.7e-14, bar 1e-13.
    cases = [(50.0, 1e-8), (200.0, 1e-6), (30.0, 1e-12),
             (500.0, 1e-8), (1000.0, 1e-4)]
    worst = 0.0
    for a, s in cases:
        x = float(chebax.gammainccinv(a, s))
        q = mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True)
        worst = max(worst, abs(float((q - mp.mpf(s)) / mp.mpf(s))))
    assert worst <= 1e-13, worst


def test_batched_bounds_broadcast():
    # low and high may be arrays even though the shapes may not. A batch
    # mixing a lower-tail interval with an upper-tail one is the case that
    # normalizes on both sides at once.
    lo, hi = jnp.asarray([1e-30, 50.0]), jnp.asarray([2e-30, 51.0])
    d = TruncatedGamma(2.0, 1.0, low=lo, high=hi)
    x = np.asarray(d.icdf(jnp.asarray([0.25, 0.5])))
    assert np.all((x >= np.asarray(lo)) & (x <= np.asarray(hi))), x
    ref = [float(TruncatedGamma(2.0, 1.0, low=1e-30, high=2e-30).icdf(0.25)),
           float(TruncatedGamma(2.0, 1.0, low=50.0, high=51.0).icdf(0.5))]
    np.testing.assert_allclose(x, ref, rtol=1e-12)
    assert np.isfinite(np.asarray(d.log_prob(jnp.asarray([1.5e-30, 50.5])))).all()
    s = np.asarray(d.sample(random.PRNGKey(7), (5,)))
    assert np.all((s >= np.asarray(lo)) & (s <= np.asarray(hi))), s


def test_infinite_bound_gradients_are_finite():
    # _bounds used to form rate * high before masking high = inf, so the
    # masked-away 0 * inf poisoned the rate cotangent of the DEFAULT
    # one-sided TruncatedGamma. Reverse mode only, which is why it went
    # unnoticed: the two forward checks below were clean all along.
    cases = [
        (lambda r: TruncatedGamma(3.0, r, low=1.0).log_prob(2.0), 1.0),
        (lambda r: TruncatedGamma(3.0, r, low=1.0).icdf(0.5), 1.0),
        (lambda c: TruncatedGamma(c, 2.0, low=1.0).log_prob(2.0), 3.0),
        (lambda lo: TruncatedGamma(3.0, 2.0, low=lo).icdf(0.5), 1.0),
        (lambda v: TruncatedStudentT(v).log_prob(0.5), 4.0),
        (lambda v: TruncatedStudentT(v).icdf(0.3), 4.0),
        (lambda lo: TruncatedStudentT(4.0, low=lo).icdf(0.3), -1.0),
    ]
    h = 1e-5
    for i, (f, th) in enumerate(cases):
        fd = float((f(th + h) - f(th - h)) / (2 * h))
        rev = float(jax.grad(f)(th))
        assert np.isfinite(rev), rev
        assert abs(rev - fd) <= 1e-5 * max(1.0, abs(fd)), (rev, fd)
        if i < 2:                     # the two the review reported
            fwd = float(jax.jacfwd(f)(th))
            assert abs(fwd - fd) <= 1e-5 * max(1.0, abs(fd)), (fwd, fd)
    # the values themselves were always fine
    d = TruncatedGamma(3.0, 1.0, low=1.0)
    assert abs(float(d.log_prob(2.0)) + 1.2231435513142084) < 1e-12
    assert abs(float(d.icdf(0.5)) - 2.8405275093459688) < 1e-9


def test_endpoint_densities_are_finite_where_the_density_is():
    # (c-1) log 0 is 0 * -inf at c = 1, where the density is the rate; same
    # for both Beta endpoints at concentration 1. The genuinely infinite and
    # genuinely zero endpoints keep their sign.
    assert abs(float(TruncatedGamma(1.0, 2.0, low=0.0, high=5.0).log_prob(0.0))
               - (np.log(2.0) - np.log(-np.expm1(-10.0)))) < 1e-12
    assert np.isposinf(
        float(TruncatedGamma(0.5, 2.0, low=0.0, high=5.0).log_prob(0.0)))
    assert np.isneginf(
        float(TruncatedGamma(2.0, 2.0, low=0.0, high=5.0).log_prob(0.0)))
    b = TruncatedBeta(1.0, 3.0)
    assert abs(float(b.log_prob(0.0)) - np.log(3.0)) < 1e-12
    assert np.isneginf(float(b.log_prob(1.0)))
    b = TruncatedBeta(3.0, 1.0)
    assert abs(float(b.log_prob(1.0)) - np.log(3.0)) < 1e-12
    assert np.isneginf(float(b.log_prob(0.0)))
    u = TruncatedBeta(1.0, 1.0)
    assert abs(float(u.log_prob(0.0))) < 1e-15
    assert abs(float(u.log_prob(1.0))) < 1e-15
    assert np.isposinf(float(TruncatedBeta(0.5, 3.0).log_prob(0.0)))


def test_validate_args_accepts_the_default_bounds():
    # constraints.real rejects inf, so validate_args used to reject
    # TruncatedStudentT's own untruncated defaults
    assert np.isfinite(
        float(TruncatedStudentT(4.0, validate_args=True).log_prob(0.5)))
    assert np.isfinite(
        float(TruncatedGamma(3.0, validate_args=True).log_prob(2.0)))
    assert np.isfinite(
        float(TruncatedBeta(2.0, 3.0, validate_args=True).log_prob(0.5)))


def test_shape_parameters_checked_at_construction():
    with pytest.raises(ValueError, match=r"concentration1 in \[0.1, 10"):
        TruncatedBeta(20.0, 3.0)
    with pytest.raises(ValueError, match=r"concentration0 in \[0.1, 10"):
        TruncatedBeta(2.0, 0.01)
    with pytest.raises(ValueError, match=r"df in \[0.2, 200"):
        TruncatedStudentT(500.0)
    with pytest.raises(ValueError, match="pergroup"):
        TruncatedGamma(jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="pergroup"):
        TruncatedBeta(jnp.asarray([1.0, 2.0]), 3.0)
    with pytest.raises(ValueError, match="pergroup"):
        TruncatedStudentT(jnp.asarray([4.0, 5.0]))
    # a traced shape cannot be checked eagerly and must still trace
    assert np.isfinite(float(jax.jit(
        lambda v: TruncatedStudentT(v, low=-1.0).log_prob(0.5))(4.0)))


def test_icdf_endpoints_are_the_bounds_exactly():
    for kind, p, lo, hi in [("gamma", (3.0, 2.0), 1.0, 6.0),
                            ("beta", (2.5, 3.5), 0.2, 0.8),
                            ("t", (4.0,), -1.0, 2.5),
                            ("gamma", (3.0, 1.0), 50.0, 51.0)]:
        d = _build(kind, p, lo, hi)
        assert float(d.icdf(0.0)) == lo
        assert float(d.icdf(1.0)) == hi


X64_OFF_RUNNER = """
import json, sys
import numpy as np
import jax.numpy as jnp
from chebax.numpyro import TruncatedBeta, TruncatedGamma, TruncatedStudentT

assert jnp.empty(()).dtype == jnp.float32, jnp.empty(()).dtype
build = {"gamma": lambda p, lo, hi: TruncatedGamma(p[0], p[1], low=lo, high=hi),
         "beta": lambda p, lo, hi: TruncatedBeta(p[0], p[1], low=lo, high=hi),
         "t": lambda p, lo, hi: TruncatedStudentT(p[0], low=lo, high=hi)}
worst = 0.0
for kind, p, lo, hi, v, q, ref in json.load(open(sys.argv[1])):
    d = build[kind](p, lo, float(hi))
    got = [float(d.log_prob(v)), float(d.cdf(v)), float(d.icdf(q))]
    assert np.isfinite(got).all(), (kind, p, lo, hi, got)
    assert lo <= got[2] <= hi, (kind, p, lo, hi, got[2])
    for g, r in zip(got, ref):
        worst = max(worst, abs(g - r) / max(abs(r), 1e-30))
# endpoints stay exact at float32 too
d = build["gamma"]([3.0, 1.0], 50.0, 51.0)
assert float(d.icdf(0.0)) == 50.0 and float(d.icdf(1.0)) == 51.0
print(worst)
"""

# float32 cannot represent the float64 deep-tail bounds (1 - 1e-7 rounds to
# 1), so the intervals here are posed at float32's own resolution.
X64_OFF_CASES = [
    ("gamma", (3.0, 1.0), 50.0, 51.0, 50.5, 0.5, (50.0, 51.0)),
    ("gamma", (3.0, 2.0), 1.0, 6.0, 2.3, 0.37, (1.0, 6.0)),
    ("gamma", (3.0, 2.0), 1.0, np.inf, 2.3, 0.37, (1.0, 1e3)),
    ("beta", (2.0, 3.0), 0.999, 0.9999, 0.9995, 0.5, (0.999, 0.9999)),
    ("beta", (2.5, 3.5), 0.2, 0.8, 0.42, 0.63, (0.2, 0.8)),
    ("t", (4.0,), 1e5, 2e5, 1.5e5, 0.5, (1e5, 2e5)),
    ("t", (4.0,), -1.0, 2.5, 0.8, 0.4, (-1.0, 2.5)),
]


def test_x64_off_deep_tails(tmp_path):
    # conftest forces x64 on for this whole suite, so the dtype jax actually
    # defaults to only gets tested in a child. Measured worst 4.7e-5
    # relative against 40-dps references, bar 2e-4.
    cases = []
    for kind, p, lo, hi, v, q, span in X64_OFF_CASES:
        ref = [float(r) for r in _trunc_ref(kind, p, lo, hi, v, q, span)]
        cases.append([kind, list(p), lo, hi, v, q, ref])
    ref_file = tmp_path / "cases.json"
    ref_file.write_text(json.dumps(cases))
    runner = tmp_path / "runner32.py"
    runner.write_text(X64_OFF_RUNNER)
    src = os.path.dirname(os.path.dirname(os.path.abspath(chebax.__file__)))
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": src}
    out = subprocess.run([sys.executable, str(runner), str(ref_file)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout) <= 2e-4, out.stdout
