"""chebax.numpyro: wiring tests for the truncated distribution classes.

These test the numpyro Distribution contract (sampling stays in bounds
and follows the truncated law, log_prob is normalized and -inf outside,
pathwise and log_prob gradients flow into the shape parameters, NUTS
runs with a latent shape). scipy is a wiring oracle at loose tolerances;
accuracy contracts live with the underlying recipes' test files.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import random

pytest.importorskip("numpyro")
scipy = pytest.importorskip("scipy")
from scipy import stats  # noqa: E402

from chebax.numpyro import (TruncatedBeta, TruncatedGamma,  # noqa: E402
                            TruncatedStudentT)


def _ks_against_cdf(d, key, n=20000):
    s = np.asarray(d.sample(key, (n,)))
    return s, stats.kstest(s, lambda v: np.asarray(d.cdf(jnp.asarray(v))))


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
    s, ks = _ks_against_cdf(d, random.PRNGKey(0))
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
    s, ks = _ks_against_cdf(d, random.PRNGKey(1))
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
    s, ks = _ks_against_cdf(d, random.PRNGKey(2))
    assert ks.pvalue > 0.01
    assert s.min() >= lo and s.max() <= hi
    # untruncated default degrades to plain t
    d0 = TruncatedStudentT(df)
    np.testing.assert_allclose(np.asarray(d0.log_prob(jnp.asarray(xs))),
                               t.logpdf(xs), rtol=1e-9)


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
