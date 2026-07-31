"""Truncated Gamma and Beta distributions for numpyro, today, from user code.

numpyro's truncated-distribution machinery covers location-scale families;
Gamma and Beta need the inverse-CDF path (numpyro#969, PR numpyro#1187),
which stalled on igammainv not existing in jax (jax#5350).
chebax.gammaincinv and chebax.betaincinv fill exactly that hole, so the
two classes below implement the full numpyro Distribution contract:
reparameterized sampling by inverse CDF, log_prob with the truncation
normalizer, cdf/icdf, and gradients through everything, including the
shape parameters. NUTS on a latent concentration works: the Gamma
normalizer differentiates through jax's own gammainc (native a-gradient),
the Beta normalizer through chebax.betainc_fn.

The chebax contract applies: shape parameters are uniform per call
(scalars, or one value per traced latent), so these classes serve scalar
or per-group shapes, not batched shape arrays. For Beta, shapes must lie
in chebax's (a, b) box [0.1, 10]^2. Rates, bounds and evaluation points
are unrestricted arrays. Per-group shapes go through chebax.pergroup.

Run:  python examples/numpyro_truncated_gamma.py   (~1 min on CPU)
Needs: pip install chebax[examples] (numpyro), scipy for the checks.
"""

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax import lax, random  # noqa: E402
from jax.scipy.special import betaln, gammainc, gammaln  # noqa: E402

import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.distributions import constraints  # noqa: E402
from numpyro.distributions.util import promote_shapes, validate_sample  # noqa: E402

import chebax  # noqa: E402


class TruncatedGamma(dist.Distribution):
    """Gamma(concentration, rate) truncated to [low, high].

    concentration must be uniform per call (a scalar or one traced
    latent); rate, low, high broadcast freely. high may be inf.
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "low": constraints.nonnegative,
        "high": constraints.positive,
    }
    reparametrized_params = ["concentration", "rate", "low", "high"]

    def __init__(self, concentration, rate=1.0, *, low=0.0, high=jnp.inf,
                 validate_args=None):
        self.concentration = jnp.asarray(concentration)
        self.rate, self.low, self.high = promote_shapes(
            jnp.asarray(rate), jnp.asarray(low), jnp.asarray(high))
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration), jnp.shape(rate), jnp.shape(low),
            jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _bounds(self):
        flo = gammainc(self.concentration, self.rate * self.low)
        # high = inf means no upper truncation; keep the argument finite
        # so the unused branch cannot poison gradients
        finite = jnp.isfinite(self.high)
        xhi = jnp.where(finite, self.rate * self.high, 1.0)
        fhi = jnp.where(finite, gammainc(self.concentration, xhi), 1.0)
        return flo, fhi

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)

    def icdf(self, q):
        flo, fhi = self._bounds()
        return chebax.gammaincinv(self.concentration,
                                  flo + q * (fhi - flo)) / self.rate

    def cdf(self, value):
        flo, fhi = self._bounds()
        f = gammainc(self.concentration, self.rate * value)
        return jnp.clip((f - flo) / (fhi - flo), 0.0, 1.0)

    @validate_sample
    def log_prob(self, value):
        flo, fhi = self._bounds()
        c = self.concentration
        base = (c * jnp.log(self.rate) + (c - 1.0) * jnp.log(value)
                - self.rate * value - gammaln(c))
        return base - jnp.log(fhi - flo)


class TruncatedBeta(dist.Distribution):
    """Beta(concentration1, concentration0) truncated to [low, high].

    Both concentrations must be uniform per call and inside chebax's
    (a, b) box [0.1, 10]^2 (tables back the normalizer's gradient and
    the inverse CDF); low, high broadcast freely.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "low": constraints.unit_interval,
        "high": constraints.unit_interval,
    }
    reparametrized_params = ["concentration1", "concentration0", "low", "high"]

    def __init__(self, concentration1, concentration0, *, low=0.0, high=1.0,
                 validate_args=None):
        self.concentration1 = jnp.asarray(concentration1)
        self.concentration0 = jnp.asarray(concentration0)
        self.low, self.high = promote_shapes(jnp.asarray(low),
                                             jnp.asarray(high))
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0),
            jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _bounds(self):
        a, b = self.concentration1, self.concentration0
        return (chebax.betainc_fn(a, b, self.low),
                chebax.betainc_fn(a, b, self.high))

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)

    def icdf(self, q):
        flo, fhi = self._bounds()
        return chebax.betaincinv(self.concentration1, self.concentration0,
                                 flo + q * (fhi - flo))

    def cdf(self, value):
        flo, fhi = self._bounds()
        f = chebax.betainc_fn(self.concentration1, self.concentration0, value)
        return jnp.clip((f - flo) / (fhi - flo), 0.0, 1.0)

    @validate_sample
    def log_prob(self, value):
        flo, fhi = self._bounds()
        a, b = self.concentration1, self.concentration0
        base = ((a - 1.0) * jnp.log(value) + (b - 1.0) * jnp.log1p(-value)
                - betaln(a, b))
        return base - jnp.log(fhi - flo)


def check_against_scipy():
    from scipy import stats

    c, r, lo, hi = 3.0, 2.0, 1.0, 6.0
    d = TruncatedGamma(c, r, low=lo, high=hi)
    xs = np.linspace(1.05, 5.95, 9)
    g = stats.gamma(c, scale=1.0 / r)
    z = g.cdf(hi) - g.cdf(lo)
    lp_ref = g.logpdf(xs) - np.log(z)
    lp = np.asarray(d.log_prob(jnp.asarray(xs)))
    err_lp = np.max(np.abs(lp - lp_ref))

    a, b, blo, bhi = 2.5, 3.5, 0.2, 0.8
    db = TruncatedBeta(a, b, low=blo, high=bhi)
    xb = np.linspace(0.25, 0.75, 9)
    be = stats.beta(a, b)
    zb = be.cdf(bhi) - be.cdf(blo)
    lpb_ref = be.logpdf(xb) - np.log(zb)
    lpb = np.asarray(db.log_prob(jnp.asarray(xb)))
    err_lpb = np.max(np.abs(lpb - lpb_ref))
    print(f"log_prob vs scipy: gamma {err_lp:.1e}, beta {err_lpb:.1e}")
    assert err_lp < 1e-12 and err_lpb < 1e-12

    # sampling follows the truncated law (KS against the exact cdf)
    key = random.PRNGKey(0)
    s = np.asarray(d.sample(key, (20000,)))
    ks = stats.kstest(s, lambda v: np.asarray(d.cdf(jnp.asarray(v))))
    sb = np.asarray(db.sample(random.PRNGKey(1), (20000,)))
    ksb = stats.kstest(sb, lambda v: np.asarray(db.cdf(jnp.asarray(v))))
    print(f"KS p-values: gamma {ks.pvalue:.3f}, beta {ksb.pvalue:.3f}")
    assert ks.pvalue > 0.01 and ksb.pvalue > 0.01
    assert float(s.min()) >= lo and float(s.max()) <= hi


def check_reparameterized_gradient():
    # d/dc E[x] for x ~ TruncatedGamma(c, 2, [1, 6]), pathwise vs
    # central differences with common random numbers
    u = random.uniform(random.PRNGKey(2), (20000,))

    def mean_sample(c):
        return jnp.mean(TruncatedGamma(c, 2.0, low=1.0, high=6.0).icdf(u))

    g = float(jax.grad(mean_sample)(3.0))
    h = 1e-5
    fd = float((mean_sample(3.0 + h) - mean_sample(3.0 - h)) / (2 * h))
    print(f"pathwise d/dconcentration E[x]: {g:+.6f} (fd {fd:+.6f})")
    assert abs(g - fd) < 1e-6


def check_nuts_recovery():
    from numpyro.infer import MCMC, NUTS

    c_true, r_true, lo, hi = 3.0, 2.0, 1.0, 6.0
    y = TruncatedGamma(c_true, r_true, low=lo, high=hi).sample(
        random.PRNGKey(3), (400,))

    def model(y):
        c = numpyro.sample("concentration", dist.LogNormal(0.0, 1.0))
        r = numpyro.sample("rate", dist.LogNormal(0.0, 1.0))
        with numpyro.plate("data", y.shape[0]):
            numpyro.sample("y", TruncatedGamma(c, r, low=lo, high=hi), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=2,
                chain_method="sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(4), np.asarray(y))
    post = mcmc.get_samples()
    cm, cs = float(post["concentration"].mean()), float(post["concentration"].std())
    rm, rs = float(post["rate"].mean()), float(post["rate"].std())
    print(f"NUTS on truncated data: concentration {cm:.2f} +- {cs:.2f} "
          f"(true {c_true}), rate {rm:.2f} +- {rs:.2f} (true {r_true})")
    assert abs(cm - c_true) < 4 * cs and abs(rm - r_true) < 4 * rs


def main():
    check_against_scipy()
    check_reparameterized_gradient()
    check_nuts_recovery()


if __name__ == "__main__":
    main()
