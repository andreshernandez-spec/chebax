"""Truncated Gamma and Beta in numpyro: chebax.numpyro in action.

numpyro's truncated-distribution machinery covers location-scale
families; Gamma and Beta need the inverse-CDF path (numpyro#969, PR
numpyro#1187), which stalled on igammainv not existing in jax
(jax#5350). chebax.numpyro ships ready-made TruncatedGamma,
TruncatedBeta and TruncatedStudentT classes built on chebax's
differentiable quantiles (see that module's docstring for the domain
contracts). This demo verifies them against scipy and runs NUTS with
LATENT shape parameters on truncated data.

Run:  python examples/numpyro_truncated_gamma.py   (~1 min on CPU)
Needs: pip install chebax[numpyro] scipy
"""

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax import random  # noqa: E402

import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from chebax.numpyro import TruncatedBeta, TruncatedGamma  # noqa: E402


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
