"""pymc on the JAX samplers without tfp: one import from chebax.

Two models that fail on a stock installation and work after
`import chebax.pytensor`:

1. Truncated Normal under nuts_sampler="numpyro". The normal log-CDF
   lowers through erfcx/erfcinv, which pytensor's JAX backend otherwise
   routes to tensorflow-probability; the plugin supplies them in plain
   jax.
2. Censored StudentT with a LATENT degrees-of-freedom. The censoring
   term is a betainc in nu, and jax's own betainc has no gradient with
   respect to its shape parameters, so NUTS cannot run at all. The
   plugin's betainc lowering carries those gradients (valid for the
   scalar-latent case, dof in [0.4, 20] via a = b = nu/2).

Needs:  pip install chebax[pytensor] pymc
Run:    python examples/pymc_censored_studentt.py
"""

import numpy as np

import chebax.pytensor  # noqa: F401  (the entire fix: registers JAX dispatches)
import pymc as pm


def main():
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Truncated("y", pm.Normal.dist(mu, 1.0), lower=-1.0, upper=2.0,
                     observed=np.array([0.3, 0.7, 1.1, -0.2, 0.5]))
        idata = pm.sample(nuts_sampler="numpyro", draws=500, tune=500,
                          chains=2, progressbar=False,
                          compute_convergence_checks=False)
    print(f"truncated Normal: posterior mu = "
          f"{float(idata.posterior['mu'].mean()):+.3f} "
          f"+- {float(idata.posterior['mu'].std()):.3f}")

    with pm.Model():
        nu = pm.Uniform("nu", 2.0, 10.0)
        pm.Censored("z", pm.StudentT.dist(nu=nu, mu=0.0, sigma=1.0),
                    lower=None, upper=1.5,
                    observed=np.array([0.3, 1.5, 1.5, -0.4, 0.9, 1.5]))
        idata = pm.sample(nuts_sampler="numpyro", draws=500, tune=500,
                          chains=2, progressbar=False,
                          compute_convergence_checks=False)
    print(f"censored StudentT with latent dof: posterior nu = "
          f"{float(idata.posterior['nu'].mean()):.2f} "
          f"+- {float(idata.posterior['nu'].std()):.2f}")


if __name__ == "__main__":
    main()
