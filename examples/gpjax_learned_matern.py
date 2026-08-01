"""A GPJax Matern kernel that learns its smoothness by gradient descent.

GPJax ships Matern12/32/52 with the smoothness frozen at construction;
gpjax#482 asked for general nu and the maintainer's answer named the
blocker: "this requires a modified Bessel function of the second kind
and its differentiation rule in JAX". chebax.matern supplies exactly
that (K_nu with a traced order), so a general kernel is the small
subclass below: nu becomes an ordinary trainable PositiveReal, and
gpjax's stock fit optimises it together with lengthscale and variance.

The demo draws noisy observations from a ground-truth Matern-5/2 GP and
recovers the smoothness from data by maximising the marginal
likelihood, starting at nu = 1.0. Contracts: nu uniform per call in
(0, 10]; sub-resolution distances are snapped to zero so the
covariance diagonal is exact (gpjax's distance helper returns ~1e-18
for identical points, never 0).

Run:    python examples/gpjax_learned_matern.py   (~1 min on CPU)
Needs:  pip install chebax gpjax optax
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import beartype.typing as tp  # noqa: E402
import optax  # noqa: E402
from paramax import AbstractUnwrappable  # noqa: E402

import gpjax  # noqa: E402
from gpjax.kernels.base import _val  # noqa: E402
from gpjax.kernels.stationary.base import StationaryKernel  # noqa: E402
from gpjax.kernels.stationary.utils import euclidean_distance  # noqa: E402
from gpjax.parameters import PositiveReal  # noqa: E402

import chebax  # noqa: E402


class Matern(StationaryKernel):
    """Matern kernel with a trainable smoothness parameter nu in (0, 10]."""

    name: str = "Matérn"
    smoothness: AbstractUnwrappable

    def __init__(self, smoothness: tp.Union[float, AbstractUnwrappable] = 1.5,
                 **kwargs):
        if isinstance(smoothness, AbstractUnwrappable):
            self.smoothness = smoothness
        else:
            self.smoothness = PositiveReal(jnp.asarray(smoothness))
        super().__init__(**kwargs)

    def __call__(self, x, y):
        x = self.slice_input(x) / _val(self.lengthscale)
        y = self.slice_input(y) / _val(self.lengthscale)
        tau = euclidean_distance(x, y)
        tau = jnp.where(tau < 1e-12, 0.0, tau)  # exact diagonal
        k = _val(self.variance) * chebax.matern(_val(self.smoothness), tau)
        return k.squeeze()


def main():
    key = jax.random.PRNGKey(0)
    nu_true, ell_true, sig2_true, noise = 2.5, 0.4, 2.0, 0.1

    # smoothness is identified by SHORT-range behavior: pair every base
    # point with a close neighbor so the design actually resolves nu
    base = jax.random.uniform(key, (150, 1), dtype=jnp.float64) * 6.0
    close = base + 0.02 * jax.random.normal(jax.random.PRNGKey(9),
                                            (150, 1), dtype=jnp.float64)
    x = jnp.sort(jnp.concatenate([base, close]), axis=0)
    n = x.shape[0]
    truth = Matern(smoothness=nu_true, lengthscale=ell_true, variance=sig2_true)
    K = truth.gram(x).as_matrix() + noise**2 * jnp.eye(n)
    y = (jnp.linalg.cholesky(K)
         @ jax.random.normal(jax.random.PRNGKey(1), (n, 1), dtype=jnp.float64))
    data = gpjax.Dataset(X=x, y=y)

    kernel = Matern(smoothness=1.0)
    prior = gpjax.gps.Prior(mean_function=gpjax.mean_functions.Zero(),
                            kernel=kernel)
    likelihood = gpjax.likelihoods.Gaussian(num_datapoints=n)
    posterior = prior * likelihood

    def nmll(p, d):
        return -gpjax.objectives.conjugate_mll(p, d)

    fitted, history = gpjax.fit(model=posterior, objective=nmll,
                                train_data=data,
                                optim=optax.adam(0.05), num_iters=1500,
                                key=jax.random.PRNGKey(2), verbose=False)

    k = fitted.prior.kernel
    print(f"true:    nu={nu_true}, ell={ell_true}, sig2={sig2_true}")
    print(f"learned: nu={float(k.smoothness.unwrap()):.3f}, "
          f"ell={float(k.lengthscale.unwrap()):.3f}, "
          f"sig2={float(k.variance.unwrap()):.3f}")
    print(f"nmll: {float(history[0]):.2f} -> {float(history[-1]):.2f}")
    assert jnp.isfinite(history[-1])
    # nu is weakly identified from finite data; the clustered design
    # gets the right neighborhood, not digit-level recovery
    assert 1.8 < float(k.smoothness.unwrap()) < 3.5


if __name__ == "__main__":
    main()
