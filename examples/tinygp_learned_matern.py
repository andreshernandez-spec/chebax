"""A tinygp custom kernel that learns the Matern smoothness from data.

tinygp ships Matern12/32/52 with the smoothness frozen at half-integer
values, like every JAX GP library: general nu needs K_nu with a traced
order and a differentiation rule, which jax does not provide.
chebax.matern supplies that, so the general kernel is the ten-line
subclass below (tinygp's custom-kernel API: implement evaluate for one
pair of inputs) and nu is just one more entry in the parameter dict the
optimizer sees.

The demo draws noisy observations from a ground-truth Matern-5/2 GP and
recovers the smoothness by maximising the marginal likelihood, starting
at nu = 1.0. Smoothness is identified by SHORT-range behavior, so every
base point gets a close neighbor (the clustered design measured in the
gpjax twin of this example); expect the right neighborhood, not
digit-level recovery. Contracts: nu uniform per call in (0, 10];
sub-resolution distances snap to zero so the covariance diagonal is
exact.

Run:    python examples/tinygp_learned_matern.py   (~1 min on CPU)
Needs:  pip install chebax tinygp optax
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import optax  # noqa: E402
import tinygp  # noqa: E402

import chebax  # noqa: E402


class Matern(tinygp.kernels.Kernel):
    """Matern correlation with trainable smoothness nu in (0, 10]."""

    nu: jax.Array
    lengthscale: jax.Array

    def evaluate(self, X1, X2):
        tau = jnp.sqrt(jnp.sum(jnp.square(jnp.atleast_1d(X1)
                                          - jnp.atleast_1d(X2))))
        tau = jnp.where(tau < 1e-12, 0.0, tau)  # exact diagonal
        return chebax.matern(self.nu, tau, self.lengthscale)


def build_gp(params, x):
    kernel = jnp.exp(params["log_sig2"]) * Matern(
        nu=jnp.exp(params["log_nu"]),
        lengthscale=jnp.exp(params["log_ell"]))
    return tinygp.GaussianProcess(kernel, x,
                                  diag=jnp.exp(2 * params["log_noise"]))


def main():
    key = jax.random.PRNGKey(0)
    nu_true, ell_true, sig2_true, noise = 2.5, 0.4, 2.0, 0.1

    base = jax.random.uniform(key, (150, 1), dtype=jnp.float64) * 6.0
    close = base + 0.02 * jax.random.normal(jax.random.PRNGKey(9),
                                            (150, 1), dtype=jnp.float64)
    x = jnp.sort(jnp.concatenate([base, close]), axis=0)
    truth = {"log_nu": jnp.log(nu_true), "log_ell": jnp.log(ell_true),
             "log_sig2": jnp.log(sig2_true), "log_noise": jnp.log(noise)}
    y = build_gp(truth, x).sample(jax.random.PRNGKey(1))

    params = {"log_nu": jnp.asarray(0.0), "log_ell": jnp.asarray(0.0),
              "log_sig2": jnp.asarray(0.0), "log_noise": jnp.asarray(-1.0)}

    @jax.jit
    def loss(p):
        return -build_gp(p, x).log_probability(y)

    opt = optax.adam(0.05)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        val, g = jax.value_and_grad(loss)(p)
        updates, s = opt.update(g, s)
        return optax.apply_updates(p, updates), s, val

    first = None
    for i in range(1500):
        params, state, val = step(params, state)
        if first is None:
            first = val
    nu = float(jnp.exp(params["log_nu"]))
    print(f"true:    nu={nu_true}, ell={ell_true}, sig2={sig2_true}")
    print(f"learned: nu={nu:.3f}, ell={float(jnp.exp(params['log_ell'])):.3f}, "
          f"sig2={float(jnp.exp(params['log_sig2'])):.3f}")
    print(f"nmll: {float(first):.2f} -> {float(val):.2f}")
    assert jnp.isfinite(val)
    assert 1.8 < nu < 3.5


if __name__ == "__main__":
    main()
