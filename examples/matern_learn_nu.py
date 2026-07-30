"""Learn a Matern kernel's smoothness nu by gradient descent.

The Matern kernel is

    k(r) = sig2 * 2^(1-nu)/Gamma(nu) * z^nu * K_nu(z),   z = sqrt(2 nu) r / ell

and learning nu by gradient needs dK_nu/dnu, which no mainstream ML library
provides. chebax.besselk_fn takes nu as a traced jax scalar, so the whole
kernel is differentiable in (nu, ell, sig2) with plain jax.grad.

This demo synthesizes noiseless kernel values at (nu=1.7, ell=0.9,
sig2=1.3), then recovers all three parameters from a different starting
point with Adam on a log-space least-squares loss.

Run:  python examples/matern_learn_nu.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402


def matern(r, nu, ell, sig2):
    z = jnp.sqrt(2.0 * nu) * r / ell
    log_c = (1.0 - nu) * jnp.log(2.0) - jax.scipy.special.gammaln(nu)
    return sig2 * jnp.exp(log_c + nu * jnp.log(z)) * chebax.besselk_fn(nu, z)


def main():
    r = jnp.linspace(0.05, 3.0, 25)
    true = dict(nu=1.7, ell=0.9, sig2=1.3)
    target = matern(r, **true)

    def loss(theta):
        nu, log_ell, log_sig2 = theta
        k = matern(r, nu, jnp.exp(log_ell), jnp.exp(log_sig2))
        return jnp.mean((jnp.log(k) - jnp.log(target)) ** 2)

    grad = jax.jit(jax.grad(loss))
    theta = jnp.array([1.0, 0.0, 0.0])  # start: nu=1.0, ell=1.0, sig2=1.0

    # plain Adam
    m = v = jnp.zeros_like(theta)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for i in range(600):
        g = grad(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** (i + 1))
        vhat = v / (1 - b2 ** (i + 1))
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)

    nu, ell, sig2 = float(theta[0]), float(jnp.exp(theta[1])), float(jnp.exp(theta[2]))
    print(f"true:      nu={true['nu']}, ell={true['ell']}, sig2={true['sig2']}")
    print(f"recovered: nu={nu:.6f}, ell={ell:.6f}, sig2={sig2:.6f}")
    return nu, ell, sig2


if __name__ == "__main__":
    main()
