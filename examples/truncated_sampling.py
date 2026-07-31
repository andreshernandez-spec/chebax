"""Reparameterized sampling of truncated Gamma, Beta and Student-t distributions.

The pattern numpyro's truncated-distribution machinery needs (see
numpyro#1365 and the truncated-distributions tutorial): with a
differentiable CDF F and quantile function F^{-1},

    x = F^{-1}( F(lo) + u * (F(hi) - F(lo)) ),   u ~ Uniform(0, 1)

is a reparameterized sample of the truncation to [lo, hi], and because
chebax's betainc/betaincinv/stdtr/stdtrit are differentiable in their shape
parameters, jax.grad flows from a Monte Carlo expectation back into
(a, b) or nu.

Run:  python examples/truncated_sampling.py
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402


def truncated_beta_sample(a, b, lo, hi, u):
    flo = chebax.betainc_fn(a, b, lo)
    fhi = chebax.betainc_fn(a, b, hi)
    return chebax.betaincinv(a, b, flo + u * (fhi - flo))


def truncated_t_sample(nu, lo, hi, u):
    flo = chebax.stdtr(nu, lo)
    fhi = chebax.stdtr(nu, hi)
    return chebax.stdtrit(nu, flo + u * (fhi - flo))


def truncated_gamma_sample(a, lo, hi, u):
    # Gamma(a, 1); a rate only rescales x and the bounds, so the unit-rate
    # case carries the general one. The CDF is jax's own gammainc; the
    # quantile is chebax's, with gradients in a on both.
    flo = jax.scipy.special.gammainc(a, lo)
    fhi = jax.scipy.special.gammainc(a, hi)
    return chebax.gammaincinv(a, flo + u * (fhi - flo))


def main():
    key = jax.random.PRNGKey(0)
    u = jax.random.uniform(key, (20000,))

    a, b, lo, hi = 2.0, 3.0, 0.2, 0.7
    x = truncated_beta_sample(a, b, lo, hi, u)
    print(f"truncated Beta({a},{b}) on [{lo},{hi}]: mean {float(x.mean()):.6f}, "
          f"all inside: {bool(((x > lo) & (x < hi)).all())}")

    grad_a = jax.grad(lambda aa: truncated_beta_sample(aa, b, lo, hi, u).mean())(a)
    print(f"d E[x] / da via reparameterization: {float(grad_a):+.6f}")

    nu, tlo, thi = 4.0, -1.0, 2.5
    t = truncated_t_sample(nu, tlo, thi, u)
    grad_nu = jax.grad(lambda n: truncated_t_sample(n, tlo, thi, u).mean())(nu)
    print(f"truncated StudentT(nu={nu}) on [{tlo},{thi}]: mean {float(t.mean()):.6f}, "
          f"d E[t]/d nu: {float(grad_nu):+.6f}")

    a, glo, ghi = 3.5, 1.0, 4.0
    g = truncated_gamma_sample(a, glo, ghi, u)
    grad_a = jax.grad(lambda aa: truncated_gamma_sample(aa, glo, ghi, u).mean())(a)
    print(f"truncated Gamma(a={a}) on [{glo},{ghi}]: mean {float(g.mean()):.6f}, "
          f"all inside: {bool(((g > glo) & (g < ghi)).all())}, "
          f"d E[x]/da: {float(grad_a):+.6f}")


if __name__ == "__main__":
    main()
