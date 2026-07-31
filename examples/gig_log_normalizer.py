"""Generalized Inverse Gaussian: a differentiable log-normalizer for JAX.

The GIG density is

    f(x; p, a, b) = (a/b)^(p/2) / (2 K_p(sqrt(ab))) * x^(p-1) e^{-(a x + b/x)/2}

on x > 0. As an exponential family its natural parameters are
eta = (p - 1, -a/2, -b/2) with sufficient statistics (ln x, x, 1/x), and the
log-normalizer is

    A = ln 2 + (p/2) ln(b/a) + ln K_p(sqrt(ab)).

Everything a fitting pipeline needs comes from derivatives of A: grad A gives
the mean statistics (E[ln x], E[x], E[1/x]), the Hessian drives Newton in the
mean-to-natural conversion, and the entropy is A - eta . grad A. The
nonstandard ingredient is d ln K_p / dp, a derivative in the ORDER of the
Bessel function.

chebax.log_besselk_fn takes the order as a traced jax scalar, so plain
jax.grad and jax.hessian work through A in all three parameters. This demo
checks the gradient identities and runs the mean-to-natural Newton
conversion: recover (p, a, b) from mean statistics alone.

Domain: |p| <= 10 (K_{-p} = K_p covers negative p), sqrt(ab) in [1e-6, ~700].
This is the plain-jax version of examples/efax_gig.py, which wires the same
math into efax's distribution classes.

Run:  python examples/gig_log_normalizer.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402


def gig_log_normalizer(p, a, b):
    z = jnp.sqrt(a * b)
    log_k = chebax.log_besselk_fn(jnp.abs(p), z)
    return jnp.log(2.0) + 0.5 * p * (jnp.log(b) - jnp.log(a)) + log_k


def log_normalizer_nat(eta):
    # eta = (p - 1, -a/2, -b/2)
    return gig_log_normalizer(eta[0] + 1.0, -2.0 * eta[1], -2.0 * eta[2])


def to_nat(p, a, b):
    return jnp.array([p - 1.0, -0.5 * a, -0.5 * b])


def from_nat(eta):
    return eta[0] + 1.0, -2.0 * eta[1], -2.0 * eta[2]


mean_stats = jax.jit(jax.grad(log_normalizer_nat))   # (E[ln x], E[x], E[1/x])
hess = jax.jit(jax.hessian(log_normalizer_nat))


def exp_to_nat(mu, theta0, iters=20):
    # Solve grad A(eta) = mu by Newton in theta = (p, ln a, ln b): the log
    # coordinates keep a, b positive with no constraint handling, and the
    # exact Jacobian comes from jax.jacfwd through besselk_fn. Backtrack on
    # the residual norm; near the solution full steps are quadratic.
    def residual(theta):
        p, la, lb = theta
        return mean_stats(to_nat(p, jnp.exp(la), jnp.exp(lb))) - mu

    jac = jax.jit(jax.jacfwd(residual))
    theta = theta0
    for i in range(iters):
        r = residual(theta)
        rn = float(jnp.linalg.norm(r))
        print(f"  newton {i}: |grad A - mu| = {rn:.2e}")
        if rn < 1e-12:
            break
        step = jnp.linalg.solve(jac(theta), r)
        t = 1.0
        while t > 1e-8:
            r1 = residual(theta - t * step)
            if bool(jnp.all(jnp.isfinite(r1))) and float(jnp.linalg.norm(r1)) < rn:
                break
            t *= 0.5
        theta = theta - t * step
    p, la, lb = theta
    return to_nat(p, jnp.exp(la), jnp.exp(lb))


def main():
    true = dict(p=2.5, a=1.5, b=3.0)
    eta_true = to_nat(**true)

    mu = mean_stats(eta_true)
    print("mean statistics at the true parameters:")
    print(f"  E[ln x] = {float(mu[0]):+.12f}")
    print(f"  E[x]    = {float(mu[1]):+.12f}")
    print(f"  E[1/x]  = {float(mu[2]):+.12f}")

    # cross-check E[x] against the classical Bessel-ratio identity
    z = jnp.sqrt(true["a"] * true["b"])
    ratio = chebax.besselk_fn(true["p"] + 1.0, z) / chebax.besselk_fn(true["p"], z)
    ex = jnp.sqrt(true["b"] / true["a"]) * ratio
    print(f"  E[x] via K_(p+1)/K_p ratio: {float(ex):+.12f}")

    print("\nmean-to-natural conversion by Newton, from (p, a, b) = (1, 1, 1):")
    eta = exp_to_nat(mu, jnp.zeros(3) + jnp.array([1.0, 0.0, 0.0]))
    p, a, b = (float(v) for v in from_nat(eta))
    print(f"  true:      p={true['p']}, a={true['a']}, b={true['b']}")
    print(f"  recovered: p={p:.12f}, a={a:.12f}, b={b:.12f}")

    def entropy(p, a, b):
        eta = to_nat(p, a, b)
        return log_normalizer_nat(eta) - eta @ mean_stats(eta)

    h, g = entropy(*jnp.array([2.5, 1.5, 3.0])), jax.grad(entropy, (0, 1, 2))(2.5, 1.5, 3.0)
    print(f"\nentropy = {float(h):.12f}, gradient = "
          f"({float(g[0]):+.6f}, {float(g[1]):+.6f}, {float(g[2]):+.6f})")
    return p, a, b


if __name__ == "__main__":
    main()
