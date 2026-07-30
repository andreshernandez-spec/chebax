"""Differentiable quantile functions: betaincinv, gammaincinv, stdtr, stdtrit.

These answer jax#2399, jax#5350 and jax#20358: inverse regularized
incomplete beta and gamma (the Beta and Gamma quantile functions), plus the
Student-t CDF and quantile built on them. All parameters are traceable
(uniform per call), so jax.grad works with respect to every argument —
which is what implicit-reparameterization gradients and copula models need.

Solvers are fixed-count safeguarded Newton (bisection fallback keeps a
bracket, so the iteration is branchless and jittable). Gradients do NOT
differentiate through the iteration: each inverse carries a custom_jvp from
the implicit function theorem,

    dx*/dp = 1 / pdf(x*),    dx*/dtheta = -(dCDF/dtheta)(x*) / pdf(x*),

with dCDF/da, dCDF/db coming from betainc_fn's traced gradients (or jax's
igamma_grad_a primitive for gamma), all evaluated at the solution.

Accuracy contract: the inverse is CDF-accurate — |CDF(x_hat) - p| at the
eps level. For extreme lower tails with a < 1 (quantiles below ~1e-12) the
bisection bracket cannot resolve x and the result saturates; typical
sampling/copula p ranges are unaffected. betaincinv needs (a, b) inside the
betainc table domain [0.1, 10]; stdtr/stdtrit therefore cover nu in
[0.2, 20]. gammaincinv uses jax's own gammainc and has no table domain.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import betainc_table as _bt
from chebax._src.recipes._common import newton_bisect as _newton_bisect
from chebax._src.recipes.betainc import (betainc_fn, eval_betainc,
                                         tensor_coefs_traced)
from chebax._src.series import ChebSeries

_eval_betainc = eval_betainc
_tensor_coefs_traced = tensor_coefs_traced


def _ln_beta(a, b):
    return (jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)
            - jax.scipy.special.gammaln(a + b))


@jax.custom_jvp
def betaincinv(a, b, p):
    """Inverse of betainc in x: the Beta(a, b) quantile function.

    a, b are (traceable) scalars in [0.1, 10], uniform per call; p any shape."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    p = jnp.asarray(p)
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    cab = ChebSeries(_tensor_coefs_traced(a, b), (0.0, _bt.XSPLIT))
    cba = ChebSeries(_tensor_coefs_traced(b, a), (0.0, _bt.XSPLIT))
    ln_b = _ln_beta(a, b)

    # Solve every element as a LOWER-tail problem of the mirrored orientation
    # (1 - p is exact for p >= 1/2): sigmoid saturates at 1 - eps/2, so upper
    # quantiles must be found as 1 - (small mirrored quantile). Quantiles
    # closer to 1 than eps still round to 1.0 on return - a representation
    # limit shared with scipy; use betaincinv(b, a, 1-p) for the distance
    # from 1. Solving in logit space keeps tail quantiles down to ~1e-300
    # resolvable, and df/du = pdf * x * (1-x) simplifies to a stable exp.
    swap = pc > 0.5
    ps2 = jnp.where(swap, 1.0 - pc, pc)

    def f_and_df(u):
        x = jax.nn.sigmoid(u)
        f = jnp.where(swap,
                      _eval_betainc(b, a, x, cba, cab),
                      _eval_betainc(a, b, x, cab, cba)) - ps2
        ls, l1s = -jax.nn.softplus(-u), -jax.nn.softplus(u)
        dfdu = jnp.exp(jnp.where(swap, b, a) * ls
                       + jnp.where(swap, a, b) * l1s - ln_b)
        return f, dfdu

    # 64 iterations: for min(a, b) <= 1/2 the log-space Newton rate |1 - 1/a|
    # is ~1, so those corners converge at bracket-shrinking speed
    u0 = jnp.where(swap, jnp.log(b / a), jnp.log(a / b)) + jnp.zeros_like(pc)
    u = _newton_bisect(f_and_df, u0, jnp.full_like(pc, -745.0),
                       jnp.full_like(pc, 745.0), 64)
    x_low = jax.nn.sigmoid(u)
    x = jnp.where(swap, 1.0 - x_low, x_low)
    return jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, 1.0, x))


@betaincinv.defjvp
def _betaincinv_jvp(primals, tangents):
    a, b, p = primals
    da, db, dp = tangents
    x = betaincinv(a, b, p)
    interior = (x > 0.0) & (x < 1.0)
    xs = jnp.where(interior, x, 0.5)
    pdf = jnp.exp((jnp.asarray(a) - 1) * jnp.log(xs)
                  + (jnp.asarray(b) - 1) * jnp.log1p(-xs) - _ln_beta(a, b))
    _, d_cdf = jax.jvp(lambda aa, bb: betainc_fn(aa, bb, xs), (a, b), (da, db))
    dx = (jnp.asarray(dp) - d_cdf) / pdf
    return x, jnp.where(interior, dx, 0.0)


@jax.custom_jvp
def gammaincinv(a, p):
    """Inverse of jax.scipy.special.gammainc in x: the Gamma(a, 1) quantile.

    a is a (traceable) positive scalar, uniform per call; p any shape. Needs
    no chebax tables (jax's own gammainc supplies values and the a-gradient)."""
    a = jnp.asarray(a)
    p = jnp.asarray(p)
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    lga = jax.scipy.special.gammaln(a)

    # solve in log space: x = e^v, so small-a tail quantiles stay resolvable
    # and df/dv = pdf * x = exp(a v - e^v - lgamma(a))
    def f_and_df(v):
        x = jnp.exp(v)
        f = jax.scipy.special.gammainc(a, x) - pc
        dfdv = jnp.exp(a * v - x - lga)
        return f, dfdv

    # Wilson-Hilferty initialization, with the exact small-x asymptotic
    # P(a, x) ~ x^a / Gamma(a+1) as the fallback when WH goes nonpositive
    z = jax.scipy.special.ndtri(pc)
    wh = a * (1.0 - 1.0 / (9.0 * a) + z / (3.0 * jnp.sqrt(a))) ** 3
    v0 = jnp.where(wh > 1e-300, jnp.log(jnp.maximum(wh, 1e-300)),
                   (jnp.log(pc) + jax.scipy.special.gammaln(a + 1.0)) / a)
    v = _newton_bisect(f_and_df, v0, jnp.full_like(pc, -745.0),
                       jnp.full_like(pc, 709.0), 40)
    return jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, jnp.inf, jnp.exp(v)))


@gammaincinv.defjvp
def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    x = gammaincinv(a, p)
    interior = jnp.isfinite(x) & (x > 0.0)
    xs = jnp.where(interior, x, 1.0)
    pdf = jnp.exp((jnp.asarray(a) - 1) * jnp.log(xs) - xs
                  - jax.scipy.special.gammaln(a))
    _, d_cdf = jax.jvp(lambda aa: jax.scipy.special.gammainc(aa, xs), (a,), (da,))
    dx = (jnp.asarray(dp) - d_cdf) / pdf
    return x, jnp.where(interior, dx, 0.0)


def stdtr(nu, t):
    """Student-t CDF with nu degrees of freedom (nu in [0.2, 20], traceable).

    jax.grad works with respect to nu as well: learnable degrees of freedom."""
    nu = jnp.asarray(nu)
    t = jnp.asarray(t)
    x = nu / (nu + t * t)
    half_tail = 0.5 * betainc_fn(0.5 * nu, jnp.asarray(0.5), x)
    return jnp.where(t >= 0.0, 1.0 - half_tail, half_tail)


def stdtrit(nu, p):
    """Student-t quantile with nu degrees of freedom (nu in [0.2, 20])."""
    nu = jnp.asarray(nu)
    p = jnp.asarray(p)
    tail2 = jnp.where(p < 0.5, 2.0 * p, 2.0 * (1.0 - p))
    x = betaincinv(0.5 * nu, jnp.asarray(0.5), tail2)
    mag = jnp.sqrt(nu * (1.0 - x) / jnp.maximum(x, 1e-300))
    return jnp.where(p < 0.5, -mag, mag)
