"""The betainc kernel without its tables: evaluation from x-series, the
inverse solve, ln B, and the Student-t density.

betainc, betaincinv and stdtr all run on the same kernel,

    I_x(a, b) = exp( a ln x + b ln(1-x) - ln a - ln B(a,b) + L(a,b,x) )

for x <= XSPLIT and the reflection I_x(a,b) = 1 - I_{1-x}(b,a) above it,
but on different tables: betainc's 3-D tensor panels, stdtr's slices at
a = 1/2. The table-free part lives here so that each family loads only its
own tables; stdtr importing betainc used to load the 15 MB tensor for a
Student-t call.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import canon_tag, newton_bisect, solver_limits

# Every table of this kernel is fitted on x in (0, XSPLIT]; the reflection
# needs 1 - x exact above it, which holds from 1/2. test_betainc checks the
# tables agree.
XSPLIT = 0.5


def ln_beta(a, b):
    return (
        jax.scipy.special.gammaln(a)
        + jax.scipy.special.gammaln(b)
        - jax.scipy.special.gammaln(a + b)
    )


def log_direct(a, b, x, lser):
    """ln I_x(a,b) for x in (0, 1/2], from the direct series."""
    return a * jnp.log(x) + b * jnp.log1p(-x) - jnp.log(a) - ln_beta(a, b) + lser(x)


def eval_betainc(a, b, x, cab, cba):
    x = jnp.asarray(x)
    interior = (x > 0.0) & (x < 1.0)
    xi = jnp.where(interior, x, 0.25)  # masked lanes get a safe dummy
    xd = jnp.where(xi <= XSPLIT, xi, 0.25)
    yd = jnp.where(xi <= XSPLIT, 0.25, 1.0 - xi)  # exact for x in [1/2, 1]
    direct = jnp.exp(log_direct(a, b, xd, cab))
    reflected = 1.0 - jnp.exp(log_direct(b, a, yd, cba))
    core = jnp.where(xi <= XSPLIT, direct, reflected)
    out = jnp.where(x <= 0.0, 0.0, jnp.where(x >= 1.0, 1.0, core))
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)


def betaincinv_core(a, b, p, cab, cba):
    """The betaincinv solve, series-agnostic: cab/cba are the x-series of
    the betainc kernel at (a, b) and (b, a), from the 3-D tensor (public
    betaincinv) or the stdtr slice tables (stdtrit). Not differentiable
    on its own; callers wrap it in a custom_jvp."""
    lim = solver_limits(canon_tag())
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    ln_b = ln_beta(a, b)

    # Solve every element as a LOWER-tail problem of the mirrored orientation
    # (1 - p is exact for p >= 1/2), in logit space so tail quantiles resolve
    # down to the dtype's floor; df/du = pdf * x * (1-x) is a stable exp.
    swap = pc > 0.5
    ps2 = jnp.where(swap, 1.0 - pc, pc)
    aa = jnp.where(swap, b, a)

    def f_and_df(u):
        x = jax.nn.sigmoid(u)
        f = (
            jnp.where(
                swap, eval_betainc(b, a, x, cba, cab), eval_betainc(a, b, x, cab, cba)
            )
            - ps2
        )
        ls, l1s = -jax.nn.softplus(-u), -jax.nn.softplus(u)
        dfdu = jnp.exp(jnp.where(swap, b, a) * ls + jnp.where(swap, a, b) * l1s - ln_b)
        return f, dfdu

    # Initialization: the generic start converges at ~1/a additive steps in
    # u, which cannot reach deep tails in a fixed count (measured: a = 1,
    # p = 1e-50 stalled at e^-64). In the tail the exact leading asymptotic
    # I_x(a,b) = x^a / (a B(a,b)) (1 + O(x)) inverts to a start within
    # O(x) relatively, and Newton converges quadratically from there.
    u_generic = jnp.where(swap, jnp.log(b / a), jnp.log(a / b)) + jnp.zeros_like(pc)
    u_asym = (jnp.log(ps2) + jnp.log(aa) + ln_b) / aa
    u0 = jnp.clip(jnp.where(ps2 < 0.01, u_asym, u_generic), lim.lo, lim.hi)
    u = newton_bisect(
        f_and_df, u0, jnp.full_like(pc, lim.lo), jnp.full_like(pc, lim.hi), lim.n_beta
    )

    # Convergence is checked, never assumed. A root at or below the bracket
    # floor underflows the dtype: return the mirrored endpoint. That is
    # read off the residual AT the floor, which is the smallest normal x;
    # a half-unit band above it in u zeroed normal quantiles, as in
    # betaincinv(1, 1, 4e-308) (review, 2026-10-26). Any other unconverged
    # element returns nan. The bar stays relative to the mass being
    # resolved: past the kernel's x = 1/2 split I comes back as
    # 1 - (the reflected orientation) and is only absolutely accurate, so
    # a target far below that absolute error is genuinely unresolvable and
    # nan is the answer (float32 hits this through stdtrit at large nu).
    f_fin, _ = f_and_df(u)
    f_lo, _ = f_and_df(jnp.full_like(u, lim.lo))
    at_floor = f_lo > 0.0
    x_low = jnp.where(at_floor, 0.0, jax.nn.sigmoid(u))
    bad = ~at_floor & (jnp.abs(f_fin) > lim.rtol * ps2)
    x = jnp.where(swap, 1.0 - x_low, x_low)
    x = jnp.where(bad, jnp.nan, x)
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, 1.0, x))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(a) | jnp.isnan(b)
    return jnp.where(oob, jnp.nan, x)


def t_logpdf(nu, t):
    # ln(1 + t^2/nu) from log|t| where t^2 would overflow (|t| ~ 1.3e154 in
    # float64, 1.8e19 in float32), where it is 2 ln|t| - ln nu to hundreds
    # of digits. The direct form is kept elsewhere: it is the one with the
    # exact zero t-derivative at t = 0, which second derivatives need.
    t2 = t * t
    ok = jnp.isfinite(t2)
    l1 = jnp.where(
        ok,
        jnp.log1p(jnp.where(ok, t2, 0.0) / nu),
        2.0 * jnp.log(jnp.where(ok, 1.0, jnp.abs(t))) - jnp.log(nu),
    )
    return (
        jax.scipy.special.gammaln(0.5 * (nu + 1.0))
        - jax.scipy.special.gammaln(0.5 * nu)
        - 0.5 * jnp.log(nu * jnp.pi)
        - 0.5 * (nu + 1.0) * l1
    )
