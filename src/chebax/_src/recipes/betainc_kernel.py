"""The betainc kernel without tables (x-series evaluation, inverse solve, ln B,
Student-t density), shared by betainc, betaincinv and stdtr so each loads only
its own tables (betainc's tensor is 15 MB).
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import canon_tag, newton_bisect, solver_limits

# every table is fitted on (0, XSPLIT]; 1 - x is exact from 1/2
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
    """The betaincinv solve on given x-series cab, cba at (a, b) and (b, a).
    Not differentiable; callers wrap it in a custom_jvp."""
    lim = solver_limits(canon_tag())
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    ln_b = ln_beta(a, b)

    # lower-tail solve of the mirrored orientation (1 - p exact for p >= 1/2),
    # in logit space so tails resolve to the dtype's floor
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

    # tail start from I_x ~ x^a / (a B(a,b)): the generic one moves ~1/a per step
    # in u (a = 1, p = 1e-50 stalled at e^-64)
    u_generic = jnp.where(swap, jnp.log(b / a), jnp.log(a / b)) + jnp.zeros_like(pc)
    u_asym = (jnp.log(ps2) + jnp.log(aa) + ln_b) / aa
    u0 = jnp.clip(jnp.where(ps2 < 0.01, u_asym, u_generic), lim.lo, lim.hi)
    u = newton_bisect(
        f_and_df, u0, jnp.full_like(pc, lim.lo), jnp.full_like(pc, lim.hi), lim.n_beta
    )

    # checked: a positive residual at the floor (smallest normal x) is an
    # underflowing root and gives the endpoint; other unconverged lanes give nan,
    # e.g. targets below the reflected side's absolute error (float32 stdtrit at
    # large nu)
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
    # 2 ln|t| - ln nu where t^2 overflows; log1p elsewhere keeps the exact zero
    # t-derivative at t = 0 that second derivatives need
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
