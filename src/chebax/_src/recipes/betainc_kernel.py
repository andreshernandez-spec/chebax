"""The betainc kernel without tables (x-series evaluation, inverse solve, ln B,
Student-t density), shared by betainc, betaincinv and stdtr so each loads only
its own tables (betainc's tensor is 15 MB).
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import canon_tag, newton_bisect, solver_limits
from chebax._src.series import chebval_either

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


def _split(x):
    interior = (x > 0.0) & (x < 1.0)
    xi = jnp.where(interior, x, 0.25)  # masked lanes get a safe dummy
    lower = xi <= XSPLIT
    xd = jnp.where(lower, xi, 0.25)
    yd = jnp.where(lower, 0.25, 1.0 - xi)  # exact for x in [1/2, 1]
    return lower, xd, yd


def _finish(x, a, b, core, lo, hi):
    out = jnp.where(x <= 0.0, lo, jnp.where(x >= 1.0, hi, core))
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)


def _lane_parts(a, b, x, cab, cba, dcab, dcba, need):
    """Per lane, the orientation it needs: ln I_t(p, q) with (p, q, t) =
    (a, b, x) below XSPLIT and (b, a, 1 - x) above, and its derivatives in a,
    b and t as values (None where need = (need_a, need_b, need_x) says no).
    One series per point (chebval_either), so the kernel runs one Clenshaw
    per value and per derivative instead of one per orientation. dcab =
    (d cab/da, d cab/db), dcba = (d cba/db, d cba/da). Values match the
    two-series form bit for bit (the padded zeros leave the recurrence
    untouched). Returns lower, L, La, Lb, Lt."""
    na, nb, nx = need
    lower, xd, yd = _split(x)
    t = jnp.where(lower, xd, yd)
    p, q = jnp.where(lower, a, b), jnp.where(lower, b, a)
    lt, l1t = jnp.log(t), jnp.log1p(-t)
    ls = chebval_either(lower, cab, xd, cba, yd)
    L = p * lt + q * l1t - jnp.where(lower, jnp.log(a), jnp.log(b)) - ln_beta(a, b) + ls
    psi = jax.scipy.special.digamma
    if na or nb:
        psi_ab = psi(a + b)
        # d/d(first) and d/d(second) of ln I_t(p, q), scalar parts selected
        d1 = lt - jnp.where(lower, 1.0 / a + psi(a), 1.0 / b + psi(b)) + psi_ab
        d2 = l1t - jnp.where(lower, psi(b), psi(a)) + psi_ab
    La = (
        jnp.where(lower, d1, d2) + chebval_either(lower, dcab[0], xd, dcba[1], yd)
        if na
        else None
    )
    Lb = (
        jnp.where(lower, d2, d1) + chebval_either(lower, dcab[1], xd, dcba[0], yd)
        if nb
        else None
    )
    Lt = (
        p / t - q / (1.0 - t) + chebval_either(lower, cab.deriv(), xd, cba.deriv(), yd)
        if nx
        else None
    )
    return lower, L, La, Lb, Lt


def _times(u, v):
    return None if u is None else u * v


def _pack(x, a, b, lo, hi, value, derivs):
    out = [_finish(x, a, b, value, lo, hi)]
    for d in derivs:
        out.append(None if d is None else _finish(x, a, b, d, 0.0, 0.0))
    return tuple(out)


def eval_betainc_parts(a, b, x, cab, cba, dcab, dcba, need):
    """I_x(a, b) and dI/da, dI/db, dI/dx as values, None where need =
    (need_a, need_b, need_x) says no. dcab = (d cab/da, d cab/db) and dcba =
    (d cba/db, d cba/da): each orientation's coefficient derivatives in its
    own first and second parameter."""
    x = jnp.asarray(x)
    na, nb, nx = need
    lower, L, La, Lb, _ = _lane_parts(a, b, x, cab, cba, dcab, dcba, (na, nb, 0))
    e = jnp.exp(L)
    sgn = jnp.where(lower, e, -e)  # I = e below, 1 - e above; dy/dx = -1
    # dI/dx is the density, from its logs: e * d ln I/dt is 0 * inf where I
    # underflows (float32, a = 100 at x = 1e-37), and above XSPLIT e carries
    # I's absolute error into the derivative
    dens = None
    if nx:
        _, xd, yd = _split(x)
        lx = jnp.where(lower, jnp.log(xd), jnp.log1p(-yd))
        l1x = jnp.where(lower, jnp.log1p(-xd), jnp.log(yd))
        dens = jnp.exp((a - 1.0) * lx + (b - 1.0) * l1x - ln_beta(a, b))
    return _pack(
        x,
        a,
        b,
        0.0,
        1.0,
        jnp.where(lower, e, 1.0 - e),
        (_times(La, sgn), _times(Lb, sgn), dens),
    )


def eval_log_betainc_parts(a, b, x, cab, cba, dcab, dcba, need):
    """ln I_x(a, b) and its derivatives as values, like eval_betainc_parts."""
    x = jnp.asarray(x)
    lower, L, La, Lb, Lt = _lane_parts(a, b, x, cab, cba, dcab, dcba, need)
    e = jnp.exp(L)
    s = -e / (1.0 - e)  # d log1p(-e) / dL above XSPLIT
    f = jnp.where(lower, 1.0, s)
    return _pack(
        x,
        a,
        b,
        -jnp.inf,
        0.0,
        jnp.where(lower, L, jnp.log1p(-e)),
        (_times(La, f), _times(Lb, f), _times(Lt, jnp.where(lower, 1.0, -s))),
    )


def eval_betainc(a, b, x, cab, cba):
    """I_x(a, b) from the two orientations' series, one per point."""
    return eval_betainc_parts(a, b, x, cab, cba, (None, None), (None, None), (0, 0, 0))[
        0
    ]


def eval_betainc_frozen(a, b, x, cab, cba):
    """I_x(a, b) with both orientations' series per element, for the frozen
    recipe: bake folds _chebval's custom_jvp into its loop and nothing else."""
    x = jnp.asarray(x)
    lower, xd, yd = _split(x)
    direct = jnp.exp(log_direct(a, b, xd, cab))
    reflected = 1.0 - jnp.exp(log_direct(b, a, yd, cba))
    return _finish(x, a, b, jnp.where(lower, direct, reflected), 0.0, 1.0)


def eval_log_betainc(a, b, x, cab, cba):
    """ln I_x(a, b), no underflow floor below XSPLIT; log1p of the reflection
    above errs ~eps/I."""
    return eval_log_betainc_parts(
        a, b, x, cab, cba, (None, None), (None, None), (0, 0, 0)
    )[0]


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
