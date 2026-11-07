"""gammaincinv_scalar and gammainccinv_scalar (jax#5350): one traceable a per
call, so jax.grad works in every argument. The solve (Newton in t = ln(x/a)),
its starts and acceptance are gammaincinv_kernel's; this module picks the ln P
and ln Q it runs on and owns the derivatives, by the implicit function theorem:

    dx*/dp = 1 / pdf(x*),    dx*/da = -(dP/da)(x*) / pdf(x*).

Each inverse solves the tail-mass half (Q for p > 1/2, where 1 - p is exact)
and checks its residual relative to min(p, 1-p), or that x is a float step
from the root; a failed solve gives nan, an underflowed quantile 0.0. ln P and
ln Q come from the gammainc tables for a in [0.1, 1e10], jax's gammainc below
0.1, and Temme's variable above 1e10 (_huge_solve). Solver constants follow
the canonical dtype (_common.solver_limits): exp saturates at 88 in float32.
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes import gammaincinv_kernel as _k
from chebax._src.recipes import gammaincinv_start as _start
from chebax._src.recipes._common import (
    canon_tag,
    float_params,
    newton_bisect,
    solver_limits,
)
from chebax._src.recipes.gammainc_small import eval_gammainc
from chebax._src.recipes.gammainc_small import eval_logs_one_series
from chebax._src.recipes.gammainc_small import series_traced as _gi_series


def _dPda_recipe(a, x):
    """dP/da through the a in [0.1, 10] tables, differentiable in x. d2/da2
    still raises, through the cond's traced fallback."""

    def p_of_a(t):
        lser, ltail = _gi_series(t)
        return eval_gammainc(t, x, lser, ltail, lower=True)

    return jax.jvp(p_of_a, (a,), (jnp.ones_like(a),))[1]


def _dPda_recipe_large(a, x):
    """dP/da through the Temme-zone tables, a > 10."""

    def p_of_a(t):
        tm, dl, du = _gl.series_traced(t)
        return _gl.eval_large(t, x, tm, dl, du, "p")

    return jax.jvp(p_of_a, (a,), (jnp.ones_like(a),))[1]


@jax.custom_jvp
def _dPda(a, x):
    """jax's igamma_grad_a with a JVP in x, d/dx dP/da = pdf (ln x - psi(a)),
    so second derivatives in p work; d2P/da2 raises."""
    return jax.lax.igamma_grad_a(a, x)


def _dPda_jvp(primals, tangents):
    a, x = primals
    da, dx = tangents
    g = _dPda(a, x)
    zero = jax.custom_derivatives.SymbolicZero
    if not isinstance(da, zero):
        raise NotImplementedError(
            "second derivative of gammaincinv with respect to a needs d2P/da2, "
            "which neither jax nor chebax provides"
        )
    if isinstance(dx, zero):
        return g, jnp.zeros_like(g)
    pdf = jnp.exp((a - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a))
    d = pdf * (jnp.log(x) - jax.scipy.special.digamma(a)) * dx
    return g, d


_dPda.defjvp(_dPda_jvp, symbolic_zeros=True)


def _dPda_dispatch(a, x):
    """dP/da from the tables inside their boxes, igamma_grad_a outside."""
    in_small = (a >= _gt.ALO) & (a <= _gt.AHI)
    in_large = (a > _gt.AHI) & (a <= _gl.AHI)
    return jax.lax.cond(
        in_small,
        lambda: _dPda_recipe(a, x),
        lambda: jax.lax.cond(
            in_large, lambda: _dPda_recipe_large(a, x), lambda: _dPda(a, x)
        ),
    )


def _gamma_log_density(a, x):
    """ln of the Gamma(a, 1) density at x > 0. a ln x - x - lnGamma(a) cancels
    (8e-6 off at a = 1e10), so past a = 100 it is -a w + ln(a / (2 pi))/2 - S(a)
    - ln x, w = lambda - 1 - ln lambda from (x - a)/a, S Stirling's to a^-5."""
    direct = (a - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a)
    big = a > 100.0
    ab = jnp.where(big, a, 1000.0)
    w, _ = _gl.w_eta((x - ab) / ab)
    ai2 = 1.0 / (ab * ab)
    rem = (1.0 / 12.0 - ai2 * (1.0 / 360.0 - ai2 / 1260.0)) / ab
    stable = -ab * w + 0.5 * jnp.log(ab / (2.0 * jnp.pi)) - rem - jnp.log(x)
    return jnp.where(big, stable, direct)


# From here up the gamma inverses solve in Temme's variable (_huge_solve)
_A_HUGE = 1e10
# Newton steps from the quantile table's start inside the tables; the second
# is the check (gammaincinv_kernel.table_solve)
_STEPS = 2
# (lambda - 1)/eta, reverting eta^2/2 = lambda - 1 - ln lambda to eta^5 (next term
# below 1e-25 for |eta| < 4e-4)
_LAM_SERIES = (1.0 / 3.0, 1.0 / 36.0, -1.0 / 270.0, 1.0 / 4320.0, 1.0 / 17010.0)


def _huge_parts(a, s, temme):
    """(ln P, ln Q, ln dP/ds, lambda - 1) at s = eta sqrt(a/2), with
    dP/ds = e^(-s^2) / sqrt(pi) e^(-1/(12 a)) eta / (lambda - 1)."""
    # sqrt(2)/sqrt(a): 2/a is subnormal from a = 9e307 and flushes to 0
    eta = s * (math.sqrt(2.0) / jnp.sqrt(a))
    r = _LAM_SERIES[-1]
    for c in _LAM_SERIES[-2::-1]:
        r = r * eta + c
    r = 1.0 + eta * r  # (lambda - 1) / eta
    # sqrt(2 pi) sqrt(a): 2 pi a overflows from a = 2.9e307
    rt = math.sqrt(2.0 * math.pi) * jnp.sqrt(a)
    lp, lq = _gl.temme_log_pq(s, s * s, temme(eta), rt)
    lden = -s * s - 0.5 * math.log(math.pi) - 1.0 / (12.0 * a) - jnp.log(r)
    return lp, lq, lden, eta * r


def _huge_solve(a, ltp, ltq, z, upper):
    """(x, relative residual, s) for a > _A_HUGE, solved in s = eta sqrt(a/2).
    In x the log density cancels to noise from a = 1e15, and past ~1e30 floats
    in ln x are wider than the distribution; in s, O(1), nothing cancels."""
    temme = _gl.series_traced(a)[0]

    def f_and_df(s):
        lp, lq, lden, _ = _huge_parts(a, s, temme)
        f = jnp.where(upper, ltq - lq, lp - ltp)
        return f, jnp.exp(lden - jnp.where(upper, lq, lp))

    # normal start: the root is within O(s^3 / sqrt(a)) of it, under 1e-3
    s0 = z / math.sqrt(2.0)
    s = newton_bisect(f_and_df, s0, s0 - 2.0, s0 + 2.0, 8)
    lp, lq, _, dm = _huge_parts(a, s, temme)
    rel = jnp.abs(jnp.expm1(jnp.where(s > 0.0, lq - ltq, lp - ltp)))
    return a + a * dm, rel, s


def _huge_dx(a, da, s, upper, dlog):
    """dx for a > _A_HUGE, implicit in s: L_s ds + L_a da = dlog for L = ln P
    (ln Q upper), dx = X_s ds + X_a da for X = a + a (lambda - 1). da None is a
    symbolic zero."""

    def lfn(ss, aa):
        lp, lq, _, _ = _huge_parts(aa, ss, _gl.series_traced(aa)[0])
        return jnp.where(upper, lq, lp)

    def xfn(ss, aa):
        return aa + aa * _huge_parts(aa, ss, _gl.series_traced(aa)[0])[3]

    one = jnp.ones_like(s)
    l_s = jax.jvp(lambda ss: lfn(ss, a), (s,), (one,))[1]
    x_s = jax.jvp(lambda ss: xfn(ss, a), (s,), (one,))[1]
    # divide first: x_s dlog overflows float32 at a = 1e12, p = 1e-33
    ratio = x_s / l_s
    if da is None:
        return ratio * dlog
    l_a = jax.jvp(lambda aa: lfn(s, aa), (a,), (jnp.ones_like(a),))[1]
    x_a = jax.jvp(lambda aa: xfn(s, aa), (a,), (jnp.ones_like(a),))[1]
    return ratio * (dlog - l_a * da) + x_a * da


def _residual(a, use_recipe):
    """(log_pq, seam, small) for this box: use_recipe (a python constant)
    "small" (a <= 10 tables), "large" (Temme zone) or None (jax's gammainc).
    seam is the x above which log_pq computes Q directly (P below); small
    says the kernel can skip its near-mode handling."""
    if use_recipe == "small":
        lser, ltail = _gi_series(a)
        seam = _gt.XS
    elif use_recipe == "large":
        tm, dl, du = _gl.series_traced(a)
        seam = a
    else:
        seam = 1.0 + a

    def log_pq(x):
        if use_recipe == "small":
            lp_in, lq_tl = eval_logs_one_series(a, x, lser, ltail)
            inner = x <= _gt.XS
            lp = jnp.where(inner, lp_in, jnp.log1p(-jnp.exp(lq_tl)))
            lq = jnp.where(inner, jnp.log1p(-jnp.exp(lp_in)), lq_tl)
            pos = x > 0.0
            return jnp.where(pos, lp, -jnp.inf), jnp.where(pos, lq, 0.0)
        if use_recipe == "large":
            return (
                _gl.eval_large(a, x, tm, dl, du, "logp"),
                _gl.eval_large(a, x, tm, dl, du, "logq"),
            )
        return (
            jnp.log(jax.scipy.special.gammainc(a, x)),
            jnp.log(jax.scipy.special.gammaincc(a, x)),
        )

    return log_pq, seam, use_recipe != "large"


def _gammainv_core(a, ltp, ltq, z, upper, use_recipe, ceiling=False):
    """The solve on this box's ln P and ln Q: use_recipe "huge" is
    _huge_solve; the others (see _residual) start from the quantile table
    and take _STEPS Newton steps, except jax's gammainc below a = 0.1, which
    the table does not cover and gets the kernel's full solve. Returns
    (x, accepted, below floor, above ceiling)."""
    lim = solver_limits(canon_tag())
    if use_recipe == "huge":
        x, rel, _ = _huge_solve(a, ltp, ltq, z, upper)
        none = jnp.zeros(x.shape, bool)
        return x, rel <= lim.rtol, none, (x == jnp.inf) if ceiling else none
    log_pq, seam, small = _residual(a, use_recipe)
    if use_recipe is None:
        return _k.solve(
            a,
            ltp,
            ltq,
            z,
            upper,
            log_pq,
            lim.n_gamma,
            ceiling=ceiling,
            seam=seam,
            small=small,
        )
    t0 = _start.start(a, ltp, ltq, z, upper)
    return _k.table_solve(
        a,
        ltp,
        ltq,
        z,
        upper,
        log_pq,
        t0,
        _STEPS,
        ceiling=ceiling,
        seam=seam,
        small=small,
    )


def _gammaincinv_solve(a, pc, use_recipe):
    """_gammainv_core targeting p, or 1 - p (exact) on the upper lanes."""
    return _gammainv_core(a, *_k.inv_targets(pc), use_recipe)


def _gammainccinv_solve(a, yc, use_recipe):
    """_gammainv_core targeting y, or 1 - y on the lower lanes; 1 - y alone
    would round to 1 below eps."""
    return _gammainv_core(a, *_k.cinv_targets(yc), use_recipe, ceiling=True)


def _dispatch_solve(solve, a, tc):
    """solve(a, tc, mode) under nested scalar conds on a's box."""
    in_small = (a >= _gt.ALO) & (a <= _gt.AHI)
    in_large = (a > _gt.AHI) & (a <= _A_HUGE)
    in_huge = (a > _A_HUGE) & (a <= _gl.AHI)
    return jax.lax.cond(
        in_small,
        lambda: solve(a, tc, "small"),
        lambda: jax.lax.cond(
            in_large,
            lambda: solve(a, tc, "large"),
            lambda: jax.lax.cond(
                in_huge, lambda: solve(a, tc, "huge"), lambda: solve(a, tc, None)
            ),
        ),
    )


@jax.custom_jvp
def _gammaincinv_scalar_cj(a, p):
    """x with P(a, x) = p, for one a; contract on gammaincinv_scalar."""
    pc = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
    x, ok, below, _ = _dispatch_solve(_gammaincinv_solve, a, pc)
    return _k.finish_inv(p, x, ok, below, ~((a > 0.0) & jnp.isfinite(a)))


def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = _gammaincinv_scalar_cj(a, p)
    interior = jnp.isfinite(x) & (x > 0.0)

    def in_x():
        xs = jnp.where(interior, x, 1.0)
        pdf = jnp.exp(_gamma_log_density(a, xs))
        # symbolic zeros: a materialized 0 da would still reach _dPda's raise
        num = dp if not isinstance(dp, zero) else jnp.zeros_like(xs)
        if not isinstance(da, zero):
            num = num - _dPda_dispatch(a, xs) * da
        return num / pdf

    def in_s():
        pc = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
        targets = _k.inv_targets(pc)
        _, _, s = _huge_solve(a, *targets)
        dpv = dp if not isinstance(dp, zero) else jnp.zeros_like(pc)
        dlog = jnp.where(targets[3], -dpv / (1.0 - pc), dpv / pc)
        return _huge_dx(a, None if isinstance(da, zero) else da, s, targets[3], dlog)

    dx = jax.lax.cond(a > _A_HUGE, in_s, in_x)
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


_gammaincinv_scalar_cj.defjvp(_gammaincinv_jvp, symbolic_zeros=True)


@jax.custom_jvp
def _gammainccinv_scalar_cj(a, y):
    """x with Q(a, x) = y, for one a; contract on gammainccinv_scalar."""
    yc = jnp.where((y > 0.0) & (y < 1.0), y, 0.5)
    x, ok, below, above = _dispatch_solve(_gammainccinv_solve, a, yc)
    return _k.finish_cinv(y, x, ok, below, above, ~((a > 0.0) & jnp.isfinite(a)))


def _gammainccinv_jvp(primals, tangents):
    a, y = primals
    da, dy = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = _gammainccinv_scalar_cj(a, y)
    interior = jnp.isfinite(x) & (x > 0.0)

    def in_x():
        xs = jnp.where(interior, x, 1.0)
        pdf = jnp.exp(_gamma_log_density(a, xs))
        # Q(a, x) = y, so dx = -(dy + dP/da da) / pdf
        num = dy if not isinstance(dy, zero) else jnp.zeros_like(xs)
        if not isinstance(da, zero):
            num = num + _dPda_dispatch(a, xs) * da
        return -num / pdf

    def in_s():
        yc = jnp.where((y > 0.0) & (y < 1.0), y, 0.5)
        targets = _k.cinv_targets(yc)
        _, _, s = _huge_solve(a, *targets)
        dyv = dy if not isinstance(dy, zero) else jnp.zeros_like(yc)
        dlog = jnp.where(targets[3], dyv / yc, -dyv / (1.0 - yc))
        return _huge_dx(a, None if isinstance(da, zero) else da, s, targets[3], dlog)

    dx = jax.lax.cond(a > _A_HUGE, in_s, in_x)
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


_gammainccinv_scalar_cj.defjvp(_gammainccinv_jvp, symbolic_zeros=True)


# float_params makes every argument a float scalar before any tangent exists
gammaincinv = float_params(_gammaincinv_scalar_cj, "gammaincinv_array")
gammainccinv = float_params(_gammainccinv_scalar_cj, "gammainccinv_array")
