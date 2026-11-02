"""Gamma quantiles with one a per call: gammaincinv_scalar and
gammainccinv_scalar (jax#5350; the entry points are in gammaincinv.py).
The Beta quantile is in betaincinv.py, on the same solver design and the
same per-dtype constants (_common.solver_limits).

a is traceable (uniform per call), so jax.grad works with respect to
every argument, which is what implicit-reparameterization gradients and
copula models need.

The solve (safeguarded Newton in t = ln(x/a)), its starts and its
acceptance test are gammaincinv_kernel's, shared with gammaincinv_array; this module picks the
ln P and ln Q it runs on (below) and owns the derivatives. Gradients do
NOT differentiate through the iteration: each inverse carries a
custom_jvp from the implicit function theorem,

    dx*/dp = 1 / pdf(x*),    dx*/da = -(dP/da)(x*) / pdf(x*),

with dP/da from the chebax recipe (or jax's igamma_grad_a primitive
outside it), evaluated at the solution.

Both inverses solve the half that carries the tail mass, never the half
that rounds against 1: gammaincinv targets Q(a, x) for p > 1/2. 1 - p is
exact for p >= 1/2, so the mirrored target costs nothing.

Guarantees instead of silent saturation: every solve checks its final CDF
residual, relative to min(p, 1-p) (the mass actually being resolved), or
that x is the root to within one float step. A quantile that underflows
the canonical dtype returns 0.0 (the scipy convention); anything else
that failed to converge returns nan. NaN inputs
propagate to nan; p outside [0, 1] returns nan; a shape that is not finite
and positive returns nan for every p; the p = 0 and 1 endpoints return the
exact distribution endpoints.

gammaincinv has no table domain: its residual evaluates the chebax gammainc recipe when a is at least 0.1
(above 10 via the Temme-zone tables; fixed-degree polynomials where
jax's gammainc re-runs a whole-array
while_loop per step) and falls back to jax's gammainc
below, picked by nested lax.conds on the scalar a. Above a = 1e10 the
solve runs in Temme's variable instead (_huge_solve). Inputs are
computed in the canonical float dtype (float64 under x64), so explicit
float32 arguments are promoted, not crashed on, and the solver constants
follow that dtype (see _common.solver_limits): float64's log-space
brackets reach to +-708, which is meaningless in float32, where exp
saturates at +-88.
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes import gammaincinv_kernel as _k
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
    """dP(a, x)/da through the gammainc tables: one polynomial where
    jax's igamma_grad_a runs a looped series. a inside [0.1, 10] only
    (the JVP's cond dispatches). Differentiable in x through the tables;
    the a direction is never requested here (the cond traces the
    fallback's NotImplementedError for a-tangents, so d2/da2 keeps its
    clear raise on every path)."""

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
    """dP(a, x)/da (regularized lower gamma), differentiable in x.

    jax's igamma_grad_a primitive has no JVP of its own, which made
    gammaincinv only once differentiable even in p (the second derivative
    reaches this term through x*(p)). The x direction has the closed form
    d/dx dP/da = pdf(a, x) * (ln x - psi(a)); the a direction (d2P/da2)
    has no closed form here and raises."""
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
    """dP/da from whichever path owns this a: the tables inside their
    boxes, jax's igamma_grad_a outside."""
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
    """ln of the Gamma(a, 1) density at x > 0.

    a ln x - x - lnGamma(a) cancels terms of size a ln a, which left the
    inverses' x-derivative 8e-6 off at a = 1e10 (and 4e-7 at 1e8). Past
    a = 100 it is -a w + ln(a / (2 pi))/2 - S(a) - ln x instead, with
    w = lambda - 1 - ln lambda from d = (x - a)/a, an exact subtraction near
    the mode, and S Stirling's remainder to a^-5 (below 1e-17 there)."""
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
# (lambda - 1)/eta = 1 + eta/3 + eta^2/36 - ..., the inverse of
# eta^2/2 = lambda - 1 - ln lambda (sympy reversion), through eta^5: above
# _A_HUGE, |eta| < 4e-4 and the next term is below 1e-25 relative
_LAM_SERIES = (1.0 / 3.0, 1.0 / 36.0, -1.0 / 270.0, 1.0 / 4320.0, 1.0 / 17010.0)


def _huge_parts(a, s, temme):
    """At s = eta sqrt(a/2): (ln P, ln Q, ln dP/ds, lambda - 1).

    dP/ds is x^a e^-x / Gamma(a) times dx/ds, which is
    e^(-s^2) / sqrt(pi) e^(-1/(12 a)) eta / (lambda - 1), Stirling's
    remainder kept to 1/(12 a) (the next term is below 1e-33 here)."""
    # sqrt(2)/sqrt(a): 2/a is subnormal from a = 9e307, flushed to 0, and
    # dx/dp was 0 there
    eta = s * (math.sqrt(2.0) / jnp.sqrt(a))
    r = _LAM_SERIES[-1]
    for c in _LAM_SERIES[-2::-1]:
        r = r * eta + c
    r = 1.0 + eta * r  # (lambda - 1) / eta
    # sqrt(2 pi) sqrt(a): 2 pi a overflows from a = 2.9e307, and dx/dp was
    # 0 there
    rt = math.sqrt(2.0 * math.pi) * jnp.sqrt(a)
    lp, lq = _gl.temme_log_pq(s, s * s, temme(eta), rt)
    lden = -s * s - 0.5 * math.log(math.pi) - 1.0 / (12.0 * a) - jnp.log(r)
    return lp, lq, lden, eta * r


def _huge_solve(a, ltp, ltq, z, upper):
    """The quantile for a > _A_HUGE, solved in s = eta sqrt(a/2).

    There x = a (1 + O(a^-1/2)) and ln x carries eps ln a. From a = 1e15
    the log density a ln x - x - lnGamma(a) cancelled to noise and every
    solve returned nan, and past ~1e30 the floats in ln x are spaced wider
    than the whole distribution (found fixing the review of 2026-10-26).
    In s, which is O(1), Temme's form gives ln P and ln Q directly, the
    density has no cancellation, and x = a + a (lambda - 1) rounds once.
    Returns (x, relative residual, s)."""
    temme = _gl.series_traced(a)[0]

    def f_and_df(s):
        lp, lq, lden, _ = _huge_parts(a, s, temme)
        f = jnp.where(upper, ltq - lq, lp - ltp)
        return f, jnp.exp(lden - jnp.where(upper, lq, lp))

    # erfc(-s)/2 = Phi(s sqrt 2), and the root is within O(s^3 / sqrt(a))
    # of that normal start, under 1e-3 here
    s0 = z / math.sqrt(2.0)
    s = newton_bisect(f_and_df, s0, s0 - 2.0, s0 + 2.0, 8)
    lp, lq, _, dm = _huge_parts(a, s, temme)
    rel = jnp.abs(jnp.expm1(jnp.where(s > 0.0, lq - ltq, lp - ltp)))
    return a + a * dm, rel, s


def _huge_dx(a, da, s, upper, dlog):
    """dx for a > _A_HUGE, implicit in s: with L = ln P on the lower lanes
    and ln Q on the upper ones, dL = L_s ds + L_a da is dlog, the target's
    own log tangent, and dx = X_s ds + X_a da for X = a + a (lambda - 1).
    Everything stays in s and a, where the density in x cancels. da is
    None for a symbolic zero."""

    def lfn(ss, aa):
        lp, lq, _, _ = _huge_parts(aa, ss, _gl.series_traced(aa)[0])
        return jnp.where(upper, lq, lp)

    def xfn(ss, aa):
        return aa + aa * _huge_parts(aa, ss, _gl.series_traced(aa)[0])[3]

    one = jnp.ones_like(s)
    l_s = jax.jvp(lambda ss: lfn(ss, a), (s,), (one,))[1]
    x_s = jax.jvp(lambda ss: xfn(ss, a), (s,), (one,))[1]
    if da is None:
        return x_s * dlog / l_s
    l_a = jax.jvp(lambda aa: lfn(s, aa), (a,), (jnp.ones_like(a),))[1]
    x_a = jax.jvp(lambda aa: xfn(s, aa), (a,), (jnp.ones_like(a),))[1]
    return x_s * (dlog - l_a * da) / l_s + x_a * da


def _gammainv_core(a, ltp, ltq, z, upper, use_recipe, ceiling=False):
    """gammaincinv_kernel.solve on the ln P and ln Q for this a's box.

    use_recipe is a python constant (each lax.cond branch traces one):
    "small" takes both logs from the a <= 10 gammainc tables, "large" from
    the Temme-zone path (a > 10, relatively accurate on both sides
    everywhere), each reconstructed once and closed over, so each step is
    a fixed-degree polynomial; None uses jax's gammainc and gammaincc
    (any a, but a whole-array while_loop per call). "huge" hands the solve
    to _huge_solve, in Temme's variable, which checks its own residual.
    Returns the kernel's (x, accepted, root below the floor, root above
    the ceiling)."""
    lim = solver_limits(canon_tag())
    if use_recipe == "huge":
        x, rel, _ = _huge_solve(a, ltp, ltq, z, upper)
        none = jnp.zeros(x.shape, bool)
        return x, rel <= lim.rtol, none, (x == jnp.inf) if ceiling else none
    # the side each log_pq computes directly: P below the seam, Q above
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

    small = use_recipe != "large"
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


def _gammaincinv_solve(a, pc, use_recipe):
    """Lower-tail entry to _gammainv_core: the target is p, exact on the
    lower lanes, and 1 - p (exact for p >= 1/2) on the upper ones."""
    return _gammainv_core(a, *_k.inv_targets(pc), use_recipe)


def _gammainccinv_solve(a, yc, use_recipe):
    """Upper-tail entry to _gammainv_core: the target is y, exact on the
    upper lanes, and 1 - y (exact for y >= 1/2) on the lower ones.

    Not a call to _gammaincinv_solve at 1 - y: that rounds to exactly 1
    for y below eps, which is the whole range this exists for."""
    return _gammainv_core(a, *_k.cinv_targets(yc), use_recipe, ceiling=True)


def _dispatch_solve(solve, a, tc):
    """Run solve(a, tc, mode) under the a-box conds: the small tables to
    10, the Temme-zone tables above, solved in Temme's variable past
    _A_HUGE, jax's gammainc below 0.1. Nested scalar conds, so only the
    taken branch executes."""
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
        # dP/da through the differentiable wrapper: second derivatives in
        # p and mixed p-a work; only d2/da2 remains unsupported (see
        # _dPda). Symbolic zeros matter here: a materialized 0 tangent
        # would still route the outer derivative through _dPda's a
        # direction.
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


# What gammaincinv.py calls. The custom_jvp rules live on the private
# names above and float_params normalizes every argument first, so an
# integer, keyword or size-one parameter is a float scalar before any
# tangent exists (see its docstring).
gammaincinv = float_params(_gammaincinv_scalar_cj, "gammaincinv_array")
gammainccinv = float_params(_gammainccinv_scalar_cj, "gammainccinv_array")
