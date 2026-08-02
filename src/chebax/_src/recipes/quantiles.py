"""Differentiable quantile functions: betaincinv, gammaincinv, stdtr, stdtrit.

These answer jax#2399, jax#5350 and jax#20358: inverse regularized
incomplete beta and gamma (the Beta and Gamma quantile functions), plus the
Student-t CDF, log-CDF and quantile built on them. All parameters are
traceable (uniform per call), so jax.grad works with respect to every
argument, which is what implicit-reparameterization gradients and copula
models need.

The beta and gamma solvers are fixed-count safeguarded Newton (bisection
fallback keeps a bracket, so the iteration is branchless and jittable),
initialized in the deep lower tail from the exact leading asymptotics
(I_x ~ x^a / (a B(a,b)), P(a,x) ~ x^a / Gamma(a+1)); Newton from those
starts converges quadratically, where the generic start needs O(|log p|/a)
additive steps and used to saturate silently. gammaincinv iterates on the
LOG probability, which is nearly linear in log x wherever the probability
is exponentially small, so a large shape in a deep tail converges in a
couple of steps instead of one e-fold per step. Gradients do NOT
differentiate through the iteration: each inverse carries a custom_jvp
from the implicit function theorem,

    dx*/dp = 1 / pdf(x*),    dx*/dtheta = -(dCDF/dtheta)(x*) / pdf(x*),

with dCDF/da, dCDF/db coming from betainc_fn's traced gradients (or jax's
igamma_grad_a primitive for gamma), all evaluated at the solution.

Both inverses solve the half that carries the tail mass, never the half
that rounds against 1: betaincinv mirrors on p > 1/2, gammaincinv targets
Q(a, x) there. 1 - p is exact for p >= 1/2, so the mirrored target costs
nothing.

Guarantees instead of silent saturation: every solve checks its final CDF
residual, relative to min(p, 1-p) (the mass actually being resolved). A
quantile that underflows the canonical dtype returns 0.0 (the scipy
convention); anything else that failed to converge returns nan. NaN inputs
propagate to nan; p outside [0, 1] returns nan; a shape parameter that is
not finite and positive returns nan for every p; the p = 0 and 1 endpoints
return the exact distribution endpoints.

stdtr/stdtrit need no solver of their own: both orientations of the
incomplete-beta reduction are used, t^2/(nu+t^2) near the median (exact
first-order behavior through t = 0, where the classic nu/(nu+t^2) form
rounds to 1 and loses everything) and nu/(nu+t^2) in the tails. The tail
orientation is evaluated from log|t| rather than from t^2, so it holds for
any t, including |t| past the point where t^2 overflows. Their JVPs use
the exact Student-t density. log_stdtr/log_stdtr_sf are the log-CDF and
log survival function, accurate where the probability itself underflows.
betaincinv needs (a, b) inside the betainc table domain [0.1, 100].
stdtr/stdtrit run on their own 2-D slice tables of the same kernel at
a = 1/2 (b-panels to 100), so they cover nu in [0.2, 200] without the
3-D tensor. gammaincinv has no table domain: its Newton residual
evaluates the chebax gammainc recipe when a is inside [0.1, 1000]
((10, 1000] via the Temme-zone tables; fixed-degree polynomials where
jax's gammainc re-runs a whole-array
while_loop per Newton iteration) and falls back to jax's gammainc
outside, picked by nested lax.conds on the scalar a. Inputs are
computed in the canonical float dtype (float64 under x64), so explicit
float32 arguments are promoted, not crashed on, and the solver constants
follow that dtype (see _limits): float64's log-space brackets reach to
+-708, which is meaningless in float32, where exp saturates at +-88.
"""

import functools
import math
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src.recipes import betainc_table as _bt
from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_large_table as _glt
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes import stdtr_table as _st
from chebax._src.recipes._common import canon_float as _canon
from chebax._src.recipes._common import canon_tag as _canon_tag
from chebax._src.recipes._common import newton_bisect as _newton_bisect
from chebax._src.recipes._common import traced_coefs
from chebax._src.recipes.betainc import _log_direct as _log_betainc_direct
from chebax._src.recipes.betainc import (betainc_fn, eval_betainc,
                                         tensor_coefs_traced,
                                         with_panel_series)
from chebax._src.recipes.gammainc import _eval_logs as _gi_logs
from chebax._src.recipes.gammainc import _traced_series as _gi_series
from chebax._src.recipes.gammainc import eval_gammainc as _eval_gammainc
from chebax._src.series import ChebSeries

_eval_betainc = eval_betainc
_tensor_coefs_traced = tensor_coefs_traced

_LN2 = math.log(2.0)

_Limits = namedtuple("_Limits", "lo hi v_hi n_beta n_gamma rtol")


@functools.lru_cache(maxsize=4)
def _limits(_tag):
    """Solver constants for the canonical float dtype, keyed per x64 mode.

    The log-space bracket is that dtype's own NORMAL exponent range. Below
    log(tiny) exp and sigmoid both flush to zero, so nothing there is
    reachable, and putting the floor at the last reachable point is what
    makes the sign of the residual there decide "root below the floor"
    instead of always reading as "not reached yet". float64's 709 ceiling
    falls out of the same rule; its old -745 floor sat 37 e-folds inside
    the flushed region, and both were 700 decades outside float32's range,
    which left every float32 solve unbracketed. Iteration counts and the
    residual bar are measured per dtype (tests/test_quantiles.py)."""
    f = np.finfo(np.dtype(_tag))
    lo = math.ceil(math.log(float(f.tiny)))
    wide = np.dtype(_tag) == np.float64
    return _Limits(lo, -lo, math.floor(math.log(float(f.max))),
                   64 if wide else 48, 40 if wide else 24,
                   1e-6 if wide else 1e-3)


def _ln_beta(a, b):
    return (jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)
            - jax.scipy.special.gammaln(a + b))


def _dPda_recipe(a, x):
    """dP(a, x)/da through the gammainc tables: one polynomial where
    jax's igamma_grad_a runs a looped series. a inside [0.1, 10] only
    (the JVP's cond dispatches). Differentiable in x through the tables;
    the a direction is never requested here (the cond traces the
    fallback's NotImplementedError for a-tangents, so d2/da2 keeps its
    clear raise on every path)."""

    def p_of_a(t):
        lser, ltail = _gi_series(t)
        return _eval_gammainc(t, x, lser, ltail, lower=True)

    return jax.jvp(p_of_a, (a,), (jnp.ones_like(a),))[1]


def _dPda_recipe_large(a, x):
    """dP/da through the Temme-zone tables, a in (10, 1000]."""

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
            "which neither jax nor chebax provides")
    if isinstance(dx, zero):
        return g, jnp.zeros_like(g)
    pdf = jnp.exp((a - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a))
    d = pdf * (jnp.log(x) - jax.scipy.special.digamma(a)) * dx
    return g, d


_dPda.defjvp(_dPda_jvp, symbolic_zeros=True)


def _betaincinv_core(a, b, p, cab, cba):
    """The betaincinv solve, series-agnostic: cab/cba are the x-series of
    the betainc kernel at (a, b) and (b, a), from the 3-D tensor (public
    betaincinv) or the stdtr slice tables (stdtrit). Not differentiable
    on its own; callers wrap it in a custom_jvp."""
    lim = _limits(_canon_tag())
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    ln_b = _ln_beta(a, b)

    # Solve every element as a LOWER-tail problem of the mirrored orientation
    # (1 - p is exact for p >= 1/2), in logit space so tail quantiles resolve
    # down to the dtype's floor; df/du = pdf * x * (1-x) is a stable exp.
    swap = pc > 0.5
    ps2 = jnp.where(swap, 1.0 - pc, pc)
    aa = jnp.where(swap, b, a)

    def f_and_df(u):
        x = jax.nn.sigmoid(u)
        f = jnp.where(swap,
                      _eval_betainc(b, a, x, cba, cab),
                      _eval_betainc(a, b, x, cab, cba)) - ps2
        ls, l1s = -jax.nn.softplus(-u), -jax.nn.softplus(u)
        dfdu = jnp.exp(jnp.where(swap, b, a) * ls
                       + jnp.where(swap, a, b) * l1s - ln_b)
        return f, dfdu

    # Initialization: the generic start converges at ~1/a additive steps in
    # u, which cannot reach deep tails in a fixed count (measured: a = 1,
    # p = 1e-50 stalled at e^-64). In the tail the exact leading asymptotic
    # I_x(a,b) = x^a / (a B(a,b)) (1 + O(x)) inverts to a start within
    # O(x) relatively, and Newton converges quadratically from there.
    u_generic = jnp.where(swap, jnp.log(b / a), jnp.log(a / b)) + jnp.zeros_like(pc)
    u_asym = (jnp.log(ps2) + jnp.log(aa) + ln_b) / aa
    u0 = jnp.clip(jnp.where(ps2 < 0.01, u_asym, u_generic), lim.lo, lim.hi)
    u = _newton_bisect(f_and_df, u0, jnp.full_like(pc, lim.lo),
                       jnp.full_like(pc, lim.hi), lim.n_beta)

    # Convergence is checked, never assumed. A solve pinned at the bracket
    # floor means the true quantile underflows the dtype (or the CDF itself
    # is indistinguishable from 0 there): return the mirrored endpoint.
    # Any other unconverged element returns nan. The bar stays relative to
    # the mass being resolved: past the kernel's x = 1/2 split I comes back
    # as 1 - (the reflected orientation) and is only absolutely accurate, so
    # a target far below that absolute error is genuinely unresolvable and
    # nan is the answer (float32 hits this through stdtrit at large nu).
    f_fin, _ = f_and_df(u)
    at_floor = u <= lim.lo + 0.5
    x_low = jnp.where(at_floor, 0.0, jax.nn.sigmoid(u))
    bad = ~at_floor & (jnp.abs(f_fin) > lim.rtol * ps2)
    x = jnp.where(swap, 1.0 - x_low, x_low)
    x = jnp.where(bad, jnp.nan, x)
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, 1.0, x))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(a) | jnp.isnan(b)
    return jnp.where(oob, jnp.nan, x)


@jax.custom_jvp
def betaincinv(a, b, p):
    """Inverse of betainc in x: the Beta(a, b) quantile function.

    a, b are (traceable) scalars in [0.1, 100], uniform per call; p any
    shape (the wide panels of increment 23 dispatch by one lax.switch).
    Quantiles below the smallest normal float of the canonical dtype return
    0.0 (and, mirrored, quantiles within eps of 1 return 1.0, the
    scipy-shared representation limit; use betaincinv(b, a, 1-p) for the
    distance from 1). A solve that fails its CDF-residual check returns nan
    instead of a wrong value."""
    a = _canon(a)
    b = _canon(b)
    p = _canon(p)
    return with_panel_series(
        a, b, lambda cab, cba: _betaincinv_core(a, b, p, cab, cba))


@betaincinv.defjvp
def _betaincinv_jvp(primals, tangents):
    a, b, p = primals
    da, db, dp = tangents
    x = betaincinv(a, b, p)
    interior = (x > 0.0) & (x < 1.0)
    xs = jnp.where(interior, x, 0.5)
    pdf = jnp.exp((_canon(a) - 1) * jnp.log(xs)
                  + (_canon(b) - 1) * jnp.log1p(-xs) - _ln_beta(_canon(a), _canon(b)))
    _, d_cdf = jax.jvp(lambda aa, bb: betainc_fn(aa, bb, xs), (a, b), (da, db))
    dx = (_canon(dp) - d_cdf) / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


def _gammaincinv_solve(a, pc, use_recipe):
    """Solve P(a, e^v) = p (lower half) or Q(a, e^v) = 1 - p (upper half)
    in v, against the LOG probability.

    Two reasons for the logs. Newton on P - p crawls exactly one e-fold
    per step wherever P is exponentially small, because the step is
    (1 - p/P)/(dlnP/dv) (measured: a = 100, p = 1e-20 was still 20
    e-folds out after 40 steps and got rejected as unconverged); ln P is
    nearly linear in v there, and the same start converges in two. And in
    the upper half P - p has no digits left at all once 1 - p is at the
    rounding of 1, while Q - (1 - p) keeps them: whichever target is the
    small one is exact (1 - p is exact for p >= 1/2), and the other is at
    least 1/2, so both carry full relative accuracy.

    The half is an elementwise mask, so both tails are evaluated per step
    and selected, the way betaincinv handles its swap. use_recipe is a
    python constant (each lax.cond branch traces one): "small" takes both
    logs from the a <= 10 gammainc tables, "large" from the Temme-zone
    path (a in (10, 1000], relatively accurate on both sides
    everywhere), each reconstructed once and closed over, so each step is
    a fixed-degree polynomial; None uses jax's gammainc and gammaincc
    (any a, but a whole-array while_loop per call).
    Returns (v, relative residual)."""
    lim = _limits(_canon_tag())
    lga = jax.scipy.special.gammaln(a)
    upper = pc > 0.5
    ltp, ltq = jnp.log(pc), jnp.log1p(-pc)
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
            lp_in, lq_tl = _gi_logs(a, x, lser, ltail)
            inner = x <= _gt.XS
            lp = jnp.where(inner, lp_in, jnp.log1p(-jnp.exp(lq_tl)))
            lq = jnp.where(inner, jnp.log1p(-jnp.exp(lp_in)), lq_tl)
            pos = x > 0.0
            return jnp.where(pos, lp, -jnp.inf), jnp.where(pos, lq, 0.0)
        if use_recipe == "large":
            return (_gl.eval_large(a, x, tm, dl, du, "logp"),
                    _gl.eval_large(a, x, tm, dl, du, "logq"))
        return (jnp.log(jax.scipy.special.gammainc(a, x)),
                jnp.log(jax.scipy.special.gammaincc(a, x)))

    def f_and_df(v):
        x = jnp.exp(v)
        lp, lq = log_pq(x)
        lden = a * v - x - lga          # ln(x pdf(x)), the density in v
        f = jnp.where(upper, ltq - lq, lp - ltp)
        return f, jnp.exp(lden - jnp.where(upper, lq, lp))

    # Initialization: Wilson-Hilferty for the bulk (it stays good into the
    # upper tail, where the target is Q); the exact leading asymptotic
    # P(a, x) = x^a / Gamma(a+1) (1 + O(x)) in the lower tail, where WH can
    # start far off for large a (measured: a = 20, p = 1e-20 started at
    # 0.56 for a root at 0.87). The log-space iteration keeps the deep
    # lower tail of a large a convergent from the asymptotic start too
    # (ln P is nearly linear in v there).
    z = jax.scipy.special.ndtri(pc)
    wh = a * (1.0 - 1.0 / (9.0 * a) + z / (3.0 * jnp.sqrt(a))) ** 3
    v_asym = (jnp.log(pc) + jax.scipy.special.gammaln(a + 1.0)) / a
    pos = wh > 0.0
    v_wh = jnp.where(pos, jnp.log(jnp.where(pos, wh, 1.0)), v_asym)
    v0 = jnp.clip(jnp.where(pc < 0.01, v_asym, v_wh), lim.lo, lim.v_hi)
    v = _newton_bisect(f_and_df, v0, jnp.full_like(pc, lim.lo),
                       jnp.full_like(pc, lim.v_hi), lim.n_gamma)
    # Judge the residual on whichever side is computed DIRECTLY, since that
    # is the side carrying relative accuracy: P below the seam, Q above it,
    # each being 1 - the other on the far side (absolutely accurate, so its
    # relative error grows like eps/Q where Q is small below the seam).
    x = jnp.exp(v)
    lp, lq = log_pq(x)
    return v, jnp.abs(jnp.expm1(jnp.where(x > seam, lq - ltq, lp - ltp)))


@jax.custom_jvp
def gammaincinv(a, p):
    """Inverse of the regularized lower gamma in x: the Gamma(a, 1) quantile.

    a is a (traceable) scalar, uniform per call; p any shape. a must be
    finite and positive: anything else (0, negative, inf, nan) returns nan
    for every p, endpoints included. For a inside [0.1, 1000] the Newton
    residual runs on the chebax gammainc tables (fixed-degree polynomials;
    (10, 1000] via the Temme-zone path); outside it falls back to jax's
    gammainc, so the domain stays all of a > 0. The upper half is solved
    against Q, so p within rounding of 1 still resolves (P(a, x) - p there
    has no digits left).
    Quantiles below the smallest normal float of the canonical dtype return
    0.0; a solve that fails its CDF-residual check returns nan instead of a
    wrong value.

    Second derivatives in p and mixed p-a derivatives work (dP/da is
    wrapped with its exact x-derivative); a pure second derivative in a
    needs d2P/da2, which is unavailable and raises with a clear message."""
    a = _canon(a)
    p = _canon(p)
    lim = _limits(_canon_tag())
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)

    # the recipe residuals serve a inside the table boxes (small tables
    # to 10, Temme-zone tables to 1000); jax's gammainc outside. Nested
    # scalar conds: only the taken branch executes.
    in_small = (a >= _gt.ALO) & (a <= _gt.AHI)
    in_large = (a > _gt.AHI) & (a <= _glt.AHI)
    v, rel = jax.lax.cond(
        in_small,
        lambda: _gammaincinv_solve(a, pc, "small"),
        lambda: jax.lax.cond(
            in_large,
            lambda: _gammaincinv_solve(a, pc, "large"),
            lambda: _gammaincinv_solve(a, pc, None)))
    at_floor = v <= lim.lo + 0.5
    x = jnp.where(at_floor, 0.0, jnp.exp(v))
    x = jnp.where(~at_floor & (rel > lim.rtol), jnp.nan, x)
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, jnp.inf, x))
    oob = (jnp.isnan(p) | (p < 0.0) | (p > 1.0)
           | ~((a > 0.0) & jnp.isfinite(a)))
    return jnp.where(oob, jnp.nan, x)


def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = gammaincinv(a, p)
    interior = jnp.isfinite(x) & (x > 0.0)
    xs = jnp.where(interior, x, 1.0)
    pdf = jnp.exp((_canon(a) - 1) * jnp.log(xs) - xs
                  - jax.scipy.special.gammaln(_canon(a)))
    # dP/da through the differentiable wrapper: second derivatives in p and
    # mixed p-a work; only d2/da2 remains unsupported (see _dPda). Symbolic
    # zeros matter here: a materialized 0 tangent would still route the
    # outer derivative through _dPda's a direction.
    num = _canon(dp) if not isinstance(dp, zero) else 0.0
    if not isinstance(da, zero):
        ac = _canon(a)
        in_small = (ac >= _gt.ALO) & (ac <= _gt.AHI)
        in_large = (ac > _gt.AHI) & (ac <= _glt.AHI)
        dPda_val = jax.lax.cond(
            in_small,
            lambda: _dPda_recipe(ac, xs),
            lambda: jax.lax.cond(in_large,
                                 lambda: _dPda_recipe_large(ac, xs),
                                 lambda: _dPda(ac, xs)))
        num = num - dPda_val * _canon(da)
    dx = num / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


gammaincinv.defjvp(_gammaincinv_jvp, symbolic_zeros=True)


def _t_logpdf(nu, t):
    # ln(1 + t^2/nu) from log|t| where t^2 would overflow (|t| ~ 1.3e154 in
    # float64, 1.8e19 in float32), where it is 2 ln|t| - ln nu to hundreds
    # of digits. The direct form is kept elsewhere: it is the one with the
    # exact zero t-derivative at t = 0, which second derivatives need.
    t2 = t * t
    ok = jnp.isfinite(t2)
    l1 = jnp.where(ok, jnp.log1p(jnp.where(ok, t2, 0.0) / nu),
                   2.0 * jnp.log(jnp.where(ok, 1.0, jnp.abs(t))) - jnp.log(nu))
    return (jax.scipy.special.gammaln(0.5 * (nu + 1.0))
            - jax.scipy.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(nu * jnp.pi) - 0.5 * (nu + 1.0) * l1)


def _half_series(hb):
    """x-series of the betainc kernel at (1/2, hb) and (hb, 1/2), from the
    stdtr slice tables: hb = nu/2 in [0.1, 100] (nu in [0.2, 200]),
    uniform per call, panel select at hb = 10. All four tables share the
    x-node count, so the selects act elementwise on coefficient vectors."""
    lo = hb <= _st.BSPLIT
    ca = jnp.where(lo,
                   traced_coefs(_st.TABLE_A_LO, _st.BLO, _st.BSPLIT, hb),
                   traced_coefs(_st.TABLE_A_HI, _st.BSPLIT, _st.BHI, hb))
    cb = jnp.where(lo,
                   traced_coefs(_st.TABLE_B_LO, _st.BLO, _st.BSPLIT, hb),
                   traced_coefs(_st.TABLE_B_HI, _st.BSPLIT, _st.BHI, hb))
    return (ChebSeries(ca, (0.0, _st.XSPLIT)),
            ChebSeries(cb, (0.0, _st.XSPLIT)))


def _stdtr_parts(nu, t):
    """(central mask, ln I_w(1/2, nu/2), ln I_x(nu/2, 1/2)).

    F is 1/2 (1 + sgn(t) I_w) near the median and 1/2 I_x in the tails,
    with w = t^2/(nu+t^2) and x = nu/(nu+t^2). w resolves F - 1/2 exactly
    through t = 0 (the classic x form rounds to 1 there and F degrades to
    exactly 0.5); x resolves the tail probability. Seam at t^2 = nu, where
    both are 1/2, which is also where each orientation leaves the tables'
    direct branch, so both logs come straight from the series.

    The tail is taken from log|t|: with z = 2 ln|t| - ln nu, x = 1/(1+e^z)
    is -softplus(z) and 1 - x is -softplus(-z), so nothing forms t^2
    (which overflows past |t| ~ 1.3e154 in float64, ~1.8e19 in float32)
    and the tail probability stays right down to where it underflows."""
    hb = 0.5 * nu
    half = jnp.asarray(0.5)
    ca, cb = _half_series(hb)
    t2 = t * t                        # may be inf; only its comparison is used
    central = t2 <= nu
    tc = jnp.where(central, t, 0.0)
    # <= 1/2 by construction, no overflow; masked lanes get a safe dummy so
    # that reverse mode does not multiply their zero cotangent by a d/dnu
    # of log(0)
    w = jnp.where(central, tc * tc / (nu + tc * tc), 0.25)
    # I <= 1, so clamping its log at 0 is exact and keeps 1 - I_w >= 0 when
    # the evaluation error swamps it (large nu just inside the seam)
    l_mid = jnp.minimum(_log_betainc_direct(half, hb, w, ca), 0.0)
    tt = jnp.where(central, jnp.sqrt(nu), jnp.abs(t))
    z = 2.0 * jnp.log(tt) - jnp.log(nu)
    lx, l1x = -jax.nn.softplus(z), -jax.nn.softplus(-z)
    l_tail = (hb * lx + 0.5 * l1x - jnp.log(hb) - _ln_beta(hb, half)
              + cb(jnp.exp(lx)))
    return central, l_mid, jnp.minimum(l_tail, 0.0)


def _stdtr_impl(nu, t):
    central, l_mid, l_tail = _stdtr_parts(nu, t)
    hp = 0.5 * jnp.exp(jnp.where(central, l_mid, l_tail))
    sgn = jnp.where(t >= 0.0, 1.0, -1.0)
    F = jnp.where(central, 0.5 + sgn * hp,
                  jnp.where(t >= 0.0, 1.0 - hp, hp))
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, F)


def _log_stdtr_impl(nu, t):
    central, l_mid, l_tail = _stdtr_parts(nu, t)
    up = t >= 0.0
    # every branch is a log of the series value, never of a difference of
    # two numbers near 1/2: below the median the lower tail is 1/2 I, above
    # it the survival half is, and log1p/expm1 keep the complement exact
    mid = jnp.where(up, jax.nn.softplus(l_mid),
                    jnp.log(-jnp.expm1(l_mid))) - _LN2
    tail = jnp.where(up, jnp.log1p(-0.5 * jnp.exp(l_tail)), l_tail - _LN2)
    L = jnp.where(central, mid, tail)
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, L)


@jax.custom_jvp
def stdtr(nu, t):
    """Student-t CDF with nu degrees of freedom (nu in [0.2, 200], traceable).

    jax.grad works with respect to nu as well: learnable degrees of freedom.
    First-order exact through t = 0 (dF/dt there is the density, not 0), and
    valid for any t, the tail being computed from log|t|."""
    return _stdtr_impl(_canon(nu), _canon(t))


@stdtr.defjvp
def _stdtr_jvp(primals, tangents):
    nu, t = primals
    dnu, dt = tangents
    nu_c, t_c = _canon(nu), _canon(t)
    F = _stdtr_impl(nu_c, t_c)
    # t-direction: the exact density (AD through the w = t^2/(nu+t^2)
    # branch hits a 0 * inf at t = 0; the density is finite there)
    dF_dt = jnp.exp(_t_logpdf(nu_c, jnp.where(jnp.isfinite(t_c), t_c, 0.0)))
    dF_dt = jnp.where(jnp.isfinite(t_c), dF_dt, 0.0)
    # nu-direction: AD of the implementation, at a safe t (t = 0 and
    # t = +-inf have exactly zero nu-sensitivity)
    t_safe = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), t_c, 1.0)
    _, dF_dnu = jax.jvp(lambda n: _stdtr_impl(n, t_safe), (nu_c,), (_canon(dnu),))
    dF_dnu = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), dF_dnu, 0.0)
    dF = dF_dt * _canon(dt) + dF_dnu
    return F, jnp.where(jnp.isnan(F), jnp.nan, dF)


@jax.custom_jvp
def log_stdtr(nu, t):
    """ln F(t; nu), the Student-t log-CDF, with no underflow floor in the
    lower tail.

    Same contract as stdtr (nu in [0.2, 200], traceable, differentiable in
    both arguments). For t <= -sqrt(nu) the tail half-probability is a
    tabulated log, returned as such: relatively accurate to arbitrarily
    negative values (ln F ~ -nu ln|t|), which is far past where F itself
    underflows. Above the median it is log1p of a half-probability that is
    at most 1/2, and F is O(1) there, so ln F is accurate too.

    Between -sqrt(nu) and the median the tables only reach F through
    1 - I_w, which is absolutely accurate, so ln F there carries the CDF's
    absolute error over F. That band is harmless for small nu (F at its far
    edge is 0.058 at nu = 4, 8e-4 at nu = 15) and is the accuracy limit for
    large nu, whose far edge sits at F ~ 4e-11 (nu = 60) or 1e-34
    (nu = 200): the tables hold the (nu/2, 1/2) series only for x <= 1/2.
    Measured float64: 3e-6 at nu = 150, t = -6 (F = 7e-9), exact by
    t = -4. Where F underflows, t is always past -sqrt(nu) and the
    directly-tabulated branch has it.

    This is the log-CDF a truncated or censored Student-t likelihood wants:
    log(stdtr(...)) underflows to -inf first, and a normalizer built as
    log(F(hi) - F(lo)) collapses to log(0) whenever both tails are deep."""
    return _log_stdtr_impl(_canon(nu), _canon(t))


@log_stdtr.defjvp
def _log_stdtr_jvp(primals, tangents):
    nu, t = primals
    dnu, dt = tangents
    nu_c, t_c = _canon(nu), _canon(t)
    L = _log_stdtr_impl(nu_c, t_c)
    fin = jnp.isfinite(t_c)
    ts = jnp.where(fin, t_c, 0.0)
    # d/dt ln F = pdf / F, taken as a difference of logs so it survives
    # where F underflows (there it tends to nu/|t|, not to 0/0)
    dL_dt = jnp.where(fin, jnp.exp(_t_logpdf(nu_c, ts) - L), 0.0)
    t_safe = jnp.where(fin & (ts != 0.0), ts, 1.0)
    _, dL_dnu = jax.jvp(lambda n: _log_stdtr_impl(n, t_safe), (nu_c,),
                        (_canon(dnu),))
    dL_dnu = jnp.where(fin & (ts != 0.0), dL_dnu, 0.0)
    dL = dL_dt * _canon(dt) + dL_dnu
    return L, jnp.where(jnp.isnan(L), jnp.nan, dL)


def log_stdtr_sf(nu, t):
    """ln (1 - F(t; nu)), the Student-t log survival function, with no
    underflow floor in the upper tail.

    The distribution is symmetric, so this is the log-CDF at -t and the
    negation is exact: no 1 - F cancellation anywhere, and the same
    accuracy in the upper tail that log_stdtr has in the lower."""
    return log_stdtr(nu, -_canon(t))


def _stdtrit_impl(nu, p):
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    upper = pc > 0.5
    # central: |t| from w = I^-1(|2p-1|; 1/2, nu/2), t = sqrt(nu w/(1-w));
    # tails: x = I^-1(2 min(p, 1-p); nu/2, 1/2), t = sqrt(nu (1-x)/x).
    # Both targets are exact in binary: 2p is a scaling, 1 - p is exact for
    # p >= 1/2, and 2p - 1 is exact for p in [1/4, 3/4]. The old tail
    # target 1 - |2p - 1| was not: at p = 1e-20 it rounded to 0 and the
    # quantile came back -inf.
    ptail = 2.0 * jnp.where(upper, 1.0 - pc, pc)
    aq = jnp.abs(2.0 * pc - 1.0)
    central = ptail >= 0.5
    ca, cb = _half_series(0.5 * nu)
    w = _betaincinv_core(jnp.asarray(0.5), 0.5 * nu,
                         jnp.where(central, aq, 0.25), ca, cb)
    x = _betaincinv_core(0.5 * nu, jnp.asarray(0.5),
                         jnp.where(central, 0.5, ptail), cb, ca)
    # sqrt(nu (1-x)) / sqrt(x), not sqrt(nu (1-x)/x): the ratio overflows
    # for x below ~1e-308 while the quantile itself is still representable
    mag = jnp.where(central, jnp.sqrt(nu * w / (1.0 - w)),
                    jnp.sqrt(nu * (1.0 - x)) / jnp.sqrt(x))
    t = jnp.where(pc < 0.5, -mag, mag)
    t = jnp.where(p <= 0.0, -jnp.inf, jnp.where(p >= 1.0, jnp.inf, t))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(nu)
    return jnp.where(oob, jnp.nan, t)


@jax.custom_jvp
def stdtrit(nu, p):
    """Student-t quantile with nu degrees of freedom (nu in [0.2, 200]).

    Closed form via betaincinv in both orientations: exact 0 at p = 1/2,
    +-inf at the endpoints, gradients from the implicit function theorem
    (dt/dp = 1/pdf is 8/3 at the median for nu = 4, not 0). The tail
    target is 2 min(p, 1-p), exact in binary, so deep-tail p resolves
    (nu = 4, p = 1e-300 gives -1.3e75). It bottoms out where the beta
    quantile behind it underflows and the +-inf endpoint takes over, which
    at small nu can be short of the representable t (nu = 1/2,
    p = 1e-100)."""
    return _stdtrit_impl(_canon(nu), _canon(p))


@stdtrit.defjvp
def _stdtrit_jvp(primals, tangents):
    nu, p = primals
    dnu, dp = tangents
    nu_c = _canon(nu)
    t = _stdtrit_impl(nu_c, _canon(p))
    finite = jnp.isfinite(t)
    ts = jnp.where(finite, t, 0.0)
    pdf = jnp.exp(_t_logpdf(nu_c, ts))
    # IFT: dt/dp = 1/pdf; dt/dnu = -(dF/dnu at t)/pdf
    t_safe = jnp.where(finite & (ts != 0.0), ts, 1.0)
    _, dF_dnu = jax.jvp(lambda n: _stdtr_impl(n, t_safe), (nu_c,), (jnp.asarray(1.0),))
    dF_dnu = jnp.where(finite & (ts != 0.0), dF_dnu, 0.0)
    dt = (_canon(dp) - dF_dnu * _canon(dnu)) / pdf
    dt = jnp.where(finite, dt, 0.0)
    return t, jnp.where(jnp.isnan(t), jnp.nan, dt)


def chi2inv(k, p):
    """Chi-squared quantile at real degrees of freedom: 2 gammaincinv(k/2, p).

    chi2(k) is Gamma(k/2, rate 1/2), so the quantile is a rescaled Gamma
    quantile: k a (traceable) scalar uniform per call, p any shape,
    differentiable in both, real non-integer dof included. k must be finite
    and positive, inheriting gammaincinv's check: k <= 0 (and inf, nan)
    gives nan for every p. The recipe residual serves k up to 2000
    (a = k/2 <= 1000 since the Temme-zone tables); beyond that the jax
    fallback takes over. Endpoints follow gammaincinv (p = 0 -> 0,
    p = 1 -> inf)."""
    return 2.0 * gammaincinv(0.5 * _canon(k), p)
