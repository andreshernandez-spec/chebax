"""Differentiable quantile functions: betaincinv, gammaincinv, gammainccinv,
chi2inv.

These answer jax#2399 and jax#5350: inverse regularized incomplete beta
and gamma (the Beta and Gamma quantile functions). The Student-t family
(jax#20358) is in stdtr.py and reuses the beta solve here. All parameters
are traceable (uniform per call), so jax.grad works with respect to every
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

betaincinv needs (a, b) inside the betainc table domain [0.1, 100].
gammaincinv has no table domain: its Newton residual
evaluates the chebax gammainc recipe when a is at least 0.1
(above 10 via the Temme-zone tables; fixed-degree polynomials where
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

from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes._common import (canon_param, canon_tag,
                                         float_params, newton_bisect)
from chebax._src.recipes.betainc import (betainc_fn, eval_betainc,
                                         with_panel_series)
from chebax._src.recipes.gammainc import _eval_logs as _gi_logs
from chebax._src.recipes.gammainc import _traced_series as _gi_series
from chebax._src.recipes.gammainc import eval_gammainc

_LN_SMALL = math.log(0.01)   # below this the lower-tail asymptotic beats WH

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
    which left every float32 solve unbracketed. The ends are ln(tiny) and
    ln(max) themselves, moved 4 ulps inward so XLA's exp stays normal and
    finite there: rounding them to whole e-folds (-708, 709) put roots
    in [tiny, e^-708] out of reach. Iteration counts and the residual bar
    are measured per dtype (tests/test_quantiles.py)."""
    dt = np.dtype(_tag).type
    f = np.finfo(dt)

    def inward(v):
        v = dt(v)
        for _ in range(4):
            v = np.nextafter(v, dt(0))
        return float(v)

    lo = inward(math.log(float(f.tiny)))
    wide = dt == np.float64
    return _Limits(lo, -lo, inward(math.log(float(f.max))),
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
            "which neither jax nor chebax provides")
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
        lambda: jax.lax.cond(in_large,
                             lambda: _dPda_recipe_large(a, x),
                             lambda: _dPda(a, x)))


def _betaincinv_core(a, b, p, cab, cba):
    """The betaincinv solve, series-agnostic: cab/cba are the x-series of
    the betainc kernel at (a, b) and (b, a), from the 3-D tensor (public
    betaincinv) or the stdtr slice tables (stdtrit). Not differentiable
    on its own; callers wrap it in a custom_jvp."""
    lim = _limits(canon_tag())
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
                      eval_betainc(b, a, x, cba, cab),
                      eval_betainc(a, b, x, cab, cba)) - ps2
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
    u = newton_bisect(f_and_df, u0, jnp.full_like(pc, lim.lo),
                       jnp.full_like(pc, lim.hi), lim.n_beta)

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


@jax.custom_jvp
def _betaincinv_cj(a, b, p):
    """Inverse of betainc in x: the Beta(a, b) quantile function.

    a, b are (traceable) scalars in [0.1, 100], uniform per call; p any
    shape (the wide panels of increment 23 dispatch by one lax.switch).
    Quantiles below the smallest normal float of the canonical dtype return
    0.0 (and, mirrored, quantiles within eps of 1 return 1.0, the
    scipy-shared representation limit; use betaincinv(b, a, 1-p) for the
    distance from 1). A solve that fails its CDF-residual check returns nan
    instead of a wrong value."""
    return with_panel_series(
        a, b, lambda cab, cba: _betaincinv_core(a, b, p, cab, cba))


@_betaincinv_cj.defjvp
def _betaincinv_jvp(primals, tangents):
    a, b, p = primals
    da, db, dp = tangents
    x = _betaincinv_cj(a, b, p)
    interior = (x > 0.0) & (x < 1.0)
    xs = jnp.where(interior, x, 0.5)
    pdf = jnp.exp((a - 1) * jnp.log(xs) + (b - 1) * jnp.log1p(-xs)
                  - _ln_beta(a, b))
    _, d_cdf = jax.jvp(lambda aa, bb: betainc_fn(aa, bb, xs), (a, b), (da, db))
    dx = (dp - d_cdf) / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


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
_LAM_SERIES = (1.0 / 3.0, 1.0 / 36.0, -1.0 / 270.0, 1.0 / 4320.0,
               1.0 / 17010.0)


def _huge_parts(a, s, temme):
    """At s = eta sqrt(a/2): (ln P, ln Q, ln dP/ds, lambda - 1).

    dP/ds is x^a e^-x / Gamma(a) times dx/ds, which is
    e^(-s^2) / sqrt(pi) e^(-1/(12 a)) eta / (lambda - 1), Stirling's
    remainder kept to 1/(12 a) (the next term is below 1e-33 here)."""
    eta = s * jnp.sqrt(2.0 / a)
    r = _LAM_SERIES[-1]
    for c in _LAM_SERIES[-2::-1]:
        r = r * eta + c
    r = 1.0 + eta * r                      # (lambda - 1) / eta
    lp, lq = _gl.temme_log_pq(s, s * s, temme(eta),
                              jnp.sqrt(2.0 * jnp.pi * a))
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
    """Solve ln P(a, e^v) = ltp (lower lanes) or ln Q(a, e^v) = ltq
    (upper lanes) in v, against the LOG probability. Shared by
    gammaincinv and gammainccinv: each passes its own input through the
    exact log (ltp = ln p, or ltq = ln y), the far side as log1p of it,
    z as ndtri of the lower-tail probability taken from whichever side
    is the accurate one, and upper marking the lanes whose target is Q.

    Two reasons for the logs. Newton on P - p crawls exactly one e-fold
    per step wherever P is exponentially small, because the step is
    (1 - p/P)/(dlnP/dv) (measured: a = 100, p = 1e-20 was still 20
    e-folds out after 40 steps and got rejected as unconverged); ln P is
    nearly linear in v there, and the same start converges in two. And on
    the upper lanes P - p has no digits left at all once the P-target is
    at the rounding of 1, while the Q-target keeps them: whichever target
    is the small one is exact, and the other is at least 1/2, so both
    carry full relative accuracy.

    The half is an elementwise mask, so both tails are evaluated per step
    and selected, the way betaincinv handles its swap. use_recipe is a
    python constant (each lax.cond branch traces one): "small" takes both
    logs from the a <= 10 gammainc tables, "large" from the Temme-zone
    path (a > 10, relatively accurate on both sides
    everywhere), each reconstructed once and closed over, so each step is
    a fixed-degree polynomial; None uses jax's gammainc and gammaincc
    (any a, but a whole-array while_loop per call).
    "huge" hands the solve to _huge_solve.
    Returns (x, relative residual, root below the floor, root above the
    ceiling), the last two from the residual's sign at the bracket ends;
    the ceiling is only checked when asked for (gammainccinv)."""
    if use_recipe == "huge":
        x, rel, _ = _huge_solve(a, ltp, ltq, z, upper)
        none = jnp.zeros(x.shape, bool)
        return x, rel, none, (x == jnp.inf) if ceiling else none
    lim = _limits(canon_tag())
    lga = jax.scipy.special.gammaln(a)
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

    # Initialization: Wilson-Hilferty for the bulk, and each tail's own
    # exact leading asymptotic where WH drifts. Below, P = x^a/Gamma(a+1)
    # (1 + O(x)), which WH can miss badly for large a (measured: a = 20,
    # p = 1e-20 started at 0.56 for a root at 0.87). Above,
    # ln Q = (a-1) ln x - x - lnGamma(a), inverted by three fixed-point
    # passes; that one needs x >> a, and once ln(1/y) < lnGamma(a) it
    # collapses to its floor, so the upper lanes take whichever of the two
    # starts higher. A start far BELOW the root is the expensive miss: the
    # logs are finite everywhere, so the safeguard never bisects and
    # Newton crawls an e-fold a step (measured through the survival
    # inverse: a = 50, y = 1e-8 stopped 5.8e-10 short from the collapsed
    # start). Overshoot costs a few steps at most. The log-space iteration
    # keeps the deep lower tail of a large a convergent from the
    # asymptotic start too (ln P is nearly linear in v there).
    wh = a * (1.0 - 1.0 / (9.0 * a) + z / (3.0 * jnp.sqrt(a))) ** 3
    v_lo = (ltp + jax.scipy.special.gammaln(a + 1.0)) / a
    r = -ltq - lga
    xa = jnp.maximum(r, 1.0)
    for _ in range(3):
        xa = jnp.maximum(r + (a - 1.0) * jnp.log(xa), 1e-3)
    v_up = jnp.log(xa)
    pos = wh > 0.0
    v_wh = jnp.where(pos, jnp.log(jnp.where(pos, wh, 1.0)),
                     jnp.where(upper, v_up, v_lo))
    v0 = jnp.where(upper, jnp.maximum(v_wh, v_up),
                   jnp.where(ltp < _LN_SMALL, v_lo, v_wh))
    v0 = jnp.clip(v0, lim.lo, lim.v_hi)
    v = newton_bisect(f_and_df, v0, jnp.full_like(v0, lim.lo),
                       jnp.full_like(v0, lim.v_hi), lim.n_gamma)
    # Judge the residual on whichever side is computed DIRECTLY, since that
    # is the side carrying relative accuracy: P below the seam, Q above it,
    # each being 1 - the other on the far side (absolutely accurate, so its
    # relative error grows like eps/Q where Q is small below the seam).
    x = jnp.exp(v)
    lp, lq = log_pq(x)
    f_lo, _ = f_and_df(jnp.full_like(v, lim.lo))
    above = jnp.zeros(v.shape, bool)
    if ceiling:
        above = f_and_df(jnp.full_like(v, lim.v_hi))[0] < 0.0
    return (x, jnp.abs(jnp.expm1(jnp.where(x > seam, lq - ltq, lp - ltp))),
            f_lo > 0.0, above)


def _inv_targets(pc):
    """(ln p, ln(1-p), ndtri(p), upper lanes) for a lower-tail target p:
    exact on the lower lanes, and 1 - p is exact for p >= 1/2."""
    return (jnp.log(pc), jnp.log1p(-pc), jax.scipy.special.ndtri(pc),
            pc > 0.5)


def _cinv_targets(yc):
    """The same for an upper-tail target y. ndtri takes y directly, by
    symmetry, since ndtri(1 - y) saturates for y below eps."""
    return (jnp.log1p(-yc), jnp.log(yc), -jax.scipy.special.ndtri(yc),
            yc < 0.5)


def _gammaincinv_solve(a, pc, use_recipe):
    """Lower-tail entry to _gammainv_core: the target is p, exact on the
    lower lanes, and 1 - p (exact for p >= 1/2) on the upper ones."""
    return _gammainv_core(a, *_inv_targets(pc), use_recipe)


def _gammainccinv_solve(a, yc, use_recipe):
    """Upper-tail entry to _gammainv_core: the target is y, exact on the
    upper lanes, and 1 - y (exact for y >= 1/2) on the lower ones.

    Not a call to _gammaincinv_solve at 1 - y: that rounds to exactly 1
    for y below eps, which is the whole range this exists for."""
    return _gammainv_core(a, *_cinv_targets(yc), use_recipe, ceiling=True)


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
                in_huge,
                lambda: solve(a, tc, "huge"),
                lambda: solve(a, tc, None))))


@jax.custom_jvp
def _gammaincinv_cj(a, p):
    """Inverse of the regularized lower gamma in x: the Gamma(a, 1) quantile.

    a is a (traceable) scalar, uniform per call; p any shape. a must be
    finite and positive: anything else (0, negative, inf, nan) returns nan
    for every p, endpoints included. From a = 0.1 up the Newton
    residual runs on the chebax gammainc tables (fixed-degree polynomials;
    above 10 via the Temme-zone path, and above 1e10 the solve itself runs
    in Temme's variable, which keeps every a up to the largest float);
    below it falls back to jax's gammainc, so the domain stays all of
    a > 0. The upper half is solved
    against Q, so p within rounding of 1 still resolves (P(a, x) - p there
    has no digits left).
    Quantiles below the smallest normal float of the canonical dtype return
    0.0; a solve that fails its CDF-residual check returns nan instead of a
    wrong value.

    Second derivatives in p and mixed p-a derivatives work (dP/da is
    wrapped with its exact x-derivative); a pure second derivative in a
    needs d2P/da2, which is unavailable and raises with a clear message.
    Above a = 1e10 first derivatives come from the implicit rule in
    Temme's variable and higher ones differentiate its fixed iteration."""
    lim = _limits(canon_tag())
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)

    # the root's place against the floor comes from the residual's sign
    # there, not a band in v (see _betaincinv_core)
    x, rel, at_floor, _ = _dispatch_solve(_gammaincinv_solve, a, pc)
    x = jnp.where(at_floor, 0.0, x)
    x = jnp.where(~at_floor & (rel > lim.rtol), jnp.nan, x)
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, jnp.inf, x))
    oob = (jnp.isnan(p) | (p < 0.0) | (p > 1.0)
           | ~((a > 0.0) & jnp.isfinite(a)))
    return jnp.where(oob, jnp.nan, x)


def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = _gammaincinv_cj(a, p)
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
        targets = _inv_targets(pc)
        _, _, s = _huge_solve(a, *targets)
        dpv = dp if not isinstance(dp, zero) else jnp.zeros_like(pc)
        dlog = jnp.where(targets[3], -dpv / (1.0 - pc), dpv / pc)
        return _huge_dx(a, None if isinstance(da, zero) else da, s,
                        targets[3], dlog)

    dx = jax.lax.cond(a > _A_HUGE, in_s, in_x)
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


_gammaincinv_cj.defjvp(_gammaincinv_jvp, symbolic_zeros=True)


@jax.custom_jvp
def _gammainccinv_cj(a, y):
    """Inverse of the regularized UPPER gamma in x: the x with Q(a, x) = y,
    the Gamma(a, 1) survival quantile.

    Same contract, dispatch and residual check as gammaincinv, with the
    roles of the tails swapped: the target enters as y, exact for the
    upper tail this exists for, and the solve runs against ln Q there.
    gammaincinv(a, 1 - y) cannot serve: 1 - y is exactly 1 for every
    y below eps, so the whole deep upper tail collapses onto one point.
    Quantiles past the largest float of the canonical dtype return inf.

    a is a (traceable) scalar, uniform per call; y any shape. Derivatives
    in both arguments, from the same implicit-function rule as
    gammaincinv's with dQ/dx = -pdf and dQ/da = -dP/da."""
    lim = _limits(canon_tag())
    interior = (y > 0.0) & (y < 1.0)
    yc = jnp.where(interior, y, 0.5)

    # at_ceil: the quantile itself overflows
    x, rel, at_floor, at_ceil = _dispatch_solve(_gammainccinv_solve, a, yc)
    x = jnp.where(at_floor, 0.0, x)
    x = jnp.where(at_ceil, jnp.inf, x)
    x = jnp.where(~(at_ceil | at_floor) & (rel > lim.rtol), jnp.nan, x)
    x = jnp.where(y <= 0.0, jnp.inf, jnp.where(y >= 1.0, 0.0, x))
    oob = (jnp.isnan(y) | (y < 0.0) | (y > 1.0)
           | ~((a > 0.0) & jnp.isfinite(a)))
    return jnp.where(oob, jnp.nan, x)


def _gammainccinv_jvp(primals, tangents):
    a, y = primals
    da, dy = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = _gammainccinv_cj(a, y)
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
        targets = _cinv_targets(yc)
        _, _, s = _huge_solve(a, *targets)
        dyv = dy if not isinstance(dy, zero) else jnp.zeros_like(yc)
        dlog = jnp.where(targets[3], dyv / yc, -dyv / (1.0 - yc))
        return _huge_dx(a, None if isinstance(da, zero) else da, s,
                        targets[3], dlog)

    dx = jax.lax.cond(a > _A_HUGE, in_s, in_x)
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


_gammainccinv_cj.defjvp(_gammainccinv_jvp, symbolic_zeros=True)


# Public entry points. The custom_jvp rules live on the private names
# above and float_params normalizes every argument first, so an integer,
# keyword or size-one parameter is a float scalar before any tangent
# exists (see its docstring).
betaincinv = float_params(_betaincinv_cj)
gammaincinv = float_params(_gammaincinv_cj)
gammainccinv = float_params(_gammainccinv_cj)


def chi2inv(k, p):
    """Chi-squared quantile at real degrees of freedom: 2 gammaincinv(k/2, p).

    chi2(k) is Gamma(k/2, rate 1/2), so the quantile is a rescaled Gamma
    quantile: k a (traceable) scalar uniform per call, p any shape,
    differentiable in both, real non-integer dof included. k must be finite
    and positive, inheriting gammaincinv's check: k <= 0 (and inf, nan)
    gives nan for every p. The recipe residual serves k from 0.2 up
    (a = k/2 >= 0.1); below that the jax fallback takes over. Endpoints
    follow gammaincinv (p = 0 -> 0, p = 1 -> inf)."""
    return 2.0 * gammaincinv(0.5 * canon_param(k, "chi2inv", "k"), p)
