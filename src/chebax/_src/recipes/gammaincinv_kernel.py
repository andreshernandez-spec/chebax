"""The Gamma quantile solve, without a table: gammaincinv_scalar runs it on
the gammainc tables (and on jax's gammainc below a = 0.1),
gammaincinv_array on the per-element patches. Each passes its own ln P and
ln Q.

The solve is safeguarded Newton (_common.newton_bisect), or Halley for
the array path (the curvature is exact, below), on the log of the
target's own tail probability, in t = ln(x/a). On 1e5 random lanes
Halley needs at most 4 steps where Newton needed 7. Newton on P - p crawls
one e-fold per step wherever P is exponentially small, since the step is
(1 - p/P)/(d ln P/dv) (measured: a = 100, p = 1e-20 was still 20 e-folds
out after 40 steps); ln P is nearly linear in ln x there, and the same
start converges in two. On the upper lanes P - p has no digits left once
the P-target rounds against 1, while the Q-target keeps them: whichever
target is the small one is exact, and the other is at least 1/2, so both
carry full relative accuracy. The half is an elementwise mask, so both
logs are evaluated per step and selected.

Why t = ln(x/a) and not v = ln x: the quantile is a (1 + O(a^-1/2)), and
v carries eps ln a of absolute error. In float32 that is 1e-6 relative to
x at a = 1e5, more than the residual bar allows in the tails, and from
a = 1e7 every solve came back nan (found writing gammaincinv_array). x is
a e^t, which rounds once, and a + a expm1(t) near the mode, which reaches
every float there (a e^t steps by a eps, two floats at a time). Each step
starts from the t that the rounded x stands for, so the slope and the
residual describe the same point: in float32 at a = 5e14 one float step
of x is 1.5 standard deviations, and a slope taken at the unrounded t sent
Newton far off.

A solve is accepted when its residual is within the bar (solver_limits'
rtol unless the caller passes one), or when the residual changes sign
between x and a neighbouring float, which makes x the root to within one
float step. Large shapes need the second: at a = 1e20 in float64 one
float step of x moves ln P by more than rtol in the bulk.
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import canon_tag, newton_bisect, solver_limits

# |t| below this, x = a + a expm1(t): x/a in (0.6, 1.65)
_NEAR = 0.5
# e^t - 1 - t = t^2 (1/2! + t/3! + ...): on |t| < 1/2 the terms past
# t^19/19! are below 1e-23
_PHI = tuple(1.0 / math.factorial(k) for k in range(2, 20))


def inv_targets(pc):
    """(ln p, ln(1-p), ndtri(p), upper lanes) for a lower-tail target p:
    exact on the lower lanes, and 1 - p is exact for p >= 1/2."""
    return (jnp.log(pc), jnp.log1p(-pc), jax.scipy.special.ndtri(pc), pc > 0.5)


def cinv_targets(yc):
    """The same for an upper-tail target y. ndtri takes y directly, by
    symmetry, since ndtri(1 - y) saturates for y below eps."""
    return (jnp.log1p(-yc), jnp.log(yc), -jax.scipy.special.ndtri(yc), yc < 0.5)


def _phi(t):
    """e^t - 1 - t, without the cancellation near t = 0."""
    near = jnp.abs(t) < _NEAR
    tn = jnp.where(near, t, 0.0)
    s = _PHI[-1]
    for c in _PHI[-2::-1]:
        s = s * tn + c
    return jnp.where(near, tn * tn * s, jnp.expm1(t) - t)


def log_xpdf(a, t, small=False):
    """ln(x pdf(x)) of Gamma(a, 1) at x = a e^t, for a > 0.

    a ln x - x - lnGamma(a) cancels terms of size a ln a, so above a = 100
    it is -a phi(t) + ln(a / (2 pi))/2 - S(a), with phi = e^t - 1 - t and
    S Stirling's remainder to a^-5 (below 1e-17 there). small (a python
    constant) says a <= 100 on every lane, which skips that form."""
    if small:
        return a * (t + jnp.log(a)) - a * jnp.exp(t) - jax.scipy.special.gammaln(a)
    stir = a > 100.0
    ab = jnp.where(stir, a, 1000.0)
    ai2 = 1.0 / (ab * ab)
    rem = (1.0 / 12.0 - ai2 * (1.0 / 360.0 - ai2 / 1260.0)) / ab
    lead = 0.5 * jnp.log(ab / (2.0 * math.pi)) - rem
    am = jnp.where(stir, 1.0, a)
    lga = jax.scipy.special.gammaln(am)
    direct = am * (t + jnp.log(am)) - am * jnp.exp(t) - lga
    return jnp.where(stir, lead - ab * _phi(t), direct)


def _start(a, ltp, ltq, z, upper, ln_a):
    """The first iterate, in t.

    Wilson-Hilferty for the bulk, and each tail's own leading asymptotic
    where WH drifts. Below, P <= x^a / Gamma(a + 1), so the start from
    that bound is never above the root, and the lower lanes take the
    larger of it and WH: WH misses the deep lower tail of a moderate a
    (a = 20, p = 1e-20 started at 0.56 for a root at 0.87), and the bound
    misses large ones (x ~ a/e at a = 1e17, where ln P is -1e16 and the
    slope has no digits). Above, ln Q = (a-1) ln x - x - lnGamma(a),
    inverted by three fixed-point passes; that one needs x >> a, and once
    ln(1/y) < lnGamma(a) it collapses to its floor, so the upper lanes
    take whichever of it and WH starts higher. A start far BELOW the root
    is the expensive miss: the logs are finite everywhere, so the
    safeguard never bisects and Newton crawls an e-fold a step (a = 50,
    y = 1e-8 stopped 5.8e-10 short from the collapsed start). Overshoot
    costs a few steps at most."""
    gammaln = jax.scipy.special.gammaln
    # WH is x = a (1 + c)^3, taken as a log so it keeps its digits at large a
    c = z / (3.0 * jnp.sqrt(a)) - 1.0 / (9.0 * a)
    pos = c > -1.0
    t_wh = 3.0 * jnp.log1p(jnp.where(pos, c, 0.0))
    t_lo = (ltp + gammaln(a + 1.0)) / a - ln_a
    r = -ltq - gammaln(a)
    xa = jnp.maximum(r, 1.0)
    for _ in range(3):
        xa = jnp.maximum(r + (a - 1.0) * jnp.log(xa), 1e-3)
    t_up = jnp.log(xa) - ln_a
    t_wh = jnp.where(pos, t_wh, jnp.where(upper, t_up, t_lo))
    return jnp.where(upper, jnp.maximum(t_wh, t_up), jnp.maximum(t_wh, t_lo))


def solve(
    a,
    ltp,
    ltq,
    z,
    upper,
    log_pq,
    iters,
    ceiling=False,
    seam=None,
    rtol=None,
    small=False,
    halley=False,
):
    """Solve ln P(a, x) = ltp on the lower lanes and ln Q(a, x) = ltq on
    the upper ones (upper) for x. log_pq(x) gives (ln P, ln Q) at the
    caller's a; z is ndtri of the lower-tail probability, for the start;
    iters is the fixed step count.

    The residual bar is rtol (solver_limits' when None, or one per
    lane), judged on the target's own side, unless seam is given: then on the side log_pq computes directly, ln P below the seam
    and ln Q above it. The scalar path needs that, since its small-a
    tables give Q below x = 8 as 1 - P, which carries no relative digits
    in float32 once Q is small (a = 0.1, y = 1e-4 is 1.4e-3 off there).

    halley (a python constant) takes Halley steps instead of Newton
    steps: fewer of them, each a little dearer, which pays where the step
    count is set from measurement (the array path) and not where it is
    fixed (the scalar path's 40).

    small (a python constant) says a <= 10 on every lane, where neither
    the rounding of x near the mode nor the Stirling density matters, so
    the solve skips both (the scalar path's small tables and jax
    fallback, 8-28% of their GPU time in experiments/09).

    Returns (x, accepted, root below the floor, root above the ceiling).
    The floor and ceiling are the canonical dtype's normal range in x, and
    the last two flags come from the residual's sign there; the ceiling is
    only checked when asked for (gammainccinv)."""
    lim = solver_limits(canon_tag())
    ln_a = jnp.log(a)
    big = a > 2.0

    def x_at(t):
        if small:
            return jnp.exp(t + ln_a), None
        # below a = 2, a e^t can be subnormal where x is not
        near = big & (jnp.abs(t) < _NEAR)
        far = jnp.where(big, a * jnp.exp(t), jnp.exp(t + ln_a))
        return jnp.where(near, a + a * jnp.expm1(jnp.where(near, t, 0.0)), far), near

    def snap(t):
        x, near = x_at(t)
        return jnp.where(near, jnp.log1p((x - a) / a), t)

    def residual(x):
        lp, lq = log_pq(x)
        return jnp.where(upper, ltq - lq, lp - ltp), jnp.where(upper, lq, lp)

    def f_and_df(t):
        x, _ = x_at(t)
        f, side = residual(x)
        # the slope of either residual in t is x pdf / (its own probability)
        df = jnp.exp(log_xpdf(a, t, small) - side)
        if not halley:
            return f, df
        # A Halley step, as a Newton step with slope df c. The curvature is
        # exact: with g = ln(x pdf), either side's log S has S' = e^(g - S)
        # up to sign, so S'' = S' (g' - S'), g' = -a expm1(t). Far from the
        # root, where c is small or nan, it is a plain Newton step.
        s1 = jnp.where(upper, -df, df)
        c = 1.0 - f * (-a * jnp.expm1(t) - s1) / (2.0 * df)
        return f, df * jnp.where(c > 0.5, c, 1.0)

    shape = jnp.shape(ltp)
    lo0 = jnp.broadcast_to(lim.lo - ln_a, shape)
    hi0 = jnp.broadcast_to(lim.v_hi - ln_a, shape)
    t0 = jnp.clip(_start(a, ltp, ltq, z, upper, ln_a), lo0, hi0)
    if not small:
        t = snap(newton_bisect(f_and_df, t0, lo0, hi0, iters, snap=snap))
    else:
        t = newton_bisect(f_and_df, t0, lo0, hi0, iters)

    # x, the floats on either side of it, and the ceiling, in one call, so
    # the program holds log_pq twice (here and in the loop) and not once
    # per point: it is most of the compile time
    x, _ = x_at(t)
    pts = [x, jnp.nextafter(x, 0.0), jnp.nextafter(x, jnp.inf)]
    lp, lq = log_pq(jnp.stack(pts + ([x_at(hi0)[0]] if ceiling else [])))
    f = jnp.where(upper, ltq - lq, lp - ltp)
    sign_change = (f[0] * f[1] <= 0.0) | (f[0] * f[2] <= 0.0)
    judged = f[0] if seam is None else jnp.where(x > seam, ltq - lq[0], lp[0] - ltp)
    bar = lim.rtol if rtol is None else rtol
    accepted = (jnp.abs(jnp.expm1(judged)) <= bar) | sign_change
    above = f[3] < 0.0 if ceiling else jnp.zeros(shape, bool)
    # the root is below the smallest normal x when P there exceeds the
    # target; at that x, P = x^a / Gamma(a + 1) to rounding, so no table
    lp_floor = a * lim.lo - jax.scipy.special.gammaln(a + 1.0)
    below = ~upper & (lp_floor > ltp)
    return x, accepted, below, above


def finish_inv(p, x, accepted, below, bad_a):
    """gammaincinv's contract on a solve of p: quantiles below the
    smallest normal float are 0.0 (the scipy convention), an unaccepted
    solve is nan, p = 0 and 1 give 0 and inf, and nan p, p outside [0, 1]
    and a bad shape give nan."""
    x = jnp.where(below, 0.0, jnp.where(accepted, x, jnp.nan))
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, jnp.inf, x))
    return jnp.where(jnp.isnan(p) | (p < 0.0) | (p > 1.0) | bad_a, jnp.nan, x)


def finish_cinv(y, x, accepted, below, above, bad_a):
    """gammainccinv's: the same, with quantiles past the largest float
    inf, and y = 0 and 1 giving inf and 0."""
    x = jnp.where(below, 0.0, jnp.where(accepted, x, jnp.nan))
    x = jnp.where(above, jnp.inf, x)
    x = jnp.where(y <= 0.0, jnp.inf, jnp.where(y >= 1.0, 0.0, x))
    return jnp.where(jnp.isnan(y) | (y < 0.0) | (y > 1.0) | bad_a, jnp.nan, x)
