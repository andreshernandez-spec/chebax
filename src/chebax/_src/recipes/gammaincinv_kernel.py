"""The Gamma quantile solve, given the caller's ln P and ln Q (the gammainc
tables for gammaincinv_scalar, per-element patches for gammaincinv_array).

Safeguarded Newton (Halley on the array path, at most 4 steps against 7) on the
log of the target's own tail, in t = ln(x/a). Newton on P - p gains one e-fold
a step where P is tiny (a = 100, p = 1e-20 still 20 e-folds out after 40); in
the log it takes two. The small target is exact and the other at least 1/2, so
both keep full relative accuracy.

t, not ln x: ln x carries eps ln a, 1e-6 relative in float32 at a = 1e5, and
every solve failed from a = 1e7. x = a + a expm1(t) near the mode reaches every
float, and each step restarts from the t the rounded x stands for (at a = 5e14
in float32 one float step is 1.5 standard deviations).

A solve is accepted within the residual bar, or when the residual changes sign
at a neighbouring float (needed at a = 1e20 in float64).
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes._common import canon_tag, newton_bisect, solver_limits

_LN2PI = math.log(2.0 * math.pi)
# |t| below this, x = a + a expm1(t): x/a in (0.6, 1.65)
_NEAR = 0.5
# e^t - 1 - t = t^2 (1/2! + t/3! + ...), terms past t^19/19! below 1e-23
_PHI = tuple(1.0 / math.factorial(k) for k in range(2, 20))


def inv_targets(pc):
    """(ln p, ln(1-p), ndtri(p), upper lanes) for a lower-tail target p."""
    return (jnp.log(pc), jnp.log1p(-pc), jax.scipy.special.ndtri(pc), pc > 0.5)


def cinv_targets(yc):
    """The same for an upper-tail y; ndtri(1 - y) would saturate below eps."""
    return (jnp.log1p(-yc), jnp.log(yc), -jax.scipy.special.ndtri(yc), yc < 0.5)


def _a_phi(a, t):
    """a (e^t - 1 - t), as (sqrt(a) t)^2 times its series near 0: e^t - 1 - t
    alone is subnormal for |t| < 2e-154, reached at a = 1e307."""
    near = jnp.abs(t) < _NEAR
    tn = jnp.where(near, t, 0.0)
    s = _PHI[-1]
    for c in _PHI[-2::-1]:
        s = s * tn + c
    # stops XLA turning st * st into a * t^2 for constant a: t^2 is the subnormal
    st = jax.lax.optimization_barrier(jnp.sqrt(a) * tn)
    return jnp.where(near, st * st * s, a * (jnp.expm1(t) - t))


def lgamma(a):
    """chebax's gammaln for the quantile paths, imported on first use so that
    gammainc_array, which reads log_xpdf, does not load its tables."""
    from chebax._src.recipes.gammaln import _gammaln_cj

    return _gammaln_cj(a)


def log_xpdf(a, t, small=False, lga=None):
    """ln(x pdf(x)) of Gamma(a, 1) at x = a e^t. Above a = 100, against the
    cancellation, -a (e^t - 1 - t) + ln(a / (2 pi))/2 - S(a), S Stirling's to
    a^-5; small (python constant) means a <= 100 everywhere. lga is lnGamma(a)
    if the caller has it (jax's gammaln otherwise; unread above a = 100)."""
    if lga is None:
        lga = jax.scipy.special.gammaln(a if small else jnp.where(a > 100.0, 1.0, a))
    if small:
        return a * (t + jnp.log(a)) - a * jnp.exp(t) - lga
    stir = a > 100.0
    ab = jnp.where(stir, a, 1000.0)
    ai2 = 1.0 / (ab * ab)
    rem = (1.0 / 12.0 - ai2 * (1.0 / 360.0 - ai2 / 1260.0)) / ab
    lead = 0.5 * jnp.log(ab / (2.0 * math.pi)) - rem
    am = jnp.where(stir, 1.0, a)
    direct = am * (t + jnp.log(am)) - am * jnp.exp(t) - jnp.where(stir, 0.0, lga)
    return jnp.where(stir, lead - _a_phi(ab, t), direct)


def x_of(a, t):
    """(x = a e^t, near-mode lanes): a + a expm1(t) for |t| < _NEAR above
    a = 2, which reaches every float near the mode; below a = 2, a e^t can be
    subnormal where x is not, so exp(t + ln a)."""
    big = a > 2.0
    near = big & (jnp.abs(t) < _NEAR)
    far = jnp.where(big, a * jnp.exp(t), jnp.exp(t + jnp.log(a)))
    return jnp.where(near, a + a * jnp.expm1(jnp.where(near, t, 0.0)), far), near


def snap_t(a, t):
    """t restarted from the t its rounded x stands for, near the mode."""
    x, near = x_of(a, t)
    return jnp.where(near, jnp.log1p((x - a) / a), t)


def _start(a, ltp, ltq, z, upper, ln_a):
    """The first t: the larger of Wilson-Hilferty and each tail's asymptotic.
    Below, P <= x^a / Gamma(a + 1) never starts above the root (WH misses a = 20,
    p = 1e-20); above, three fixed-point passes on ln Q. Starting far below the
    root is the costly miss: Newton crawls an e-fold a step (a = 50, y = 1e-8
    stopped 5.8e-10 short); overshoot costs a few steps."""
    # WH is x = a (1 + c)^3, taken as a log so it keeps its digits at large a
    c = z / (3.0 * jnp.sqrt(a)) - 1.0 / (9.0 * a)
    pos = c > -1.0
    t_wh = 3.0 * jnp.log1p(jnp.where(pos, c, 0.0))
    # (ln p + lnGamma(a + 1))/a - ln a, by Stirling above a = 100 since
    # lnGamma(a + 1) overflows from a ~ 1e305
    big = a > 100.0
    ab = jnp.where(big, a, 1e3)
    am = jnp.where(big, 1.0, a)
    stirling = (_LN2PI + jnp.log(ab)) / (2.0 * ab) + 1.0 / (12.0 * ab * ab)
    t_lo = jnp.where(
        big,
        ltp / ab - 1.0 + stirling,
        (ltp + lgamma(am) + jnp.log(am)) / am - jnp.log(am),
    )
    r = -ltq - lgamma(a)
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
    *,
    ceiling=False,
    seam=None,
    rtol=None,
    small=False,
    halley=False,
):
    """x with ln P = ltp (ln Q = ltq on upper lanes), log_pq(x) giving both, z
    ndtri for the start, iters fixed steps. Returns (x, accepted, root below the
    normal floor, root above the ceiling, checked only with ceiling).

    The bar is rtol (default solver_limits'), judged on the target's side, or
    with seam on the side log_pq computes directly: the small tables' Q below
    x = 8 is 1 - P (a = 0.1, y = 1e-4 is 1.4e-3 off in float32). halley and
    small are python constants; small (a <= 10) skips the near-mode rounding and
    the Stirling density, 8-28% of GPU time (experiments/09)."""
    lim = solver_limits(canon_tag())
    ln_a = jnp.log(a)
    lga = lgamma(a)

    def x_at(t):
        if small:
            return jnp.exp(t + ln_a), None
        return x_of(a, t)

    def snap(t):
        return snap_t(a, t)

    def residual(x):
        lp, lq = log_pq(x)
        return jnp.where(upper, ltq - lq, lp - ltp), jnp.where(upper, lq, lp)

    def f_and_df(t):
        x, _ = x_at(t)
        f, side = residual(x)
        # the slope of either residual in t is x pdf / (its own probability)
        df = jnp.exp(log_xpdf(a, t, small, lga) - side)
        if not halley:
            return f, df
        # Halley as Newton with slope df c, exact curvature S'' = S' (g' - S'),
        # g = ln(x pdf); plain Newton where c is small or nan
        s1 = jnp.where(upper, -df, df)
        c = 1.0 - f * (-a * jnp.expm1(t) - s1) / (2.0 * df)
        return f, df * jnp.where(c > 0.5, c, 1.0)

    shape = jnp.shape(ltp)
    lo0 = jnp.broadcast_to(lim.lo - ln_a, shape)
    # at least x = a: within 4 ulps of the largest float the ceiling is below the mode
    hi0 = jnp.broadcast_to(jnp.maximum(lim.v_hi - ln_a, 0.0), shape)
    t0 = jnp.clip(_start(a, ltp, ltq, z, upper, ln_a), lo0, hi0)
    if not small:
        t = snap(newton_bisect(f_and_df, t0, lo0, hi0, iters, snap=snap))
    else:
        t = newton_bisect(f_and_df, t0, lo0, hi0, iters)

    # one log_pq call for all points: each copy of log_pq is compile time
    x, _ = x_at(t)
    pts = [x, jnp.nextafter(x, 0.0), jnp.nextafter(x, jnp.inf)]
    lp, lq = log_pq(jnp.stack(pts + ([x_at(hi0)[0]] if ceiling else [])))
    f = jnp.where(upper, ltq - lq, lp - ltp)
    sign_change = (f[0] * f[1] <= 0.0) | (f[0] * f[2] <= 0.0)
    judged = f[0] if seam is None else jnp.where(x > seam, ltq - lq[0], lp[0] - ltp)
    bar = lim.rtol if rtol is None else rtol
    accepted = (jnp.abs(jnp.expm1(judged)) <= bar) | sign_change
    above = f[3] < 0.0 if ceiling else jnp.zeros(shape, bool)
    # root below the smallest normal x, where P = x^a / Gamma(a + 1) to rounding
    lp_floor = a * lim.lo - lga - ln_a
    below = ~upper & (lp_floor > ltp)
    return x, accepted, below, above


def table_solve(
    a,
    ltp,
    ltq,
    z,
    upper,
    log_pq,
    t0,
    steps,
    *,
    ceiling=False,
    seam=None,
    rtol=None,
    small=False,
    start_bar=None,
):
    """solve's contract from a start t0 within ~1e-8 (gammaincinv_start): steps
    Newton steps in t, the last one's residual, the previous one's result,
    being the check: inside the bar (rtol, judged as in solve), or the step
    below rounding in t. With one step the start is judged instead: residual
    within start_bar max(1, |ln target|) (python float, None for no check), or
    the step within 64 floats in t; with none nothing is. Returns (x,
    accepted, root below the normal floor, root above the ceiling)."""
    lim = solver_limits(a.dtype.name)
    eps = jnp.finfo(a.dtype).eps
    ln_a = jnp.log(a)
    lga = lgamma(a)
    lo0 = lim.lo - ln_a
    hi0 = jnp.maximum(lim.v_hi - ln_a, 0.0)
    bar = lim.rtol if rtol is None else rtol

    def x_at(t):
        if small:
            return jnp.exp(t + ln_a)
        return x_of(a, t)[0]

    t = jnp.clip(t0, lo0, hi0)
    accepted = jnp.ones(jnp.shape(t), bool)
    above = jnp.zeros(jnp.shape(t), bool)
    for k in range(steps):
        if not small:
            t = snap_t(a, t)
        x = x_at(t)
        lp, lq = log_pq(x)
        f = jnp.where(upper, ltq - lq, lp - ltp)
        dt = f / jnp.exp(log_xpdf(a, t, small, lga) - jnp.where(upper, lq, lp))
        judged = f if seam is None else jnp.where(x > seam, ltq - lq, lp - ltp)
        if k > 0:
            accepted = (jnp.abs(jnp.expm1(judged)) <= bar) | (
                jnp.abs(dt) <= 2.0 * eps * jnp.maximum(1.0, jnp.abs(t))
            )
        elif steps == 1 and start_bar is not None:
            # where x cannot resolve the start's 1e-8 (a > 1e10), within 64 floats
            lt = jnp.where(upper, ltq, ltp)
            accepted = (
                jnp.abs(judged) <= start_bar * jnp.maximum(1.0, jnp.abs(lt))
            ) | (jnp.abs(dt) <= 64.0 * eps * jnp.maximum(1.0, jnp.abs(t)))
        t = t - dt
        if ceiling:
            above = above | (t > hi0)
        t = jnp.clip(t, lo0, hi0)
    x = x_at(t if small else snap_t(a, t))
    # root below the smallest normal x, where P = x^a / Gamma(a + 1) to rounding
    lp_floor = a * lim.lo - lga - ln_a
    below = ~upper & (lp_floor > ltp)
    return x, accepted, below, above


def finish_inv(p, x, accepted, below, bad_a):
    """gammaincinv's endpoints, 0.0 for underflow and nan for failed solves."""
    x = jnp.where(below, 0.0, jnp.where(accepted, x, jnp.nan))
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, jnp.inf, x))
    return jnp.where(jnp.isnan(p) | (p < 0.0) | (p > 1.0) | bad_a, jnp.nan, x)


def finish_cinv(y, x, accepted, below, above, bad_a):
    """gammainccinv's, with inf past the largest float."""
    x = jnp.where(below, 0.0, jnp.where(accepted, x, jnp.nan))
    x = jnp.where(above, jnp.inf, x)
    x = jnp.where(y <= 0.0, jnp.inf, jnp.where(y >= 1.0, 0.0, x))
    return jnp.where(jnp.isnan(y) | (y < 0.0) | (y > 1.0) | bad_a, jnp.nan, x)
