"""gammaincinv_array and gammainccinv_array: a table of the quantile itself
(gammaincinv_array_gen: one 2-D patch per element, 1e-8 in x) and then Newton
steps on gammainc_array's ln P and ln Q, one patch per step whatever the
spread of a (the scalar path's lax.conds run every branch under vmap). Two
steps in float64: the second's residual, the first's result, is the check;
none in float32, where the table is within rounding.

Both logs are relatively accurate, so the bar is 256 eps |ln target| (or a
step below rounding in t, the root within one float of x), and a start the
step does not bring inside it gives nan. a >= 0.1 (nan below, at inf and
nan); above 1e36 (1e18 in float32) the quantile is a.

Derivatives: implicit function theorem on the same _log_pq, dx = (d lt - L_a da)
/ L_x with L = ln P (ln Q upper), both AD modes, second order in a too. 1/pdf
takes its value from exp(lt) |L_x| and its derivatives from the closed form;
above a = 1e3 both come from the root found again, unrounded (_closed_root).
Higher derivatives are AD of the same forms, nan where x^2 underflows.
"""

import functools

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_array as _ga
from chebax._src.recipes import gammaincinv_array_table as _t
from chebax._src.recipes import gammaincinv_kernel as _k
from chebax._src.recipes._common import solver_limits

_STEPS = {"float64": 2, "float32": 0}
# above this a, 1/pdf from the closed form at the root, in _REFINE Halley steps
_CLOSED = 1e3
_REFINE = 6
# the residual bar, in units of eps |ln target|
_BAR = 256.0
# above this a, |z| / sqrt(a) < eps/4 for every p, so the quantile rounds to a;
# solving there meets subnormal 1/a from a = 4.5e307
_FLAT = {"float64": 1e36, "float32": 1e18}
_LO0, _LO1, _LO2, _CE1, _CE2, _UP0, _UP1, _UP2 = range(len(_t.NAMES))


@functools.lru_cache(maxsize=2)
def _tables(dtype):
    """(patch blocks, parameter cuts, argument cuts, ln lambda series) in
    the compute dtype."""
    dt = np.dtype(dtype)
    return tuple(
        np.ascontiguousarray(v, dt) for v in (_t.TABLE, _t.PCUTS, _t.YCUTS, _t.LAM)
    )


def _domain(a):
    """(a outside the tables, a with those lanes moved inside)."""
    bad = ~((a >= _ga._t.ALO) & (a < jnp.inf))
    return bad, jnp.where(bad, 1.0, a)


def _x_w(a, r):
    """x with x - (a-1) ln x = r, four Newton steps from max(r, 1.5 a + 1);
    the upper zone's leading part (q = e^-r / Gamma(a) to first order)."""
    x = jnp.maximum(r, 1.5 * a + 1.0)
    for _ in range(4):
        g = x - (a - 1.0) * jnp.log(x) - r
        x = jnp.maximum(x - g / (1.0 - (a - 1.0) / x), 0.5 * x)
    return x


def _start(a, ltp, ltq, z, upper):
    """t0 = ln(x0/a) from the quantile table (zones in the generator's
    docstring); relatively accurate at large a, where t ~ z/sqrt(a)."""
    tab, pcuts, ycuts, lam = (jnp.asarray(v) for v in _tables(a.dtype.name))
    npar, narg, offset = (jnp.asarray(v) for v in (_t.NPAR, _t.NARG, _t.OFFSET))
    ky, kp = tab.shape[1:]
    gammaln = jax.scipy.special.gammaln

    panel = jnp.where(a < 1.0, 0, jnp.where(a < _t.ASPLIT, 1, 2))
    # lower: s = (p Gamma(a+1))^(1/a) / (a+1), as a log
    ln_y = (ltp + gammaln(a + 1.0)) / a
    s = jnp.exp(ln_y - jnp.log1p(a))
    s_max = jnp.where(a < 1.0, _t.S_SMALL, _t.S_BIG)
    # centre: Temme's eta0
    eta0 = z / jnp.sqrt(a)
    # upper: the Lambert-W-type leading part
    r = -ltq - gammaln(a)
    x_w = _x_w(a, r)
    tau = (a + 1.0) / x_w
    kind = jnp.where(s <= s_max, 0, jnp.where((a >= 1.0) & (eta0 <= _t.EHI), 1, 2))
    zone = jnp.select(
        [kind == 0, kind == 1],
        [_LO0 + panel, _CE1 + jnp.maximum(panel - 1, 0)],
        _UP0 + panel,
    )
    w = 1.0 / a
    par = jnp.select([kind == 1, panel == 2], [w, _t.ASPLIT / a], jnp.log1p(a))
    arg = jnp.select([kind == 0, kind == 1], [s, eta0], tau)
    i, sp = _ga._locate(par, pcuts[zone], npar[zone])
    j, sy = _ga._locate(arg, ycuts[zone], narg[zone])
    block = tab[offset[zone] + i * narg[zone] + j]
    f = jnp.einsum(
        "...kj,...k,...j->...", block, _ga._basis(sy, ky), _ga._basis(sp, kp)
    )

    # centre: eta = eta0 + w eps, then ln lambda = eta g(eta)
    eta = jnp.clip(eta0 + w * f, _t.LAM_BAND[0], _t.LAM_BAND[1])
    te = 2.0 * (eta - _t.LAM_BAND[0]) / (_t.LAM_BAND[1] - _t.LAM_BAND[0]) - 1.0
    b1 = b2 = jnp.zeros_like(te)
    for c in lam[:0:-1]:
        b1, b2 = 2.0 * te * b1 - b2 + c, b1
    g = te * b1 - b2 + lam[0]
    ln_a = jnp.log(a)
    return jnp.select(
        [kind == 0, kind == 1],
        [ln_y - ln_a + jnp.log1p(f), eta * g],
        jnp.log(x_w) - ln_a + jnp.log1p(f),
    )


def _solve(a, targets, ceiling=False):
    """(x, accepted, root below the normal floor, root above the ceiling):
    the table start and _STEPS Newton steps in t = ln(x/a) on the log of the
    target's own tail, each from the t its rounded x stands for."""
    ltp, ltq, z, upper = targets
    lim = solver_limits(a.dtype.name)
    eps = jnp.finfo(a.dtype).eps
    ln_a = jnp.log(a)
    lo0 = lim.lo - ln_a
    hi0 = jnp.maximum(lim.v_hi - ln_a, 0.0)
    t = jnp.clip(_start(a, ltp, ltq, z, upper), lo0, hi0)
    bar = _BAR * eps * jnp.maximum(1.0, jnp.abs(jnp.where(upper, ltq, ltp)))
    accepted = jnp.ones_like(t, dtype=bool)
    above = jnp.zeros_like(t, dtype=bool)
    for k in range(_STEPS[a.dtype.name]):
        t = _k.snap_t(a, t)
        x, _ = _k.x_of(a, t)
        # a + 0 x keeps a-only work in the loop: XLA's hoisted copy is wrong on
        # AVX-512 CPUs (jax 0.11), x 8e-9 off at a = 30, p = 0.956, nan elsewhere
        lp, lq = _ga._log_pq(a + 0.0 * jnp.where(jnp.isfinite(x), x, 0.0), x)
        f = jnp.where(upper, ltq - lq, lp - ltp)
        dt = f / jnp.exp(_k.log_xpdf(a, t) - jnp.where(upper, lq, lp))
        if k > 0:
            # the residual the previous step left is inside the bar, or this
            # step is below rounding in t; the table start itself is not judged
            accepted = (jnp.abs(jnp.expm1(f)) <= bar) | (
                jnp.abs(dt) <= 2.0 * eps * jnp.maximum(1.0, jnp.abs(t))
            )
        t = t - dt
        above = above | (ceiling & (t > hi0))
        t = jnp.clip(t, lo0, hi0)
    x, _ = _k.x_of(a, _k.snap_t(a, t))
    # root below the smallest normal x, where P = x^a / Gamma(a + 1) to rounding
    lp_floor = a * lim.lo - jax.scipy.special.gammaln(a + 1.0)
    below = ~upper & (lp_floor > ltp)
    return x, accepted, below, above


@jax.custom_jvp
def inv(a, p):
    """x with P(a, x) = p per element; contract on gammaincinv_array."""
    bad, a = _domain(a)
    pc = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
    x, ok, below, _ = _solve(a, _k.inv_targets(pc))
    flat = a > _FLAT[a.dtype.name]
    x = jnp.where(flat, a, x)
    return _k.finish_inv(p, x, ok | flat, below & ~flat, bad)


@jax.custom_jvp
def cinv(a, y):
    """x with Q(a, x) = y per element; contract on gammainccinv_array."""
    bad, a = _domain(a)
    yc = jnp.where((y > 0.0) & (y < 1.0), y, 0.5)
    x, ok, below, above = _solve(a, _k.cinv_targets(yc), ceiling=True)
    flat = a > _FLAT[a.dtype.name]
    x = jnp.where(flat, a, x)
    return _k.finish_cinv(y, x, ok | flat, below & ~flat, above & ~flat, bad)


def _log_pdf(a, x):
    """ln pdf(x) of Gamma(a, 1), x > 0, with an uncancelled x-derivative:
    direct to a = 100, log_xpdf(a, t) - ln x above."""
    stir = a > 100.0
    ab = jnp.where(stir, a, 1e3)
    xb = jnp.where(stir, x, ab)
    d = (xb - ab) / ab
    near = jnp.abs(d) < 0.5
    far = jnp.log(jnp.where(near, 1.0, xb / ab))
    t = jnp.where(near, _ga._log1p(jnp.where(near, d, 0.0)), far)
    am = jnp.where(stir, 2.0, a)
    xm = jnp.where(stir, 1.0, x)
    direct = (am - 1.0) * jnp.log(xm) - xm - jax.scipy.special.gammaln(am)
    return jnp.where(stir, _k.log_xpdf(ab, t) - jnp.log(xb), direct)


def _closed_root(a, upper, lt, z):
    """(1/pdf, dx/da) at the unrounded root for a > _CLOSED, nan if unconverged.
    At the float x, L_x is off by eps sqrt(a) (1 + |z|), 0.3% at a = 1e30, so the
    root is found again in t: Halley from Wilson-Hilferty with closed-form slope
    and curvature, at most 3 steps on 2e4 lanes per dtype."""
    closed = a > _CLOSED
    ac = jnp.where(closed, a, 2.0 * _CLOSED)
    up = upper & closed
    ltc = jnp.where(closed, lt, jnp.log(0.5))
    zc = jnp.where(closed, z, 0.0)
    t0 = 3.0 * jnp.log1p(zc / (3.0 * jnp.sqrt(ac)) - 1.0 / (9.0 * ac))

    def step(_, carry):
        t, _ = carry
        # a + 0 t: no a-only work hoisted out of the loop (see _solve)
        av = ac + 0.0 * t

        # x = a only sets the endpoint lanes; a (1 + d) can overflow
        lp, lq = _ga._log_pq(av, av, jnp.expm1(t))
        lv = jnp.where(up, lq, lp)
        # closed-form slope: a _log_pq derivative here is most of the compile time
        l1 = jnp.where(up, -1.0, 1.0) * jnp.exp(_k.log_xpdf(ac, t) - lv)
        f = lv - ltc
        c = 1.0 - f * (-ac * jnp.expm1(t) - l1) / (2.0 * l1)
        return t - f / l1 / jnp.where(c > 0.5, c, 1.0), f

    start = (t0, jnp.full_like(t0, jnp.inf))
    t, f = jax.lax.fori_loop(0, _REFINE, step, start)
    bar = _BAR * jnp.finfo(ac.dtype).eps * jnp.maximum(1.0, jnp.abs(ltc))
    inv_pdf = jnp.exp(jnp.log(ac) + t - _k.log_xpdf(ac, t))

    # dx/da = (1 + d) - (dL/d ln a) / (dL/dd): slopes in a and x carry 1/a,
    # subnormal from a ~ 2e307
    def side(aa, d):
        lp, lq = _ga._log_pq(aa, aa, d)
        return jnp.where(up, lq, lp)

    d = jnp.expm1(t)
    zero = jnp.zeros_like(ac)
    tans = (jnp.stack([ac, zero]), jnp.stack([zero, jnp.ones_like(ac)]))
    _, (l_lna, l_d) = jax.vmap(lambda u, v: jax.jvp(side, (ac, d), (u, v)))(*tans)
    dxda = 1.0 + d - l_lna / l_d
    ok = jnp.abs(f) <= bar
    return jnp.where(ok, inv_pdf, jnp.nan), jnp.where(ok, dxda, jnp.nan)


def _implicit(a, x, upper, lt, z):
    """(interior, 1/pdf, dx/da) at the root x; other lanes are placeholders."""
    interior = jnp.isfinite(x) & (x > 0.0)
    _, a = _domain(a)
    xs = jnp.where(interior, x, a)

    def side(aa, xx):
        lp, lq = _ga._log_pq(aa, xx)
        return jnp.where(upper, lq, lp)

    # one batched jvp, so one copy of _log_pq's derivative in the program
    one, zero = jnp.ones_like(xs), jnp.zeros_like(xs)
    tans = (jnp.stack([zero, one]), jnp.stack([one, zero]))
    _, (l_x, l_a) = jax.vmap(lambda u, v: jax.jvp(side, (a, xs), (u, v)))(*tans)
    # value from exp(lt) |L_x|, derivatives from the closed form: differentiating
    # L_x cancels two 1/x terms (d2x/dp2 98304 for 1 at a = 1, p = 1e-20)
    cf = _log_pdf(a, xs)
    ln_pdf = cf + jax.lax.stop_gradient(lt + jnp.log(jnp.abs(l_x)) - cf)
    inv_pdf, dxda = jnp.exp(-ln_pdf), -l_a / l_x
    closed = a > _CLOSED

    def at_root():
        inv_c, dxda_c = _closed_root(a, upper, lt, z)
        return jnp.where(closed, inv_c, inv_pdf), jnp.where(closed, dxda_c, dxda)

    inv_pdf, dxda = jax.lax.cond(jnp.any(closed), at_root, lambda: (inv_pdf, dxda))
    return interior, inv_pdf, dxda


def _jvp(fn, targets, sign, primals, tangents):
    a, q = primals
    da, dq = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = fn(a, q)
    qc = jnp.where((q > 0.0) & (q < 1.0), q, 0.5)
    ltp, ltq, z, upper = targets(qc)
    lt = jnp.where(upper, ltq, ltp)
    interior, inv_pdf, dxda = _implicit(a, x, upper, lt, z)

    # masks on the coefficients, not the tangent, so the rule stays linear
    def coef(c):
        return jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, c, 0.0))

    dx = jnp.zeros_like(x)
    if not isinstance(da, zero):
        dx = dx + coef(dxda) * da
    if not isinstance(dq, zero):
        # dx/dp = 1/pdf, dx/dy = -1/pdf
        dx = dx + coef(sign * inv_pdf) * dq
    return x, dx


inv.defjvp(lambda p, t: _jvp(inv, _k.inv_targets, 1.0, p, t), symbolic_zeros=True)
cinv.defjvp(lambda p, t: _jvp(cinv, _k.cinv_targets, -1.0, p, t), symbolic_zeros=True)
