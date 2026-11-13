"""gammaincinv_array and gammainccinv_array: a table of the quantile itself
(gammaincinv_array_gen: one 2-D patch per element, 1e-8 in x) and then a
Newton step on gammainc_array's ln P and ln Q, one patch whatever the spread
of a (the scalar path's lax.conds run every branch under vmap). One step in
float64, which from 1e-8 lands within ln P's own error, as two did; none in
float32, where the table is within rounding.

The float64 check is the start's residual: within 1e-6 max(1, |ln target|)
(100x the worst over a = 0.1 to 1e30, both tails), or the step within 64
floats in t, where x cannot resolve the table's 1e-8 (a > 1e10). A start
outside both gives nan. a >= 0.1 (nan below, at inf and nan); above 1e36
(1e18 in float32) the quantile is a.

Derivatives: implicit function theorem on the same _log_pq, dx = (d lt - L_a da)
/ L_x with L = ln P (ln Q upper), both AD modes, second order in a too. With a
differentiated, L_x and L_a come from one batched jvp of _log_pq; with p alone,
L_x = +-pdf/F in closed form and one ln F at x. 1/pdf takes its value from
exp(lt) / |L_x| and its derivatives from the closed form; above a = 1e3 both
come from the unrounded root, one Newton step in t from the float x
(_closed_root). Higher derivatives are AD of the same forms, nan where x^2
underflows.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_array as _ga
from chebax._src.recipes import gammaincinv_start as _s
from chebax._src.recipes import gammaincinv_kernel as _k
from chebax._src.recipes._common import log1p

# float64: one Newton step from the table's 1e-8 lands within ln P's own
# error, so the start's residual is the check; float32: the table is within
# rounding
_STEPS = {"float64": 1, "float32": 0}
# the start's residual bar, in units of max(1, |ln target|): 100x the worst seen
_START_BAR = 1e-6
# above _CLOSED the derivatives are taken at the unrounded root: one Newton step
# from the t the float x stands for, exact while x's rounding squared is below
# the dtype's (eps sqrt(a) (1 + |z|) below ~1e-8, to _ONE_STEP); above, _REFINE
# Halley steps, from Wilson-Hilferty beyond _WH, where x cannot resolve the root
# and WH is within 1e-10 of it, judged on 256 eps |ln target|
_CLOSED = 1e3
_ONE_STEP = {"float64": 1e11, "float32": 1e4}
_REFINE = 2
_WH = 1e15
_BAR = 256.0
# above this a, |z| / sqrt(a) < eps/4 for every p, so the quantile rounds to a;
# solving there meets subnormal 1/a from a = 4.5e307
_FLAT = {"float64": 1e36, "float32": 1e18}


def _domain(a):
    """(a outside the tables, a with those lanes moved inside)."""
    bad = ~((a >= _ga._t.ALO) & (a < jnp.inf))
    return bad, jnp.where(bad, 1.0, a)


def _solve(a, targets, ceiling=False):
    """(x, accepted, root below the normal floor, root above the ceiling)."""
    ltp, ltq, z, upper = targets

    def log_pq(x):
        # a + 0 x keeps a-only work in the loop: XLA's hoisted copy is wrong on
        # AVX-512 CPUs (jax 0.11), x 8e-9 off at a = 30, p = 0.956, nan elsewhere
        return _ga._log_pq(a + 0.0 * jnp.where(jnp.isfinite(x), x, 0.0), x)

    eps = jnp.finfo(a.dtype).eps
    bar = _BAR * eps * jnp.maximum(1.0, jnp.abs(jnp.where(upper, ltq, ltp)))
    t0 = _s.start(a, ltp, ltq, z, upper)
    steps = _STEPS[a.dtype.name]
    return _k.table_solve(
        a, *targets, log_pq, t0, steps, ceiling=ceiling, rtol=bar, start_bar=_START_BAR
    )


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
    direct to a = 10, log_xpdf(a, t) - ln x above."""
    stir = a > _k._STIR
    ab = jnp.where(stir, a, 1e3)
    xb = jnp.where(stir, x, ab)
    d = (xb - ab) / ab
    near = jnp.abs(d) < 0.5
    far = jnp.log(jnp.where(near, 1.0, xb / ab))
    t = jnp.where(near, log1p(jnp.where(near, d, 0.0)), far)
    am = jnp.where(stir, 2.0, a)
    xm = jnp.where(stir, 1.0, x)
    direct = (am - 1.0) * jnp.log(xm) - xm - _k.lgamma(am)
    # Stirling's form, no lnGamma
    return jnp.where(stir, _k.log_xpdf(ab, t, lga=0.0) - jnp.log(xb), direct)


def _closed_root(a, x, upper, lt, z, lv, need_da):
    """(1/pdf, dx/da) at the unrounded root for a > _CLOSED, nan if unconverged;
    lv is ln F at the float x, dx/da only with need_da (python bool). At the
    float x, L_x is off by eps sqrt(a) (1 + |z|), 0.3% at a = 1e30, so the root
    is taken in t from the t the float x stands for (constants above)."""
    closed = a > _CLOSED
    ac = jnp.where(closed, a, 2.0 * _CLOSED)
    up = upper & closed
    sgn = jnp.where(up, -1.0, 1.0)
    ltc = jnp.where(closed, lt, jnp.log(0.5))
    lvc = jnp.where(closed, lv, ltc)
    xc = jnp.where(closed, x, ac)
    # one Newton step: the slope of ln F in t is x pdf / F (Stirling's form)
    t_x = log1p((xc - ac) / ac)
    t = t_x - (lvc - ltc) / (sgn * jnp.exp(_k.log_xpdf(ac, t_x, lga=0.0) - lvc))
    far = closed & (ac > _ONE_STEP[ac.dtype.name])

    def refine():
        zc = jnp.where(far, z, 0.0)
        t_wh = 3.0 * jnp.log1p(zc / (3.0 * jnp.sqrt(ac)) - 1.0 / (9.0 * ac))

        def step(_, carry):
            t, _ = carry
            # a + 0 t: no a-only work hoisted out of the loop (see _solve)
            av = ac + 0.0 * t
            # x = a only sets the endpoint lanes; a (1 + d) can overflow
            lp, lq = _ga._log_pq(av, av, jnp.expm1(t))
            lv = jnp.where(up, lq, lp)
            l1 = sgn * jnp.exp(_k.log_xpdf(ac, t, lga=0.0) - lv)
            f = lv - ltc
            c = 1.0 - f * (-ac * jnp.expm1(t) - l1) / (2.0 * l1)
            return t - f / l1 / jnp.where(c > 0.5, c, 1.0), f

        t0 = jnp.where(ac <= _WH, t_x, t_wh)
        tf, f = jax.lax.fori_loop(0, _REFINE, step, (t0, jnp.full_like(t0, jnp.inf)))
        bar = _BAR * jnp.finfo(ac.dtype).eps * jnp.maximum(1.0, jnp.abs(ltc))
        return jnp.where(far, tf, t), ~far | (jnp.abs(f) <= bar)

    t, ok = jax.lax.cond(
        jnp.any(far), refine, lambda: (t, jnp.ones(jnp.shape(t), bool))
    )
    inv_pdf = jnp.exp(jnp.log(ac) + t - _k.log_xpdf(ac, t, lga=0.0))
    if not need_da:
        return jnp.where(ok, inv_pdf, jnp.nan), jnp.zeros_like(inv_pdf)

    # dx/da = (1 + d) - (dL/d ln a) / (dL/dd): slopes in a and x carry 1/a,
    # subnormal from a ~ 2e307. Both slopes from the table, as in _implicit
    d = jnp.expm1(t)

    def side(aa, dd):
        lp, lq = _ga._log_pq(aa, aa, dd)
        return jnp.where(up, lq, lp)

    zero = jnp.zeros_like(ac)
    tans = (jnp.stack([ac, zero]), jnp.stack([zero, jnp.ones_like(ac)]))
    _, (l_lna, l_d) = jax.vmap(lambda u, v: jax.jvp(side, (ac, d), (u, v)))(*tans)
    dxda = 1.0 + d - l_lna / l_d
    return jnp.where(ok, inv_pdf, jnp.nan), jnp.where(ok, dxda, jnp.nan)


def _implicit(a, x, upper, lt, z, need_da):
    """(interior, 1/pdf, dx/da) at the root x, dx/da only with need_da (python
    bool, zeros otherwise); other lanes are placeholders."""
    interior = jnp.isfinite(x) & (x > 0.0)
    _, a = _domain(a)
    xs = jnp.where(interior, x, a)

    def side(aa, xx):
        lp, lq = _ga._log_pq(aa, xx)
        return jnp.where(upper, lq, lp)

    # 1/pdf takes its value from exp(lt) / |L_x| (the pdf at the float x moved
    # to the root to first order) and its derivatives from the closed form:
    # differentiating L_x cancels two 1/x terms (d2x/dp2 98304 for 1 at a = 1,
    # p = 1e-20)
    cf = _log_pdf(a, xs)
    if need_da:
        # L_x and L_a both from the table, one batched jvp: their errors cancel
        # in dx/da = -L_a / L_x, a closed-form L_x's do not (with closed forms
        # here and in _closed_root, 3e-14 against 1e-15 for a in [20, 1e4],
        # experiments/53's references)
        one, zero = jnp.ones_like(xs), jnp.zeros_like(xs)
        tans = (jnp.stack([zero, one]), jnp.stack([one, zero]))
        lvs, (l_x, l_a) = jax.vmap(lambda u, v: jax.jvp(side, (a, xs), (u, v)))(*tans)
        lv = lvs[0]
        inv_pdf = jnp.exp(
            -(cf + jax.lax.stop_gradient(lt + jnp.log(jnp.abs(l_x)) - cf))
        )
        dxda = -l_a / l_x
    else:
        # L_x = +-pdf/F in closed form: one ln F at x, no derivative of the table
        lv = side(a, xs)
        inv_pdf = jnp.exp(-(cf + jax.lax.stop_gradient(lt - lv)))
        dxda = jnp.zeros_like(xs)
    closed = a > _CLOSED

    def at_root():
        inv_c, dxda_c = _closed_root(a, xs, upper, lt, z, lv, need_da)
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
    interior, inv_pdf, dxda = _implicit(a, x, upper, lt, z, not isinstance(da, zero))

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
