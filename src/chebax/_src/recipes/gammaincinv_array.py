"""Gamma quantiles with a per element: gammaincinv_array and
gammainccinv_array (inputs arrive normalized and broadcast from
gammaincinv.py).

The scalar path picks its tables with lax.conds on its one a, and under
vmap those conds run every branch for every element. This path runs the
shared solve (gammaincinv_kernel) on gammainc_array's ln P and ln Q, so a
step costs one patch evaluation per element whatever the spread of a,
with no loop and no lgamma inside the residual.

The residual bar is tighter than the scalar path's: both logs are
relatively accurate here, so a converged solve sits at the rounding of
its target's log, and the bar is 256 eps |ln target| (or x within one
float step of the root). A solve that stops short, as one did under an
XLA miscompile (below), returns nan instead of a quantile 1e-7 off.

The contract is the scalar path's, on gammainc_array's domain: a >= 0.1,
and below that, for a = inf and for a = nan, the quantile is nan (the
scalar path falls back to jax's gammainc below 0.1). The solve keeps its
resolution for any finite a, since the kernel works in t = ln(x/a), and
above a = 1e36 (1e18 in float32) the quantile is a itself (_FLAT).

_ITERS is twice the most (Halley) steps any of 1e5 random lanes needed
for a tail probability within the residual bar of the 14-step answer: 4
in float64 and 3 in float32, for a log-uniform on [0.1, 1e30] (1e15 in
float32) and p log-uniform down to the smallest normal float, half of
them survival targets. Newton needed 7 and 4. The scalar path runs
solver_limits' 40 and 24.

Derivatives come from the implicit function theorem at the root x, with
L = ln P on the lower lanes and ln Q on the upper ones, so that L(a, x)
is the target's log lt: dx = (d lt - L_a da) / L_x. Both partials are AD
of the same _log_pq the solve inverted, in both AD modes; second
derivatives in a work here, where the scalar path needs d2P/da2 from jax
and raises. 1/pdf takes its value from exp(lt) |L_x|, finite where p or
L_x alone would not be, and its derivatives from the closed-form density
(_implicit). Above a = 1e3 both derivatives come from the root found
again, unrounded (_closed_root). Second derivatives are tested; higher
ones are AD of the same forms, and are nan where x^2 underflows.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_array as _ga
from chebax._src.recipes import gammaincinv_kernel as _k

_ITERS = {"float64": 8, "float32": 6}
# above this a, 1/pdf comes from the closed form at the unrounded root,
# reached in _REFINE Halley steps
_CLOSED = 1e3
_REFINE = 6
# the residual bar, in units of eps |ln target|
_BAR = 256.0
# Above this a every p in (0, 1) has |z| / sqrt(a) < eps/4 (|z| <= 38.5,
# 14.1 in float32, down to the smallest subnormal p), so the quantile
# rounds to a itself. The solve there would only meet the top of the
# float range: from a = 4.5e307, 1/a is subnormal, jit flushed it to 0 in
# (x - a)/a, and the float-step check rejected the right x
_FLAT = {"float64": 1e36, "float32": 1e18}


def _domain(a):
    """(a outside the tables, a with those lanes moved inside)."""
    bad = ~((a >= _ga._t.ALO) & (a < jnp.inf))
    return bad, jnp.where(bad, 1.0, a)


def _solve(a, targets, ceiling=False):
    def log_pq(x):
        # a + 0 x keeps every a-only step of _log_pq inside the Newton loop.
        # Hoisted out of it (XLA's while-loop invariant code motion), they
        # came back wrong on AVX-512 CPUs with jax 0.11's CPU fusion
        # emitters: x off by 8e-9 at a = 30, p = 0.956 among 128 lanes,
        # and nan on others. 0 x does not fold for floats.
        return _ga._log_pq(a + 0.0 * jnp.where(jnp.isfinite(x), x, 0.0), x)

    ltp, ltq, _, upper = targets
    eps = jnp.finfo(a.dtype).eps
    bar = _BAR * eps * jnp.maximum(1.0, jnp.abs(jnp.where(upper, ltq, ltp)))
    iters = _ITERS[a.dtype.name]
    return _k.solve(a, *targets, log_pq, iters, ceiling=ceiling, rtol=bar, halley=True)


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
    """ln pdf(x) of Gamma(a, 1) for x > 0, in forms whose x-derivative,
    (a - 1)/x - 1, has no cancellation: directly to a = 100, and above as
    log_xpdf(a, t) - ln x in t = ln(x/a)."""
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
    """(1/pdf, dx/da) at the unrounded root for the lanes with a > _CLOSED
    (the others solve a harmless stand-in), nan where it did not converge.

    L_x at the float x is L_x at the root only to within its change over
    one float step, about eps sqrt(a) (1 + |z|): 0.3% at a = 1e30, and at
    a = 1e35, p = 1e-20 x rounds to a itself, nine standard deviations
    from the root. So the root is found again in t = ln(x/a), unrounded,
    with _log_pq reading d = e^t - 1 exactly: Halley steps from the
    Wilson-Hilferty point, whose error at these shapes is O(z^3/a), with
    slope and curvature in closed form, L' = +-e^(g - L) and
    L'' = L' (g' - L') for g = ln(x pdf), g' = -a expm1(t). At most 3
    were needed on 2e4 random lanes per dtype from a = 1e3 to the largest
    float. The density is then in closed form at that t."""
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

        # x = a only sets the endpoint lanes, and a (1 + d) can overflow
        lp, lq = _ga._log_pq(av, av, jnp.expm1(t))
        lv = jnp.where(up, lq, lp)
        # the slope in closed form, as in the solve: no derivative of
        # _log_pq in the loop, which is most of the compile time
        l1 = jnp.where(up, -1.0, 1.0) * jnp.exp(_k.log_xpdf(ac, t) - lv)
        f = lv - ltc
        c = 1.0 - f * (-ac * jnp.expm1(t) - l1) / (2.0 * l1)
        return t - f / l1 / jnp.where(c > 0.5, c, 1.0), f

    start = (t0, jnp.full_like(t0, jnp.inf))
    t, f = jax.lax.fori_loop(0, _REFINE, step, start)
    bar = _BAR * jnp.finfo(ac.dtype).eps * jnp.maximum(1.0, jnp.abs(ltc))
    inv_pdf = jnp.exp(jnp.log(ac) + t - _k.log_xpdf(ac, t))

    # dx/da at the same root, from L's slopes in ln a and in d at fixed d:
    # with x = a (1 + d), dx/da = (1 + d) - (dL/d ln a) / (dL/dd). Slopes
    # in a and x carry tangents of size 1/a, below the smallest normal
    # float from a ~ 2e307, where they were flushed to 0 and dx/da was nan
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
    """(interior, 1/pdf, dx/da) at the root x, from L = ln P on the lower
    lanes and ln Q on the upper ones. Lanes without an interior root get
    placeholders the caller masks."""
    interior = jnp.isfinite(x) & (x > 0.0)
    _, a = _domain(a)
    xs = jnp.where(interior, x, a)

    def side(aa, xx):
        lp, lq = _ga._log_pq(aa, xx)
        return jnp.where(upper, lq, lp)

    # L_x and L_a from one jvp batched over two unit tangents, so the
    # program holds one copy of _log_pq's derivative and not two. da only
    # multiplies the result, which keeps the rule linear in it (reverse
    # mode transposes that)
    one, zero = jnp.ones_like(xs), jnp.zeros_like(xs)
    tans = (jnp.stack([zero, one]), jnp.stack([one, zero]))
    _, (l_x, l_a) = jax.vmap(lambda u, v: jax.jvp(side, (a, xs), (u, v)))(*tans)
    # pdf = exp(lt) |L_x| at the root: the tables' own slope, finite where p
    # or L_x alone would not be. Differentiated again it puts two terms of
    # size 1/x against each other (d2x/dp2 was 98304 for 1 at a = 1,
    # p = 1e-20), so its derivatives come from the closed form, whose
    # x-derivative is (a - 1)/x - 1, and only its value from L_x
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

    # the masks go in the coefficients, not on the tangent: a nan or 0
    # placed on the tangent is not linear in it, and reverse mode then
    # returned 0 where x is nan
    def coef(c):
        return jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, c, 0.0))

    dx = jnp.zeros_like(x)
    if not isinstance(da, zero):
        dx = dx + coef(dxda) * da
    if not isinstance(dq, zero):
        # P = p (or Q = y) at the root, so dx/dp = 1/pdf and dx/dy = -1/pdf
        dx = dx + coef(sign * inv_pdf) * dq
    return x, dx


inv.defjvp(lambda p, t: _jvp(inv, _k.inv_targets, 1.0, p, t), symbolic_zeros=True)
cinv.defjvp(lambda p, t: _jvp(cinv, _k.cinv_targets, -1.0, p, t), symbolic_zeros=True)
