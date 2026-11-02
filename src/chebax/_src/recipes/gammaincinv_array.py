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
resolution for any finite a, since the kernel works in t = ln(x/a).

_ITERS is twice the most (Halley) steps any of 1e5 random lanes needed
for a tail probability within the residual bar of the 14-step answer: 4
in float64 and 3 in float32, for a log-uniform on [0.1, 1e30] (1e15 in
float32) and p log-uniform down to the smallest normal float, half of
them survival targets. Newton needed 7 and 4. The scalar path runs
solver_limits' 40 and 24.

Derivatives come from the implicit function theorem at the root x, with
L = ln P on the lower lanes and ln Q on the upper ones, so that L(a, x)
is the target's log lt: dx = (d lt - L_a da) / L_x. Both partials are AD
of the same _log_pq the solve inverted, so the result is the derivative
of the inverse of that approximation, in both AD modes and to any order;
second derivatives in a work here, where the scalar path needs d2P/da2
from jax and raises. The density is exp(lt) |L_x|, which stays finite
where p or L_x alone would not.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_array as _ga
from chebax._src.recipes import gammaincinv_kernel as _k

_ITERS = {"float64": 8, "float32": 6}
# above this a, 1/pdf comes from the closed form at the unrounded root
_CLOSED = 1e3
# the residual bar, in units of eps |ln target|
_BAR = 256.0


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
    return _k.solve(a, *targets, log_pq, iters, ceiling, rtol=bar, halley=True)


@jax.custom_jvp
def inv(a, p):
    """x with P(a, x) = p per element; contract on gammaincinv_array."""
    bad, a = _domain(a)
    pc = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
    x, ok, below, _ = _solve(a, _k.inv_targets(pc))
    return _k.finish_inv(p, x, ok, below, bad)


@jax.custom_jvp
def cinv(a, y):
    """x with Q(a, x) = y per element; contract on gammainccinv_array."""
    bad, a = _domain(a)
    yc = jnp.where((y > 0.0) & (y < 1.0), y, 0.5)
    x, ok, below, above = _solve(a, _k.cinv_targets(yc), ceiling=True)
    return _k.finish_cinv(y, x, ok, below, above, bad)


def _implicit(a, x, upper, lt):
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
    ls, (l_x, l_a) = jax.vmap(lambda u, v: jax.jvp(side, (a, xs), (u, v)))(*tans)
    # pdf = exp(lt) |L_x| at the root, which stays finite where p or L_x
    # alone would not
    inv_pdf = jnp.exp(-(lt + jnp.log(jnp.abs(l_x))))
    # L_x at the float x is L_x at the root only to within its change over
    # one float step, about eps sqrt(a) (1 + |z|): 0.3% at a = 1e30. Above
    # _CLOSED take the density in closed form at the root itself, reached
    # from x by one Halley step in t = ln(x/a). The curvature is exact:
    # with g = ln(x pdf), L' = e^(g - L), so L'' = L' (g' - L') and
    # g' = -a expm1(t). A Newton step alone left 3.4e-4 in dx/dp at
    # a = 1e30, p = 1e-20 (one float step moves ln P by 0.25 there)
    # Other lanes get inputs at the mode: at a = 0.1, p = 1e-5 the form's
    # partials overflow (x ~ 1e-50), and 0 * inf under the select made
    # d2x/dp2 nan
    closed = a > _CLOSED
    ac = jnp.where(closed, a, 2.0 * _CLOSED)
    xc = jnp.where(closed, xs, ac)
    r = jnp.where(closed, lt - ls[0], 0.0)
    d1 = jnp.where(closed, xs * l_x, 1.0)
    t = _ga._log1p((xc - ac) / ac)
    d2 = d1 * (-ac * jnp.expm1(t) - d1)
    t = t + r / d1 / (1.0 + r * d2 / (2.0 * d1 * d1))
    inv_closed = jnp.exp(jnp.log(ac) + t - _k.log_xpdf(ac, t))
    inv_pdf = jnp.where(closed, inv_closed, inv_pdf)
    return interior, inv_pdf, -l_a / l_x


def _jvp(fn, targets, sign, primals, tangents):
    a, q = primals
    da, dq = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = fn(a, q)
    qc = jnp.where((q > 0.0) & (q < 1.0), q, 0.5)
    ltp, ltq, _, upper = targets(qc)
    interior, inv_pdf, dxda = _implicit(a, x, upper, jnp.where(upper, ltq, ltp))

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
