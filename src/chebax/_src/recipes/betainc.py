"""Regularized incomplete beta I_x(a, b) for a, b in [0.1, 100], x in [0, 1].

The library's first two-parameter recipe. Structure (DLMF 8.17.7):

    I_x(a, b) = exp( a ln x + b ln(1-x) - ln a - ln B(a,b) + L(a,b,x) )

for x <= 1/2, where L = ln 2F1(a+b, 1; a+1; x) comes from a baked
(x, a, b) tensor (each x-coefficient is a 2-D Chebyshev series in (a, b));
for x > 1/2 the reflection I_x(a,b) = 1 - I_{1-x}(b,a) is used, which
needs the tensor reconstructed at BOTH parameter orders. 1 - x is exact in
f64 for x in [1/2, 1], so the reflection costs no accuracy; users who need
the upper tail 1 - I_x(a, b) to relative accuracy should evaluate
betainc(b, a)(1 - x) themselves.

The (a, b) box is [0.1, 100]^2 since increment 23: four tensor panels
split at 10 per axis (the original [0.1, 10]^2 tensor is the low-low
panel, so in-box evaluation is unchanged to the bit), dispatched by one
lax.switch on the scalar parameters; only the taken branch executes and
each panel keeps its own degrees. Panel edges are ordinary points to
~1e-14 (both neighbors fit the same function).

betainc(a, b) is the eager instance (cached, numpy reconstruction).
betainc_fn(a, b, x) takes a and b as traced scalars (uniform per call,
inside [0.1, 100], unchecked under trace): jax.grad works with respect to
both shape parameters — dI/da and dI/db, which jax itself lacks
(jax#38610) — flowing through gammaln/digamma and chebder along the
parameter axes of the tensor. dI/dx is AD through the formula; its exact
value is the Beta density, which the tests use as an oracle.

Endpoints are handled by hard selects (I(0) = 0, I(1) = 1 exactly, interior
lanes see raw x); gradients at exactly 0 and 1 are masked to the interior
formula's limits and can be infinite in exact math anyway (a < 1 or b < 1).

eval_betainc and tensor_coefs_traced are internal-but-shared: the quantile
toolkit builds betaincinv on them.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.pytree import Recipe
from chebax._src.recipes import betainc_table as _bt
from chebax._src.recipes import betainc_wide_table as _bw
from chebax._src.recipes._common import canon_tag, check_range, edge_slope
from chebax._src.series import ChebSeries, _chebval


# panel registry: (a-high?, b-high?) -> (tensor, a-box, b-box). The
# LOxLO panel is the original tensor, so in-box users pay nothing; the
# wide panels (increment 23) extend the box to [0.1, 100]^2 and are
# reached only through the runtime switch below.
_PANELS = {
    (0, 0): (_bt.TENSOR, (_bt.ALO, _bt.AHI), (_bt.ALO, _bt.AHI)),
    (0, 1): (_bw.TENSOR_LOHI, (_bw.ALO, _bw.ASPLIT), (_bw.ASPLIT, _bw.AHI)),
    (1, 0): (_bw.TENSOR_HILO, (_bw.ASPLIT, _bw.AHI), (_bw.ALO, _bw.ASPLIT)),
    (1, 1): (_bw.TENSOR_HIHI, (_bw.ASPLIT, _bw.AHI), (_bw.ASPLIT, _bw.AHI)),
}


def _smap(v, box=(_bt.ALO, _bt.AHI)):
    return 2.0 * (v - box[0]) / (box[1] - box[0]) - 1.0


def _tensor_coefs_from(tensor, abox, bbox, a, b):
    """x-coefficient vector of ln F at (a, b): Clenshaw in a, then b. numpy."""
    sa, sb = _smap(a, abox), _smap(b, bbox)
    out = []
    for k in range(tensor.shape[0]):
        vb = [algorithms.chebval(sa, tensor[k, :, n]) for n in range(tensor.shape[2])]
        out.append(algorithms.chebval(sb, np.array(vb)))
    return np.array(out)


def _tensor_coefs(a, b):
    return _tensor_coefs_from(_bt.TENSOR, (_bt.ALO, _bt.AHI),
                              (_bt.ALO, _bt.AHI), a, b)


def _panel_of(v):
    return 1 if v > _bw.ASPLIT else 0


def _eager_series_pair(a, b):
    """(cab, cba) ChebSeries at concrete (a, b), panel-dispatched. numpy."""
    ta, ab_, bb_ = _PANELS[(_panel_of(a), _panel_of(b))]
    tb, ab2, bb2 = _PANELS[(_panel_of(b), _panel_of(a))]
    return (ChebSeries(_tensor_coefs_from(ta, ab_, bb_, a, b), (0.0, _bt.XSPLIT)),
            ChebSeries(_tensor_coefs_from(tb, ab2, bb2, b, a), (0.0, _bt.XSPLIT)))


def _tensor_coefs_traced_from(tensor, abox, bbox, a, b):
    sa, sb = _smap(a, abox), _smap(b, bbox)
    t = jnp.asarray(tensor)
    va = jax.vmap(jax.vmap(lambda row: _chebval(sa, row, (-1.0, 1.0))))(
        jnp.swapaxes(t, 1, 2))                      # (NX, NB)
    return jax.vmap(lambda row: _chebval(sb, row, (-1.0, 1.0)))(va)  # (NX,)


def tensor_coefs_traced(a, b):
    return _tensor_coefs_traced_from(_bt.TENSOR, (_bt.ALO, _bt.AHI),
                                     (_bt.ALO, _bt.AHI), a, b)


def with_panel_series(a, b, make):
    """Run make(cab, cba) under the panel switch: a, b traced scalars in
    [0.1, 100], one lax.switch on which quadrant they sit in (only the
    taken branch executes; branch-internal series lengths differ freely
    because make's OUTPUT shape is panel-independent). The reflection
    needs both orientations, so each branch builds (a, b) and (b, a)
    from their own, possibly different, panels."""

    def branch(pa, pb):
        def f():
            ta, ab_, bb_ = _PANELS[(pa, pb)]
            tb, ab2, bb2 = _PANELS[(pb, pa)]
            cab = ChebSeries(_tensor_coefs_traced_from(ta, ab_, bb_, a, b),
                             (0.0, _bt.XSPLIT))
            cba = ChebSeries(_tensor_coefs_traced_from(tb, ab2, bb2, b, a),
                             (0.0, _bt.XSPLIT))
            return make(cab, cba)
        return f

    idx = ((a > _bw.ASPLIT).astype(jnp.int32) * 2
           + (b > _bw.ASPLIT).astype(jnp.int32))
    return jax.lax.switch(idx, [branch(0, 0), branch(0, 1),
                                branch(1, 0), branch(1, 1)])


def _log_direct(a, b, x, lser):
    """ln I_x(a,b) for x in (0, 1/2], from the direct series."""
    lnB = (jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)
           - jax.scipy.special.gammaln(a + b))
    return a * jnp.log(x) + b * jnp.log1p(-x) - jnp.log(a) - lnB + lser(x)


def eval_betainc(a, b, x, cab, cba):
    x = jnp.asarray(x)
    interior = (x > 0.0) & (x < 1.0)
    xi = jnp.where(interior, x, 0.25)          # masked lanes get a safe dummy
    xd = jnp.where(xi <= _bt.XSPLIT, xi, 0.25)
    yd = jnp.where(xi <= _bt.XSPLIT, 0.25, 1.0 - xi)   # exact for x in [1/2, 1]
    direct = jnp.exp(_log_direct(a, b, xd, cab))
    reflected = 1.0 - jnp.exp(_log_direct(b, a, yd, cba))
    core = jnp.where(xi <= _bt.XSPLIT, direct, reflected)
    out = jnp.where(x <= 0.0, 0.0, jnp.where(x >= 1.0, 1.0, core))
    out = out + edge_slope(_edge_density(a, b, x), x)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)


def _edge_density(a, b, x):
    """The Beta density AT an endpoint, 0 in between.

    Both endpoint values are picked by a hard select, so AD reads a slope
    of 0 there even where the density is exact and finite: grad of
    betainc_fn(1, 3, .) at x = 0 gave 0 for a true 3 (review,
    2026-08-02). f(0) = 1/B(1, b) = b when a = 1, and f(1) = 1/B(a, 1) = a
    when b = 1; either side is 0 above its 1 and infinite below it."""
    d0 = jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, b))
    d1 = jnp.where(b < 1.0, jnp.inf, jnp.where(b > 1.0, 0.0, a))
    return jnp.where(x == 0.0, d0, jnp.where(x == 1.0, d1, 0.0))


@jax.tree_util.register_pytree_node_class
class BetaInc(Recipe):
    """Callable I_x(a, b) on x in [0, 1]. Build with betainc(a, b)."""

    _static_fields = ("a", "b")
    _series_fields = ("cab", "cba")

    def _post_init(self):
        self.a = float(self.a)
        self.b = float(self.b)

    def __call__(self, x):
        return eval_betainc(self.a, self.b, x, self.cab, self.cba)


@functools.lru_cache(maxsize=128)
def _betainc_cached(a, b, _tag):
    from chebax import domains as _dom
    a = check_range("betainc", "a", a, *_dom.BETAINC[:2])
    b = check_range("betainc", "b", b, *_dom.BETAINC[:2])
    return BetaInc(a, b, *_eager_series_pair(a, b))


def betainc_fn(a, b, x):
    """I_x(a, b) with a, b as (traceable) scalars, differentiable in all three.

    a and b must be uniform per call and inside [0.1, 100] (unchecked
    under trace). Reconstruction costs two tensor contractions per call
    (of the panel the parameters land in), not per point; jit
    constant-folds them when a, b are static."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    x = jnp.asarray(x)
    return with_panel_series(a, b,
                             lambda cab, cba: eval_betainc(a, b, x, cab, cba))


def log_betainc_fn(a, b, x):
    """ln I_x(a, b), same contract as betainc_fn but with no underflow
    floor in the lower tail.

    For x <= 1/2 the tables already hold the log (betainc_fn exponentiates
    it), so ln I evaluates directly: valid to arbitrarily negative values
    (ln I ~ a ln x as x -> 0), differentiable in a, b and x, shapes
    uniform per call in [0.1, 100]^2. For x > 1/2 the reflection gives
    ln(1 - exp(.)) via log1p: absolutely accurate in I, so the error in
    ln I is ~eps/I there - fine where I is O(1), NOT a deep-tail path.
    Both deep tails at full log accuracy: lower tail directly (x <= 1/2),
    upper tail ln(1 - I_x(a, b)) = log_betainc_fn(b, a, 1 - x), where
    1 - x is exact in f64 for x >= 1/2. x <= 0 returns -inf, x >= 1
    returns 0. This is the log-CDF a censored likelihood wants when the
    censored mass is tiny (log(betainc_fn(...)) underflows first)."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    x = jnp.asarray(x)
    return with_panel_series(a, b,
                             lambda cab, cba: _log_eval(a, b, x, cab, cba))


def _log_eval(a, b, x, cab, cba):
    interior = (x > 0.0) & (x < 1.0)
    xi = jnp.where(interior, x, 0.25)
    xd = jnp.where(xi <= _bt.XSPLIT, xi, 0.25)
    yd = jnp.where(xi <= _bt.XSPLIT, 0.25, 1.0 - xi)
    direct = _log_direct(a, b, xd, cab)
    refl = jnp.log1p(-jnp.exp(_log_direct(b, a, yd, cba)))
    core = jnp.where(xi <= _bt.XSPLIT, direct, refl)
    out = jnp.where(x <= 0.0, -jnp.inf, jnp.where(x >= 1.0, 0.0, core))
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)


def betainc(a, b):
    """I_x(a, b) for a, b in [0.1, 100]. Cached per (a, b); no mpmath.

    a, b may be python numbers or concrete jax scalars; the bounded cache
    is keyed per x64 mode."""
    return _betainc_cached(float(a), float(b), canon_tag())
