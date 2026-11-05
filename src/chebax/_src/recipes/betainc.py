"""Regularized incomplete beta I_x(a, b) for a, b in [0.1, 100], x in [0, 1].

For x <= 1/2 (DLMF 8.17.7)

    I_x(a, b) = exp( a ln x + b ln(1-x) - ln a - ln B(a,b) + L(a,b,x) ),

L = ln 2F1(a+b, 1; a+1; x) from an (x, a, b) tensor; above 1/2 the reflection
1 - I_{1-x}(b, a), with 1 - x exact. For the upper tail to relative accuracy use
betainc(b, a, 1 - x). Four tensor panels split at 10 per axis, one lax.switch
on the scalar parameters; panel edges agree to ~1e-14. Differentiable in a and
b (jax lacks dI/da, jax#38610) and x: the custom_jvp rules compute the
derivatives as values inside the switch, since reverse mode through it costs
~120 ms per parameter on the GPU (experiments/29). This module owns the
tables; the kernel is in betainc_kernel.py, shared with betaincinv and stdtr.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from chebax import domains as _dom
from chebax._src import algorithms
from chebax._src.pytree import Recipe
from chebax._src.recipes import betainc_table as _bt
from chebax._src.recipes import betainc_wide_table as _bw
from chebax._src.recipes._common import (
    canon_param,
    canon_tag,
    check_range,
    deprecated_factory,
    edge_slope,
)
from chebax._src.recipes.betainc_kernel import (
    eval_betainc,
    eval_betainc_parts,
    eval_log_betainc_parts,
    log_direct,
)
from chebax._src.series import ChebSeries, _chebval
from jax.custom_derivatives import SymbolicZero

# (a-high?, b-high?) -> (tensor, a-box, b-box); the wide panels are reached
# only through the runtime switch
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
    return _tensor_coefs_from(_bt.TENSOR, (_bt.ALO, _bt.AHI), (_bt.ALO, _bt.AHI), a, b)


def _panel_of(v):
    return 1 if v > _bw.ASPLIT else 0


def _eager_series_pair(a, b):
    """(cab, cba) ChebSeries at concrete (a, b), panel-dispatched. numpy."""
    ta, ab_, bb_ = _PANELS[(_panel_of(a), _panel_of(b))]
    tb, ab2, bb2 = _PANELS[(_panel_of(b), _panel_of(a))]
    return (
        ChebSeries(_tensor_coefs_from(ta, ab_, bb_, a, b), (0.0, _bt.XSPLIT)),
        ChebSeries(_tensor_coefs_from(tb, ab2, bb2, b, a), (0.0, _bt.XSPLIT)),
    )


def _tensor_coefs_traced_from(tensor, abox, bbox, a, b):
    sa, sb = _smap(a, abox), _smap(b, bbox)
    t = jnp.asarray(tensor)
    va = jax.vmap(jax.vmap(lambda row: _chebval(sa, row, (-1.0, 1.0))))(
        jnp.swapaxes(t, 1, 2)
    )  # (NX, NB)
    return jax.vmap(lambda row: _chebval(sb, row, (-1.0, 1.0)))(va)  # (NX,)


def tensor_coefs_traced(a, b):
    return _tensor_coefs_traced_from(
        _bt.TENSOR, (_bt.ALO, _bt.AHI), (_bt.ALO, _bt.AHI), a, b
    )


@functools.lru_cache(maxsize=None)
def _deriv_tensors(panel):
    """The panel tensor's Chebyshev derivatives along a and b, in the
    parameters' own units. numpy, once per panel."""
    tensor, abox, bbox = _PANELS[panel]
    der = np.polynomial.chebyshev.chebder
    return (
        der(tensor, axis=1) * (2.0 / (abox[1] - abox[0])),
        der(tensor, axis=2) * (2.0 / (bbox[1] - bbox[0])),
    )


def _coef_parts(panel, a, b, need):
    """Coefficient vector at (a, b) and, where need says so, its a- and
    b-derivatives (None otherwise): contractions of the panel's tensor and its
    derivative tensors, independent of N."""
    tensor, abox, bbox = _PANELS[panel]
    contract = lambda t: _tensor_coefs_traced_from(t, abox, bbox, a, b)
    derivs = _deriv_tensors(panel)
    return (contract(tensor),) + tuple(
        contract(d) if n else None for d, n in zip(derivs, need)
    )


def _panel_switch(a, b, branch):
    idx = (a > _bw.ASPLIT).astype(jnp.int32) * 2 + (b > _bw.ASPLIT).astype(jnp.int32)
    return jax.lax.switch(idx, [branch(0, 0), branch(0, 1), branch(1, 0), branch(1, 1)])


def with_panel_series(a, b, make):
    """make(cab, cba) under one lax.switch on the (a, b) quadrant; each branch
    builds both orientations from their own panels, and make's output shape is
    panel-independent. Reverse mode through the switch is slow (experiments/29):
    differentiate through with_panel_parts instead."""

    def branch(pa, pb):
        def f():
            ta, ab_, bb_ = _PANELS[(pa, pb)]
            tb, ab2, bb2 = _PANELS[(pb, pa)]
            cab = ChebSeries(
                _tensor_coefs_traced_from(ta, ab_, bb_, a, b), (0.0, _bt.XSPLIT)
            )
            cba = ChebSeries(
                _tensor_coefs_traced_from(tb, ab2, bb2, b, a), (0.0, _bt.XSPLIT)
            )
            return make(cab, cba)

        return f

    return _panel_switch(a, b, branch)


def with_panel_parts(a, b, need, make):
    """make(cab, cba, dcab, dcba) under the panel switch: dcab = (d cab/da,
    d cab/db) and dcba = (d cba/db, d cba/da) as series, None where need =
    (need_a, need_b) is False. For custom_jvp rules that compute derivatives
    as values inside the switch and apply the tangents outside: reverse mode
    then never transposes the switch."""

    def branch(pa, pb):
        def f():
            series = lambda c: None if c is None else ChebSeries(c, (0.0, _bt.XSPLIT))
            c, c1, c2 = _coef_parts((pa, pb), a, b, need)
            d, d1, d2 = _coef_parts((pb, pa), b, a, need[::-1])
            return make(
                series(c), series(d), (series(c1), series(c2)), (series(d1), series(d2))
            )

        return f

    return _panel_switch(a, b, branch)


def _parts_rule(parts_fn, primals, tangents):
    """custom_jvp rule from parts_fn(a, b, x, ..., need) -> (value, dI/da,
    dI/db, dI/dx), None for the derivatives whose tangents are symbolic zeros:
    only the needed ones are traced."""
    a, b, x = primals
    need = tuple(type(t) is not SymbolicZero for t in tangents)
    parts = with_panel_parts(
        a, b, need[:2], lambda *s: parts_fn(a, b, x, *s, need=need)
    )
    out = jnp.zeros_like(parts[0])
    for v, t in zip(parts[1:], tangents):
        if v is not None:
            out = out + v * t
    return parts[0], out


@jax.custom_jvp
def _betainc_cj(a, b, x):
    return with_panel_series(a, b, lambda cab, cba: eval_betainc(a, b, x, cab, cba))


@functools.partial(_betainc_cj.defjvp, symbolic_zeros=True)
def _betainc_jvp(primals, tangents):
    return _parts_rule(eval_betainc_parts, primals, tangents)


@jax.custom_jvp
def _log_betainc_cj(a, b, x):
    return with_panel_series(a, b, lambda cab, cba: _log_eval(a, b, x, cab, cba))


@functools.partial(_log_betainc_cj.defjvp, symbolic_zeros=True)
def _log_betainc_jvp(primals, tangents):
    return _parts_rule(eval_log_betainc_parts, primals, tangents)


def with_edge_slopes(a, b, x, out):
    """Splice the endpoint densities into a CDF, outside the panel switch: a
    custom_jvp inside a switch branch has no transpose under vmap (pergroup)."""
    return out + edge_slope(_edge_density(a, b, x), x)


def _edge_density(a, b, x):
    """The Beta density at an endpoint, 0 in between: f(0) = b at a = 1, f(1) = a
    at b = 1, 0 above 1 and inf below."""
    d0 = jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, b))
    d1 = jnp.where(b < 1.0, jnp.inf, jnp.where(b > 1.0, 0.0, a))
    return jnp.where(x == 0.0, d0, jnp.where(x == 1.0, d1, 0.0))


@jax.tree_util.register_pytree_node_class
class BetaInc(Recipe):
    """Callable I_x(a, b) on x in [0, 1]. Build with betainc_frozen(a, b)."""

    _static_fields = ("a", "b")
    _series_fields = ("cab", "cba")

    def _post_init(self):
        self.a = float(self.a)
        self.b = float(self.b)

    def __call__(self, x):
        return with_edge_slopes(
            self.a, self.b, x, eval_betainc(self.a, self.b, x, self.cab, self.cba)
        )


@functools.lru_cache(maxsize=128)
def _betainc_cached(a, b, _tag):
    a = check_range("betainc", "a", a, *_dom.BETAINC[:2])
    b = check_range("betainc", "b", b, *_dom.BETAINC[:2])
    return BetaInc(a, b, *_eager_series_pair(a, b))


def betainc_scalar(a, b, x):
    """I_x(a, b) for traceable scalar a, b in [0.1, 100] (unchecked under trace),
    differentiable in all three; two tensor contractions per call."""
    a = canon_param(a, "betainc_scalar", "a")
    b = canon_param(b, "betainc_scalar", "b")
    return with_edge_slopes(a, b, x, _betainc_cj(a, b, jnp.asarray(x)))


def log_betainc_scalar(a, b, x):
    """ln I_x(a, b) as betainc_scalar, no underflow floor in the lower tail. For
    x <= 1/2 the tables hold the log; above, log1p of the reflection errs ~eps/I,
    so for the upper tail use log_betainc_scalar(b, a, 1 - x). x <= 0 gives -inf,
    x >= 1 gives 0."""
    a = canon_param(a, "log_betainc_scalar", "a")
    b = canon_param(b, "log_betainc_scalar", "b")
    return _log_betainc_cj(a, b, jnp.asarray(x))


def _log_eval(a, b, x, cab, cba):
    interior = (x > 0.0) & (x < 1.0)
    xi = jnp.where(interior, x, 0.25)
    xd = jnp.where(xi <= _bt.XSPLIT, xi, 0.25)
    yd = jnp.where(xi <= _bt.XSPLIT, 0.25, 1.0 - xi)
    direct = log_direct(a, b, xd, cab)
    refl = jnp.log1p(-jnp.exp(log_direct(b, a, yd, cba)))
    core = jnp.where(xi <= _bt.XSPLIT, direct, refl)
    out = jnp.where(x <= 0.0, -jnp.inf, jnp.where(x >= 1.0, 0.0, core))
    return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)


def betainc(a, b, x=None):
    """I_x(a, b) for one pair a, b in [0.1, 100] (betainc_scalar; no per-element
    form, use chebax.pergroup). x=None is the deprecated 0.4 factory."""
    if x is None:
        return deprecated_factory("betainc", betainc_frozen, a, b)
    return betainc_scalar(a, b, x)


def log_betainc(a, b, x):
    """ln I_x(a, b) for one pair a, b in [0.1, 100]: log_betainc_scalar."""
    return log_betainc_scalar(a, b, x)


def betainc_frozen(a, b):
    """I_x(a, b) for concrete a, b in [0.1, 100], cached per (a, b)."""
    return _betainc_cached(float(a), float(b), canon_tag())
