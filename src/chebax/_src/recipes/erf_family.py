"""dawsn and erfcx. dawsn is x E(|x|) on equal panels up to |x| = 6,
with a (6/x)^2 tail beyond. erfcx is t h(t) with t = K/(K + |x|), h from one of NP
t-panels picked by floor(NP t), and erfcx(-x) = 2 e^{x^2} - erfcx(x) below
0 with x^2 split exactly (rounded, it costs up to x^2 eps: 5.7e-14 at
x = -22.75). |x| is a select, since jnp.abs zeroes the derivative at 0.
The JVPs reuse D' = 1 - 2 x D in the inner region and
erfcx' = 2 x erfcx - 2/sqrt(pi) below zero. In the tails these identities
cancel; differentiated table coefficients give stable slopes instead.
"""

import functools
import math
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import erf_table as _et
from chebax._src.recipes._common import canon_float, canon_tag

_Tables = namedtuple("_Tables", "dawsn tail dtail erfcx derfcx")
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


@functools.lru_cache(maxsize=4)
def _tables_cached(_tag):
    # The first call may be inside gammainc's cond; cached arrays must escape that trace.
    with jax.ensure_compile_time_eval():
        ch = np.polynomial.chebyshev
        g, p = _et.DAWSN_G, _et.ERFCX_P
        dg = ch.chebder(g)
        # G + 2 t G', t = (v+1)/2; h + t h', t = (2j+v+1)/(2 NP).
        dtail = ch.chebadd(g, 2.0 * ch.chebadd(dg, ch.chebmulx(dg)))
        derfcx = []
        for j, row in enumerate(p):
            d = ch.chebder(row)
            derfcx.append(ch.chebadd(row, ch.chebadd((2 * j + 1) * d, ch.chebmulx(d))))
        derfcx = np.array(derfcx)
        f32 = _tag == "float32"
        ng = 5 if f32 else 8
        # One lookup for a gradient-only call: h below zero, h + t h' above.
        derfcx = np.concatenate((p[:, :ng], derfcx[:, :ng]))
        return _Tables(
            canon_float(_et.DAWSN_P[:, :5] if f32 else _et.DAWSN_P),
            canon_float(g[:7] if f32 else g[:12]),
            canon_float(dtail[:7] if f32 else dtail[:14]),
            canon_float(p[:, :5] if f32 else p),
            canon_float(derfcx),
        )


def _tables():
    """Values and stable slope coefficients, keyed per x64 mode."""
    return _tables_cached(canon_tag())


def _panel_coord(y, n):
    i = jnp.minimum(jnp.floor(y), n - 1)
    return i.astype(jnp.int32), 2.0 * (y - i) - 1.0


def _clenshaw(v, c):
    # Keep each gathered coefficient separate; stacking them materializes an N x K array.
    b1 = b2 = jnp.zeros_like(v)
    for k in range(len(c) - 1, 0, -1):
        b1, b2 = c[k] - b2 + 2.0 * v * b1, b1
    return c[0] - b2 + v * b1


def _panel_value(panels, i, v):
    return _clenshaw(
        v, [jnp.take(panels[:, k], i, mode="clip") for k in range(panels.shape[1])]
    )


@jax.custom_jvp
def _exp_sq(x):
    """e^{x^2} with x^2 = hi + lo, hi exact by a bit mask (no fma can move
    the split); lo reaches 2^-25 x^2, too big for 1 + lo to stand in for e^lo."""
    dt = x.dtype
    it, low = (jnp.int64, 27) if dt == jnp.float64 else (jnp.int32, 12)
    # an infinite x would make x - x_hi nan; e^{x^2} is inf long before
    inf = jnp.isinf(x)
    xs = jnp.where(inf, 0.0, x)
    bits = jax.lax.bitcast_convert_type(xs, it)
    xh = jax.lax.bitcast_convert_type(bits & ~((1 << low) - 1), dt)
    # e^hi (1 + expm1(lo)), not e^hi e^lo: under jit XLA folds a product of
    # exps into the exp of the sum, which rounds x^2 again. |lo| <= 2.1e-5
    # in float64, so three expm1 terms suffice; float32 needs five.
    e = jnp.exp(xh * xh)
    lo = (xs - xh) * (xs + xh)
    last = 1.0 / 6.0
    if dt == jnp.float32:
        last = last + lo * (1.0 / 24.0 + lo / 120.0)
    y = e + e * (lo * (1.0 + lo * (0.5 + lo * last)))
    return jnp.where(inf | jnp.isinf(e), jnp.inf, y)


@_exp_sq.defjvp
def _exp_sq_jvp(primals, tangents):
    # through the split, reverse mode gives inf * 0 = nan once e^{x^2}
    # overflows at an x whose lo is exactly 0 (x = -27); the derivative is
    # 2 x e^{x^2} whatever the split
    (x,), (dx,) = primals, tangents
    y = _exp_sq(x)
    return y, 2.0 * x * y * dx


@jax.custom_jvp
def _dawsn_cj(x):
    tables = _tables()
    ax = jnp.where(x >= 0, x, -x)
    inner = ax <= _et.XS
    xi, xo = jnp.where(inner, ax, _et.XS), jnp.where(inner, _et.XS, ax)
    i, v = _panel_coord(xi * (_et.NPD / _et.XS), _et.NPD)
    p = _panel_value(tables.dawsn, i, v)
    g = tables.tail
    tail = _clenshaw(2.0 * (_et.XS / xo) ** 2 - 1.0, [g[k] for k in range(g.size)])
    # D(x)/x differs from 1 by < eps/24; preserve normal x at the underflow limit.
    small = ax < 0.25 * math.sqrt(jnp.finfo(x.dtype).eps)
    value = jnp.where(small, x, jnp.where(inner, x, 0.0) * p)
    return jnp.where(inner, value, jnp.sign(x) * tail / xo)


@_dawsn_cj.defjvp
def _dawsn_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = _dawsn_cj(x)
    ax = jnp.where(x >= 0, x, -x)
    inner = ax <= _et.XS
    xi, yi = jnp.where(inner, x, 0.0), jnp.where(inner, y, 0.0)
    xo = jnp.where(inner, _et.XS, ax)
    v = 2.0 * (_et.XS / xo) ** 2 - 1.0
    c = _tables().dtail
    tail = -_clenshaw(v, [c[k] for k in range(c.size)]) / xo / xo
    return y, jnp.where(inner, 1.0 - 2.0 * xi * yi, tail) * dx


@jax.custom_jvp
def _erfcx_cj(x):
    panels = _tables().erfcx
    ax = jnp.where(x >= 0, x, -x)
    t = _et.K / (_et.K + ax)
    i, v = _panel_coord(_et.NP * t, _et.NP)
    pos = t * _panel_value(panels, i, v)
    # clamp the reflection's exp argument for the masked (x >= 0) lanes:
    # exp(x^2) = inf there would poison gradients via 0 * inf. The split
    # runs in the result's dtype (integer and f16 inputs promote).
    xn = jnp.where(x < 0, x, 0.0).astype(pos.dtype)
    return jnp.where(x >= 0, pos, 2.0 * _exp_sq(xn) - pos)


@_erfcx_cj.defjvp
def _erfcx_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = _erfcx_cj(x)
    ax = jnp.where(x >= 0, x, -x)
    t = _et.K / (_et.K + ax)
    i, v = _panel_coord(_et.NP * t, _et.NP)
    negative = x < 0
    row = jnp.where(negative, i, i + _et.NP)
    q = _panel_value(_tables().derfcx, row, v)
    pos = -(t * t / _et.K) * q
    xn = jnp.where(negative, x, 0.0)
    e = _exp_sq(xn)
    # Expanded 2 x Y - 2/sqrt(pi); the bounded correction must avoid inf * 0.
    xc = jnp.where(jnp.isfinite(e), xn, 0.0)
    neg = 4.0 * xn * e - (2.0 * xc * t) * q - _TWO_OVER_SQRT_PI
    return y, jnp.where(negative, neg, pos) * dx


def dawsn(x):
    """Dawson's integral D(x) = e^{-x^2} int_0^x e^{t^2} dt, any real x."""
    return _dawsn_cj(canon_float(x))


def erfcx(x):
    """e^{x^2} erfc(x), any real x; inf below x ~ -26.6."""
    return _erfcx_cj(canon_float(x))
