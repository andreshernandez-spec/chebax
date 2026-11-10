"""gammaln and digamma for x >= 0, from 1/32-octave panels below 64 and
Stirling's series above (the tables and their design are in gammaln_gen).
Float64 floors, all relative: lgamma 3.5e-16, next to its zeros at 1 and 2
too; digamma 5.8e-16, next to its zero too; trigamma 7.2e-16
(tests/test_gammaln.py).

XLA evaluates every branch of a select for every element, so the cost is
the sum of the branches. The value's branches share one log, which serves
both lgamma(x) = lgamma(x + 1) - log x below 1 and Stirling's term, and
use one division, Stirling's; the panel lookup is one set of gathers into
the lgamma tables laid end to end. jax's gammaln (XLA's Lanczos lgamma)
compiles to two logs, a log1p, a sine and nine divisions per element.

The derivative, digamma, comes from its own panel table with its own log
and division, not from the value's series. A program returning value and
gradient would otherwise read the value's gathered coefficients twice, and
XLA writes such shared gathers to memory instead of fusing them into either
consumer: value_and_grad took 40 ns per element on one float64 CPU core
against 17 ns with the separate table. Apart from x the two share nothing,
and each fuses into one loop (experiments/44). digamma's own derivative
differentiates the PSI series, so its value_and_grad pays that cost: 43 ns
against 9 for its value, still 3.4x faster than jax's.
"""

import functools
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammaln_table as _t
from chebax._src.recipes._common import canon_float, canon_tag

_Tables = namedtuple("_Tables", "lgamma psi stirling dstirling x0_hi x0_lo")
_G_END = 2.0 ** _t.META["g_octaves"]
_LOG_XB = float(np.log(_t.XB))


@functools.lru_cache(maxsize=4)
def _tables_cached(tag):
    with jax.ensure_compile_time_eval():
        f32 = tag == "float32"
        n, n_psi = (4, 5) if f32 else (_t.G.shape[1], _t.PSI.shape[1])
        na = 2 if f32 else _t.STIRLING.size
        a = _t.STIRLING[:na]
        # d/dx sum_k A_k x^-(2k+1) = -x^-2 sum_k (2k+1) A_k x^-2k
        da = -(2 * np.arange(na) + 1) * a
        # digamma's zero split for this dtype, so x - x0_hi is exact near it
        x0_hi = float(np.float32(_t.X0_HI)) if f32 else _t.X0_HI
        x0_lo = (_t.X0_HI - x0_hi) + _t.X0_LO
        return _Tables(
            canon_float(np.concatenate((_t.G, _t.LG))[:, :n]),
            canon_float(_t.PSI[:, :n_psi]),
            canon_float(a),
            canon_float(da),
            canon_float(x0_hi),
            canon_float(x0_lo),
        )


def _tables():
    """Panel coefficients and Stirling's series, keyed per x64 mode."""
    return _tables_cached(canon_tag())


def _horner(p, c):
    out = c[-1]
    for k in range(c.size - 2, -1, -1):
        out = c[k] + p * out
    return out


def _clenshaw(c, s):
    b1 = b2 = jnp.zeros_like(s)
    for ck in c[:0:-1]:
        b1, b2 = ck - b2 + 2.0 * s * b1, b1
    return c[0] - b2 + s * b1


def _per_coefficient_series(panels, i, s):
    n = panels.shape[1]
    return _clenshaw([jnp.take(panels[:, k], i, mode="clip") for k in range(n)], s)


def _row_series(panels, i, s):
    rows = jnp.take(panels, i, axis=0, mode="clip")
    return _clenshaw([rows[..., k] for k in range(panels.shape[1])], s)


def _series(panels, i, s):
    """Panel i's series at s. On the CPU, one gather per coefficient fuses
    into the recurrence, and one row gather instead runs 1.5-4.2x slower. On
    the GPU, the index those gathers share is written to memory first
    (float32 values at 0.57x jax's), and one row gather reads it once
    (experiments/44)."""
    return jax.lax.platform_dependent(
        panels, i, s, cpu=_per_coefficient_series, default=_row_series
    )


def _locate(x):
    """Branch masks, x as xl, and each element's panel i and coordinate s in
    [-1, 1]. xl is nan below 0, so every value and derivative is nan there."""
    big = x >= _t.XB
    small = x < 1.0
    # added, not selected: a select would give the nan lanes a zero derivative
    xl = x + jnp.where(x >= 0.0, 0.0, jnp.nan)
    y = jnp.where(small, xl + 1.0, jnp.where(big, 1.0, xl))
    if y.dtype == jnp.float64:
        it, mb, bias = jnp.int64, 52, 1023
    else:
        it, mb, bias = jnp.int32, 23, 127
    # y = 2^e (1 + m), m in [0, 1): the octave e from the exponent bits
    e = (jax.lax.bitcast_convert_type(y, it) >> mb) - bias
    scale = jax.lax.bitcast_convert_type((bias - e) << mb, y.dtype)
    u = (y * scale - 1.0) * _t.K
    j = jnp.floor(u)
    i = e.astype(jnp.int32) * _t.K + j.astype(jnp.int32)
    return big, small, xl, y, i, 2.0 * (u - j) - 1.0


def _lgamma(x):
    tab = _tables()
    big, small, xl, y, i, s = _locate(x)
    f = _series(tab.lgamma, i, s)
    log_x = jnp.log(xl)
    # (x - 1)(x - 2) g(x) on [1, 4), x (x - 1) g(x + 1) - log x below 1
    q = jnp.where(small, xl, xl - 1.0)
    w = jnp.where(y < _G_END, q * (q - 1.0), 1.0)
    tabled = w * f - jnp.where(small, log_x, 0.0)
    xb = jnp.where(big, xl, _t.XB)
    ib = 1.0 / xb
    lb = jnp.where(big, log_x, _LOG_XB)
    series = ib * _horner(ib * ib, tab.stirling)
    stirling = (xb - 0.5) * (lb - 1.0) + _t.STIRLING_C + series
    return jnp.where(big, stirling, tabled)


def _digamma(x):
    tab = _tables()
    big, small, xl, y, i, s = _locate(x)
    # (y - x0) r(y) on [1, 2), relative next to the zero x0. Not
    # ((y - hi) - lo) r: XLA folds that into y - (hi + lo), which drops lo.
    r = _series(tab.psi, i, s)
    root = y < 2.0
    f = r * jnp.where(root, y - tab.x0_hi, 1.0) - r * jnp.where(root, tab.x0_lo, 0.0)
    # psi(x + 1) - 1/x below 1; the division also serves Stirling. Elsewhere
    # its argument is 1, so no masked lane differentiates an infinite 1/x.
    inv = 1.0 / jnp.where(small | big, xl, 1.0)
    tabled = f - jnp.where(small, inv, 0.0)
    ib = jnp.where(big, inv, 1.0 / _t.XB)
    p = ib * ib
    lb = jnp.log(jnp.where(big, xl, _t.XB))
    stirling = lb - 0.5 * ib + p * _horner(p, tab.dstirling)
    return jnp.where(big, stirling, tabled)


@jax.custom_jvp
def _gammaln_cj(x):
    return _lgamma(x)


@_gammaln_cj.defjvp
def _gammaln_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return _lgamma(x), _digamma(x) * dx


def gammaln(x):
    """log Gamma(x) for x >= 0: inf at 0 and at inf, nan below 0 (where jax's
    gammaln gives log|Gamma(x)|). The derivative is digamma, from its own
    table."""
    return _gammaln_cj(canon_float(x))


def digamma(x):
    """psi(x) = d/dx log Gamma(x) for x >= 0: -inf at 0, inf at inf, nan
    below 0 (where jax's digamma reflects). Relative next to its zero at
    1.4616. Its derivatives (trigamma and up) differentiate the table."""
    return _digamma(canon_float(x))
