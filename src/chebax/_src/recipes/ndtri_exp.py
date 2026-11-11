"""ndtri_exp(y) = ndtri(e^y) for y <= 0, from one table of panels (the
design is in ndtri_exp_gen). The centre, e^-1.5 <= e^y <= 1 - e^-1.5, is
(y + log 2) h(y + log 2), relative next to the median's zero. The tails are
-+ s g(s), s = sqrt(-2 u), u = y below the centre and log(1 - e^y) above
it, so neither forms e^y where it underflows or 1 - e^y where it rounds
to 0; g = 1 from s = 2^30. Float64 floors, all relative: z 3.8e-16, dz/dy
6.3e-16, d2z/dy2 5.6e-15 (tests/test_ndtri_exp.py).

XLA evaluates every branch of a select for every element, so the branches
share one series: centre and tail panels lie end to end in one table, and
each element finds its own. The value costs one expm1, one log and one or
two sqrts per element and no division; the derivative has its own table
and one division. Against jax's ndtri(exp(y)), which is -inf below y = -745
and loses digits as y nears 0 (experiments/45): value_and_grad 2.0-2.4x
faster on one CPU core and 3.2-4.0x on the GPU, the value 0.92-1.10x on
the CPU and 1.1-1.9x on the GPU.
"""

import functools
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import ndtri_exp_table as _t
from chebax._src.recipes._common import canon_float, canon_tag
from chebax._src.recipes._common import panel_series as _series

_Tables = namedtuple("_Tables", "value slope ln2_hi ln2_lo")
_SHIFT = _t.LN2_HI - _t.D_LO  # centre panels start at y = D_LO - log 2


@functools.lru_cache(maxsize=4)
def _tables_cached(tag):
    with jax.ensure_compile_time_eval():
        f32 = tag == "float32"
        n = 5 if f32 else _t.P.shape[1]
        n_d = 5 if f32 else _t.DP.shape[1]
        # log 2 split for this dtype, so y + ln2_hi is exact in the centre
        ln2_hi = float(np.float32(_t.LN2_HI)) if f32 else _t.LN2_HI
        ln2_lo = (_t.LN2_HI - ln2_hi) + _t.LN2_LO
        return _Tables(
            canon_float(_t.P[:, :n]),
            canon_float(_t.DP[:, :n_d]),
            canon_float(ln2_hi),
            canon_float(ln2_lo),
        )


def _tables():
    """Panel coefficients and the log 2 split, keyed per x64 mode."""
    return _tables_cached(canon_tag())


def _layout(dtype):
    """Integer type, mantissa bits and exponent bias of a float dtype."""
    if dtype == jnp.float64:
        return jnp.int64, 52, 1023
    return jnp.int32, 23, 127


def _estimated_u(y, u):
    """u for the panel index, without the log: above the centre, log(1 - e^y)
    = log(-y) + y/2 + y^2/24 + O(y^4), log(-y) from the exponent bits and
    LN_EST on the mantissa. It is off by at most 6e-4 there (LN_EST_ERR,
    the y^4 term, and float32 rounding of the exponent's part), and s^2 >= 3
    with panels at least s/32 wide, so s lies within 1.3% of a half-width of
    the panel picked, where its series still holds. u goes unused: reading it
    would wait for the log."""
    lower = y < _t.YC
    upper = y > _t.Y_HI
    x = jnp.where(upper, -y, 1.0)
    it, mb, bias = _layout(y.dtype)
    bits = jax.lax.bitcast_convert_type(x, it)
    mant = (bits & ((1 << mb) - 1)) | (bias << mb)
    m = jax.lax.bitcast_convert_type(mant, y.dtype) - 1.0
    log_m = _t.LN_EST[-1]
    for c in _t.LN_EST[-2::-1]:
        log_m = c + m * log_m
    log_x = ((bits >> mb) - bias).astype(y.dtype) * _t.LN2_HI + log_m
    yu = jnp.where(upper, y, 0.0)
    return jnp.where(lower, y, jnp.where(upper, log_x + yu * (0.5 + yu / 24.0), _t.YC))


def _exact_u(y, u):
    return u


def _index_u(y, u):
    """u for picking the tail panel. XLA's CPU backend has no gather
    instruction: it moves each panel index to an integer register and loads
    the coefficients lane by lane, so an index that waits for log(1 - e^y)
    puts one serial chain on every element. The CPU takes the estimate, and
    so does CUDA in float32, where the program for value and gradient then
    fuses whole; in float64 there the estimate costs more than it saves
    (experiments/45). Other platforms are unmeasured and keep u."""
    cuda = _estimated_u if y.dtype == jnp.float32 else _exact_u
    return jax.lax.platform_dependent(
        y, u, cpu=_estimated_u, cuda=cuda, default=_exact_u
    )


def _locate(y):
    """Region masks, 1 - e^y, the tail's s, and each element's panel i and
    coordinate v in [-1, 1]."""
    lower = y < _t.YC
    centre = ~lower & (y <= _t.Y_HI)
    # -expm1, not 0 - expm1: with the subtraction, XLA's GPU program writes
    # each element's panel index to memory before the gather
    q = -jnp.expm1(y)
    # the centre's lanes take YC, which keeps the tail on its panels
    u = jnp.where(lower, y, jnp.where(centre, _t.YC, jnp.log(q)))
    # 2 sqrt(-u/2): exact scalings, so -2u never overflows
    s = 2.0 * jnp.sqrt(-0.5 * u)
    # the panel from s_i, the coordinate in it from s; they differ only
    # above the centre, where _index_u estimates
    s_i = 2.0 * jnp.sqrt(-0.5 * _index_u(y, u))
    big_i = s_i >= _t.S_BIG
    st = jnp.where(big_i, 1.5, s_i)
    it, mb, bias = _layout(y.dtype)
    # st = 2^e (1 + m), m in [0, 1): the octave e from the exponent bits
    e = (jax.lax.bitcast_convert_type(st, it) >> mb) - bias
    scale = jax.lax.bitcast_convert_type((bias - e) << mb, y.dtype)
    j = jnp.floor((st * scale - 1.0) * _t.K)
    w = (jnp.where(big_i, 1.5, s) * scale - 1.0) * _t.K
    i_tail = (e * _t.K - _t.J0 + _t.NC).astype(jnp.int32) + j.astype(jnp.int32)
    # one add, so eager and jitted programs put an edge in the same panel
    wc = (jnp.where(centre, y, _t.YC) + _SHIFT) * _t.WC
    jc = jnp.floor(wc)
    i = jnp.where(centre, jc.astype(jnp.int32), i_tail)
    v = 2.0 * jnp.where(centre, wc - jc, w - j) - 1.0
    return centre, lower, q, s, s >= _t.S_BIG, i, v


def _value(y):
    tab = _tables()
    centre, lower, _, s, big, i, v = _locate(y)
    f = _series(tab.value, i, v)
    # f (y + hi) + f lo, not f ((y + hi) + lo): XLA folds that into
    # y + (hi + lo), which drops lo
    yc = jnp.where(centre, y, -tab.ln2_hi)
    mid = f * (yc + tab.ln2_hi) + f * tab.ln2_lo
    a = s * jnp.where(big, 1.0, f)
    return jnp.where(centre, mid, jnp.where(lower, -a, a))


def _slope(y):
    """dz/dy: the table's value in the centre, a'(s)/s below it and
    a'(s)/s e^y/(1 - e^y) above it."""
    tab = _tables()
    centre, lower, q, s, big, i, v = _locate(y)
    f = _series(tab.slope, i, v)
    upper = ~(centre | lower)
    da = jnp.where(big, 1.0, f)
    num = da * jnp.where(upper, 1.0 - q, 1.0)
    den = s * jnp.where(upper, q, 1.0)
    # y = 0: s q = inf * 0, the slope is +inf
    tail = jnp.where(q == 0.0, jnp.inf, num / den)
    return jnp.where(centre, f, tail)


@jax.custom_jvp
def _ndtri_exp_cj(y):
    return _value(y)


@_ndtri_exp_cj.defjvp
def _ndtri_exp_jvp(primals, tangents):
    (y,), (dy,) = primals, tangents
    return _value(y), _slope(y) * dy


def ndtri_exp(y):
    """ndtri(e^y), the standard normal quantile of the log-probability y,
    for y <= 0: -inf at -inf, +inf at 0, nan above 0. Relative next to its
    zero at y = -log 2. The derivative, e^y / phi(z), has its own table."""
    return _ndtri_exp_cj(canon_float(y))
