"""The Marcum Q-function of order 1 and its complement, in logs, a and b
broadcast: log_marcum_q1, log_marcum_p1.

Q_1(a, b) = int_b^inf r e^(-(r^2 + a^2)/2) I_0(a r) dr and P_1 = 1 - Q_1. For
the standard Rice distribution (a = nu/sigma, b = value/sigma) P_1 is the CDF
and Q_1 the survival function. With xi = a b and d = b - a,

  b >= a, rho = a/b:  Q = e^(-d^2/2) S_Q
  b <  a, rho = b/a:  P = e^(-d^2/2) S_P

S_Q, S_P positive sums of e^(-xi) I_k(xi) (marcum_gen). One patch of
marcum_gen's table per element gives the smaller tail's ln S: BULK at xi <=
E1^2 as ln S_Q or ln(2 S_P/(rho xi)), MEDIAN and TAIL as ln(2 S/erfcx(eta/sqrt
2)) with S = S_Q or S_P/rho, eta = |d| sqrt(rho); CORNER gives ln P at small b
on b >= a. The other is log1mexp of it: on b < a P <= 1/2, and on b >= a
outside CORNER P >= 0.31.

Derivatives are the closed forms dP/db = b e^(-d^2/2) i0e(xi) and dP/da =
-b e^(-d^2/2) i1e(xi), divided by the probability as exponentials of log
differences, the e^(-d^2/2) left out of both where the probability is the
scaled tail. Second derivatives are AD of that rule.

a enters through |a| (Q_1 is even in a); b <= 0 gives ln P = -inf, ln Q = 0,
nan in either input nan.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import marcum_table as _t
from chebax._src.recipes._common import canon_float, locate_static, log1p, patch_sum
from chebax._src.recipes.erf_family import erfcx

_BULK, _MEDIAN, _TAIL, _CORNER = range(len(_t.NAMES))
_LN2 = math.log(2.0)
_LN_2PI = math.log(2.0 * math.pi)
_RSQRT2 = 1.0 / math.sqrt(2.0)
_E1, _B_C = _t.E1, _t.B_C
_CUTS = tuple(
    (tuple(_t.PCUTS[z][: _t.NPAR[z] + 1]), tuple(_t.YCUTS[z][: _t.NARG[z] + 1]))
    for z in range(len(_t.NAMES))
)


@functools.lru_cache(maxsize=2)
def _table(dtype):
    dt = np.dtype(dtype)
    tab = _t.TABLE if dt == np.float64 else _t.TABLE[:, : _t.KY32, : _t.KP32]
    return np.ascontiguousarray(tab, dt)


def _per_chart(z, values):
    out = jnp.asarray(values[0], jnp.result_type(*values))
    for k in range(1, len(values)):
        out = jnp.where(z == k, values[k], out)
    return out


def _log1mexp(v):
    """ln(1 - e^v), v <= 0."""
    near = v > -_LN2
    return jnp.where(
        near,
        jnp.log(-jnp.expm1(jnp.where(near, v, -1.0))),
        log1p(-jnp.exp(jnp.where(near, -1.0, v))),
    )


def _own_sum(tab, n, sy, sp, f):
    return -patch_sum(-tab, n, sy, sp)


def _shared_sum(tab, n, sy, sp, f):
    return f


def _slope_sum(tab, n, sy, sp, f):
    """The patch sum the slopes read. On the CPU their own, from the negated
    table (the same numbers): sharing the value's, XLA's program for
    value_and_grad gathers each of the 144 coefficients in its own loop
    through memory, 4.1x slower in float64 and 1.6x in float32. On the GPU
    the second sum costs 1.3x and 1.4x (experiments/52); other platforms
    share."""
    return jax.lax.platform_dependent(
        tab, n, sy, sp, f, cpu=_own_sum, default=_shared_sum
    )


def _parts(a, b):
    """ln P, ln Q, and for the slopes ln S of the small tail (P on b < a
    and in CORNER, else Q), small tail = e^(ln S - extra - d^2/2)."""
    nan = jnp.isnan(a) | jnp.isnan(b)
    a = jnp.where(a < 0.0, -a, a)
    pos = b > 0.0
    b = jnp.where(pos, b, 1.0)
    upper = b >= a
    corner = upper & (b <= _B_C)
    hi = jnp.where(upper, b, a)
    lo = jnp.where(upper, a, b)
    rho = lo / hi
    xi = a * b
    d = b - a
    d2h = 0.5 * d * d
    # not sqrt(rho): lo/hi can underflow where xi and eta are ordinary numbers
    eta = jnp.abs(d) * jnp.sqrt(lo) / jnp.sqrt(hi)

    bulk = xi <= _E1 * _E1
    tail = ~bulk & (eta >= _E1)
    z = jnp.where(
        corner, _CORNER, jnp.where(bulk, _BULK, jnp.where(tail, _TAIL, _MEDIAN))
    )
    side = jnp.where(upper, 0, 1)  # CORNER (b >= a) has one side
    # masked lanes get values inside every chart's box
    xi_m = jnp.where(bulk, 2.0 * _E1 * _E1, xi)
    eps = jax.lax.rsqrt(xi_m)
    t = 1.0 / (1.0 + eta)
    par = _per_chart(z, (rho, eps, rho, rho * rho))
    arg = _per_chart(z, (xi, eta, t, b * b))
    i, sp = locate_static(par, tuple(c[0] for c in _CUTS), z)
    j, sy = locate_static(arg, tuple(c[1] for c in _CUTS), z)
    npar = _per_chart(z, _t.NPAR)
    narg = _per_chart(z, _t.NARG)
    n = _per_chart(z, _t.OFFSET) + side * npar * narg + i * narg + j
    tab = jnp.asarray(_table(b.dtype.name))

    # the small tail's ln S (Q or P with e^(-shift) taken out)
    ln_b = jnp.log(b)
    ln_b2h = 2.0 * ln_b - _LN2  # b^2 underflows below 1e-154
    ln_half_erfcx = jnp.log(erfcx(eta * _RSQRT2)) - _LN2
    ln_rho = jnp.log(jnp.where(lo > 0.0, lo, 1.0)) - jnp.log(hi)
    ln_rho = jnp.where(lo > 0.0, ln_rho, -jnp.inf)

    def ln_s_of(f):
        return jnp.select(
            [corner, bulk & upper, bulk],
            [ln_b2h + f, f, ln_b2h + f],
            f + ln_half_erfcx + jnp.where(upper, 0.0, ln_rho),
        )

    f = patch_sum(tab, n, sy, sp)
    ln_s = ln_s_of(f)
    ln_s_slope = ln_s_of(_slope_sum(tab, n, sy, sp, f))
    # the shift beyond d^2/2, kept apart so the slopes never add d^2/2 back
    extra = jnp.where(corner, xi, 0.0)
    small = ln_s - extra - d2h
    small_is_p = ~upper | corner
    other = _log1mexp(small)
    log_p = jnp.where(small_is_p, small, other)
    log_q = jnp.where(small_is_p, other, small)
    log_p = jnp.where(pos, log_p, -jnp.inf)
    log_q = jnp.where(pos, log_q, 0.0)
    log_p = jnp.where(nan, jnp.nan, log_p)
    log_q = jnp.where(nan, jnp.nan, log_q)
    return log_p, log_q, ln_s_slope, extra, small_is_p, xi, a, b, d2h, pos


def _slopes(parts, want_p):
    """(d/da, d/db) of ln P (want_p) or ln Q, a >= 0 branch."""
    log_p, log_q, ln_s, extra, small_is_p, xi, a_abs, b, d2h, pos = parts
    ln_b = jnp.log(b)
    # past xi = 1e17 i0e and i1e are 1/sqrt(2 pi xi) in float64; from ln a +
    # ln b the slopes stay finite where xi overflows
    far = xi > 1e17
    ln_far = -0.5 * (_LN_2PI + jnp.log(a_abs) + ln_b)
    xi_n = jnp.where(far, 1.0, xi)
    i0 = jax.scipy.special.i0e(xi_n)
    ln_i0 = jnp.where(far, ln_far, jnp.log(i0))
    # I_1/I_0, xi/2 at 0: as a ratio, not ln i1e, so AD of the rule sees
    # d/da through a = 0
    ratio = jnp.where(far, 1.0, jax.scipy.special.i1e(xi_n) / i0)
    # F the wanted probability: is it the small tail?
    f_small = small_is_p == want_p
    # ln(b e^(-d^2/2) / F), e^(-d^2/2) left out of both where F is the small tail
    log_f = log_p if want_p else log_q
    base = jnp.where(f_small, ln_b + extra - ln_s, ln_b - d2h - log_f)
    sign = 1.0 if want_p else -1.0
    d_x = sign * jnp.exp(base + ln_i0)
    d_a = -d_x * ratio
    return jnp.where(pos, d_a, 0.0), jnp.where(pos, d_x, 0.0)


def _make(want_p):
    @jax.custom_jvp
    def fn(a, b):
        parts = _parts(a, b)
        return parts[0] if want_p else parts[1]

    @fn.defjvp
    def _jvp(primals, tangents):
        a, b = primals
        ta, tx = tangents
        parts = _parts(a, b)
        d_a, d_x = _slopes(parts, want_p)
        d_a = jnp.where(a < 0.0, -d_a, d_a)
        out = parts[0] if want_p else parts[1]
        return out, d_a * ta + d_x * tx

    return fn


_log_cdf = _make(True)
_log_sf = _make(False)


def log_marcum_q1(a, b):
    """ln Q_1(a, b), the Marcum Q-function of order 1, a and b broadcast; the
    log survival function of the standard Rice distribution at b, shape a.
    Differentiable in both."""
    a, b = jnp.broadcast_arrays(canon_float(a), canon_float(b))
    return _log_sf(a, b)


def log_marcum_p1(a, b):
    """ln(1 - Q_1(a, b)), the log CDF of the standard Rice distribution at b,
    shape a."""
    a, b = jnp.broadcast_arrays(canon_float(a), canon_float(b))
    return _log_cdf(a, b)
