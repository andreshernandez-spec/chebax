"""Per-element gammainc: P(a, x), Q = 1 - P and their logs with a
different a >= 0.1 on every element, for chebax.gammainc_array and its
siblings (inputs arrive normalized and broadcast from gammainc.py).

The scalar path (gammainc_scalar) collapses each table's a-axis once per
call, which is the right cost when a is one value. With a different a on
every element that collapse would run per element, so this path evaluates
one small 2-D Chebyshev patch per element instead, from the table
gammainc_array_gen builds (zones, variables and formulas are in its
docstring). Per element: pick the zone, locate the patch (a few compares),
gather one KY x KP block, sum it against the two Chebyshev bases, add the
zone's prefactor. No lgamma and no loop.

Each zone computes the probability that is small there directly, as a log
(ln P in A0 and L, ln Q in B0 and U, the Temme zone's smaller side through
erfcx), and the other one as log1p(-exp(direct)). The far side loses a
factor of (direct probability)/(far probability), which only matters in A0:
there Q = 1 - P falls to 0.024 (a = 0.1, x = 1.1), and ln Q is 8.4e-15
against 7.8e-16 everywhere else (tests/test_gammainc_array.py). The scalar
path takes 1 - P down to Q ~ 1e-6, in its wedge just below x = 8 at small
a. Neither log underflows (ln P reaches -1e5 and beyond).

The Temme zone needs phi = d - log1p(d), d = (x - a)/a, near d = 0, where
that form cancels (relative error ~2 eps/|d|). gammainc_large's w_eta
switches to a series at |d| = 1e-3, which suits float64; float32 would lose
4 digits there. This path uses z = d/(2 + d), from log1p(d) = 2 atanh(z),
which gives phi = 2 z^2 S(z) with S = 1 + 2/3 z + z^2 + 4/5 z^3 + ...
(1 for even powers, (k+1)/(k+2) for odd k), cancellation-free in both
dtypes; eta = 2 z sqrt(S) then needs no sign and no sqrt(0).

Endpoints as in gammainc: x <= 0 gives P = 0, Q = 1 (ln P = -inf), x = inf
gives P = 1, Q = 0, and nan propagates. a < 0.1, a = inf and a = nan give
nan. Gradients in a and x, forward and reverse, are AD through the patch
polynomial and the prefactors, with two slopes written out: ln(x/a) and an
exp whose slope is 0 where it underflows. Every branch is fed masked x on
the lanes it does not own, so nothing outside a lane's own zone can turn
its slope into nan. The far tails keep their relative accuracy in the
slopes too (d ln Q/da ~ ln(x/a) at x = 1e100). The one infinity is d ln P/dx
where its true value, ~ a/x, overflows. At x = 0 the slope in x is spliced
in by hand (the density, as in gammainc), and x < 0 and x = inf are flat.

Dtype: the active jax float dtype, like every canonicalized chebax
function. float32 (x64 off) evaluates the leading KY32 x KP32 block of the
same patches, which the generator picks so the dropped coefficients stay
under float32's roundoff.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_array_table as _t
from chebax._src.recipes._common import edge_slope
from chebax._src.recipes.erf_family import erfcx

_A0, _B0, _L1, _U1, _L2, _U2, _T = range(len(_t.NAMES))
_LN2PI = math.log(2.0 * math.pi)
# the series of S(z) to z^24: on |z| <= 0.2 its tail is below 2e-17, and
# the direct form beyond loses at most ~10 eps there
_Z_SER = 0.2
_S = tuple(1.0 if k % 2 == 0 else (k + 1) / (k + 2) for k in range(25))


@functools.lru_cache(maxsize=2)
def _tables(dtype):
    """(patch blocks, parameter cuts, argument cuts) in the compute dtype."""
    dt = np.dtype(dtype)
    tab = _t.TABLE if dt == np.float64 else _t.TABLE[:, : _t.KY32, : _t.KP32]
    return np.ascontiguousarray(tab, dt), _t.PCUTS.astype(dt), _t.YCUTS.astype(dt)


def _basis(s, n):
    """T_0(s), ..., T_{n-1}(s) along a new last axis."""
    b = [jnp.ones_like(s), s]
    for _ in range(2, n):
        b.append(2.0 * s * b[-1] - b[-2])
    return jnp.stack(b[:n], axis=-1)


def _locate(val, cuts, n):
    """Interval index of val in each row of cuts (n intervals, padded by
    repeating the last cut) and val mapped onto [-1, 1] inside it."""
    k = jnp.clip(jnp.sum(val[..., None] >= cuts[..., 1:], axis=-1), 0, n - 1)
    lo = jnp.take_along_axis(cuts, k[..., None], axis=-1)[..., 0]
    hi = jnp.take_along_axis(cuts, k[..., None] + 1, axis=-1)[..., 0]
    return k, 2.0 * (val - lo) / (hi - lo) - 1.0


def _log1p(x):
    """log1p for x in (-1, inf). XLA's float64 log1p is up to 128 ulp off
    on (-0.63, -0.25) (jax 0.11 on CPU, increment 34), which put 5.5e-14
    into ln P through phi; below -1/4, 1 + x rounds by at most an ulp of
    0.75 and log is accurate (3.7e-16 measured). The textbook fix,
    x log(u)/(u - 1), does not survive XLA: it simplifies u - 1 to x."""
    return jnp.where(x < -0.25, jnp.log(1.0 + x), jnp.log1p(x))


def _phi_eta(d):
    """phi = d - log1p(d) and eta = sign(d) sqrt(2 phi), smooth through 0."""
    z = d / (2.0 + d)
    ser = jnp.abs(z) <= _Z_SER
    zs = jnp.where(ser, z, 0.0)
    s = _S[-1]
    for c in _S[-2::-1]:
        s = s * zs + c
    # direct lanes only; the dummy keeps log1p and sqrt off 0
    dd = jnp.where(ser, 1.0, d)
    phi_d = dd - _log1p(dd)
    eta_d = jnp.where(dd < 0.0, -1.0, 1.0) * jnp.sqrt(2.0 * phi_d)
    return (
        jnp.where(ser, 2.0 * zs * zs * s, phi_d),
        jnp.where(ser, 2.0 * zs * jnp.sqrt(s), eta_d),
    )


@jax.custom_jvp
def _log_ratio(x, a):
    """ln(x/a) for x, a > 0: the log of the ratio where that is a normal
    number, else ln x - ln a, so a lower tail whose x/a underflows keeps
    its log (x = 1e-36, a = 1000 in float32)."""
    r = x / a
    normal = r >= np.finfo(r.dtype).tiny
    return jnp.where(
        normal, jnp.log(jnp.where(normal, r, 1.0)), jnp.log(x) - jnp.log(a)
    )


@_log_ratio.defjvp
def _log_ratio_jvp(primals, tangents):
    # written out: through the ratio, AD forms x/a^2 and 1/r, which leave
    # the float range at its ends and lost the a-term (off by 1 at
    # x = 1e-306, a = 10, and by 0.6 at x = 1.7e308, a = 2.5)
    x, a = primals
    tx, ta = tangents
    return _log_ratio(x, a), tx / x - ta / a


@jax.custom_jvp
def _exp0(v):
    """exp(v) with slope exactly 0 where exp(v) is 0."""
    return jnp.exp(v)


@_exp0.defjvp
def _exp0_jvp(primals, tangents):
    # where x/a underflows, d ln P/dx = a/x can exceed the float range while
    # P is 0, and the product rule gives 0 * inf = nan in forward mode
    (v,), (t,) = primals, tangents
    e = jnp.exp(v)
    return e, jnp.where(e == 0.0, 0.0, e * t)


def _log_pq(a, x):
    """(ln P, ln Q) for a, x of one shape and float dtype; no x = 0 slope."""
    tab, pcuts, ycuts = (jnp.asarray(v) for v in _tables(a.dtype.name))
    ky, kp = tab.shape[1:]
    npar, narg, offset = (jnp.asarray(v) for v in (_t.NPAR, _t.NARG, _t.OFFSET))

    good_a = (a >= _t.ALO) & (a < jnp.inf)
    a = jnp.where(good_a, a, 1.0)
    ok = (x > 0.0) & (x < jnp.inf)
    xx = jnp.where(ok, x, 1.0)
    # lam only picks the zone: where x/a underflows (x = 1e-36, a = 1000 in
    # float32) it still reads as the lower zone
    lam = xx / a
    big = a > _t.ASPLIT
    zone = jnp.where(
        a < 1.0,
        jnp.where(xx <= a + 1.0, _A0, _B0),
        jnp.where(
            lam <= _t.LAM_LO,
            jnp.where(big, _L2, _L1),
            jnp.where(lam >= _t.LAM_HI, jnp.where(big, _U2, _U1), _T),
        ),
    )
    in_a0, in_b0, in_t = zone == _A0, zone == _B0, zone == _T
    is_low = (zone == _L1) | (zone == _L2)
    is_up = (zone == _U1) | (zone == _U2)
    uses_u = in_a0 | in_b0 | (zone == _L1) | (zone == _U1)

    # Each branch sees x on its own lanes and a harmless value elsewhere. A
    # select zeroes an unused branch's cotangent, but 0 * inf is nan, so a
    # division there that overflows under AD (a/x at x = 1e-20 in float32)
    # must not happen at all.
    x_a0 = jnp.where(in_a0, xx, a + 1.0)
    x_b0 = jnp.where(in_b0, xx, a + 1.0)
    x_lo = jnp.where(is_low, xx, a)
    x_up = jnp.where(is_up, xx, a)
    x_lu = jnp.where(is_low | is_up, xx, a)
    x_t = jnp.where(in_t, xx, a)

    # Temme coordinates from d = (x - a)/a, which keeps x - a exact
    phi_t, eta = _phi_eta((x_t - a) / a)

    par = jnp.select([uses_u, in_t], [_log1p(a), 1.0 / a], _t.ASPLIT / a)
    arg = jnp.select(
        [in_a0, in_b0, is_low, is_up],
        [x_a0 / (a + 1.0), (a + 1.0) / x_b0, x_lo / a, a / x_up],
        eta,
    )
    i, sp = _locate(par, pcuts[zone], npar[zone])
    j, sy = _locate(arg, ycuts[zone], narg[zone])
    block = tab[offset[zone] + i * narg[zone] + j]
    f = jnp.einsum("...kj,...k,...j->...", block, _basis(sy, ky), _basis(sp, kp))

    ln_a = jnp.log(a)
    half_ln_2pi_a = 0.5 * (ln_a + _LN2PI)
    # the Temme zone's small side in log space, through erfcx so neither
    # erfc nor e^(-y^2) underflows: ln Q = -y^2 + ln(erfcx(y)/2 + T/rt)
    # for y >= 0, ln P the same with -y and -T; y^2 = a phi
    y = eta * jnp.sqrt(0.5 * a)
    sign = jnp.where(y < 0.0, -1.0, 1.0)
    rt = jnp.sqrt(2.0 * math.pi * a)
    yt = jnp.where(in_t, jnp.abs(y), 0.0)
    ft = jnp.where(in_t, f, 0.0)
    temme = -a * phi_t + jnp.log(0.5 * erfcx(yt) + sign * ft / rt)

    # Below and above the Temme zone, -a phi = -(x - a) + a ln lambda. In
    # this form d/da is ln lambda up to O(1) terms; differentiating -a phi
    # as a product cancels two terms of size x/a, which at x = 1e8 in
    # float32 left 13.6 for 17.7.
    ln_lam = _log_ratio(x_lu, a)
    direct = jnp.select(
        [in_a0, in_b0, is_low, is_up],
        [
            a * jnp.log(x_a0) - x_a0 + f,
            (a - 1.0) * jnp.log(x_b0) - x_b0 + ln_a + f,
            a * ln_lam - (x_lu - a) - half_ln_2pi_a + f,
            (a - 1.0) * ln_lam - (x_lu - a) - half_ln_2pi_a + f,
        ],
        temme,
    )
    is_p = in_a0 | is_low | (in_t & (y < 0.0))
    other = _log1p(-_exp0(direct))
    lp = jnp.where(is_p, direct, other)
    lq = jnp.where(is_p, other, direct)

    top = x == jnp.inf
    lp = jnp.where(ok, lp, jnp.where(top, 0.0, -jnp.inf))
    lq = jnp.where(ok, lq, jnp.where(top, -jnp.inf, 0.0))
    bad = jnp.isnan(x) | ~good_a
    return jnp.where(bad, jnp.nan, lp), jnp.where(bad, jnp.nan, lq)


def _density0(a):
    """Gamma density at x = 0: inf below a = 1, 1 at it, 0 above."""
    return jnp.where(a < 1.0, jnp.inf, jnp.where(a > 1.0, 0.0, 1.0))


def p(a, x):
    lp, _ = _log_pq(a, x)
    return _exp0(lp) + edge_slope(jnp.where(x == 0.0, _density0(a), 0.0), x)


def q(a, x):
    _, lq = _log_pq(a, x)
    return _exp0(lq) + edge_slope(jnp.where(x == 0.0, -_density0(a), 0.0), x)


def log_p(a, x):
    lp, _ = _log_pq(a, x)
    # d ln P/dx = density/P ~ a/x, so the slope at x = 0 is inf for every a
    return lp + edge_slope(jnp.where(x == 0.0, jnp.inf, 0.0), x)


def log_q(a, x):
    _, lq = _log_pq(a, x)
    # Q = 1 at x = 0, so d ln Q/dx there is just -density
    return lq + edge_slope(jnp.where(x == 0.0, -_density0(a), 0.0), x)
