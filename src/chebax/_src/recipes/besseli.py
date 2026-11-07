"""I_v for real v in [0, 10] on x >= 0, split at x = 8:

    x <= 8: I = (x/2)^v / Gamma(v+1) [1 + z/(4(v+1)) (1 + z R(v,z))], z = x^2
    x > 8:  I = exp(Lt(v, 8/x)) e^x / sqrt(2 pi x)

R is an entire 16-term series (10 in float32), avoiding the logarithm's
nearby complex singularities and preserving the first two terms at zero.
Lt = ln[sqrt(2 pi x) e^-x I_v]. The original inner log table is used for
order derivatives, whose near-zero correction needs that representation.

sqrt(2 pi) and sqrt(x) stay separate, since 2 pi x overflows at x ~ 2.9e307.
scaled=True gives e^-x I_v (scipy's ive), finite past x ~ 709. f64 floor
~eps v |ln(x/2)|, from the prefactor pow.

Near v = 0, dlnI/dv (-K_0/I_0 at v = 0) is below the ~3e-16 absolute noise of
the tables' first v-derivative, so below |dlnI/dv| ~ 1e-3 (3e-2 in float32) it
is the exact -K_0/I_0 plus the second v-derivative integrated from 0 by
two-point Gauss.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.pytree import Recipe
from chebax._src.recipes import besseli_table as _it
from chebax._src.recipes._common import (
    canon_param,
    canon_float,
    canon_tag,
    check_range,
    deprecated_factory,
    digamma64,
    frozen_cache,
    log_half,
    param_coefs,
    param_coefs_der,
    pow_half,
    renamed_order,
    traced_coefs,
)
from chebax._src.recipes.besselk import besselk_frozen, besselk_scalar
from chebax._src.series import ChebSeries

# e^x overflows at x ~ 709.78, I_v(x) only at x ~ 713
_X_SPLIT_EXP = 700.0

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# the tables' second v-derivative, with the [0, VMAX] map's chain factor
_D2_IN = np.array([algorithms.chebder(algorithms.chebder(r)) for r in _it.TABLE_IN])
_D2_TAIL = np.array([algorithms.chebder(algorithms.chebder(r)) for r in _it.TABLE_TAIL])
_D2_SCALE = (2.0 / _it.VMAX) ** 2

# two-point Gauss-Legendre nodes on [0, v], as fractions of v
_G_LO = 0.5 - 0.5 / math.sqrt(3.0)
_G_HI = 0.5 + 0.5 / math.sqrt(3.0)

# besseli_ratio's series below, scaled I's above (where (x/2)^v >= 1)
_RATIO_XS = 2.0


def _canon_is_f64():
    return jnp.empty(()).dtype == jnp.float64


def _trigamma64(z):
    """float64 psi'(z), z >= 1: recurrence to 30, then A&S 6.4.12 (< 1e-18)."""
    acc = 0.0
    while z < 30.0:
        acc += 1.0 / (z * z)
        z += 1.0
    w = 1.0 / z
    zi = w * w
    tail = 1 / 6 - zi * (1 / 30 - zi * (1 / 42 - zi * (1 / 30 - zi * (5 / 66))))
    return acc + w + 0.5 * zi + w * zi * tail


def _dlog_floor():
    """|dlnI/dv| below which the tables' first v-derivative is all noise."""
    return 1e-3 if _canon_is_f64() else 3e-2


def _tail_i(x, log_value, scaled):
    if scaled:
        # Two square roots keep the denominator finite at the dtype's upper end.
        return jnp.exp(log_value) / (_SQRT_2PI * jnp.sqrt(x))
    cap = _X_SPLIT_EXP if log_value.dtype == jnp.float64 else 80.0
    xc = jnp.where(x > cap, cap, x)
    split = jnp.exp(log_value) / jnp.sqrt(2 * jnp.pi * xc) * jnp.exp(xc)
    combined = jnp.exp(log_value + x - _LOG_SQRT_2PI - 0.5 * jnp.log(x))
    return jnp.where(x > cap, combined, split)


def _eval_i(v, inv_gamma, x, lh, lt, scaled):
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    static_zero = isinstance(v, (int, float)) and v == 0.0
    if static_zero:
        # the power rule gives 0 * (x/2)^-1 at 0, a nan gradient
        pref = 1.0
    elif isinstance(v, (int, float)):
        pref = pow_half(xi, v)
    else:
        # mask only v == 0 at x == 0 (0 * inf in reverse mode); elsewhere the
        # power keeps its v derivative ln(x/2), which dI/dv needs
        bad = (jnp.asarray(v) == 0.0) & (xi == 0.0)
        pref = jnp.where(bad, 1.0, pow_half(jnp.where(bad, 1.0, xi), v))
    inner = jnp.exp(lh(xi * xi)) * pref * inv_gamma
    if scaled:
        inner = inner * jnp.exp(-xi)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    lt_val = lt(_it.XS / xo)
    tail = _tail_i(xo, lt_val, scaled)
    out = jnp.where(x <= _it.XS, inner, tail)
    if not scaled:
        out = jnp.where(jnp.isinf(x), jnp.inf, out)
    return jnp.where(jnp.isnan(x), jnp.nan, out)


def _eval_value(v, inv_gamma, x, r, lt, scaled):
    inner = x <= _it.XS
    xi = jnp.where(inner, x, _it.XS)
    xo = jnp.where(inner, _it.XS, x)
    if isinstance(v, (int, float)):
        power = int(v) if float(v).is_integer() else v
        pref = pow_half(xi, power)
    else:
        zero = (v == 0.0) & (xi == 0.0)
        one = (v == 1.0) & (xi == 0.0)
        # The generic second power derivative has 0 * inf at v=1, x=0.
        pref = pow_half(jnp.where(zero | one, 1.0, xi), v)
        pref = jnp.where(zero, 1.0, jnp.where(one, xi / 2.0, pref))
    z = xi * xi
    entire = 1.0 + (z / (4.0 * (v + 1.0))) * (1.0 + z * r(z))
    small = pref * inv_gamma * entire
    tail = _tail_i(xo, lt(_it.XS / xo), scaled)
    if scaled:
        small = small * jnp.exp(-xi)
    value = jnp.where(inner, small, tail)
    if not scaled:
        value = jnp.where(jnp.isinf(x), jnp.inf, value)
    return jnp.where(jnp.isnan(x), jnp.nan, value)


def _value_series(v, reconstruct):
    f32 = canon_tag() == "float32"
    n, nt = (10, 24) if f32 else (16, 48)
    return (
        ChebSeries(reconstruct(_it.VALUE_IN[:n], 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(reconstruct(_it.TABLE_TAIL[:nt], 0.0, _it.VMAX, v), (0.0, 1.0)),
    )


def _dlog_i(x, lh_nu, lt_nu, psi):
    """dlnI_v/dv from the tables' first v-derivative (psi = digamma(v+1))."""
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    return jnp.where(
        x <= _it.XS, lh_nu(xi * xi) + log_half(xi) - psi, lt_nu(_it.XS / xo)
    )


def _int_dlog2(v, x, lh2a, lt2a, psi1a, lh2b, lt2b, psi1b):
    """int_0^v d2 lnI_s/ds2 ds by two-point Gauss, exact through cubics (a
    one-point rule misses the odd K-sized part). psi1 is trigamma from the inner
    region's -lnGamma(v+1). v factors out, so v = 0 gives exactly 0."""
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    z, t = xi * xi, _it.XS / xo
    return (
        0.5
        * v
        * jnp.where(
            x <= _it.XS, (lh2a(z) - psi1a) + (lh2b(z) - psi1b), lt2a(t) + lt2b(t)
        )
    )


def _combine_dnu(val, ratio, k0, d_first, integ, dv=1.0):
    """dI_v/dv times dv: integ * val - k0 * ratio (the v = 0 term as
    -K_0 I_v/I_0) below the floor, else d_first, already scaled by dv. The
    0 * inf at v = 0 past I_v's overflow is zeroed there only, to keep d2I/dv2."""
    safe = jnp.where(jnp.isinf(val) & (integ == 0.0), 0.0, val)
    return jnp.where(
        jnp.abs(integ) < _dlog_floor(), (integ * safe - k0 * ratio) * dv, d_first
    )


@jax.tree_util.register_pytree_node_class
class BesselI(Recipe):
    """Callable I_v (or e^-x I_v if scaled) on x >= 0. Build with besseli_frozen()."""

    _static_fields = ("v", "scaled")
    _series_fields = ("r", "lt")

    def _post_init(self):
        self.v = float(self.v)
        self.scaled = bool(self.scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)

    def __call__(self, x):
        return _eval_value(
            self.v, self._inv_gamma, jnp.asarray(x), self.r, self.lt, self.scaled
        )


@jax.tree_util.register_pytree_node_class
class BesselIdnu(Recipe):
    """Callable dI_v/dv (or e^-x dI_v/dv if scaled). Build with besseli_dnu_frozen()."""

    _static_fields = ("v", "scaled")
    _series_fields = (
        "lh",
        "lt",
        "lh_nu",
        "lt_nu",
        "lh2a",
        "lt2a",
        "lh2b",
        "lt2b",
        "i0",
        "k0",
    )

    def _post_init(self):
        self.v = float(self.v)
        self.scaled = bool(self.scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)
        self._psi = digamma64(self.v + 1.0)
        self._psi1a = _trigamma64(_G_LO * self.v + 1.0)
        self._psi1b = _trigamma64(_G_HI * self.v + 1.0)

    def __call__(self, x):
        x = jnp.asarray(x)
        val = _eval_i(self.v, self._inv_gamma, x, self.lh, self.lt, self.scaled)
        if self.v == 0.0:
            ratio = 1.0
        else:
            sc = (
                val
                if self.scaled
                else _eval_i(self.v, self._inv_gamma, x, self.lh, self.lt, True)
            )
            ratio = sc / self.i0(x)
        k0 = self.k0(x) * jnp.exp(-x) if self.scaled else self.k0(x)
        d_first = val * _dlog_i(x, self.lh_nu, self.lt_nu, self._psi)
        integ = _int_dlog2(
            self.v,
            x,
            self.lh2a,
            self.lt2a,
            self._psi1a,
            self.lh2b,
            self.lt2b,
            self._psi1b,
        )
        out = _combine_dnu(val, ratio, k0, d_first, integ)
        if self.scaled:
            # the masked tail gives nan at +inf, where the limit is 0
            out = jnp.where(x == jnp.inf, 0.0, out)
        return out


@frozen_cache
def _besseli_cached(v, scaled, _tag):
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    return BesselI(
        v,
        scaled,
        *_value_series(v, param_coefs),
    )


def _d2_series(v):
    """The second-v-derivative series (inner, tail) at order v."""
    return (
        ChebSeries(_D2_SCALE * param_coefs(_D2_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(_D2_SCALE * param_coefs(_D2_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
    )


@frozen_cache
def _besseli_dnu_cached(v, scaled, _tag):
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    lh2a, lt2a = _d2_series(_G_LO * v)
    lh2b, lt2b = _d2_series(_G_HI * v)
    return BesselIdnu(
        v,
        scaled,
        ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs_der(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
        lh2a,
        lt2a,
        lh2b,
        lt2b,
        _besseli_cached(0.0, True, canon_tag()),
        besselk_frozen(0.0),
    )


def _besseli_impl(nu, x, scaled):
    inv_gamma = jnp.exp(-jax.scipy.special.gammaln(nu + 1.0))
    r, lt = _value_series(nu, traced_coefs)
    return _eval_value(nu, inv_gamma, x, r, lt, scaled)


def _besseli_log_impl(nu, x, scaled):
    inv_gamma = jnp.exp(-jax.scipy.special.gammaln(nu + 1.0))
    return _eval_i(
        nu,
        inv_gamma,
        x,
        ChebSeries(traced_coefs(_it.TABLE_IN, 0.0, _it.VMAX, nu), (0.0, _it.ZMAX)),
        ChebSeries(traced_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, nu), (0.0, 1.0)),
        scaled,
    )


def _d2_series_traced(nu):
    return (
        ChebSeries(
            _D2_SCALE * traced_coefs(_D2_IN, 0.0, _it.VMAX, nu), (0.0, _it.ZMAX)
        ),
        ChebSeries(_D2_SCALE * traced_coefs(_D2_TAIL, 0.0, _it.VMAX, nu), (0.0, 1.0)),
    )


def _besseli_dnu_impl(nu, x, scaled, val, d_ad, dnu):
    """dI_nu/dnu * dnu with nu traced; d_ad is the AD route, already scaled."""
    zero = jnp.zeros_like(nu)
    ratio = _besseli_impl(nu, x, True) / _besseli_impl(zero, x, True)
    k0 = besselk_scalar(zero, x)
    if scaled:
        k0 = k0 * jnp.exp(-x)
    sa, sb = _G_LO * nu, _G_HI * nu
    lh2a, lt2a = _d2_series_traced(sa)
    lh2b, lt2b = _d2_series_traced(sb)
    integ = _int_dlog2(
        nu,
        x,
        lh2a,
        lt2a,
        jax.scipy.special.polygamma(1, sa + 1.0),
        lh2b,
        lt2b,
        jax.scipy.special.polygamma(1, sb + 1.0),
    )
    return _combine_dnu(val, ratio, k0, d_ad, integ, dnu)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _besseli_scalar_cj(nu, x, scaled):
    return _besseli_impl(nu, x, scaled)


@functools.partial(_besseli_scalar_cj.defjvp, symbolic_zeros=True)
def _besseli_scalar_jvp(scaled, primals, tangents):
    nu, x = primals
    dnu, dx = tangents
    val = _besseli_impl(nu, x, scaled)
    tangent = jnp.zeros_like(val)
    if not isinstance(dx, jax.custom_derivatives.SymbolicZero):
        _, tangent = jax.jvp(lambda xx: _besseli_impl(nu, xx, scaled), (x,), (dx,))
    if not isinstance(dnu, jax.custom_derivatives.SymbolicZero):
        # AD on dnu itself: a unit tangent scaled afterwards stalls XLA's simplifier.
        _, d_ad = jax.jvp(lambda n: _besseli_log_impl(n, x, scaled), (nu,), (dnu,))
        tangent = tangent + _besseli_dnu_impl(nu, x, scaled, val, d_ad, dnu)
    return val, tangent


def besseli_scalar(nu, x, scaled=False):
    """I_nu(x), nu a traced scalar in [0, 10] (unchecked under trace),
    differentiable in both to second order, also near nu = 0."""
    return _besseli_scalar_cj(
        canon_param(nu, "besseli_scalar", "nu"), canon_float(x), bool(scaled)
    )


def besseli(nu=None, x=None, scaled=False, *, v=None):
    """I_nu(x) for one order nu in [0, 10] (besseli_scalar). Without x, or with
    a bool for x, it is the deprecated 0.4 factory, and v= the 0.4 spelling of
    nu. chebax.pergroup gives one order per group."""
    nu = renamed_order("besseli", nu, v)
    if x is None or isinstance(x, bool):
        scaled = scaled if x is None else x
        return deprecated_factory("besseli", besseli_frozen, nu, scaled=scaled)
    return besseli_scalar(nu, x, scaled)


def besseli_frozen(v, scaled=False):
    """I_v on x >= 0 for a concrete v in [0, 10] (e^-x I_v if scaled). Cached."""
    return _besseli_cached(float(v), bool(scaled), canon_tag())


def besseli_dnu(v, x=None, scaled=False):
    """dI_v/dv at x for a concrete v; for a traced v, differentiate
    besseli_scalar. Without x it is the deprecated 0.4 factory."""
    if x is None or isinstance(x, bool):
        scaled = scaled if x is None else x
        return deprecated_factory("besseli_dnu", besseli_dnu_frozen, v, scaled=scaled)
    return besseli_dnu_frozen(v, scaled)(x)


def besseli_dnu_frozen(v, scaled=False):
    """dI_v/dv on x > 0 for v in [0, 10] (times e^-x if scaled); -K_0(x) at
    v = 0, with K clamped below x = 1e-6."""
    return _besseli_dnu_cached(float(v), bool(scaled), canon_tag())


def _ratio_series(nu, x):
    """I_{nu+1}/I_nu as x/(2(nu+1)) times a ratio of the two ascending sums,
    with no power or Gamma alone (scaled I's both underflow at small x)."""
    y = (0.5 * x) ** 2
    # term k ~ 1/(k!)^2 at y = 1, the crossover
    n = 16 if _canon_is_f64() else 9
    a0 = a1 = s0 = s1 = 1.0
    for k in range(1, n + 1):
        a0 = a0 * y / (k * (nu + k))
        a1 = a1 * y / (k * (nu + 1.0 + k))
        s0 = s0 + a0
        s1 = s1 + a1
    return 0.5 * x / (nu + 1.0) * (s1 / s0)


def besseli_ratio(nu, x):
    """I_{nu+1}(x) / I_nu(x), the von Mises mean resultant length at nu = 0
    (vMF: nu = d/2 - 1), for nu in [0, 9] uniform per call and x >= 0; 0 at
    x = 0, 1 at +inf. The series below x = 2, scaled besseli_scalar above."""
    nu = canon_param(nu, "besseli_ratio", "nu")
    x = jnp.asarray(x)
    near = x <= _RATIO_XS
    xs = jnp.where(near, x, 0.0)
    # inf and nan take the dummy too, or their gradients come back nan
    xt = jnp.where(near | ~jnp.isfinite(x), 2.0 * _RATIO_XS, x)
    r = jnp.where(
        near,
        _ratio_series(nu, xs),
        besseli_scalar(nu + 1.0, xt, scaled=True) / besseli_scalar(nu, xt, scaled=True),
    )
    r = jnp.where(jnp.isinf(x), 1.0, r)
    return jnp.where(jnp.isnan(x) | jnp.isnan(nu), jnp.nan, r)
