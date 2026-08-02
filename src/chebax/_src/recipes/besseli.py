"""I_v for real v in [0, 10] on x >= 0, from baked log-tables.

Two regions, one hard select at x = 8:

- x in [0, 8]:  I = exp(Lh(v, x^2)) * (x/2)^v / Gamma(v+1), with
                Lh = ln[Gamma(v+1) (x/2)^(-v) I_v(x)] tabulated in z = x^2
                (the factored entire part, logged: I > 0 everywhere).
- x > 8:        I = exp(Lt(v, 8/x)) * e^x / sqrt(2 pi x), with
                Lt = ln[sqrt(2 pi x) e^(-x) I_v(x)].

sqrt(2 pi) and sqrt(x) stay separate factors and the combined tail logs
them separately, so every finite x works: 2 pi x overflows at x ~ 2.9e307
while e^-x I_v(x) ~ 1/sqrt(2 pi x) is still perfectly representable there.

scaled=True returns e^(-x) I_v(x) (scipy's ive): the tail then never forms
e^x, so it stays finite past x ~ 709 where the unscaled value correctly
overflows to inf. The f64 floor is the familiar prefactor pow term,
~eps * v * |ln(x/2)|.

dI_v/dv is I_v times its log derivative, and near v = 0 that derivative is
an exponentially small number over an exponentially large one (-K_0/I_0 at
v = 0, DLMF 10.38.2). The tables' first v-derivative carries ~3e-16 of
ABSOLUTE noise and cannot resolve it, so below |dlnI/dv| ~ 1e-3 (3e-2 in
float32) the log derivative is rebuilt from the SECOND v-derivative, which
is O(1/x) and well conditioned: integrate it from 0 by two-point Gauss and
add the exact -K_0/I_0 at v = 0.

besseli(v)/besseli_dnu(v) mirror the besselj API; besseli_fn(nu, x)
additionally takes nu as a traced jax scalar (uniform per call, in [0, 10],
unchecked under trace), differentiable in both arguments.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.pytree import Recipe
from chebax._src.recipes import besseli_table as _it
from chebax._src.recipes._common import (canon_tag, check_range, digamma64, param_coefs,
                                         param_coefs_der, traced_coefs)
from chebax._src.recipes.besselk import besselk, besselk_fn
from chebax._src.series import ChebSeries

# above this the tail switches to a single combined exp: e^x alone overflows
# at x ~ 709.78 while I_v(x) stays representable to x ~ 713
_X_SPLIT_EXP = 700.0

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# second v-derivative of the tables: chebder twice along the parameter axis,
# with the [0, VMAX] -> [-1, 1] map's chain factor
_D2_IN = np.array([algorithms.chebder(algorithms.chebder(r)) for r in _it.TABLE_IN])
_D2_TAIL = np.array([algorithms.chebder(algorithms.chebder(r)) for r in _it.TABLE_TAIL])
_D2_SCALE = (2.0 / _it.VMAX) ** 2

# two-point Gauss-Legendre nodes on [0, v], as fractions of v
_G_LO = 0.5 - 0.5 / math.sqrt(3.0)
_G_HI = 0.5 + 0.5 / math.sqrt(3.0)

# crossover for besseli_ratio's cancelled small-x series. Above it the two
# scaled I's have (x/2)^v >= 1 and cannot underflow; at it the series argument
# (x/2)^2 is 1, where the terms fall like 1/(k!)^2
_RATIO_XS = 2.0


def _canon_is_f64():
    return jnp.empty(()).dtype == jnp.float64


def _trigamma64(z):
    """float64 psi'(z) for z >= 1: recurrence up to z >= 30, then the
    Bernoulli series through z^-11 (A&S 6.4.12). Truncation < 1e-18."""
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


def _eval_i(v, inv_gamma, x, lh, lt, scaled):
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    static_zero = isinstance(v, (int, float)) and v == 0.0
    if static_zero:
        # (x/2)^0 = 1, but the generic power rule is 0 * (x/2)^-1 at the
        # origin: NaN gradient for a constant prefactor
        pref = 1.0
    elif isinstance(v, (int, float)):
        pref = jnp.power(xi / 2, v)
    else:
        # traced v == 0 at the origin: constant prefactor 1, and the power
        # must see a safe dummy input too, or reverse mode transposes its
        # 0 * inf coefficient into NaN through the select. Only that one
        # corner is masked: for x > 0 the power keeps its v derivative
        # ln(x/2), which dI/dv needs.
        bad = (jnp.asarray(v) == 0.0) & (xi == 0.0)
        pref = jnp.where(bad, 1.0, jnp.power(jnp.where(bad, 1.0, xi) / 2, v))
    inner = jnp.exp(lh(xi * xi)) * pref * inv_gamma
    if scaled:
        inner = inner * jnp.exp(-xi)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    lt_val = lt(_it.XS / xo)
    if scaled:
        # 2 pi xo overflows past x ~ 2.9e307 and takes the whole quotient to
        # 0; two separate sqrts cost the same two roundings and cannot
        tail = jnp.exp(lt_val) / (_SQRT_2PI * jnp.sqrt(xo))
    else:
        # keep e^x as its own correctly rounded factor while it fits (a
        # combined exp costs eps*x of relative error); past the exp
        # overflow point only the combined form can express the result, and
        # its log has to split 2 pi off x for the same overflow reason
        xc = jnp.where(xo > _X_SPLIT_EXP, _X_SPLIT_EXP, xo)
        split = jnp.exp(lt_val) / jnp.sqrt(2 * jnp.pi * xc) * jnp.exp(xc)
        combined = jnp.exp(lt_val + xo - _LOG_SQRT_2PI - 0.5 * jnp.log(xo))
        tail = jnp.where(xo > _X_SPLIT_EXP, combined, split)
    out = jnp.where(x <= _it.XS, inner, tail)
    if not scaled:
        out = jnp.where(jnp.isinf(x), jnp.inf, out)
    return jnp.where(jnp.isnan(x), jnp.nan, out)


def _dlog_i(x, lh_nu, lt_nu, psi):
    """dlnI_v/dv from the tables' first v-derivative (psi = digamma(v+1))."""
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    return jnp.where(x <= _it.XS,
                     lh_nu(xi * xi) + jnp.log(xi / 2) - psi,
                     lt_nu(_it.XS / xo))


def _int_dlog2(v, x, lh2a, lt2a, psi1a, lh2b, lt2b, psi1b):
    """dlnI_v/dv minus its value at v = 0, as int_0^v d2 lnI_s/ds2 ds.

    Two-point Gauss, exact through cubics: that also covers the odd K-sized
    part of the integrand, which a one-point rule would miss. The series are
    the tables' second v-derivative at the two nodes, psi1 the trigamma the
    inner region's -lnGamma(v+1) contributes there. v factors out, so v = 0
    integrates to exactly 0 whatever the series say."""
    xi = jnp.where(x <= _it.XS, x, _it.XS)
    xo = jnp.where(x <= _it.XS, _it.XS, x)
    z, t = xi * xi, _it.XS / xo
    return 0.5 * v * jnp.where(x <= _it.XS,
                               (lh2a(z) - psi1a) + (lh2b(z) - psi1b),
                               lt2a(t) + lt2b(t))


def _combine_dnu(val, ratio, k0, d_first, integ, dv=1.0):
    """dI_v/dv from the two routes, times an order tangent dv.

    integ * val - k0 * ratio is I_v * [dlnI/dv|_0 + int_0^v d2lnI/ds2] with
    the v = 0 term kept as -K_0 * I_v/I_0, so nothing underflows where K_0
    and I_v separately still fit. v = 0 gives integ == 0 exactly, so past
    the x where I_v overflows that product is 0 * inf; substitute 0 there
    only, or the v derivative of the whole branch (d2I/dv2) goes with it.
    d_first, the first-derivative route, arrives already scaled by dv."""
    safe = jnp.where(jnp.isinf(val) & (integ == 0.0), 0.0, val)
    return jnp.where(jnp.abs(integ) < _dlog_floor(),
                     (integ * safe - k0 * ratio) * dv, d_first)


@jax.tree_util.register_pytree_node_class
class BesselI(Recipe):
    """Callable I_v (or e^-x I_v if scaled) on x >= 0. Build with besseli()."""

    _static_fields = ("v", "scaled")
    _series_fields = ("lh", "lt")

    def _post_init(self):
        self.v = float(self.v)
        self.scaled = bool(self.scaled)
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)

    def __call__(self, x):
        return _eval_i(self.v, self._inv_gamma, jnp.asarray(x), self.lh, self.lt, self.scaled)


@jax.tree_util.register_pytree_node_class
class BesselIdnu(Recipe):
    """Callable dI_v/dv (or e^-x dI_v/dv if scaled). Build with besseli_dnu()."""

    _static_fields = ("v", "scaled")
    _series_fields = ("lh", "lt", "lh_nu", "lt_nu",
                      "lh2a", "lt2a", "lh2b", "lt2b", "i0", "k0")

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
            sc = val if self.scaled else _eval_i(self.v, self._inv_gamma, x,
                                                 self.lh, self.lt, True)
            ratio = sc / self.i0(x)
        k0 = self.k0(x) * jnp.exp(-x) if self.scaled else self.k0(x)
        d_first = val * _dlog_i(x, self.lh_nu, self.lt_nu, self._psi)
        integ = _int_dlog2(self.v, x, self.lh2a, self.lt2a, self._psi1a,
                           self.lh2b, self.lt2b, self._psi1b)
        return _combine_dnu(val, ratio, k0, d_first, integ)


@functools.lru_cache(maxsize=128)
def _besseli_cached(v, scaled, _tag):
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    return BesselI(v, scaled,
                   ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
                   ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)))


def _d2_series(v):
    """The two second-v-derivative series (inner, tail) at order v."""
    return (ChebSeries(_D2_SCALE * param_coefs(_D2_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
            ChebSeries(_D2_SCALE * param_coefs(_D2_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)))


@functools.lru_cache(maxsize=128)
def _besseli_dnu_cached(v, scaled, _tag):
    v = check_range("besseli", "v", v, 0.0, _it.VMAX)
    lh2a, lt2a = _d2_series(_G_LO * v)
    lh2b, lt2b = _d2_series(_G_HI * v)
    return BesselIdnu(
        v, scaled,
        ChebSeries(param_coefs(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_it.TABLE_IN, 0.0, _it.VMAX, v), (0.0, _it.ZMAX)),
        ChebSeries(param_coefs_der(_it.TABLE_TAIL, 0.0, _it.VMAX, v), (0.0, 1.0)),
        lh2a, lt2a, lh2b, lt2b,
        _besseli_cached(0.0, True, canon_tag()), besselk(0.0),
    )


def _besseli_fn_impl(nu, x, scaled):
    inv_gamma = jnp.exp(-jax.scipy.special.gammaln(nu + 1.0))
    return _eval_i(nu, inv_gamma, x,
                   ChebSeries(traced_coefs(_it.TABLE_IN, 0.0, _it.VMAX, nu), (0.0, _it.ZMAX)),
                   ChebSeries(traced_coefs(_it.TABLE_TAIL, 0.0, _it.VMAX, nu), (0.0, 1.0)),
                   scaled)


def _d2_series_traced(nu):
    return (ChebSeries(_D2_SCALE * traced_coefs(_D2_IN, 0.0, _it.VMAX, nu), (0.0, _it.ZMAX)),
            ChebSeries(_D2_SCALE * traced_coefs(_D2_TAIL, 0.0, _it.VMAX, nu), (0.0, 1.0)))


def _besseli_dnu_impl(nu, x, scaled, val, d_ad, dnu):
    """dI_nu/dnu * dnu with nu traced; d_ad is the AD route, already scaled."""
    zero = jnp.zeros_like(nu)
    ratio = _besseli_fn_impl(nu, x, True) / _besseli_fn_impl(zero, x, True)
    k0 = besselk_fn(zero, x)
    if scaled:
        k0 = k0 * jnp.exp(-x)
    sa, sb = _G_LO * nu, _G_HI * nu
    lh2a, lt2a = _d2_series_traced(sa)
    lh2b, lt2b = _d2_series_traced(sb)
    integ = _int_dlog2(nu, x, lh2a, lt2a, jax.scipy.special.polygamma(1, sa + 1.0),
                       lh2b, lt2b, jax.scipy.special.polygamma(1, sb + 1.0))
    return _combine_dnu(val, ratio, k0, d_ad, integ, dnu)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _besseli_fn(nu, x, scaled):
    return _besseli_fn_impl(nu, x, scaled)


@_besseli_fn.defjvp
def _besseli_fn_jvp(scaled, primals, tangents):
    nu, x = primals
    dnu, dx = tangents
    val = _besseli_fn_impl(nu, x, scaled)
    _, d_x = jax.jvp(lambda xx: _besseli_fn_impl(nu, xx, scaled), (x,), (dx,))
    # nu direction: AD of the table reconstruction where the log derivative
    # is big enough to survive the tables' absolute noise, the integrated
    # second derivative below that (see the module docstring). AD runs on
    # dnu itself; a unit tangent scaled back out afterwards leaves XLA's
    # algebraic simplifier churning on the jitted gradient.
    _, d_ad = jax.jvp(lambda n: _besseli_fn_impl(n, x, scaled), (nu,), (dnu,))
    return val, d_x + _besseli_dnu_impl(nu, x, scaled, val, d_ad, dnu)


def besseli_fn(nu, x, scaled=False):
    """I_nu(x) with nu a (traceable) scalar, differentiable in both arguments.

    nu must be uniform per call and inside [0, 10] (unchecked under trace).
    Uses exp(gammaln) for the traced 1/Gamma(nu+1). The nu gradient at and
    near nu = 0 comes from the small-order path (exact -K_0(x) at nu = 0,
    clamped below x = 1e-6 with the K table), so it is also correct to
    second order in nu."""
    return _besseli_fn(jnp.asarray(nu), jnp.asarray(x), bool(scaled))


def besseli(v, scaled=False):
    """I_v on x >= 0 for real v in [0, 10] (e^-x I_v if scaled). Cached.

    v may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode."""
    return _besseli_cached(float(v), bool(scaled), canon_tag())


def besseli_dnu(v, scaled=False):
    """dI_v/dv on x > 0 for real v in [0, 10] (times e^-x if scaled).

    v = 0 reduces to the exact identity dI_v/dv|_0 = -K_0(x) (clamped below
    x = 1e-6 with the K table); small v > 0 rides the same small-order
    path."""
    return _besseli_dnu_cached(float(v), bool(scaled), canon_tag())


def _ratio_series(nu, x):
    """I_{nu+1}/I_nu from the ascending series with the prefactors cancelled.

    I_nu = (x/2)^nu/Gamma(nu+1) * sum_k (x^2/4)^k/(k! (nu+1)_k), so the ratio
    is x/(2(nu+1)) times a ratio of two sums that both start at 1. No power
    and no Gamma is ever evaluated on its own, which is what the direct
    quotient of two scaled I's gets wrong (both underflow at small x)."""
    y = (0.5 * x) ** 2
    # worst case y = 1 at the crossover, where term k is about 1/(k!)^2
    n = 16 if _canon_is_f64() else 9
    a0 = a1 = s0 = s1 = 1.0
    for k in range(1, n + 1):
        a0 = a0 * y / (k * (nu + k))
        a1 = a1 * y / (k * (nu + 1.0 + k))
        s0 = s0 + a0
        s1 = s1 + a1
    return 0.5 * x / (nu + 1.0) * (s1 / s0)


def besseli_ratio(nu, x):
    """I_{nu+1}(x) / I_nu(x), the Bessel ratio of circular statistics.

    A(kappa) = besseli_ratio(0.0, kappa) is the von Mises mean resultant
    length; the vMF version uses nu = d/2 - 1. nu must be uniform per call
    and inside [0, 9] (nu + 1 rides the same [0, 10] tables; unchecked
    under trace); x >= 0. The ratio is 0 at x = 0, increases toward 1, and
    is exactly 1 at x = +inf.

    Two branches: below x = 2 the cancelled series _ratio_series, so
    nothing underflows and the slope at the origin is the exact
    1/(2(nu+1)); from x = 2 up the quotient of the scaled besseli_fn
    values, whose (x/2)^nu prefactors are >= 1 there and whose e^x factors
    cancel."""
    nu = jnp.asarray(nu)
    x = jnp.asarray(x)
    near = x <= _RATIO_XS
    xs = jnp.where(near, x, 0.0)
    # inf and NaN ride the table branch's dummy too, or their gradients
    # transpose back through the select as NaN
    xt = jnp.where(near | ~jnp.isfinite(x), 2.0 * _RATIO_XS, x)
    r = jnp.where(near, _ratio_series(nu, xs),
                  besseli_fn(nu + 1.0, xt, scaled=True) / besseli_fn(nu, xt, scaled=True))
    r = jnp.where(jnp.isinf(x), 1.0, r)
    return jnp.where(jnp.isnan(x) | jnp.isnan(nu), jnp.nan, r)
