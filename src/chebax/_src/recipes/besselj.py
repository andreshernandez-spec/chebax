"""J_v for real v in [0, 10] on x >= 0, from v-tables in three regions with hard
seams (../../bessel/experiments/04):

    x <= 8:       J = (x/2)^v / Gamma(v+1) g_v(x^2),  g_v = 0F1(; v+1; -z/4),
                  so accuracy near 0 is relative
    8 < x <= 30:  J_v(x) directly (g_v spans six decades at large v)
    x > 30:       J = sqrt(2/(pi x)) (cos(x) A + sin(x) B), A = P cos(phi) +
                  Q sin(phi), B = P sin(phi) - Q cos(phi), phi = (v/2 + 1/4) pi,
                  P, Q in t = (30/x)^2

The phase x - phi is never formed, so it costs no eps x. dJ/dv comes from the
tables' v-derivative (outer: dA/dv, dB/dv gain -(pi/2) B, +(pi/2) A).
Validated against mpmath to x = 1e4. dJ/dv is log-singular at x = 0 (nan at
v = 0, x = 0).
"""

import functools
import math

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import besselj_table as _tab
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.recipes._common import (
    canon_tag,
    check_range,
    deprecated_factory,
    digamma64,
    param_coefs,
    param_coefs_der,
)
from chebax._src.series import ChebSeries


def _regions_for(domain):
    """The regions covering [lo, hi], as a tuple of "in", "mid", "out"."""
    if domain is None:
        return ("in", "mid", "out")
    lo, hi = domain
    regs = []
    if lo <= _ext.MID_X0:
        regs.append("in")
    if hi > _ext.MID_X0 and lo <= _ext.MID_X1:
        regs.append("mid")
    if hi > _ext.MID_X1:
        regs.append("out")
    return tuple(regs)


class _JBase(Recipe):
    """Shared v-derived constants and the three-region plumbing."""

    def _post_init(self):
        self.v = float(self.v)
        if getattr(self, "domain", None) is not None:
            self.domain = (float(self.domain[0]), float(self.domain[1]))
        self.regions = _regions_for(getattr(self, "domain", None))
        self._inv_gamma = 1.0 / math.gamma(self.v + 1.0)
        phi = (self.v / 2 + 0.25) * math.pi
        self._cphi = math.cos(phi)
        self._sphi = math.sin(phi)

    # the active branch sees raw x up to and at its seam (min/max ties halve
    # the gradient there); the others see clamped constants
    def _xin(self, x):
        return jnp.where(x <= _ext.MID_X0, x, _ext.MID_X0)

    def _xmid(self, x):
        return jnp.where(
            x <= _ext.MID_X0, _ext.MID_X0, jnp.where(x <= _ext.MID_X1, x, _ext.MID_X1)
        )

    def _xout(self, x):
        return jnp.where(x <= _ext.MID_X1, _ext.MID_X1, x)

    def _inner(self, x):
        xi = self._xin(x)
        # integer powers avoid 0 * inf in higher derivatives at 0
        power = int(self.v) if self.v.is_integer() else self.v
        pref = jnp.power(xi / 2, power)
        return self._inv_gamma * pref * self.g(xi * xi)

    def _mid(self, x):
        return self.jm(self._xmid(x))

    def _outer_pq(self, x):
        xo = self._xout(x)
        s = _ext.XS / xo
        t = s * s
        return xo, self.p(t), s * self.q(t)

    @staticmethod
    def _select(x, inner, mid, outer):
        return jnp.where(
            x <= _ext.MID_X0, inner, jnp.where(x <= _ext.MID_X1, mid, outer)
        )

    def _trim(self, x, y):
        """nan outside a declared domain, where a trimmed instance would
        silently extrapolate a region's polynomial."""
        if self.domain is None:
            return y
        lo, hi = self.domain
        return jnp.where((x >= lo) & (x <= hi), y, jnp.nan)


@jax.tree_util.register_pytree_node_class
class BesselJ(_JBase):
    """Callable J_v on x >= 0 or its domain. Build with besselj_frozen."""

    _static_fields = ("v", "domain")
    _series_fields = ("g", "jm", "p", "q")

    def _outer(self, x):
        xo, P, Q = self._outer_pq(x)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        return (jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo)) * (
            jnp.cos(xo) * a + jnp.sin(xo) * b
        )

    def __call__(self, x):
        x = jnp.asarray(x)
        if self.regions == ("in",):
            return self._trim(x, self._inner(x))
        if self.regions == ("mid",):
            return self._trim(x, self._mid(x))
        if self.regions == ("out",):
            return self._trim(x, self._outer(x))
        return self._trim(
            x, self._select(x, self._inner(x), self._mid(x), self._outer(x))
        )


@jax.tree_util.register_pytree_node_class
class BesselJdnu(_JBase):
    """Callable dJ_v/dv on x > 0 or its domain. Build with besselj_dnu_frozen."""

    _static_fields = ("v", "domain")
    _series_fields = ("g", "jm", "p", "q", "gnu", "jmnu", "pnu", "qnu")

    def _post_init(self):
        super()._post_init()
        self._psi = digamma64(self.v + 1.0)

    def _inner_dnu(self, x):
        xi = self._xin(x)
        z = xi * xi
        pref = self._inv_gamma * jnp.power(xi / 2, self.v)
        return pref * ((jnp.log(xi / 2) - self._psi) * self.g(z) + self.gnu(z))

    def _outer_dnu(self, x):
        xo, P, Q = self._outer_pq(x)
        s = _ext.XS / xo
        Pn = self.pnu(s * s)
        Qn = s * self.qnu(s * s)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        an = Pn * self._cphi + Qn * self._sphi - (0.5 * math.pi) * b
        bn = Pn * self._sphi - Qn * self._cphi + (0.5 * math.pi) * a
        return (jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo)) * (
            jnp.cos(xo) * an + jnp.sin(xo) * bn
        )

    def __call__(self, x):
        x = jnp.asarray(x)
        if self.regions == ("in",):
            return self._trim(x, self._inner_dnu(x))
        if self.regions == ("mid",):
            return self._trim(x, self.jmnu(self._xmid(x)))
        if self.regions == ("out",):
            return self._trim(x, self._outer_dnu(x))
        return self._trim(
            x,
            self._select(
                x, self._inner_dnu(x), self.jmnu(self._xmid(x)), self._outer_dnu(x)
            ),
        )


def _series_at(v):
    return (
        ChebSeries(param_coefs(_tab.TABLE, 0.0, _tab.VMAX, v), (0.0, _tab.ZMAX)),
        ChebSeries(
            param_coefs(_ext.TABLE_MID, 0.0, _tab.VMAX, v), (_ext.MID_X0, _ext.MID_X1)
        ),
        ChebSeries(param_coefs(_ext.TABLE_P, 0.0, _tab.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs(_ext.TABLE_QS, 0.0, _tab.VMAX, v), (0.0, 1.0)),
    )


def _canon_domain(name, domain):
    if domain is None:
        return None
    lo, hi = (float(d) for d in domain)
    if not (0.0 <= lo < hi):
        raise ValueError(
            f"{name} needs domain=(lo, hi) with 0 <= lo < hi, " f"got ({lo}, {hi})"
        )
    return (lo, hi)


@functools.lru_cache(maxsize=128)
def _besselj_cached(v, domain, _tag):
    v = check_range("besselj", "v", v, 0.0, _tab.VMAX)
    return BesselJ(v, domain, *_series_at(v))


def besselj(v, x=None, domain=None):
    """J_v(x) for a concrete v in [0, 10], not traceable. Without x it is the
    deprecated 0.4 factory."""
    if x is None:
        return deprecated_factory("besselj", besselj_frozen, v, domain=domain)
    return besselj_frozen(v, domain)(x)


def besselj_frozen(v, domain=None):
    """J_v on x >= 0 for a concrete v in [0, 10] (raises outside). Cached.

    domain=(lo, hi) evaluates only the regions covering it, nan outside:
    2.5x f64 / 3.0x f32 faster for the inner region alone (experiments/04)."""
    return _besselj_cached(float(v), _canon_domain("besselj", domain), canon_tag())


@functools.lru_cache(maxsize=128)
def _besselj_dnu_cached(v, domain, _tag):
    v = check_range("besselj", "v", v, 0.0, _tab.VMAX)
    return BesselJdnu(
        v,
        domain,
        *_series_at(v),
        ChebSeries(param_coefs_der(_tab.TABLE, 0.0, _tab.VMAX, v), (0.0, _tab.ZMAX)),
        ChebSeries(
            param_coefs_der(_ext.TABLE_MID, 0.0, _tab.VMAX, v),
            (_ext.MID_X0, _ext.MID_X1),
        ),
        ChebSeries(param_coefs_der(_ext.TABLE_P, 0.0, _tab.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_ext.TABLE_QS, 0.0, _tab.VMAX, v), (0.0, 1.0)),
    )


def besselj_dnu(v, x=None, domain=None):
    """dJ_v/dv at x for a concrete v. Without x it is the deprecated 0.4
    factory."""
    if x is None:
        return deprecated_factory("besselj_dnu", besselj_dnu_frozen, v, domain=domain)
    return besselj_dnu_frozen(v, domain)(x)


def besselj_dnu_frozen(v, domain=None):
    """dJ_v/dv on x > 0 for v in [0, 10]; domain= as in besselj_frozen."""
    return _besselj_dnu_cached(
        float(v), _canon_domain("besselj_dnu", domain), canon_tag()
    )
