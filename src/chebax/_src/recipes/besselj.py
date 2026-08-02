"""J_v for real v in [0, 10] on x >= 0, from baked nu-tables.

Three regions, assembled with predicated selects (both seams are hard
switches, per ../../bessel/experiments/04):

- x in [0, 8]:   J = (x/2)^v / Gamma(v+1) * g_v(x^2), g_v = 0F1(; v+1; -z/4)
                 entire, one short Chebyshev series in z (the M2 table). The
                 factorization carries the x^v branch point, so accuracy near
                 0 is relative, not just absolute.
- x in (8, 30]:  J_v(x) as a direct Chebyshev series in x (no factorization
                 needed away from 0; direct fitting keeps the table
                 sup-accurate at large v where g_v spans six decades).
- x > 30:        J = sqrt(2/(pi x)) (P cos w - Q sin w) with the exact
                 modulus functions P, Q tabulated in t = (30/x)^2. The phase
                 w = x - (v/2 + 1/4) pi is never formed: with phi = (v/2 +
                 1/4) pi baked into instantiation constants,
                     J = sqrt(2/(pi x)) (cos(x) A + sin(x) B),
                     A = P cos(phi) + Q sin(phi),  B = P sin(phi) - Q cos(phi)
                 so sincos(x) does its own argument reduction and the eps*x
                 phase-subtraction error never appears.

Instantiation reconstructs every coefficient vector at v by Clenshaw in v
across the table rows (numpy, cached, no mpmath). dJ/dx flows through the
series custom_jvp; besselj_dnu gives dJ/dv via chebder along the v axis (for
the outer region, dA/dv and dB/dv pick up -(pi/2) B and +(pi/2) A from the
phi rotation). Validated against mpmath to x = 1e4; the outer form has no
upper limit beyond sincos accuracy. dJ/dv is log-singular at x = 0 (real for
v = 0, where besselj_dnu returns nan at exactly 0). v outside [0, 10] raises
at instantiation.
"""

import functools
import math

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import besselj_table as _tab
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.recipes._common import (canon_tag, check_range, digamma64,
                                         param_coefs, param_coefs_der)
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

    # Branch inputs: the active branch must see the raw x right up to and
    # including its own seam, or grad halves there (min/max ties split the
    # tangent). Inactive branches get clamped constants; where() masks their
    # gradients out.
    def _xin(self, x):
        return jnp.where(x <= _ext.MID_X0, x, _ext.MID_X0)

    def _xmid(self, x):
        return jnp.where(x <= _ext.MID_X0, _ext.MID_X0,
                         jnp.where(x <= _ext.MID_X1, x, _ext.MID_X1))

    def _xout(self, x):
        return jnp.where(x <= _ext.MID_X1, _ext.MID_X1, x)

    def _inner(self, x):
        xi = self._xin(x)
        # (x/2)^0 = 1, but the generic power rule is 0 * (x/2)^-1 at the
        # origin: NaN gradient for a constant prefactor
        pref = 1.0 if self.v == 0.0 else jnp.power(xi / 2, self.v)
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
        return jnp.where(x <= _ext.MID_X0, inner, jnp.where(x <= _ext.MID_X1, mid, outer))

    def _trim(self, x, y):
        """nan outside a declared domain. A trimmed instance evaluates
        only its own regions, so anything outside them would come back as
        that region's polynomial extrapolated, which is silently wrong;
        the whole point of naming a domain is that the caller promised to
        stay inside it. The branch inputs are already clamped, so this
        masks the OUTPUT and no gradient reaches a lane it should not."""
        if self.domain is None:
            return y
        lo, hi = self.domain
        return jnp.where((x >= lo) & (x <= hi), y, jnp.nan)


@jax.tree_util.register_pytree_node_class
class BesselJ(_JBase):
    """Callable J_v on x >= 0, or on `domain` when one was given.
    Build with besselj(v) or besselj(v, domain=(lo, hi))."""

    _static_fields = ("v", "domain")
    _series_fields = ("g", "jm", "p", "q")

    def _outer(self, x):
        xo, P, Q = self._outer_pq(x)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        return ((jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo))
                * (jnp.cos(xo) * a + jnp.sin(xo) * b))

    def __call__(self, x):
        x = jnp.asarray(x)
        if self.regions == ("in",):
            return self._trim(x, self._inner(x))
        if self.regions == ("mid",):
            return self._trim(x, self._mid(x))
        if self.regions == ("out",):
            return self._trim(x, self._outer(x))
        return self._trim(x, self._select(x, self._inner(x), self._mid(x),
                                          self._outer(x)))


@jax.tree_util.register_pytree_node_class
class BesselJdnu(_JBase):
    """Callable dJ_v/dv on x > 0, or on `domain` when one was given.
    Build with besselj_dnu(v) or besselj_dnu(v, domain=(lo, hi))."""

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
        return ((jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo))
                * (jnp.cos(xo) * an + jnp.sin(xo) * bn))

    def __call__(self, x):
        x = jnp.asarray(x)
        if self.regions == ("in",):
            return self._trim(x, self._inner_dnu(x))
        if self.regions == ("mid",):
            return self._trim(x, self.jmnu(self._xmid(x)))
        if self.regions == ("out",):
            return self._trim(x, self._outer_dnu(x))
        return self._trim(x, self._select(x, self._inner_dnu(x),
                                          self.jmnu(self._xmid(x)),
                                          self._outer_dnu(x)))


def _series_at(v):
    return (
        ChebSeries(param_coefs(_tab.TABLE, 0.0, _tab.VMAX, v), (0.0, _tab.ZMAX)),
        ChebSeries(param_coefs(_ext.TABLE_MID, 0.0, _tab.VMAX, v), (_ext.MID_X0, _ext.MID_X1)),
        ChebSeries(param_coefs(_ext.TABLE_P, 0.0, _tab.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs(_ext.TABLE_QS, 0.0, _tab.VMAX, v), (0.0, 1.0)),
    )


def _canon_domain(name, domain):
    if domain is None:
        return None
    lo, hi = (float(d) for d in domain)
    if not (0.0 <= lo < hi):
        raise ValueError(f"{name} needs domain=(lo, hi) with 0 <= lo < hi, "
                         f"got ({lo}, {hi})")
    return (lo, hi)


@functools.lru_cache(maxsize=128)
def _besselj_cached(v, domain, _tag):
    v = check_range("besselj", "v", v, 0.0, _tab.VMAX)
    return BesselJ(v, domain, *_series_at(v))


def besselj(v, domain=None):
    """J_v on x >= 0 for real v in [0, 10]. Cached per order; no mpmath.

    v may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode.

    domain=(lo, hi) narrows the instance to the regions covering [lo, hi]
    and drops the rest of the evaluation: the full callable computes all
    three regions and selects, so an x-range inside one of them pays
    about three times the arithmetic it needs (measured 2.5x f64 / 3.0x
    f32 for the inner region alone, experiments/04). Outside the declared
    domain the result is nan, since the regions that would have answered
    are the ones that were dropped. The tables and their accuracy are
    unchanged; this only removes work."""
    return _besselj_cached(float(v), _canon_domain("besselj", domain),
                           canon_tag())


@functools.lru_cache(maxsize=128)
def _besselj_dnu_cached(v, domain, _tag):
    v = check_range("besselj", "v", v, 0.0, _tab.VMAX)
    return BesselJdnu(
        v, domain, *_series_at(v),
        ChebSeries(param_coefs_der(_tab.TABLE, 0.0, _tab.VMAX, v), (0.0, _tab.ZMAX)),
        ChebSeries(param_coefs_der(_ext.TABLE_MID, 0.0, _tab.VMAX, v), (_ext.MID_X0, _ext.MID_X1)),
        ChebSeries(param_coefs_der(_ext.TABLE_P, 0.0, _tab.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_ext.TABLE_QS, 0.0, _tab.VMAX, v), (0.0, 1.0)),
    )


def besselj_dnu(v, domain=None):
    """dJ_v/dv on x > 0 for real v in [0, 10] (the order gradient).

    domain=(lo, hi) trims the regions exactly as besselj's does."""
    return _besselj_dnu_cached(float(v), _canon_domain("besselj_dnu", domain),
                               canon_tag())
