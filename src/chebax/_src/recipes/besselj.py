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
import numpy as np

from chebax._src import algorithms
from chebax._src.recipes import besselj_table as _tab
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.series import ChebSeries


def _digamma(z):
    """float64 psi(z) for z >= 1: recurrence up to z >= 15, then the Bernoulli
    series through z^-12 (A&S 6.3.5 / 6.3.18). Truncation < 1e-16 there."""
    acc = 0.0
    while z < 15.0:
        acc -= 1.0 / z
        z += 1.0
    zi = 1.0 / (z * z)
    tail = 1 / 12 - zi * (1 / 120 - zi * (1 / 252 - zi * (1 / 240 - zi * (1 / 132 - zi * (691 / 32760)))))
    return acc + math.log(z) - 0.5 / z - zi * tail


def _nu_eval(table, v):
    """Coefficient vector at order v: Clenshaw in v across table rows."""
    t = 2.0 * v / _tab.VMAX - 1.0
    return np.array([algorithms.chebval(t, row) for row in table])


def _nu_eval_der(table, v):
    """d/dv of the coefficient vector, via chebder along the v axis."""
    t = 2.0 * v / _tab.VMAX - 1.0
    d = np.array([algorithms.chebval(t, algorithms.chebder(row)) for row in table])
    return d * (2.0 / _tab.VMAX)


def _check_order(v):
    v = float(v)
    if not 0.0 <= v <= _tab.VMAX:
        raise ValueError(f"besselj table covers v in [0, {_tab.VMAX}], got {v}")
    return v


def _series_at(v):
    return dict(
        g=ChebSeries(_nu_eval(_tab.TABLE, v), (0.0, _tab.ZMAX)),
        jm=ChebSeries(_nu_eval(_ext.TABLE_MID, v), (_ext.MID_X0, _ext.MID_X1)),
        p=ChebSeries(_nu_eval(_ext.TABLE_P, v), (0.0, 1.0)),
        q=ChebSeries(_nu_eval(_ext.TABLE_QS, v), (0.0, 1.0)),
    )


class _Constants:
    """Shared v-derived scalars, recomputed deterministically on unflatten."""

    def _init_constants(self, v):
        self.v = float(v)
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
        return self._inv_gamma * jnp.power(xi / 2, self.v) * self.g(xi * xi)

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


@jax.tree_util.register_pytree_node_class
class BesselJ(_Constants):
    """Callable J_v on x >= 0. Build with besselj(v)."""

    def __init__(self, v, g, jm, p, q):
        self._init_constants(v)
        self.g, self.jm, self.p, self.q = g, jm, p, q

    def __call__(self, x):
        x = jnp.asarray(x)
        xo, P, Q = self._outer_pq(x)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        outer = jnp.sqrt(2.0 / (jnp.pi * xo)) * (jnp.cos(xo) * a + jnp.sin(xo) * b)
        return self._select(x, self._inner(x), self._mid(x), outer)

    def astype(self, dtype):
        return BesselJ(self.v, *(s.astype(dtype) for s in (self.g, self.jm, self.p, self.q)))

    def __repr__(self):
        return f"BesselJ(v={self.v}, dtype={self.g.coef.dtype})"

    def tree_flatten(self):
        return (self.g, self.jm, self.p, self.q), self.v

    @classmethod
    def tree_unflatten(cls, v, children):
        obj = object.__new__(cls)
        obj._init_constants(v)
        obj.g, obj.jm, obj.p, obj.q = children
        return obj


@jax.tree_util.register_pytree_node_class
class BesselJdnu(_Constants):
    """Callable dJ_v/dv on x > 0. Build with besselj_dnu(v)."""

    def __init__(self, v, g, jm, p, q, gnu, jmnu, pnu, qnu):
        self._init_constants(v)
        self._psi = _digamma(self.v + 1.0)
        self.g, self.jm, self.p, self.q = g, jm, p, q
        self.gnu, self.jmnu, self.pnu, self.qnu = gnu, jmnu, pnu, qnu

    def __call__(self, x):
        x = jnp.asarray(x)
        xi = self._xin(x)
        z = xi * xi
        pref = self._inv_gamma * jnp.power(xi / 2, self.v)
        inner = pref * ((jnp.log(xi / 2) - self._psi) * self.g(z) + self.gnu(z))

        mid = self.jmnu(self._xmid(x))

        xo, P, Q = self._outer_pq(x)
        s = _ext.XS / xo
        Pn = self.pnu(s * s)
        Qn = s * self.qnu(s * s)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        an = Pn * self._cphi + Qn * self._sphi - (0.5 * math.pi) * b
        bn = Pn * self._sphi - Qn * self._cphi + (0.5 * math.pi) * a
        outer = jnp.sqrt(2.0 / (jnp.pi * xo)) * (jnp.cos(xo) * an + jnp.sin(xo) * bn)
        return self._select(x, inner, mid, outer)

    def astype(self, dtype):
        return BesselJdnu(self.v, *(s.astype(dtype) for s in (
            self.g, self.jm, self.p, self.q, self.gnu, self.jmnu, self.pnu, self.qnu)))

    def __repr__(self):
        return f"BesselJdnu(v={self.v}, dtype={self.g.coef.dtype})"

    def tree_flatten(self):
        return (self.g, self.jm, self.p, self.q, self.gnu, self.jmnu, self.pnu, self.qnu), self.v

    @classmethod
    def tree_unflatten(cls, v, children):
        obj = object.__new__(cls)
        obj._init_constants(v)
        obj._psi = _digamma(obj.v + 1.0)
        obj.g, obj.jm, obj.p, obj.q, obj.gnu, obj.jmnu, obj.pnu, obj.qnu = children
        return obj


@functools.lru_cache(maxsize=None)
def besselj(v):
    """J_v on x >= 0 for real v in [0, 10]. Cached per order; no mpmath."""
    v = _check_order(v)
    return BesselJ(v, **_series_at(v))


@functools.lru_cache(maxsize=None)
def besselj_dnu(v):
    """dJ_v/dv on x > 0 for real v in [0, 10] (the order gradient)."""
    v = _check_order(v)
    return BesselJdnu(
        v, **_series_at(v),
        gnu=ChebSeries(_nu_eval_der(_tab.TABLE, v), (0.0, _tab.ZMAX)),
        jmnu=ChebSeries(_nu_eval_der(_ext.TABLE_MID, v), (_ext.MID_X0, _ext.MID_X1)),
        pnu=ChebSeries(_nu_eval_der(_ext.TABLE_P, v), (0.0, 1.0)),
        qnu=ChebSeries(_nu_eval_der(_ext.TABLE_QS, v), (0.0, 1.0)),
    )
