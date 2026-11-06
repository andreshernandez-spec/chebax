"""Y_v for real v in [0, 10] on x in [1e-6, inf), three regions split at 5, 30:

    x <= 5:       mu = v - floor(v) in [0, 1); tables of (x/2)^mu Y_mu and
                  (x/2)^(mu+1) Y_{mu+1} in ln x, lifted by floor(v) steps of
                  Y_{k+1} = (2k/x) Y_k - Y_{k-1}, stable once k >= x (at a seam
                  of 8 the early steps cost 1.4e-10). No J_{+-v} connection
                  formula, so integer orders are ordinary points.
    5 < x <= 30:  Y_v directly
    x > 30:       Y = sqrt(2/(pi x)) (sin(x) A - cos(x) B), besselj's A, B

|Y_v| reaches ~6e67 at v = 10, x = 1e-6, past float32. x clamps below 1e-6.
floor(v) sets the recurrence depth, so the order cannot be traced (use besselj,
besselk or besseli for traced orders).
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.recipes import bessely_table as _yt
from chebax._src.recipes._common import (
    canon_tag,
    check_range,
    deprecated_factory,
    frozen_cache,
    param_coefs,
    param_coefs_der,
)
from chebax._src.series import ChebSeries


def _split_order(v):
    n = int(math.floor(v))
    return n, v - n  # mu in [0, 1)


class _YBase(Recipe):
    def _post_init(self):
        self.v = float(self.v)
        self.n, self.mu = _split_order(self.v)
        phi = (self.v / 2 + 0.25) * math.pi
        self._cphi = math.cos(phi)
        self._sphi = math.sin(phi)

    def _xin(self, x):
        return jnp.where(x < _yt.XMIN, _yt.XMIN, jnp.where(x <= _yt.X0, x, _yt.X0))

    def _xmid(self, x):
        return jnp.where(x <= _yt.X0, _yt.X0, jnp.where(x <= _yt.X1, x, _yt.X1))

    def _xout(self, x):
        return jnp.where(x <= _yt.X1, _yt.X1, x)

    @staticmethod
    def _select(x, inner, mid, outer):
        return jnp.where(x <= _yt.X0, inner, jnp.where(x <= _yt.X1, mid, outer))

    def _outer_pq(self, x):
        xo = self._xout(x)
        s = _ext.XS / xo
        t = s * s
        return xo, self.p(t), s * self.q(t)


@jax.tree_util.register_pytree_node_class
class BesselY(_YBase):
    """Callable Y_v on [1e-6, inf). Build with bessely_frozen(v)."""

    _static_fields = ("v",)
    _series_fields = ("ta", "tb", "ymid", "p", "q")

    def __call__(self, x):
        x = jnp.asarray(x)
        xi = self._xin(x)
        u = jnp.log(xi)
        yp = self.ta(u) * jnp.power(xi / 2, -self.mu)
        yq = self.tb(u) * jnp.power(xi / 2, -(self.mu + 1.0))
        for k in range(1, self.n):
            yq, yp = (2.0 * (self.mu + k) / xi) * yq - yp, yq
        inner = yp if self.n == 0 else yq

        mid = self.ymid(self._xmid(x))

        xo, P, Q = self._outer_pq(x)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        outer = (jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo)) * (
            jnp.sin(xo) * a - jnp.cos(xo) * b
        )
        return self._select(x, inner, mid, outer)


@jax.tree_util.register_pytree_node_class
class BesselYdnu(_YBase):
    """Callable dY_v/dv on [1e-6, inf); inner region by the paired recurrence
    dY_{k+1} = (2/x) Y_k + (2k/x) dY_k - dY_{k-1}. Build with bessely_dnu_frozen."""

    _static_fields = ("v",)
    _series_fields = (
        "ta",
        "tb",
        "ymid",
        "p",
        "q",
        "ta_mu",
        "tb_mu",
        "ymid_nu",
        "p_nu",
        "q_nu",
    )

    def __call__(self, x):
        x = jnp.asarray(x)
        xi = self._xin(x)
        u = jnp.log(xi)
        lg = jnp.log(xi / 2)
        pw_a = jnp.power(xi / 2, -self.mu)
        pw_b = jnp.power(xi / 2, -(self.mu + 1.0))
        a0, b0 = self.ta(u), self.tb(u)
        yp, yq = a0 * pw_a, b0 * pw_b
        pd = (self.ta_mu(u) - lg * a0) * pw_a
        qd = (self.tb_mu(u) - lg * b0) * pw_b
        for k in range(1, self.n):
            c = 2.0 * (self.mu + k) / xi
            yq_new = c * yq - yp
            qd_new = (2.0 / xi) * yq + c * qd - pd
            yp, yq, pd, qd = yq, yq_new, qd, qd_new
        inner = pd if self.n == 0 else qd

        mid = self.ymid_nu(self._xmid(x))

        xo, P, Q = self._outer_pq(x)
        s = _ext.XS / xo
        Pn = self.p_nu(s * s)
        Qn = s * self.q_nu(s * s)
        a = P * self._cphi + Q * self._sphi
        b = P * self._sphi - Q * self._cphi
        an = Pn * self._cphi + Qn * self._sphi - (0.5 * math.pi) * b
        bn = Pn * self._sphi - Qn * self._cphi + (0.5 * math.pi) * a
        outer = (jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo)) * (
            jnp.sin(xo) * an - jnp.cos(xo) * bn
        )
        return self._select(x, inner, mid, outer)


def _base_series(v):
    n, mu = _split_order(v)
    return (
        ChebSeries(param_coefs(_yt.TABLE_A, 0.0, 1.0, mu), (_yt.U0, _yt.U1)),
        ChebSeries(param_coefs(_yt.TABLE_B, 0.0, 1.0, mu), (_yt.U0, _yt.U1)),
        ChebSeries(param_coefs(_yt.TABLE_MID, 0.0, _yt.VMAX, v), (_yt.X0, _yt.X1)),
        ChebSeries(param_coefs(_ext.TABLE_P, 0.0, _yt.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs(_ext.TABLE_QS, 0.0, _yt.VMAX, v), (0.0, 1.0)),
    )


@frozen_cache
def _bessely_cached(v, _tag):
    v = check_range("bessely", "v", v, 0.0, _yt.VMAX)
    return BesselY(v, *_base_series(v))


@frozen_cache
def _bessely_dnu_cached(v, _tag):
    v = check_range("bessely", "v", v, 0.0, _yt.VMAX)
    n, mu = _split_order(v)
    return BesselYdnu(
        v,
        *_base_series(v),
        ChebSeries(param_coefs_der(_yt.TABLE_A, 0.0, 1.0, mu), (_yt.U0, _yt.U1)),
        ChebSeries(param_coefs_der(_yt.TABLE_B, 0.0, 1.0, mu), (_yt.U0, _yt.U1)),
        ChebSeries(param_coefs_der(_yt.TABLE_MID, 0.0, _yt.VMAX, v), (_yt.X0, _yt.X1)),
        ChebSeries(param_coefs_der(_ext.TABLE_P, 0.0, _yt.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_ext.TABLE_QS, 0.0, _yt.VMAX, v), (0.0, 1.0)),
    )


def bessely(v, x=None):
    """Y_v(x) for a concrete v in [0, 10]. Without x it is the deprecated 0.4
    factory."""
    if x is None:
        return deprecated_factory("bessely", bessely_frozen, v)
    return bessely_frozen(v)(x)


def bessely_frozen(v):
    """Y_v on [1e-6, inf) for a concrete v in [0, 10]. Cached."""
    return _bessely_cached(float(v), canon_tag())


def bessely_dnu(v, x=None):
    """dY_v/dv at x for a concrete v. Without x it is the deprecated 0.4
    factory."""
    if x is None:
        return deprecated_factory("bessely_dnu", bessely_dnu_frozen, v)
    return bessely_dnu_frozen(v)(x)


def bessely_dnu_frozen(v):
    """dY_v/dv on [1e-6, inf) for v in [0, 10]."""
    return _bessely_dnu_cached(float(v), canon_tag())
