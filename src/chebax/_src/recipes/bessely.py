"""Y_v for real v in [0, 10] on x in [1e-6, inf), from baked tables.

Three regions, hard selects at 5 and 30:

- x in [1e-6, 5]: reduce to small order. With mu = v - floor(v) in [0, 1)
  (floor, not round: the (x/2)^mu scaling only removes the singular branch
  for mu >= 0), the baked pair T_a = (x/2)^mu Y_mu, T_b = (x/2)^(mu+1)
  Y_{mu+1} (tabulated in u = ln x) gives the base values, and floor(v)
  unrolled steps of the upward recurrence Y_{k+1} = (2k/x) Y_k - Y_{k-1}
  lift to order v. Upward is Y's stable direction once k is at or beyond x
  (it amplifies the dominant solution, which is Y itself); the seam sits at
  x = 5 rather than 8 because steps taken while k < x are neutral-to-
  cancelling and at x = 8 they cost 1.4e-10 (measured). The step count is
  fixed at instantiation, so the loop is branchless. This is Temme's
  strategy with the series replaced by tables, and it never touches the
  J_{+-v} connection formula, so integer orders are ordinary points (S3/R6).
- x in (5, 30]: Y_v fitted directly in x.
- x > 30: no tables of its own - besselj's P, Q give
  Y = sqrt(2/(pi x)) (sin(x) A - cos(x) B) with the same A, B constants.

|Y_v| reaches ~6e67 at (v=10, x=1e-6): fine in f64, overflows f32 (astype
is for moderate subdomains). Below XMIN = 1e-6 the input clamps. No traced-
order variant: the recurrence depth floor(v) is structural, so v cannot be
a tracer here (use besselj/besselk/besseli for traced orders).
"""

import functools
import math

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.recipes import bessely_table as _yt
from chebax._src.recipes._common import (canon_tag, check_range, param_coefs,
                                         param_coefs_der)
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
    """Callable Y_v on [1e-6, inf). Build with bessely(v)."""

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
        outer = (jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo)) * (jnp.sin(xo) * a - jnp.cos(xo) * b)
        return self._select(x, inner, mid, outer)


@jax.tree_util.register_pytree_node_class
class BesselYdnu(_YBase):
    """Callable dY_v/dv on [1e-6, inf). Build with bessely_dnu().

    Inner region: differentiate the base pair (chebder in mu, minus the
    ln(x/2) prefactor term) and carry (Y, dY) through the paired recurrence
    dY_{k+1} = (2/x) Y_k + (2k/x) dY_k - dY_{k-1}."""

    _static_fields = ("v",)
    _series_fields = ("ta", "tb", "ymid", "p", "q",
                      "ta_mu", "tb_mu", "ymid_nu", "p_nu", "q_nu")

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
        outer = (jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(xo)) * (jnp.sin(xo) * an - jnp.cos(xo) * bn)
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


@functools.lru_cache(maxsize=128)
def _bessely_cached(v, _tag):
    v = check_range("bessely", "v", v, 0.0, _yt.VMAX)
    return BesselY(v, *_base_series(v))


@functools.lru_cache(maxsize=128)
def _bessely_dnu_cached(v, _tag):
    v = check_range("bessely", "v", v, 0.0, _yt.VMAX)
    n, mu = _split_order(v)
    return BesselYdnu(
        v, *_base_series(v),
        ChebSeries(param_coefs_der(_yt.TABLE_A, 0.0, 1.0, mu), (_yt.U0, _yt.U1)),
        ChebSeries(param_coefs_der(_yt.TABLE_B, 0.0, 1.0, mu), (_yt.U0, _yt.U1)),
        ChebSeries(param_coefs_der(_yt.TABLE_MID, 0.0, _yt.VMAX, v), (_yt.X0, _yt.X1)),
        ChebSeries(param_coefs_der(_ext.TABLE_P, 0.0, _yt.VMAX, v), (0.0, 1.0)),
        ChebSeries(param_coefs_der(_ext.TABLE_QS, 0.0, _yt.VMAX, v), (0.0, 1.0)),
    )


def bessely(v):
    """Y_v on [1e-6, inf) for real v in [0, 10]. Cached per order; no mpmath.

    v may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode."""
    return _bessely_cached(float(v), canon_tag())


def bessely_dnu(v):
    """dY_v/dv on [1e-6, inf) for real v in [0, 10] (the order gradient)."""
    return _bessely_dnu_cached(float(v), canon_tag())
