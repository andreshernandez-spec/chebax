"""Spherical Bessel j_n, y_n for integer n in [0, 9] (jax#18119), as
sqrt(pi/(2x)) J_{n+1/2}(x) and sqrt(pi/(2x)) Y_{n+1/2}(x). The inner j_n is
sqrt(pi) / (2^{n+1} Gamma(n+3/2)) x^n g(x^2), exact at 0 in value and gradient.
"""

import functools
import math

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes._common import canon_tag, deprecated_factory
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.recipes import bessely_table as _yt
from chebax._src.recipes.besselj import besselj_frozen
from chebax._src.recipes.bessely import bessely_frozen

_NMAX = 9  # n + 1/2 must stay inside the cylindrical tables' [0, 10]


@jax.tree_util.register_pytree_node_class
class _SphericalJ(Recipe):
    _static_fields = ("n",)
    _series_fields = ("inst",)

    def _post_init(self):
        self.n = int(self.n)
        self._c0 = math.sqrt(math.pi) / (2.0 ** (self.n + 1) * math.gamma(self.n + 1.5))

    def __call__(self, x):
        x = jnp.asarray(x)
        inner_region = x <= _ext.MID_X0
        xi = jnp.where(inner_region, x, _ext.MID_X0)
        inner = self._c0 * xi**self.n * self.inst.g(xi * xi)
        xs = jnp.where(inner_region, 1.0, x)  # masked lanes get a safe dummy
        outer = jnp.sqrt(jnp.pi / (2 * xs)) * self.inst(xs)
        out = jnp.where(inner_region, inner, outer)
        # the masked outer evaluation gives nan at +inf, where j_n -> 0
        out = jnp.where(x == jnp.inf, 0.0, out)
        return jnp.where(jnp.isnan(x), jnp.nan, out)


@jax.tree_util.register_pytree_node_class
class _SphericalY(Recipe):
    _static_fields = ("n",)
    _series_fields = ("inst",)

    def _post_init(self):
        self.n = int(self.n)

    def __call__(self, x):
        x = jnp.asarray(x)
        # clamp the prefactor with Y: a raw one gives -1e13 for y_1(1e-8) ~ -1e16
        xc = jnp.where(x < _yt.XMIN, _yt.XMIN, x)
        val = jnp.sqrt(jnp.pi / (2 * xc)) * self.inst(xc)
        out = jnp.where(x > 0, val, -jnp.inf)
        out = jnp.where(x == jnp.inf, 0.0, out)  # same 1/x decay
        return jnp.where(jnp.isnan(x), jnp.nan, out)


def _check_n(n):
    if n != int(n) or not 0 <= int(n) <= _NMAX:
        raise ValueError(
            f"spherical Bessel wrappers cover integer n in [0, {_NMAX}], got {n}"
        )
    return int(n)


@functools.lru_cache(maxsize=128)
def _spherical_jn_cached(n, _tag):
    return _SphericalJ(n, besselj_frozen(n + 0.5))


def spherical_jn(n, x=None):
    """j_n(x), n in [0, 9]. Without x it is the deprecated 0.4 factory."""
    if x is None:
        return deprecated_factory("spherical_jn", spherical_jn_frozen, n)
    return spherical_jn_frozen(n)(x)


def spherical_jn_frozen(n):
    """j_n on x >= 0 for integer n in [0, 9]. Exact values and gradients at 0."""
    return _spherical_jn_cached(_check_n(n), canon_tag())


@functools.lru_cache(maxsize=128)
def _spherical_yn_cached(n, _tag):
    return _SphericalY(n, bessely_frozen(n + 0.5))


def spherical_yn(n, x=None):
    """y_n(x), n in [0, 9]. Without x it is the deprecated 0.4 factory."""
    if x is None:
        return deprecated_factory("spherical_yn", spherical_yn_frozen, n)
    return spherical_yn_frozen(n)(x)


def spherical_yn_frozen(n):
    """y_n for integer n in [0, 9]; x clamps at 1e-6, x <= 0 returns -inf."""
    return _spherical_yn_cached(_check_n(n), canon_tag())
