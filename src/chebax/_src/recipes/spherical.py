"""Spherical Bessel functions j_n and y_n for integer n in [0, 9].

One-line wrappers over the half-integer cylindrical tables (jax#18119):

    j_n(x) = sqrt(pi / (2x)) J_{n+1/2}(x),
    y_n(x) = sqrt(pi / (2x)) Y_{n+1/2}(x),

so they inherit the cylindrical recipes' domains and accuracy. Domain
x >= 0 (j_n) and x > 0 (y_n); x = 0 returns the exact limit for j_n
(1 for n = 0, else 0; the gradient at exactly 0 is masked and reads 0,
which is wrong only for n = 1 where j_1'(0) = 1/3) and -inf for y_n.
"""

import functools

import jax
import jax.numpy as jnp

from chebax._src.pytree import Recipe
from chebax._src.recipes.besselj import besselj
from chebax._src.recipes.bessely import bessely

_NMAX = 9  # n + 1/2 must stay inside the cylindrical tables' [0, 10]


@jax.tree_util.register_pytree_node_class
class _Spherical(Recipe):
    _static_fields = ("n", "limit0")
    _series_fields = ("inst",)

    def _post_init(self):
        self.n = int(self.n)
        self.limit0 = float(self.limit0)

    def __call__(self, x):
        x = jnp.asarray(x)
        xs = jnp.where(x > 0, x, 1.0)  # masked lanes get a safe dummy
        val = jnp.sqrt(jnp.pi / (2 * xs)) * self.inst(xs)
        return jnp.where(x > 0, val, self.limit0)


def _check_n(n):
    if n != int(n) or not 0 <= int(n) <= _NMAX:
        raise ValueError(f"spherical Bessel wrappers cover integer n in [0, {_NMAX}], got {n}")
    return int(n)


@functools.lru_cache(maxsize=None)
def spherical_jn(n):
    """j_n on x >= 0 for integer n in [0, 9]."""
    n = _check_n(n)
    return _Spherical(n, 1.0 if n == 0 else 0.0, besselj(n + 0.5))


@functools.lru_cache(maxsize=None)
def spherical_yn(n):
    """y_n on x > 0 for integer n in [0, 9] (x = 0 returns -inf)."""
    n = _check_n(n)
    return _Spherical(n, -jnp.inf, bessely(n + 0.5))
