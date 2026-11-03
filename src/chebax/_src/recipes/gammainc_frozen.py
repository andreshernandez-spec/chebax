"""Frozen gammainc: P(a, .) and Q(a, .) for one concrete a, coefficients built
once in numpy and cached per a and x64 mode. Recipe pytrees: they jit,
differentiate in x and bake. gammainc_small for a <= 10, gammainc_large above.
"""

import functools

import jax

from chebax import domains as _dom
from chebax._src.pytree import Recipe
from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_small as _gs
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes._common import check_range


@jax.tree_util.register_pytree_node_class
class GammaIncP(Recipe):
    """Callable P(a, x) on x in [0, inf), a <= 10. Build with gammainc_frozen(a)."""

    _static_fields = ("a",)
    _series_fields = ("lser", "ltail")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return _gs.eval_gammainc(self.a, x, self.lser, self.ltail, lower=True)


@jax.tree_util.register_pytree_node_class
class GammaIncQ(Recipe):
    """Callable Q(a, x) on x in [0, inf), a <= 10. Build with gammaincc_frozen(a)."""

    _static_fields = ("a",)
    _series_fields = ("lser", "ltail")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return _gs.eval_gammainc(self.a, x, self.lser, self.ltail, lower=False)


@jax.tree_util.register_pytree_node_class
class GammaIncLargeP(Recipe):
    """Callable P(a, x), a > 10. Build with gammainc_frozen(a)."""

    _static_fields = ("a",)
    _series_fields = ("temme", "dlow", "dup")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return _gl.eval_large(self.a, x, self.temme, self.dlow, self.dup, "p")


@jax.tree_util.register_pytree_node_class
class GammaIncLargeQ(Recipe):
    """Callable Q(a, x), a > 10. Build with gammaincc_frozen(a)."""

    _static_fields = ("a",)
    _series_fields = ("temme", "dlow", "dup")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return _gl.eval_large(self.a, x, self.temme, self.dlow, self.dup, "q")


@functools.lru_cache(maxsize=128)
def p_at(a, _tag):
    a = check_range("gammainc_frozen", "a", a, *_dom.GAMMAINC[:2])
    if a > _gt.AHI:
        return GammaIncLargeP(a, *_gl.series_eager(a))
    return GammaIncP(a, *_gs.series_eager(a))


@functools.lru_cache(maxsize=128)
def q_at(a, _tag):
    a = check_range("gammaincc_frozen", "a", a, *_dom.GAMMAINC[:2])
    if a > _gt.AHI:
        return GammaIncLargeQ(a, *_gl.series_eager(a))
    return GammaIncQ(a, *_gs.series_eager(a))
