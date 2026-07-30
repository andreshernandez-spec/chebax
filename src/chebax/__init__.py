"""chebax: differentiable Chebyshev approximants for special functions.

Full float64 accuracy needs jax's x64 mode (jax.config.update("jax_enable_x64", True)).
"""

from chebax._src.generate import fit
from chebax._src.recipes.besseli import besseli, besseli_dnu, besseli_fn
from chebax._src.recipes.erf_family import dawsn, erfcx
from chebax._src.recipes.besselj import besselj, besselj_dnu
from chebax._src.recipes.besselk import besselk, besselk_dnu, besselk_fn
from chebax._src.recipes.bessely import bessely, bessely_dnu
from chebax._src.series import ChebSeries, PiecewiseCheb

__version__ = "0.1.0.dev0"
__all__ = ["ChebSeries", "PiecewiseCheb", "besseli", "besseli_dnu", "besseli_fn",
           "besselj", "besselj_dnu", "besselk", "besselk_dnu", "besselk_fn",
           "bessely", "bessely_dnu", "dawsn", "erfcx", "fit", "__version__"]
