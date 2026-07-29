"""chebax: differentiable Chebyshev approximants for special functions.

Full float64 accuracy needs jax's x64 mode (jax.config.update("jax_enable_x64", True)).
"""

from chebax._src.generate import fit
from chebax._src.series import ChebSeries, PiecewiseCheb

__version__ = "0.1.0.dev0"
__all__ = ["ChebSeries", "PiecewiseCheb", "fit", "__version__"]
