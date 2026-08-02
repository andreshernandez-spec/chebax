"""chebax: differentiable Chebyshev approximants for special functions.

Full float64 accuracy needs jax's x64 mode (jax.config.update("jax_enable_x64", True)).
"""

from chebax._src.generate import fit
from chebax._src.recipes.besseli import (besseli, besseli_dnu, besseli_fn,
                                          besseli_ratio)
from chebax._src.recipes.erf_family import dawsn, erfcx
from chebax._src.recipes.lambertw import lambertw
from chebax._src.recipes.matern import matern
from chebax._src.recipes.spherical import spherical_jn, spherical_yn
from chebax._src.recipes.besselj import besselj, besselj_dnu
from chebax._src.recipes.besselk import (besselk, besselk_dnu, besselk_fn,
                                         log_besselk_fn)
from chebax._src.recipes.bessely import bessely, bessely_dnu
from chebax._src.recipes.betainc import betainc, betainc_fn, log_betainc_fn
from chebax._src.recipes.gammainc import (gammainc, gammainc_fn, gammaincc,
                                          gammaincc_fn, log_gammainc_fn,
                                          log_gammaincc_fn)
from chebax._src.recipes.hyp1f1 import hyp1f1, hyp1f1_fn, log_hyp1f1_fn
from chebax._src.recipes.quantiles import (betaincinv, chi2inv,
                                           gammainccinv, gammaincinv,
                                           log_stdtr, log_stdtr_sf, stdtr,
                                           stdtrit)
from chebax._src.recipes.vonmises import vonmises_cdf, vonmises_icdf
from chebax._src.pergroup import pergroup
from chebax._src.series import ChebSeries, PiecewiseCheb

# single-sourced from pyproject.toml via package metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("chebax")
except Exception:  # not installed (e.g. running from a source checkout)
    __version__ = "0+unknown"
__all__ = ["ChebSeries", "PiecewiseCheb", "besseli", "besseli_dnu", "besseli_fn",
           "besseli_ratio", "besselj", "besselj_dnu", "besselk", "besselk_dnu", "besselk_fn",
           "log_besselk_fn",
           "bessely", "bessely_dnu", "betainc", "betainc_fn", "betaincinv",
           "chi2inv", "dawsn", "erfcx", "fit", "gammainc", "gammainc_fn", "gammaincc",
           "gammaincc_fn", "gammainccinv", "gammaincinv", "hyp1f1", "hyp1f1_fn", "lambertw",
           "log_betainc_fn", "log_hyp1f1_fn",
           "matern",
           "log_gammainc_fn", "log_gammaincc_fn", "log_stdtr", "log_stdtr_sf",
           "pergroup",
           "spherical_jn",
           "spherical_yn", "stdtr", "stdtrit", "vonmises_cdf", "vonmises_icdf",
           "__version__"]

# submodule re-export so `import chebax; chebax.bake` works as documented
# (bake reads __version__, so this import must stay below its definition)
from chebax import bake  # noqa: E402,F401

__all__.append("bake")
