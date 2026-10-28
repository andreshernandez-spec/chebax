"""chebax: differentiable Chebyshev approximants for special functions.

Full float64 accuracy needs jax's x64 mode (jax.config.update("jax_enable_x64", True)).

Every public name is imported on first use (a module __getattr__, PEP 562),
so `import chebax` loads no recipe and no table, and a program that calls
two or three functions loads only their modules and tables.
"""

import importlib
from typing import TYPE_CHECKING

# single-sourced from pyproject.toml via package metadata
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("chebax")
except Exception:  # not installed (e.g. running from a source checkout)
    __version__ = "0+unknown"

# defining module -> the public names it holds
_LAZY = {
    "chebax._src.generate": ("fit",),
    "chebax._src.recipes.besseli": (
        "besseli",
        "besseli_dnu",
        "besseli_fn",
        "besseli_ratio",
    ),
    "chebax._src.recipes.erf_family": ("dawsn", "erfcx"),
    "chebax._src.recipes.lambertw": ("lambertw",),
    "chebax._src.recipes.matern": ("matern",),
    "chebax._src.recipes.spherical": ("spherical_jn", "spherical_yn"),
    "chebax._src.recipes.besselj": ("besselj", "besselj_dnu"),
    "chebax._src.recipes.besselk": (
        "besselk",
        "besselk_dnu",
        "besselk_fn",
        "log_besselk_fn",
    ),
    "chebax._src.recipes.bessely": ("bessely", "bessely_dnu"),
    "chebax._src.recipes.betainc": ("betainc", "betainc_fn", "log_betainc_fn"),
    "chebax._src.recipes.betaincinv": ("betaincinv",),
    "chebax._src.recipes.gammainc": (
        "gammainc",
        "gammainc_fn",
        "gammaincc",
        "gammaincc_fn",
        "log_gammainc_fn",
        "log_gammaincc_fn",
    ),
    "chebax._src.recipes.gammaincinv": ("chi2inv", "gammainccinv", "gammaincinv"),
    "chebax._src.recipes.hyp1f1": ("hyp1f1", "hyp1f1_fn", "log_hyp1f1_fn"),
    "chebax._src.recipes.stdtr": ("log_stdtr", "log_stdtr_sf", "stdtr", "stdtrit"),
    "chebax._src.recipes.tweedie": ("log_wright_bessel0", "tweedie_logpdf"),
    "chebax._src.recipes.vonmises": ("vonmises_cdf", "vonmises_icdf"),
    "chebax._src.pergroup": ("pergroup",),
    "chebax._src.series": ("ChebSeries", "PiecewiseCheb"),
}
_WHERE = {name: mod for mod, names in _LAZY.items() for name in names}
# submodules that `import chebax; chebax.bake` reaches, as documented
_SUBMODULES = ("bake", "domains")

__all__ = [
    "ChebSeries",
    "PiecewiseCheb",
    "besseli",
    "besseli_dnu",
    "besseli_fn",
    "besseli_ratio",
    "besselj",
    "besselj_dnu",
    "besselk",
    "besselk_dnu",
    "besselk_fn",
    "log_besselk_fn",
    "bessely",
    "bessely_dnu",
    "betainc",
    "betainc_fn",
    "betaincinv",
    "chi2inv",
    "dawsn",
    "erfcx",
    "fit",
    "gammainc",
    "gammainc_fn",
    "gammaincc",
    "gammaincc_fn",
    "gammainccinv",
    "gammaincinv",
    "hyp1f1",
    "hyp1f1_fn",
    "lambertw",
    "log_betainc_fn",
    "log_hyp1f1_fn",
    "matern",
    "log_gammainc_fn",
    "log_gammaincc_fn",
    "log_stdtr",
    "log_stdtr_sf",
    "pergroup",
    "spherical_jn",
    "spherical_yn",
    "stdtr",
    "stdtrit",
    "vonmises_cdf",
    "vonmises_icdf",
    "log_wright_bessel0",
    "tweedie_logpdf",
    "__version__",
    "bake",
]


def __getattr__(name):
    if name in _WHERE:
        value = getattr(importlib.import_module(_WHERE[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # cached as a plain global, so __getattr__ runs once per name
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_WHERE) | set(_SUBMODULES))


# what type checkers and IDEs read; tests/test_lazy_imports.py keeps it
# equal to _LAZY
if TYPE_CHECKING:
    from chebax._src.generate import fit
    from chebax._src.recipes.besseli import (
        besseli,
        besseli_dnu,
        besseli_fn,
        besseli_ratio,
    )
    from chebax._src.recipes.erf_family import dawsn, erfcx
    from chebax._src.recipes.lambertw import lambertw
    from chebax._src.recipes.matern import matern
    from chebax._src.recipes.spherical import spherical_jn, spherical_yn
    from chebax._src.recipes.besselj import besselj, besselj_dnu
    from chebax._src.recipes.besselk import (
        besselk,
        besselk_dnu,
        besselk_fn,
        log_besselk_fn,
    )
    from chebax._src.recipes.bessely import bessely, bessely_dnu
    from chebax._src.recipes.betainc import betainc, betainc_fn, log_betainc_fn
    from chebax._src.recipes.betaincinv import betaincinv
    from chebax._src.recipes.gammainc import (
        gammainc,
        gammainc_fn,
        gammaincc,
        gammaincc_fn,
        log_gammainc_fn,
        log_gammaincc_fn,
    )
    from chebax._src.recipes.gammaincinv import chi2inv, gammainccinv, gammaincinv
    from chebax._src.recipes.hyp1f1 import hyp1f1, hyp1f1_fn, log_hyp1f1_fn
    from chebax._src.recipes.stdtr import log_stdtr, log_stdtr_sf, stdtr, stdtrit
    from chebax._src.recipes.tweedie import log_wright_bessel0, tweedie_logpdf
    from chebax._src.recipes.vonmises import vonmises_cdf, vonmises_icdf
    from chebax._src.pergroup import pergroup
    from chebax._src.series import ChebSeries, PiecewiseCheb
    from chebax import bake, domains
