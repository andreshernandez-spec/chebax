"""chebax: differentiable Chebyshev approximants for special functions.

Full float64 accuracy needs jax's x64 mode (jax.config.update("jax_enable_x64", True)).

Every public name is imported on first use (a module __getattr__, PEP 562),
so `import chebax` loads no recipe and no table, and a program that calls
two or three functions loads only their modules and tables.
"""

import importlib
import warnings
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
        "besseli_dnu_frozen",
        "besseli_frozen",
        "besseli_ratio",
        "besseli_scalar",
    ),
    "chebax._src.recipes.erf_family": ("dawsn", "erfcx"),
    "chebax._src.recipes.gammaln": ("gammaln",),
    "chebax._src.recipes.lambertw": ("lambertw",),
    "chebax._src.recipes.matern": ("matern",),
    "chebax._src.recipes.spherical": (
        "spherical_jn",
        "spherical_jn_frozen",
        "spherical_yn",
        "spherical_yn_frozen",
    ),
    "chebax._src.recipes.besselj": (
        "besselj",
        "besselj_dnu",
        "besselj_dnu_frozen",
        "besselj_frozen",
    ),
    "chebax._src.recipes.besselk": (
        "besselk",
        "besselk_dnu",
        "besselk_dnu_frozen",
        "besselk_frozen",
        "besselk_scalar",
        "log_besselk",
        "log_besselk_scalar",
    ),
    "chebax._src.recipes.bessely": (
        "bessely",
        "bessely_dnu",
        "bessely_dnu_frozen",
        "bessely_frozen",
    ),
    "chebax._src.recipes.betainc": (
        "betainc",
        "betainc_frozen",
        "betainc_scalar",
        "log_betainc",
        "log_betainc_scalar",
    ),
    "chebax._src.recipes.betaincinv": ("betaincinv",),
    "chebax._src.recipes.gammainc": (
        "gammainc",
        "gammainc_array",
        "gammainc_frozen",
        "gammainc_scalar",
        "gammaincc",
        "gammaincc_array",
        "gammaincc_frozen",
        "gammaincc_scalar",
        "log_gammainc",
        "log_gammainc_array",
        "log_gammainc_scalar",
        "log_gammaincc",
        "log_gammaincc_array",
        "log_gammaincc_scalar",
    ),
    "chebax._src.recipes.gammaincinv": (
        "chi2inv",
        "chi2inv_array",
        "chi2inv_scalar",
        "gammainccinv",
        "gammainccinv_array",
        "gammainccinv_scalar",
        "gammaincinv",
        "gammaincinv_array",
        "gammaincinv_scalar",
    ),
    "chebax._src.recipes.hyp1f1": (
        "hyp1f1",
        "hyp1f1_frozen",
        "hyp1f1_scalar",
        "log_hyp1f1",
        "log_hyp1f1_scalar",
    ),
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
    "besseli",
    "besseli_dnu",
    "besseli_dnu_frozen",
    "besseli_frozen",
    "besseli_ratio",
    "besseli_scalar",
    "besselj",
    "besselj_dnu",
    "besselj_dnu_frozen",
    "besselj_frozen",
    "besselk",
    "besselk_dnu",
    "besselk_dnu_frozen",
    "besselk_frozen",
    "besselk_scalar",
    "bessely",
    "bessely_dnu",
    "bessely_dnu_frozen",
    "bessely_frozen",
    "betainc",
    "betainc_frozen",
    "betainc_scalar",
    "betaincinv",
    "ChebSeries",
    "chi2inv",
    "chi2inv_array",
    "chi2inv_scalar",
    "dawsn",
    "erfcx",
    "fit",
    "gammainc",
    "gammainc_array",
    "gammainc_frozen",
    "gammainc_scalar",
    "gammaincc",
    "gammaincc_array",
    "gammaincc_frozen",
    "gammaincc_scalar",
    "gammainccinv",
    "gammainccinv_array",
    "gammainccinv_scalar",
    "gammaincinv",
    "gammaincinv_array",
    "gammaincinv_scalar",
    "gammaln",
    "hyp1f1",
    "hyp1f1_frozen",
    "hyp1f1_scalar",
    "lambertw",
    "log_besselk",
    "log_besselk_scalar",
    "log_betainc",
    "log_betainc_scalar",
    "log_gammainc",
    "log_gammainc_array",
    "log_gammainc_scalar",
    "log_gammaincc",
    "log_gammaincc_array",
    "log_gammaincc_scalar",
    "log_hyp1f1",
    "log_hyp1f1_scalar",
    "log_stdtr",
    "log_stdtr_sf",
    "log_wright_bessel0",
    "matern",
    "pergroup",
    "PiecewiseCheb",
    "spherical_jn",
    "spherical_jn_frozen",
    "spherical_yn",
    "spherical_yn_frozen",
    "stdtr",
    "stdtrit",
    "tweedie_logpdf",
    "vonmises_cdf",
    "vonmises_icdf",
    "__version__",
    "bake",
]


# the 0.4 names of the scalar paths, kept for one release with a warning
_RENAMED = {
    "besseli_fn": "besseli_scalar",
    "besselk_fn": "besselk_scalar",
    "log_besselk_fn": "log_besselk_scalar",
    "betainc_fn": "betainc_scalar",
    "log_betainc_fn": "log_betainc_scalar",
    "gammainc_fn": "gammainc_scalar",
    "gammaincc_fn": "gammaincc_scalar",
    "log_gammainc_fn": "log_gammainc_scalar",
    "log_gammaincc_fn": "log_gammaincc_scalar",
    "hyp1f1_fn": "hyp1f1_scalar",
    "log_hyp1f1_fn": "log_hyp1f1_scalar",
}


def __getattr__(name):
    if name in _RENAMED:
        new = _RENAMED[name]
        warnings.warn(
            f"chebax.{name} is deprecated: it is chebax.{new} now, and "
            f"chebax.{new.removesuffix('_scalar')} takes any parameter shape",
            DeprecationWarning,
            stacklevel=2,
        )
        # not cached, so every access warns
        return __getattr__(new)
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
        besseli_dnu_frozen,
        besseli_frozen,
        besseli_ratio,
        besseli_scalar,
    )
    from chebax._src.recipes.erf_family import dawsn, erfcx
    from chebax._src.recipes.gammaln import gammaln
    from chebax._src.recipes.lambertw import lambertw
    from chebax._src.recipes.matern import matern
    from chebax._src.recipes.spherical import (
        spherical_jn,
        spherical_jn_frozen,
        spherical_yn,
        spherical_yn_frozen,
    )
    from chebax._src.recipes.besselj import (
        besselj,
        besselj_dnu,
        besselj_dnu_frozen,
        besselj_frozen,
    )
    from chebax._src.recipes.besselk import (
        besselk,
        besselk_dnu,
        besselk_dnu_frozen,
        besselk_frozen,
        besselk_scalar,
        log_besselk,
        log_besselk_scalar,
    )
    from chebax._src.recipes.bessely import (
        bessely,
        bessely_dnu,
        bessely_dnu_frozen,
        bessely_frozen,
    )
    from chebax._src.recipes.betainc import (
        betainc,
        betainc_frozen,
        betainc_scalar,
        log_betainc,
        log_betainc_scalar,
    )
    from chebax._src.recipes.betaincinv import betaincinv
    from chebax._src.recipes.gammainc import (
        gammainc,
        gammainc_array,
        gammainc_frozen,
        gammainc_scalar,
        gammaincc,
        gammaincc_array,
        gammaincc_frozen,
        gammaincc_scalar,
        log_gammainc,
        log_gammainc_array,
        log_gammainc_scalar,
        log_gammaincc,
        log_gammaincc_array,
        log_gammaincc_scalar,
    )
    from chebax._src.recipes.gammaincinv import (
        chi2inv,
        chi2inv_array,
        chi2inv_scalar,
        gammainccinv,
        gammainccinv_array,
        gammainccinv_scalar,
        gammaincinv,
        gammaincinv_array,
        gammaincinv_scalar,
    )
    from chebax._src.recipes.hyp1f1 import (
        hyp1f1,
        hyp1f1_frozen,
        hyp1f1_scalar,
        log_hyp1f1,
        log_hyp1f1_scalar,
    )
    from chebax._src.recipes.stdtr import log_stdtr, log_stdtr_sf, stdtr, stdtrit
    from chebax._src.recipes.tweedie import log_wright_bessel0, tweedie_logpdf
    from chebax._src.recipes.vonmises import vonmises_cdf, vonmises_icdf
    from chebax._src.pergroup import pergroup
    from chebax._src.series import ChebSeries, PiecewiseCheb
    from chebax import bake, domains
