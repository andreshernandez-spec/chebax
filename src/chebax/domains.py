"""The parameter box each public family serves, read off the table module
that governs the computation, so a widened box cannot be true in the recipe
and stale elsewhere (chebax.numpyro once kept rejecting TruncatedBeta(20, 3)
after the tables grew to [0.1, 100]).

    from chebax.domains import BETAINC
    BETAINC.lo, BETAINC.hi, BETAINC.contains(20.0)

`hi` is math.inf where a family has no upper end; `contains` is for concrete
floats, the traceable checks are in the recipes. Each box is built on first
access (PEP 562) and imports its own table only.
"""

import math
from collections import namedtuple
from typing import TYPE_CHECKING

__all__ = [
    "Domain",
    "BESSELI",
    "BESSELJ",
    "BESSELK",
    "BESSELY",
    "BETAINC",
    "GAMMAINC",
    "HYP1F1",
    "STUDENT_T_DF",
    "TWEEDIE_P",
    "VONMISES_KAPPA",
    "WRIGHT_RHO",
]


class Domain(namedtuple("Domain", "lo hi what")):
    """Closed interval [lo, hi] a family's parameter must lie in."""

    __slots__ = ()

    def contains(self, v):
        return self.lo <= float(v) <= self.hi

    def __str__(self):
        hi = "inf" if math.isinf(self.hi) else f"{self.hi:g}"
        return f"[{self.lo:g}, {hi}]"


def _betainc():
    from chebax._src.recipes import betainc_wide_table as t

    return Domain(t.ALO, t.AHI, "betainc a and b")


def _gammainc():
    from chebax._src.recipes import gammainc_large_table as glt
    from chebax._src.recipes import gammainc_table as gt

    # the Temme-zone tables run in v = ASPLIT / a down to VLO, and VLO = 0
    # is a = inf
    hi = math.inf if glt.VLO == 0.0 else glt.ASPLIT / glt.VLO
    return Domain(gt.ALO, hi, "gammainc a")


def _hyp1f1():
    from chebax._src.recipes import hyp1f1_table as t

    return Domain(t.ALO, t.AHI, "hyp1f1 a and b")


def _student_t_df():
    from chebax._src.recipes import stdtr_table as t

    return Domain(2.0 * t.BLO, 2.0 * t.BHI, "Student-t df")


def _tweedie_p():
    from chebax._src.recipes import tweedie_table as t

    return Domain(t.P_LO, t.P_HI, "Tweedie power p")


def _wright_rho():
    from chebax._src.recipes import tweedie_table as t

    return Domain(t.RHO_MIN, t.RHO_MAX, "log_wright_bessel0 rho")


def _vonmises_kappa():
    from chebax._src.recipes import vonmises_table as t

    return Domain(0.0, t.KMAX, "von Mises kappa")


def _besseli():
    from chebax._src.recipes import besseli_table as t

    return Domain(0.0, t.VMAX, "besseli order")


def _besselj():
    from chebax._src.recipes import besselj_table as t

    return Domain(0.0, t.VMAX, "besselj order")


def _besselk():
    from chebax._src.recipes import besselk_table as t

    return Domain(0.0, t.VMAX, "besselk order")


def _bessely():
    from chebax._src.recipes import bessely_table as t

    return Domain(0.0, t.VMAX, "bessely order")


_BUILD = {
    "BETAINC": _betainc,
    "GAMMAINC": _gammainc,
    "HYP1F1": _hyp1f1,
    "STUDENT_T_DF": _student_t_df,
    "TWEEDIE_P": _tweedie_p,
    "WRIGHT_RHO": _wright_rho,
    "VONMISES_KAPPA": _vonmises_kappa,
    "BESSELI": _besseli,
    "BESSELJ": _besselj,
    "BESSELK": _besselk,
    "BESSELY": _bessely,
}


def __getattr__(name):
    build = _BUILD.get(name)
    if build is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # cached as a plain global, so __getattr__ runs once per name
    value = globals()[name] = build()
    return value


def __dir__():
    return sorted(set(globals()) | set(_BUILD))


if TYPE_CHECKING:
    BESSELI: Domain
    BESSELJ: Domain
    BESSELK: Domain
    BESSELY: Domain
    BETAINC: Domain
    GAMMAINC: Domain
    HYP1F1: Domain
    STUDENT_T_DF: Domain
    TWEEDIE_P: Domain
    VONMISES_KAPPA: Domain
    WRIGHT_RHO: Domain
