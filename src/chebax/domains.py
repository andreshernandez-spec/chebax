"""The parameter box each public family serves, in one place.

    from chebax.domains import BETAINC
    BETAINC.lo, BETAINC.hi          # 0.1, 100.0
    BETAINC.contains(20.0)          # True

Every one of these is read off the table module that actually governs the
computation, so a widened box cannot be true in the recipe and false
somewhere else. That is exactly what went wrong: chebax.numpyro imported
the [0.1, 10] panel constant and kept rejecting TruncatedBeta(20, 3)
after the tables grew to [0.1, 100], while its own docstring advertised
the wide box (review, 2026-08-02). The recipes, the numpyro and pytensor
integrations, and the constraint objects now all consume this module.

`hi` is math.inf where a family has no upper end (gammainc's Temme-zone
tables run to a = inf). `contains` is written for concrete floats: the
runtime checks are elsewhere and stay traceable.
"""

import math
from collections import namedtuple

from chebax._src.recipes import besseli_table as _it
from chebax._src.recipes import besselj_table as _jt
from chebax._src.recipes import besselk_table as _kt
from chebax._src.recipes import bessely_table as _yt
from chebax._src.recipes import betainc_wide_table as _bw
from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes import hyp1f1_table as _ht
from chebax._src.recipes import stdtr_table as _st
from chebax._src.recipes import vonmises_table as _vt

__all__ = ["Domain", "BESSELI", "BESSELJ", "BESSELK", "BESSELY", "BETAINC",
           "GAMMAINC", "HYP1F1", "STUDENT_T_DF", "VONMISES_KAPPA"]


class Domain(namedtuple("Domain", "lo hi what")):
    """Closed interval [lo, hi] a family's parameter must lie in."""

    __slots__ = ()

    def contains(self, v):
        return self.lo <= float(v) <= self.hi

    def __str__(self):
        hi = "inf" if math.isinf(self.hi) else f"{self.hi:g}"
        return f"[{self.lo:g}, {hi}]"


BETAINC = Domain(_bw.ALO, _bw.AHI, "betainc a and b")
GAMMAINC = Domain(_gt.ALO, _gl.AHI, "gammainc a")
HYP1F1 = Domain(_ht.ALO, _ht.AHI, "hyp1f1 a and b")
STUDENT_T_DF = Domain(2.0 * _st.BLO, 2.0 * _st.BHI, "Student-t df")
VONMISES_KAPPA = Domain(0.0, _vt.KMAX, "von Mises kappa")
BESSELI = Domain(0.0, _it.VMAX, "besseli order")
BESSELJ = Domain(0.0, _jt.VMAX, "besselj order")
BESSELK = Domain(0.0, _kt.VMAX, "besselk order")
BESSELY = Domain(0.0, _yt.VMAX, "bessely order")
