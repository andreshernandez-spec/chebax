"""Generator for the noncentral t table. Build-time only; mpmath inside.

nct.py writes the log-density as a saddle-point part, hh's table T and a
nu-only remainder

    B(nu) = R(nu) - R(nu/2),  R(x) = lnGamma(x) - (x - 1/2) ln x + x - ln(2 pi)/2,

Binet's remainder of Stirling's lnGamma (nct.py derives the split). B is
-1/(12 nu) + O(nu^-3) at large nu and tends to -ln(2)/2 at nu = 0, smooth in
ln nu, whose nearest singularities are pi off the real axis. Below NU_S it
is NPAN equal panels in ln nu on [ln NU_LO, ln NU_S], each fitted on NN
nodes and kept to NTP terms: the dropped terms sum to under 1e-17 on every
panel (16-19 needed, measured here). From NU_S up, Stirling's series
B = sum_k A_k (1 - 2^(2k+1)) nu^-(2k+1), A_k = B_(2k+2) / ((2k+2)(2k+1)), to
NA terms: the next is 2e-22 at NU_S. float32 keeps the leading K32 terms of
each panel, the fewest whose dropped sum is under TOL32. Regenerate with:

    python -m chebax._src.recipes.nct_gen
"""

import numpy as np

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    table_path,
    write_table_module,
)

NU_LO = 0.1  # hh's lower end
NU_S = 64.0  # Stirling's series from here up
NPAN = 4  # panels in ln nu on [NU_LO, NU_S]
NN = 32  # nodes per panel fit
NTP = 20  # terms kept per panel
NA = 6  # Stirling terms
TOL64 = 1e-17
TOL32 = 2e-8


def binet_diff(mp, nu):
    def r(x):
        return (
            mp.loggamma(x) - (x - mp.mpf(1) / 2) * mp.log(x) + x - mp.log(2 * mp.pi) / 2
        )

    return r(nu) - r(nu / 2)


def generate():
    import mpmath as mp

    with mp.workdps(DPS):
        lo, hi = mp.log(mp.mpf(NU_LO)), mp.log(mp.mpf(NU_S))
        w = (hi - lo) / NPAN
        panels, k32 = [], 1
        for p in range(NPAN):
            a = lo + p * w
            ls = [a + (u + 1) * w / 2 for u in nodes(mp, NN)]
            c = dct(mp, [binet_diff(mp, mp.exp(v)) for v in ls])
            tail = [mp.fsum(abs(v) for v in c[k:]) for k in range(NN)]
            if tail[NTP] > TOL64:
                raise RuntimeError(f"panel {p} drops {float(tail[NTP]):.1e}")
            k32 = max(k32, next(k for k in range(NN) if tail[k] < TOL32))
            panels.append([float(v) for v in c[:NTP]])
        stir = [
            float(
                mp.bernoulli(2 * k + 2)
                / ((2 * k + 2) * (2 * k + 1))
                * (1 - mp.mpf(2) ** (2 * k + 1))
            )
            for k in range(NA)
        ]
    return np.array(panels), np.array(stir), k32


def main(out_dir=None):
    panels, stir, k32 = generate()
    write_table_module(
        table_path(out_dir, __file__, "nct_table.py"),
        "chebax._src.recipes.nct_gen",
        "B_PANELS[p, k] the k-th coefficient of B(nu) = R(nu) - R(nu/2) in ln nu "
        "on panel p of NPAN equal panels of [ln NU_LO, ln NU_S]; B_STIRLING[k] "
        "the coefficient of nu^-(2k+1) from NU_S up",
        {"npan": NPAN, "nn": NN, "ntp": NTP, "na": NA, "tol32": TOL32},
        {"NU_LO": NU_LO, "NU_S": NU_S, "NPAN": NPAN, "K32": k32},
        {"B_PANELS": panels, "B_STIRLING": stir},
    )


if __name__ == "__main__":
    main()
