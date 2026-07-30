"""Generator for the von Mises CDF table. Build-time only; mpmath inside.

The centered (mu = 0) von Mises CDF on theta in [-pi, pi] has no closed
form. Its deviation from uniform,

    G(theta; kappa) = F(theta; kappa) - 1/2 - theta/(2 pi),

is odd in theta and vanishes at theta = +-pi, so (the dawsn move)
G = theta * H(theta^2) with H even-in-theta, smooth, and tabulated on
w = theta^2 in [0, pi^2]. The kappa -> inf boundary layer of width
~1/sqrt(kappa) sits at the LEFT endpoint of the w-interval, where
Chebyshev nodes cluster; the second axis is r = sqrt(kappa) on
[0, sqrt(50)], which keeps the layer's motion uniform (measured degrees:
w ~92 at kappa=50, r ~68; raw kappa needed >64; reproduce via
experiments/03_degree_measurement.py). Samples come from mp.quad of the
defining integral at 40 dps.

Regenerate with:  python -m chebax._src.recipes.vonmises_gen  (~5 min)
"""

import math
import pathlib

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, table_path, write_table_module)

KMAX_VM = 50.0
NW_VM = 104
NK_VM = 80


def generate_table():
    import mpmath as mp

    with mp.workdps(DPS):
        W = mp.pi ** 2
        rmax = mp.sqrt(KMAX_VM)
        tw = nodes(mp, NW_VM)
        rows = []
        for t in nodes(mp, NK_VM):
            r = (t + 1) / 2 * rmax
            kappa = r * r
            i0 = mp.besseli(0, kappa)
            samples = []
            for s in tw:
                w = (s + 1) / 2 * W
                th = mp.sqrt(w)
                F = mp.quad(lambda u: mp.exp(kappa * mp.cos(u)), [-mp.pi, th]) / (2 * mp.pi * i0)
                samples.append((F - mp.mpf(1) / 2 - th / (2 * mp.pi)) / th)
            rows.append(dct(mp, samples))
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    write_table_module(
        table_path(out_dir, __file__, "vonmises_table.py"),
        "chebax._src.recipes.vonmises_gen",
        "TABLE[k, j] = j-th r-coefficient (r = sqrt(kappa) in [0, RMAX]) of "
        "the k-th w-coefficient (w = theta^2 in [0, WMAX])",
        {"nw": NW_VM, "nk": NK_VM, "kmax": KMAX_VM},
        {"KMAX": KMAX_VM, "WMAX": math.pi ** 2, "RMAX": math.sqrt(KMAX_VM)},
        {"TABLE": generate_table()})


if __name__ == "__main__":
    main()
