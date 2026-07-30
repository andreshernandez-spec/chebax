"""Generator for the dawsn/erfcx tables. Build-time only; mpmath inside.

Both are parameter-free, so these are plain 1-D fits — no order tables.

- dawsn: odd and entire, D(x) = x E(x^2) with E fitted on z in [0, 36]
  (degree 37 measured); tail x D(x) = G(t), t = (6/x)^2, degree 10. The
  seam sits at 6, not 4: the recessive e^{-x^2} component (~1e-7 at x=4)
  measurably slows the tail fit there, while e^{-36} is below f64.
- erfcx: positive and smooth, fitted directly on x in [0, 6] (degree 32);
  tail x erfcx(x) = H(t), degree 9. Negative x is a runtime reflection,
  erfcx(-x) = 2 e^{x^2} - erfcx(x), which correctly overflows past
  x ~ -26.6.

Regenerate with:  python -m chebax._src.recipes.erf_gen
"""

import pathlib

import numpy as np

from chebax._src.recipes._gen_common import (DPS, dct, nodes,
                                             write_table_module)

XS_E = 6.0
NZ_DAWSN = 44
NT_DAWSN = 16
NC_ERFCX = 40
NT_ERFCX = 16


def generate_tables():
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS_E)
        zmax = xs * xs

        def dawsn(x):
            return mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)

        def erfcx(x):
            return mp.erfc(x) * mp.exp(x * x)

        e = dct(mp, [(lambda x: dawsn(x) / x)(mp.sqrt((t + 1) / 2 * zmax))
                     for t in nodes(mp, NZ_DAWSN)])
        g = dct(mp, [(lambda x: x * dawsn(x))(xs / mp.sqrt((t + 1) / 2))
                     for t in nodes(mp, NT_DAWSN)])
        c = dct(mp, [erfcx((t + 1) / 2 * xs) for t in nodes(mp, NC_ERFCX)])
        h = dct(mp, [(lambda x: x * erfcx(x))(xs / mp.sqrt((t + 1) / 2))
                     for t in nodes(mp, NT_ERFCX)])
    tof = lambda row: np.array([float(v) for v in row])
    return tof(e), tof(g), tof(c), tof(h)


def main():
    e, g, c, h = generate_tables()
    write_table_module(
        pathlib.Path(__file__).with_name("erf_table.py"),
        "chebax._src.recipes.erf_gen",
        "1-D coefficient vectors; DAWSN_E on z = x^2 in [0, XS^2], ERFCX_C "
        "on x in [0, XS], the two tails on t = (XS/x)^2 in [0, 1]",
        {"nz_dawsn": NZ_DAWSN, "nt_dawsn": NT_DAWSN,
         "nc_erfcx": NC_ERFCX, "nt_erfcx": NT_ERFCX, "xs": XS_E},
        {"XS": XS_E},
        {"DAWSN_E": e, "DAWSN_G": g, "ERFCX_C": c, "ERFCX_H": h})


if __name__ == "__main__":
    main()
