"""Generator for the dawsn/erfcx tables. Build-time only; mpmath inside.

Both are parameter-free, so these are plain 1-D fits — no order tables.

- dawsn: odd and entire, D(x) = x E(x^2) with E fitted on z in [0, 36]
  (degree 37 measured); tail x D(x) = G(t), t = (6/x)^2, degree 10. The
  seam sits at 6, not 4: the recessive e^{-x^2} component (~1e-7 at x=4)
  measurably slows the tail fit there, while e^{-36} is below f64.
- erfcx: one series for all x >= 0 in t = K / (K + x), the map of
  Schonfelder (Math. Comp. 32, 1978): erfcx(x) = t h(t), with
  h(t) = erfcx(x) (K + x) / K smooth on [0, 1], h(1) = 1 and
  h(0) = 1 / (K sqrt(pi)). No seam, so no select for x >= 0. Terms above
  2^-56 min h (experiments/22): 26 at K = 3, 25-26 for K in [2.5, 5],
  30 at K = 1.5; 32 nodes. Negative x is a runtime reflection,
  erfcx(-x) = 2 e^{x^2} - erfcx(x), which correctly overflows past
  x ~ -26.6.
  ERFCX_TL holds each coefficient's rounding residue c - fl(c). Near
  t = 0 the sum cancels (h(0) = 0.188 from c0 = 0.471, c1 = 0.377, ...),
  so a kernel that evaluates the last Clenshaw steps in double-word
  arithmetic needs the leading coefficients past f64; the jax runtime
  does not use them.

Regenerate with:  python -m chebax._src.recipes.erf_gen
"""

import pathlib

import numpy as np

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    table_path,
    write_table_module,
)

XS_E = 6.0
NZ_DAWSN = 44
NT_DAWSN = 16
K_ERFCX = 3.0
NT_ERFCX = 32


def generate_tables():
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS_E)
        zmax = xs * xs

        def dawsn(x):
            return mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)

        k = mp.mpf(K_ERFCX)

        def h(u):
            t = (u + 1) / 2
            x = k * (1 - t) / t
            return mp.erfc(x) * mp.exp(x * x) * (k + x) / k

        e = dct(
            mp,
            [
                (lambda x: dawsn(x) / x)(mp.sqrt((t + 1) / 2 * zmax))
                for t in nodes(mp, NZ_DAWSN)
            ],
        )
        g = dct(
            mp,
            [
                (lambda x: x * dawsn(x))(xs / mp.sqrt((t + 1) / 2))
                for t in nodes(mp, NT_DAWSN)
            ],
        )
        c = dct(mp, [h(u) for u in nodes(mp, NT_ERFCX)])
        cl = [v - mp.mpf(float(v)) for v in c]
    tof = lambda row: np.array([float(v) for v in row])
    return tof(e), tof(g), tof(c), tof(cl)


def main(out_dir=None):
    e, g, c, cl = generate_tables()
    write_table_module(
        table_path(out_dir, __file__, "erf_table.py"),
        "chebax._src.recipes.erf_gen",
        "1-D coefficient vectors; DAWSN_E on z = x^2 in [0, XS^2], DAWSN_G "
        "on t = (XS/x)^2 in [0, 1], ERFCX_T on t = K/(K + x) in [0, 1] with "
        "ERFCX_TL its rounding residues",
        {
            "nz_dawsn": NZ_DAWSN,
            "nt_dawsn": NT_DAWSN,
            "nt_erfcx": NT_ERFCX,
            "xs": XS_E,
            "k": K_ERFCX,
        },
        {"XS": XS_E, "K": K_ERFCX},
        {"DAWSN_E": e, "DAWSN_G": g, "ERFCX_T": c, "ERFCX_TL": cl},
    )


if __name__ == "__main__":
    main()
