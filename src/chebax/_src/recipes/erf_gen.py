"""Generator for the dawsn/erfcx tables. Build-time only; mpmath inside.

Both are parameter-free, so these are plain 1-D fits — no order tables.

- dawsn: odd and entire, D(x) = x E(x^2) with E fitted on z in [0, 36]
  (degree 37 measured); tail x D(x) = G(t), t = (6/x)^2, degree 10. The
  seam sits at 6, not 4: the recessive e^{-x^2} component (~1e-7 at x=4)
  measurably slows the tail fit there, while e^{-36} is below f64.
- erfcx: x >= 0 is mapped onto t = K / (K + x) in (0, 1] (the map of
  Schonfelder, Math. Comp. 32, 1978), erfcx(x) = t h(t), with
  h(t) = erfcx(x) (K + x) / K smooth on [0, 1], h(1) = 1 and
  h(0) = 1 / (K sqrt(pi)). One series in t needs 26 terms above
  2^-56 min h; it is split into NP equal t-panels instead, each fitted on
  24 nodes and kept to NTP terms (experiments/22). For values 32 panels
  need 8 terms (16 need 10, 64 need 7); the ninth brings the derivative's
  worst case from 6.8e-15 to 7.7e-16 (at the panel edges, where a
  truncated series' derivative errs most). A panel is picked by floor(NP t), with
  no select on x. Fewer dependent steps per point is what makes it
  faster, on XLA and in a compiled kernel run one element at a time.
  Negative x is a runtime reflection, erfcx(-x) = 2 e^{x^2} - erfcx(x),
  which correctly overflows past x ~ -26.6.
  ERFCX_PL holds each coefficient's rounding residue c - fl(c), for
  kernels that carry the leading coefficient past f64 (pytensor's Numba
  backend); the jax runtime does not use them.

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
NP_ERFCX = 32  # t-panels
NN_ERFCX = 24  # nodes per panel fit
NTP_ERFCX = 9  # terms kept per panel: 8 for values, 9 for the derivative


def generate_tables():
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS_E)
        zmax = xs * xs

        def dawsn(x):
            return mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)

        k = mp.mpf(K_ERFCX)

        def h(t):
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
        c, cl = [], []
        for j in range(NP_ERFCX):
            lo = mp.mpf(j) / NP_ERFCX
            row = dct(
                mp,
                [h(lo + (u + 1) / (2 * NP_ERFCX)) for u in nodes(mp, NN_ERFCX)],
            )[:NTP_ERFCX]
            c.append([float(v) for v in row])
            cl.append([float(v - mp.mpf(float(v))) for v in row])
    tof = lambda row: np.array([float(v) for v in row])
    return tof(e), tof(g), np.array(c), np.array(cl)


def main(out_dir=None):
    e, g, c, cl = generate_tables()
    write_table_module(
        table_path(out_dir, __file__, "erf_table.py"),
        "chebax._src.recipes.erf_gen",
        "1-D coefficient vectors; DAWSN_E on z = x^2 in [0, XS^2], DAWSN_G "
        "on t = (XS/x)^2 in [0, 1]; ERFCX_P[j, k] the k-th coefficient of h on "
        "the t-panel [j/NP, (j+1)/NP], t = K/(K + x), ERFCX_PL its rounding "
        "residues",
        {
            "nz_dawsn": NZ_DAWSN,
            "nt_dawsn": NT_DAWSN,
            "np_erfcx": NP_ERFCX,
            "nn_erfcx": NN_ERFCX,
            "ntp_erfcx": NTP_ERFCX,
            "xs": XS_E,
            "k": K_ERFCX,
        },
        {"XS": XS_E, "K": K_ERFCX, "NP": NP_ERFCX},
        {"DAWSN_E": e, "DAWSN_G": g, "ERFCX_P": c, "ERFCX_PL": cl},
    )


if __name__ == "__main__":
    main()
