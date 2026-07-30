"""Generator for the besseli log-tables. Build-time only; mpmath inside.

I_v is positive and non-oscillatory, so it takes K's log treatment with J's
factored form:

- Inner, z = x^2 in [0, 64]:
      Lh(v, z) = ln[ Gamma(v+1) (x/2)^(-v) I_v(x) ]
  the log of the entire part h_v(z) = 0F1(; v+1; z/4). The z-degree is set
  by the complex zeros of I_v (for v=0 the nearest sits at z = -j_{0,1}^2
  ~ -5.78, giving rho ~ 1.8 and measured degree ~64); one v-panel suffices,
  there is no Gamma-pole feature (measured v-degree 52).

- Tail, t = 8/x on (0, 1]:
      Lt(v, t) = ln[ sqrt(2 pi x) e^(-x) I_v(x) ]
  The recessive e^(-2x) component is largest at half-integer v (measured
  worst t-degree 39, at v = 0.5); still converges cleanly at 48 nodes.

Both stages stay in mpmath (the M1 floor rule). Measured degrees reproduce
via experiments/03_degree_measurement.py. Regenerate with:

    python -m chebax._src.recipes.besseli_gen
"""

import pathlib

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, table_path, write_table_module)
from chebax._src.recipes.besselj_gen import VMAX

ZMAX_I = 64.0
XS_I = 8.0
NZ_I = 72
NV_I = 64
NT_I = 48
NVT_I = 32


def generate_inner_table():
    import mpmath as mp

    with mp.workdps(DPS):
        zmax = mp.mpf(ZMAX_I)

        def lh(v, z):
            x = mp.sqrt(z)
            return mp.log(mp.besseli(v, x)) - v * mp.log(x / 2) + mp.loggamma(v + 1)

        tz = nodes(mp, NZ_I)
        rows = [dct(mp, [lh(v, (t + 1) / 2 * zmax) for t in tz])
                for v in [(s + 1) / 2 * mp.mpf(VMAX) for s in nodes(mp, NV_I)]]
        return to_f64(param_fit(mp, rows))


def generate_tail_table():
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS_I)

        def lt(v, t):
            x = xs / t
            return mp.log(mp.besseli(v, x)) - x + mp.log(mp.sqrt(2 * mp.pi * x))

        tt = nodes(mp, NT_I)
        rows = [dct(mp, [lt(v, (t + 1) / 2) for t in tt])
                for v in [(s + 1) / 2 * mp.mpf(VMAX) for s in nodes(mp, NVT_I)]]
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    write_table_module(
        table_path(out_dir, __file__, "besseli_table.py"),
        "chebax._src.recipes.besseli_gen",
        "T[k, j] = j-th v-coefficient (v in [0, VMAX]) of the k-th argument "
        "coefficient (z = x^2 in [0, ZMAX] for TABLE_IN; t = XS/x in [0, 1] "
        "for TABLE_TAIL)",
        {"nz": NZ_I, "nv": NV_I, "nt_tail": NT_I, "nv_tail": NVT_I,
         "zmax": ZMAX_I, "xs": XS_I, "vmax": VMAX},
        {"ZMAX": ZMAX_I, "XS": XS_I, "VMAX": VMAX},
        {"TABLE_IN": generate_inner_table(), "TABLE_TAIL": generate_tail_table()})


if __name__ == "__main__":
    main()
