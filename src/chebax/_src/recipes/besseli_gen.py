"""Generator for Bessel I's factored values and order-derivative log tables.

The direct inner value is 0F1 = 1 + z/(4(v+1)) (1 + z R(v,z)), so its
constant and linear terms stay exact near zero. R is entire in z on [0,64];
16 terms (10 in float32) suffice, measured by experiment 38. The log tables
below retain the stable order derivative and the scaled tail.

Inner, z = x^2 in [0, 64]: Lh(v, z) = ln[Gamma(v+1) (x/2)^(-v) I_v(x)], the
log of the entire part 0F1(; v+1; z/4); the nearest complex zero of I_0, at
z = -5.78, sets degree ~64 in z, and one v-panel suffices (degree 52). Tail,
t = 8/x on (0, 1]: Lt(v, t) = ln[sqrt(2 pi x) e^(-x) I_v(x)], worst t-degree
39 at v = 0.5, where the recessive e^(-2x) part is largest. Both stages stay
in mpmath; degrees reproduce via experiments/03. Regenerate with:

    python -m chebax._src.recipes.besseli_gen
"""

import pathlib

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    param_fit,
    to_f64,
    table_path,
    write_table_module,
)
from chebax._src.recipes.besselj_gen import VMAX

ZMAX_I = 64.0
XS_I = 8.0
NZ_I = 72
NV_I = 64
NT_I = 48
NVT_I = 32
NN_VALUE = 24
NT_VALUE = 16


def generate_value_table():
    """R(v,z) = (4(v+1)(0F1(;v+1;z/4)-1)/z - 1)/z."""
    import mpmath as mp

    with mp.workdps(DPS):
        rows = []
        for s in nodes(mp, NV_I):
            v = (s + 1) * mp.mpf(VMAX) / 2
            values = []
            for t in nodes(mp, NN_VALUE):
                z = (t + 1) * mp.mpf(ZMAX_I) / 2
                values.append(((mp.hyp0f1(v + 1, z / 4) - 1) * 4 * (v + 1) / z - 1) / z)
            rows.append(dct(mp, values)[:NT_VALUE])
        return to_f64(param_fit(mp, rows))


def generate_inner_table():
    import mpmath as mp

    with mp.workdps(DPS):
        zmax = mp.mpf(ZMAX_I)

        def lh(v, z):
            x = mp.sqrt(z)
            return mp.log(mp.besseli(v, x)) - v * mp.log(x / 2) + mp.loggamma(v + 1)

        tz = nodes(mp, NZ_I)
        rows = [
            dct(mp, [lh(v, (t + 1) / 2 * zmax) for t in tz])
            for v in [(s + 1) / 2 * mp.mpf(VMAX) for s in nodes(mp, NV_I)]
        ]
        return to_f64(param_fit(mp, rows))


def generate_tail_table():
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS_I)

        def lt(v, t):
            x = xs / t
            return mp.log(mp.besseli(v, x)) - x + mp.log(mp.sqrt(2 * mp.pi * x))

        tt = nodes(mp, NT_I)
        rows = [
            dct(mp, [lt(v, (t + 1) / 2) for t in tt])
            for v in [(s + 1) / 2 * mp.mpf(VMAX) for s in nodes(mp, NVT_I)]
        ]
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    write_table_module(
        table_path(out_dir, __file__, "besseli_table.py"),
        "chebax._src.recipes.besseli_gen",
        "T[k, j] = j-th v-coefficient (v in [0, VMAX]) of the k-th argument "
        "coefficient (z = x^2 in [0, ZMAX] for TABLE_IN; t = XS/x in [0, 1] "
        "for TABLE_TAIL). VALUE_IN[k, j] holds R(v,z) in the same z/v coordinates",
        {
            "nz": NZ_I,
            "nv": NV_I,
            "nt_tail": NT_I,
            "nv_tail": NVT_I,
            "zmax": ZMAX_I,
            "xs": XS_I,
            "vmax": VMAX,
            "nn_value": NN_VALUE,
            "nt_value": NT_VALUE,
        },
        {"ZMAX": ZMAX_I, "XS": XS_I, "VMAX": VMAX},
        {
            "TABLE_IN": generate_inner_table(),
            "TABLE_TAIL": generate_tail_table(),
            "VALUE_IN": generate_value_table(),
        },
    )


if __name__ == "__main__":
    main()
