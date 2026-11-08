"""Generator for Bessel I's factored values and order-derivative log tables.

The direct inner value is 0F1 = 1 + z/(4(v+1)) (1 + z R(v,z)), so its
constant and linear terms stay exact near zero. R is entire in z on [0,64];
16 terms (10 in float32) suffice, measured by experiment 38.

The direct tail is A(v,t) = sqrt(2 pi x) exp(-x) I_v(x), t = 8/x.
Fit B = (A(v,t) - A(v,1))/(1-t), with log A(v,1) fitted separately in v.
Anchoring at t=1 avoids cancellation when A(v,1) is small at large v.
Experiment 39 checks 32 terms (12 in float32), from 64 t-nodes and 40
v-nodes. Unscaled values retain the log tail, truncated to the same lengths
for values; the full log tables below retain the stable order derivative.

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
NN_TAIL_VALUE = 64
NT_TAIL_VALUE = 32
NV_TAIL_VALUE = 40


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


def generate_value_tail_tables():
    import mpmath as mp

    with mp.workdps(DPS):
        rows, log_edges = [], []
        xs = mp.mpf(XS_I)
        for s in nodes(mp, NV_TAIL_VALUE):
            v = (s + 1) * mp.mpf(VMAX) / 2
            edge = mp.sqrt(2 * mp.pi * xs) * mp.exp(-xs) * mp.besseli(v, xs)
            log_edges.append(mp.log(edge))
            values = []
            for u in nodes(mp, NN_TAIL_VALUE):
                t = (u + 1) / 2
                x = xs / t
                a = mp.sqrt(2 * mp.pi * x) * mp.exp(-x) * mp.besseli(v, x)
                values.append((a - edge) / (1 - t))
            rows.append(dct(mp, values)[:NT_TAIL_VALUE])
        return to_f64(param_fit(mp, rows)), to_f64([dct(mp, log_edges)])


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
    value_tail, value_edge_log = generate_value_tail_tables()
    write_table_module(
        table_path(out_dir, __file__, "besseli_table.py"),
        "chebax._src.recipes.besseli_gen",
        "T[k, j] = j-th v-coefficient (v in [0, VMAX]) of the k-th argument "
        "coefficient (z = x^2 in [0, ZMAX] for TABLE_IN; t = XS/x in [0, 1] "
        "for TABLE_TAIL). VALUE_IN holds R(v,z); VALUE_TAIL holds B(v,t); "
        "VALUE_EDGE_LOG holds log A(v,1), all with the same v map",
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
            "nn_tail_value": NN_TAIL_VALUE,
            "nt_tail_value": NT_TAIL_VALUE,
            "nv_tail_value": NV_TAIL_VALUE,
        },
        {"ZMAX": ZMAX_I, "XS": XS_I, "VMAX": VMAX},
        {
            "TABLE_IN": generate_inner_table(),
            "TABLE_TAIL": generate_tail_table(),
            "VALUE_IN": generate_value_table(),
            "VALUE_TAIL": value_tail,
            "VALUE_EDGE_LOG": value_edge_log,
        },
    )


if __name__ == "__main__":
    main()
