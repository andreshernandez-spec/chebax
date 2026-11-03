"""Generator for the bessely tables. Build-time only; mpmath inside.

Y_v is singular at 0 with a v-dependent exponent, changes sign, and its
connection formula cancels near integer v, so (Temme's route) the tables
hold the pair Y_mu, Y_(mu+1) for mu = v - floor(v) in [0, 1), and the
runtime lifts to v by the upward recurrence, which amplifies the dominant
solution Y and is stable (lift error <= 4e-16 at the x = 5 seam); floor(v)
is fixed at instantiation. floor, not round: a negative mu leaves a
(x/2)^(-2|mu|) piece that leaked 1e-11 of noise. Inner, u = ln x on
[ln 1e-6, ln 5]: T_a = (x/2)^mu Y_mu and T_b = (x/2)^(mu+1) Y_(mu+1), bounded
and smooth. Mid, x in [5, 30]: Y_v directly in x. Tail: besselj's P and Q.
Both stages stay in mpmath; degrees reproduce via experiments/03.
Regenerate with:

    python -m chebax._src.recipes.bessely_gen
"""

import math
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

XMIN_Y = 1e-6
X0_Y = 5.0  # NOT 8: the upward lift is neutral-to-cancelling while k < x, and
# at x=8 the accumulated cancellation cost 1.4e-10; at x=5 the
# measured lift error is <= 3.9e-16 for every worst order.
X1_Y = 30.0
NU_Y = 96  # u-nodes (worst measured degree 82, T_b near mu=1)
NM_Y = 48  # mu-nodes on [0, 1] (worst measured 33; no panels needed)
NX_MID_Y = 68  # x-degree 60 measured at v=9.97 on [5, 30]
NV_MID_Y = 48


def generate_inner_tables():
    import mpmath as mp

    with mp.workdps(DPS):
        u0, u1 = mp.log(mp.mpf(f"{XMIN_Y:.0e}")), mp.log(mp.mpf(X0_Y))
        tu = nodes(mp, NU_Y)
        munodes = [(t + 1) / 2 for t in nodes(mp, NM_Y)]
        rows_a, rows_b = [], []
        for mu in munodes:
            sa, sb = [], []
            for t in tu:
                x = mp.exp(u0 + (t + 1) / 2 * (u1 - u0))
                sa.append((x / 2) ** mu * mp.bessely(mu, x))
                sb.append((x / 2) ** (mu + 1) * mp.bessely(mu + 1, x))
            rows_a.append(dct(mp, sa))
            rows_b.append(dct(mp, sb))
        return to_f64(param_fit(mp, rows_a)), to_f64(param_fit(mp, rows_b))


def generate_mid_table():
    import mpmath as mp

    with mp.workdps(DPS):
        x0, x1 = mp.mpf(X0_Y), mp.mpf(X1_Y)
        tx = nodes(mp, NX_MID_Y)
        rows = [
            dct(mp, [mp.bessely(v, (t + 1) / 2 * (x1 - x0) + x0) for t in tx])
            for v in [(s + 1) / 2 * mp.mpf(VMAX) for s in nodes(mp, NV_MID_Y)]
        ]
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    ta, tb = generate_inner_tables()
    mid = generate_mid_table()
    write_table_module(
        table_path(out_dir, __file__, "bessely_table.py"),
        "chebax._src.recipes.bessely_gen",
        "T[k, j] = j-th parameter coefficient of the k-th argument "
        "coefficient; TABLE_A/TABLE_B have mu in [0, 1], argument u = ln x "
        "in [U0, U1]; TABLE_MID has v in [0, VMAX], argument x in [X0, X1]",
        {
            "nu": NU_Y,
            "nm": NM_Y,
            "nx_mid": NX_MID_Y,
            "nv_mid": NV_MID_Y,
            "xmin": XMIN_Y,
            "x0": X0_Y,
            "x1": X1_Y,
            "vmax": VMAX,
        },
        {
            "XMIN": XMIN_Y,
            "X0": X0_Y,
            "X1": X1_Y,
            "VMAX": VMAX,
            "U0": math.log(XMIN_Y),
            "U1": math.log(X0_Y),
        },
        {"TABLE_A": ta, "TABLE_B": tb, "TABLE_MID": mid},
    )


if __name__ == "__main__":
    main()
