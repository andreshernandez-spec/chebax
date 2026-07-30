"""Generator for the bessely tables. Build-time only; mpmath inside.

Y_v combines every hard case: singular at x = 0 with a v-dependent exponent,
sign-changing (so no log-tabulation), and defined through the J_{+-v}
connection formula that cancels catastrophically near integer v (the R6/S3
structure). The classical resolution (Temme) adapted to tables:

- Reduce to small order: mu = v - floor(v) in [0, 1). The floor (not round)
  decomposition matters: the (x/2)^mu scaling below only removes the
  singular branch when mu >= 0 — for negative mu the divergent piece of
  Y_mu goes like (x/2)^{+mu}, T_a blows up like (x/2)^{-2|mu|} (~1e3 at
  mu=-0.3), and that magnitude leaks ~1e-11 of noise into every
  mu-reconstruction (measured; the first cut of this generator had it).
  On [0, 1) the pair Y_mu, Y_{mu+1} is tabulated DIRECTLY (mpmath at 40 dps
  absorbs the connection-formula cancellation during generation; the tables
  never see a pole), then the runtime lifts to v with the upward recurrence
  Y_{k+1} = (2k/x) Y_k - Y_{k-1}, which amplifies the dominant solution
  (= Y itself) and is therefore stable; measured lift error from f64 base
  pairs is <= 4e-16 at the x = 5 seam. The step count floor(v) is fixed at
  instantiation, so the unrolled loop stays branchless.

- Scale out the singular growth: the tables hold
      T_a(mu, u) = (x/2)^mu     Y_mu(x)
      T_b(mu, u) = (x/2)^(mu+1) Y_{mu+1}(x)
  in u = ln x on [ln 1e-6, ln 5]: bounded O(10) quantities, smooth in both
  variables (Y_0's log divergence becomes a linear-in-u term).

- Mid, x in [5, 30]: Y_v fitted directly in x (same reasoning as besselj's
  mid region). Tail, x > 30: NO new tables — besselj's P, Q give
  Y = sqrt(2/(pi x)) (sin(x) A - cos(x) B) with the same A, B constants.

Both fit stages stay in mpmath (the M1 floor rule). Measured degrees
reproduce via experiments/03_degree_measurement.py. Regenerate with:

    python -m chebax._src.recipes.bessely_gen
"""

import math
import pathlib

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, table_path, write_table_module)
from chebax._src.recipes.besselj_gen import VMAX

XMIN_Y = 1e-6
X0_Y = 5.0   # NOT 8: the upward lift is neutral-to-cancelling while k < x, and
             # at x=8 the accumulated cancellation cost 1.4e-10; at x=5 the
             # measured lift error is <= 3.9e-16 for every worst order.
X1_Y = 30.0
NU_Y = 96    # u-nodes (worst measured degree 82, T_b near mu=1)
NM_Y = 48    # mu-nodes on [0, 1] (worst measured 33; no panels needed)
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
        rows = [dct(mp, [mp.bessely(v, (t + 1) / 2 * (x1 - x0) + x0) for t in tx])
                for v in [(s + 1) / 2 * mp.mpf(VMAX) for s in nodes(mp, NV_MID_Y)]]
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
        {"nu": NU_Y, "nm": NM_Y, "nx_mid": NX_MID_Y, "nv_mid": NV_MID_Y,
         "xmin": XMIN_Y, "x0": X0_Y, "x1": X1_Y, "vmax": VMAX},
        {"XMIN": XMIN_Y, "X0": X0_Y, "X1": X1_Y, "VMAX": VMAX,
         "U0": math.log(XMIN_Y), "U1": math.log(X0_Y)},
        {"TABLE_A": ta, "TABLE_B": tb, "TABLE_MID": mid})


if __name__ == "__main__":
    main()
