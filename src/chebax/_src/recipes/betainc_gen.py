"""Generator for the betainc tensor table. Build-time only; mpmath inside.

The regularized incomplete beta factors as (DLMF 8.17.7)

    I_x(a, b) = x^a (1-x)^b / (a B(a,b)) * F(a, b, x),
    F = 2F1(a + b, 1; a + 1; x),

with the reflection I_x(a,b) = 1 - I_{1-x}(b,a) covering x > 1/2, so the
table only spans x in [0, 1/2], where F's singularity at x = 1 stays far
away. F is positive (all series terms positive), and tabulating ln F
flattens the tensor: raw F spans 833x over the (a,b) corners while ln F
stays in [0.1, 6.7], so f64 coefficient noise cannot poison small values
(the besselk lesson). This is the library's first TWO-parameter table: each
x-coefficient is a 2-D Chebyshev series over (a, b) in [0.1, 10]^2.
Measured degrees: x <= 19, a <= 58 (the costly axis, near small a),
b <= 20 (reproduce via experiments/03_degree_measurement.py).

Three fit stages, all in mpmath (the M1 floor rule): DCT in x per (a,b)
node pair, then per x-coefficient a DCT along b, then along a. The runtime
contracts in the reverse order: Clenshaw in a, then b, then x.

Regenerate with:  python -m chebax._src.recipes.betainc_gen  (~5 min)
"""

import pathlib

import numpy as np

from chebax._src.recipes._gen_common import (DPS, dct, nodes,
                                             write_table_module)

ALO, AHI = 0.1, 10.0
XSPLIT = 0.5
NX_BETA = 24
NA_BETA = 72
NB_BETA = 28


def generate_table():
    import mpmath as mp

    with mp.workdps(DPS):
        alo, ahi = mp.mpf(f"{ALO}"), mp.mpf(f"{AHI}")
        anodes = [alo + (t + 1) / 2 * (ahi - alo) for t in nodes(mp, NA_BETA)]
        bnodes = [alo + (t + 1) / 2 * (ahi - alo) for t in nodes(mp, NB_BETA)]
        xnodes = [(t + 1) / 4 for t in nodes(mp, NX_BETA)]

        # stage 1: x-coefficients of ln F at every (a, b) node pair
        rows = [[dct(mp, [mp.log(mp.hyp2f1(a + b, 1, a + 1, x)) for x in xnodes])
                 for b in bnodes] for a in anodes]

        # stage 2 (along b) and stage 3 (along a), per x-coefficient k
        tensor = []
        for k in range(NX_BETA):
            g1 = [dct(mp, [rows[i][j][k] for j in range(NB_BETA)])
                  for i in range(NA_BETA)]                      # NA x NB
            tk = [dct(mp, [g1[i][n] for i in range(NA_BETA)])
                  for n in range(NB_BETA)]                      # NB x NA
            tensor.append([[float(tk[n][m]) for n in range(NB_BETA)]
                           for m in range(NA_BETA)])            # NA x NB
    return np.array(tensor)  # (NX, NA, NB)


def main():
    write_table_module(
        pathlib.Path(__file__).with_name("betainc_table.py"),
        "chebax._src.recipes.betainc_gen",
        "TENSOR[k, m, n] = (m-th a-coefficient, n-th b-coefficient) of the "
        "k-th x-coefficient of ln 2F1(a+b, 1; a+1; x); a, b in [ALO, AHI], "
        "x in [0, XSPLIT]",
        {"nx": NX_BETA, "na": NA_BETA, "nb": NB_BETA,
         "ab_domain": (ALO, AHI), "xsplit": XSPLIT},
        {"ALO": ALO, "AHI": AHI, "XSPLIT": XSPLIT},
        {"TENSOR": generate_table()})


if __name__ == "__main__":
    main()
