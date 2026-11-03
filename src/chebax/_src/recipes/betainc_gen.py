"""Generator for the betainc tensor table. Build-time only; mpmath inside.

I_x(a, b) = x^a (1-x)^b / (a B(a, b)) F, F = 2F1(a+b, 1; a+1; x) (DLMF
8.17.7), with the reflection I_x(a, b) = 1 - I_(1-x)(b, a) above x = 1/2, so
the table spans x in [0, 1/2]. It holds ln F, which stays in [0.1, 6.7]
where F spans 833x, as a 2-D series over (a, b) in [0.1, 10]^2 per
x-coefficient: degrees x <= 19, a <= 58 (near small a), b <= 20
(experiments/03). Three DCT stages in mpmath, x then b then a; the runtime
contracts a, b, then x. Regenerate with:

    python -m chebax._src.recipes.betainc_gen      # ~5 min
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
        rows = [
            [
                dct(mp, [mp.log(mp.hyp2f1(a + b, 1, a + 1, x)) for x in xnodes])
                for b in bnodes
            ]
            for a in anodes
        ]

        # stage 2 (along b) and stage 3 (along a), per x-coefficient k
        tensor = []
        for k in range(NX_BETA):
            g1 = [
                dct(mp, [rows[i][j][k] for j in range(NB_BETA)]) for i in range(NA_BETA)
            ]  # NA x NB
            tk = [
                dct(mp, [g1[i][n] for i in range(NA_BETA)]) for n in range(NB_BETA)
            ]  # NB x NA
            tensor.append(
                [[float(tk[n][m]) for n in range(NB_BETA)] for m in range(NA_BETA)]
            )  # NA x NB
    return np.array(tensor)  # (NX, NA, NB)


def main(out_dir=None):
    write_table_module(
        table_path(out_dir, __file__, "betainc_table.py"),
        "chebax._src.recipes.betainc_gen",
        "TENSOR[k, m, n] = (m-th a-coefficient, n-th b-coefficient) of the "
        "k-th x-coefficient of ln 2F1(a+b, 1; a+1; x); a, b in [ALO, AHI], "
        "x in [0, XSPLIT]",
        {
            "nx": NX_BETA,
            "na": NA_BETA,
            "nb": NB_BETA,
            "ab_domain": (ALO, AHI),
            "xsplit": XSPLIT,
        },
        {"ALO": ALO, "AHI": AHI, "XSPLIT": XSPLIT},
        {"TENSOR": generate_table()},
    )


if __name__ == "__main__":
    main()
