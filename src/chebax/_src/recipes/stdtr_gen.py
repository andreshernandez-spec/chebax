"""Generator for the stdtr slice tables. Build-time only; mpmath inside.

stdtr and stdtrit evaluate betainc only at (1/2, nu/2) and (nu/2, 1/2), so
two 2-D slices of the kernel L = ln 2F1(a+b, 1; a+1; x), orientation A =
L(1/2, b, x) and B = L(b, 1/2, x), extend Student-t to b in [0.1, 100] at
~7.6k coefficients where the full tensor widening needs ~650k
(experiments/11). Each has two raw-b panels split at 10 (worst tails,
experiments/03: A x 18/53, b 21/35; B x 18/18, a 53/45). Every table is
fitted on NX x-nodes; orientation B keeps NX_B x-coefficients, since its
dropped tail is under 1e-15 in the value and the x-derivative over both
panels (experiments/24: value 19, d/dt 22, d/db 18; orientation A's hi
panel needs all 68). The runtime's panel select works elementwise, so the
two tables of an orientation share a length.
Regenerate with:

    python -m chebax._src.recipes.stdtr_gen      # ~4 min
"""

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    param_fit,
    to_f64,
    table_path,
    write_table_module,
)

XSPLIT_T = 0.5
BLO, BSPLIT, BHI = 0.1, 10.0, 100.0
NX = 68  # x-nodes, shared (worst measured tail 53, orientation A hi)
NX_B = 22  # x-coefficients kept for orientation B (measured tail 18)
NB_A_LO = 28  # b-nodes per panel (worst measured: 21)
NB_A_HI = 44  # (35)
NB_B_LO = 66  # (53)
NB_B_HI = 56  # (45)


def _l(mp, a, b, x):
    return mp.log(mp.hyp2f1(a + b, 1, a + 1, x))


def _table(mp, first_half, blo, bhi, nb, nx_keep=NX):
    h = mp.mpf(1) / 2
    tx = nodes(mp, NX)
    lo, hi = mp.mpf(blo), mp.mpf(bhi)
    rows = []
    for tb in nodes(mp, nb):
        b = lo + (tb + 1) / 2 * (hi - lo)
        a1, b1 = (h, b) if first_half else (b, h)
        rows.append(
            dct(mp, [_l(mp, a1, b1, (t + 1) / 2 * mp.mpf(XSPLIT_T)) for t in tx])
        )
    return to_f64(param_fit(mp, rows))[:nx_keep]


def main(out_dir=None):
    import mpmath as mp

    with mp.workdps(DPS):
        a_lo = _table(mp, True, BLO, BSPLIT, NB_A_LO)
        a_hi = _table(mp, True, BSPLIT, BHI, NB_A_HI)
        b_lo = _table(mp, False, BLO, BSPLIT, NB_B_LO, NX_B)
        b_hi = _table(mp, False, BSPLIT, BHI, NB_B_HI, NX_B)
    write_table_module(
        table_path(out_dir, __file__, "stdtr_table.py"),
        "chebax._src.recipes.stdtr_gen",
        "T[k, j] = j-th b-coefficient of the k-th x coefficient; TABLE_A_* "
        "hold L(1/2, b, x), TABLE_B_* hold L(b, 1/2, x); *_LO has b in "
        "[BLO, BSPLIT], *_HI b in [BSPLIT, BHI]; x in [0, XSPLIT] for all",
        {
            "nx": NX,
            "nx_b": NX_B,
            "nb_a_lo": NB_A_LO,
            "nb_a_hi": NB_A_HI,
            "nb_b_lo": NB_B_LO,
            "nb_b_hi": NB_B_HI,
            "blo": BLO,
            "bsplit": BSPLIT,
            "bhi": BHI,
            "xsplit": XSPLIT_T,
        },
        {"BLO": BLO, "BSPLIT": BSPLIT, "BHI": BHI, "XSPLIT": XSPLIT_T},
        {
            "TABLE_A_LO": a_lo,
            "TABLE_A_HI": a_hi,
            "TABLE_B_LO": b_lo,
            "TABLE_B_HI": b_hi,
        },
    )


if __name__ == "__main__":
    main()
