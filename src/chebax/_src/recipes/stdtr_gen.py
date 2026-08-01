"""Generator for the stdtr slice tables. Build-time only; mpmath inside.

stdtr/stdtrit evaluate betainc only at (1/2, nu/2) and (nu/2, 1/2), so a
pair of 2-D slices of the betainc kernel L = ln 2F1(a+b, 1; a+1; x)
extends the Student-t range far beyond the 3-D tensor's [0.1, 10] box at
a thousandth of the widening cost (experiments/11: full widening needs
~650k coefficients and breaks the CI regen budget; these four tables
hold ~13k). Orientation A is L(1/2, b, x) (the direct branch and the
central-w branch), orientation B is L(b, 1/2, x) (the reflection and the
tail branch); each spans b in [0.1, 100] in two raw-b panels split at 10
(measured worst tails, experiments/03: A x 18/53, b 21/35; B x 18/18,
a 53/45; raw beats log on the b-like axis, panels beat global transforms
on the a-like axis). All four tables share NX x-nodes so the runtime's
panel select works elementwise on the coefficient vectors.

Regenerate with:

    python -m chebax._src.recipes.stdtr_gen
"""

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, table_path,
                                             write_table_module)

XSPLIT_T = 0.5
BLO, BSPLIT, BHI = 0.1, 10.0, 100.0
NX = 68        # x-nodes, shared (worst measured tail 53, orientation A hi)
NB_A_LO = 28   # b-nodes per panel (worst measured: 21)
NB_A_HI = 44   # (35)
NB_B_LO = 66   # (53)
NB_B_HI = 56   # (45)


def _l(mp, a, b, x):
    return mp.log(mp.hyp2f1(a + b, 1, a + 1, x))


def _table(mp, first_half, blo, bhi, nb):
    h = mp.mpf(1) / 2
    tx = nodes(mp, NX)
    lo, hi = mp.mpf(blo), mp.mpf(bhi)
    rows = []
    for tb in nodes(mp, nb):
        b = lo + (tb + 1) / 2 * (hi - lo)
        a1, b1 = (h, b) if first_half else (b, h)
        rows.append(dct(mp, [_l(mp, a1, b1, (t + 1) / 2 * mp.mpf(XSPLIT_T))
                             for t in tx]))
    return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    import mpmath as mp

    with mp.workdps(DPS):
        a_lo = _table(mp, True, BLO, BSPLIT, NB_A_LO)
        a_hi = _table(mp, True, BSPLIT, BHI, NB_A_HI)
        b_lo = _table(mp, False, BLO, BSPLIT, NB_B_LO)
        b_hi = _table(mp, False, BSPLIT, BHI, NB_B_HI)
    write_table_module(
        table_path(out_dir, __file__, "stdtr_table.py"),
        "chebax._src.recipes.stdtr_gen",
        "T[k, j] = j-th b-coefficient of the k-th x coefficient; TABLE_A_* "
        "hold L(1/2, b, x), TABLE_B_* hold L(b, 1/2, x); *_LO has b in "
        "[BLO, BSPLIT], *_HI b in [BSPLIT, BHI]; x in [0, XSPLIT] for all",
        {"nx": NX, "nb_a_lo": NB_A_LO, "nb_a_hi": NB_A_HI,
         "nb_b_lo": NB_B_LO, "nb_b_hi": NB_B_HI,
         "blo": BLO, "bsplit": BSPLIT, "bhi": BHI, "xsplit": XSPLIT_T},
        {"BLO": BLO, "BSPLIT": BSPLIT, "BHI": BHI, "XSPLIT": XSPLIT_T},
        {"TABLE_A_LO": a_lo, "TABLE_A_HI": a_hi,
         "TABLE_B_LO": b_lo, "TABLE_B_HI": b_hi})


if __name__ == "__main__":
    main()
