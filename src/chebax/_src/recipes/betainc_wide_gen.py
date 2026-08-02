"""Generator for the wide betainc panels. Build-time only; mpmath inside.

Widens betainc's (a, b) box from [0.1, 10]^2 to [0.1, 100]^2 with three
additional tensor panels of the same kernel L = ln 2F1(a+b, 1; a+1; x)
(x in [0, 1/2], reflection beyond), raw axes per panel, split at 10:

    LOxHI  a in [0.1, 10] x b in [10, 100]   (x deg <= 62, a 52, b 49)
    HIxLO  a in [10, 100] x b in [0.1, 10]   (x deg <= 19, a 49, b 15)
    HIxHI  a in [10, 100] x b in [10, 100]   (x deg <= 62, a 50, b 49)

Degrees measured in experiments/11 (2026-08-01): the a-axis carries the
pole structure, so per-panel raw a (deg ~50) beats a global log
transform (deg 95) and far beats global raw (175); b is mild and log-b
is WORSE; x grows 19 -> 62 where the CDF transition x* = a/(a+b) is
interior and sharp. The existing [0.1, 10]^2 tensor stays untouched as
the fourth (LOxLO) panel, so in-box users pay nothing: the runtime
dispatches with one lax.switch on the scalar (a, b). Each panel keeps
its own node counts; branch shapes never meet.

Storage is dense f64 (~600k coefficients, ~11 MB module). Tucker
compression was considered and DECLINED for checked-in tables: the
factorization needs an SVD, float SVDs are BLAS-build-dependent, and
the bit-for-bit regeneration contract is deterministic today precisely
because every fit stage is pure mpmath; an mpmath SVD of these
unfoldings is computationally infeasible. Revisit only with a
deterministic factorization.

Regenerate with:

    python -m chebax._src.recipes.betainc_wide_gen
"""

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, table_path,
                                             write_table_module)

ALO, ASPLIT, AHI = 0.1, 10.0, 100.0
XSPLIT_W = 0.5
# (name, a-range, b-range, x-nodes, a-nodes, b-nodes); ~20% margin on
# the measured worst tails
PANELS = (
    ("TENSOR_LOHI", (0.1, 10.0), (10.0, 100.0), 76, 64, 60),
    ("TENSOR_HILO", (10.0, 100.0), (0.1, 10.0), 24, 60, 20),
    ("TENSOR_HIHI", (10.0, 100.0), (10.0, 100.0), 76, 62, 60),
)


def _l(mp, a, b, x):
    return mp.log(mp.hyp2f1(a + b, 1, a + 1, x))


def panel_rows(mp, arange, brange, nx, na, nb, a_index):
    """The (a_index)-th a-node's (b, x)-coefficient sheet, mp exact.

    Split out so the CI regeneration check can rebuild single sheets
    bit-for-bit without paying for the full tensor."""
    alo, ahi = mp.mpf(arange[0]), mp.mpf(arange[1])
    blo, bhi = mp.mpf(brange[0]), mp.mpf(brange[1])
    ta = nodes(mp, na)[a_index]
    a = alo + (ta + 1) / 2 * (ahi - alo)
    tx = nodes(mp, nx)
    rows = []
    for tb in nodes(mp, nb):
        b = blo + (tb + 1) / 2 * (bhi - blo)
        rows.append(dct(mp, [_l(mp, a, b, (t + 1) / 2 * mp.mpf(XSPLIT_W))
                             for t in tx]))
    return rows


def generate_panel(mp, arange, brange, nx, na, nb):
    """Full (x, a, b) tensor: T[k, i, j] = (a-node i, b-node j) fit of the
    k-th x coefficient, then the (a, b) directions fitted per x-slot."""
    import numpy as np

    sheets = [panel_rows(mp, arange, brange, nx, na, nb, i) for i in range(na)]
    # sheets[i][j][k]: a-node i, b-node j, x-coefficient k. Fit b per
    # (i, k), then a per (k, j'): both stages stay in mp.
    out = np.empty((nx, na, nb))
    bfit = [[dct(mp, [sheets[i][j][k] for j in range(nb)]) for k in range(nx)]
            for i in range(na)]
    for k in range(nx):
        for j in range(nb):
            col = dct(mp, [bfit[i][k][j] for i in range(na)])
            for i in range(na):
                out[k, i, j] = float(col[i])
    return out


def main(out_dir=None):
    import mpmath as mp

    tables = {}
    with mp.workdps(DPS):
        for name, ar, br, nx, na, nb in PANELS:
            tables[name] = generate_panel(mp, ar, br, nx, na, nb)
    write_table_module(
        table_path(out_dir, __file__, "betainc_wide_table.py"),
        "chebax._src.recipes.betainc_wide_gen",
        "T[k, i, j] = (i-th a-coefficient, j-th b-coefficient) of the k-th "
        "x coefficient; TENSOR_LOHI has a in [ALO, ASPLIT] x b in "
        "[ASPLIT, AHI], TENSOR_HILO the transpose ranges, TENSOR_HIHI both "
        "high; x in [0, XSPLIT] for all; the [0.1, 10]^2 panel is the "
        "original betainc_table.TENSOR",
        {f"n_{n.lower()}": (nx, na, nb) for n, _, _, nx, na, nb in PANELS},
        {"ALO": ALO, "ASPLIT": ASPLIT, "AHI": AHI, "XSPLIT": XSPLIT_W},
        tables)


if __name__ == "__main__":
    main()
