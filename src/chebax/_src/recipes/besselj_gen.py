"""Generators for the besselj tables. Build-time only; mpmath inside.

Inner (besselj_table.py): the degree-24 coefficients of g_v(z) =
0F1(; v+1; -z/4) on z in [0, 64] at 64 nodes in v on [0, 10], each then
fitted in v (experiments/02). Mid (besselj_table_ext.py), x in [8, 30]: J_v
fitted directly in x, which has no branch point inside and keeps the table
sup-accurate at large v, where g_v spans six decades by x = 30. Outer,
x >= 30, t = (30/x)^2: the modulus functions P = sqrt(pi x/2) (J cos w +
Y sin w) and Q/s, Q = sqrt(pi x/2) (Y cos w - J sin w), w = x - (v/2 + 1/4)
pi, s = 30/x, so J = sqrt(2/(pi x)) (P cos w - Q sin w); the runtime never
forms w (eps x of phase) and expands the cosine instead. Every fit stage
stays in mpmath; degrees reproduce via experiments/03. Regenerate with:

    python -m chebax._src.recipes.besselj_gen
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

# inner table
NZ = 25  # z-degree 24, covers the worst case (v=0) with margin
NV = 64  # v-nodes, shared by all tables
ZMAX = 64.0  # z = x^2, x in [0, 8]
VMAX = 10.0

# M3 tables
MID_X0 = 8.0
MID_X1 = 30.0
NX_MID = 42  # x-degree 41; measured tail at degree 40 is ~1e-19
XS = 30.0  # outer switch point; also the seam with the mid region
NT_OUT = 12  # t-degree 11; measured tail is below 1e-16 from degree ~8


def _vnodes(mp):
    return [(t + 1) / 2 * mp.mpf(VMAX) for t in nodes(mp, NV)]


def generate_table():
    """Inner table: coefficients of g_v on z in [0, ZMAX], as series in v."""
    import mpmath as mp

    with mp.workdps(DPS):
        zmax = mp.mpf(ZMAX)
        tz = nodes(mp, NZ)
        rows = [
            dct(mp, [mp.hyp0f1(v + 1, -((t + 1) / 2 * zmax) / 4) for t in tz])
            for v in _vnodes(mp)
        ]
        table = param_fit(mp, rows)
    return to_f64(table)


def generate_mid_table():
    """Mid table: coefficients of J_v on x in [MID_X0, MID_X1], as series in v."""
    import mpmath as mp

    with mp.workdps(DPS):
        x0, x1 = mp.mpf(MID_X0), mp.mpf(MID_X1)
        tx = nodes(mp, NX_MID)
        rows = [
            dct(mp, [mp.besselj(v, (t + 1) / 2 * (x1 - x0) + x0) for t in tx])
            for v in _vnodes(mp)
        ]
        table = param_fit(mp, rows)
    return to_f64(table)


def _pq(mp, v, x):
    w = x - (v / 2 + mp.mpf(1) / 4) * mp.pi
    J, Y = mp.besselj(v, x), mp.bessely(v, x)
    r = mp.sqrt(mp.pi * x / 2)
    return r * (J * mp.cos(w) + Y * mp.sin(w)), r * (Y * mp.cos(w) - J * mp.sin(w))


def generate_outer_tables():
    """Outer tables: coefficients of P and Q/s on t in [0, 1], as series in v."""
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS)
        # sanity-lock the sign convention before fitting anything
        v0, x0 = mp.mpf("3.3"), mp.mpf(47)
        P0, Q0 = _pq(mp, v0, x0)
        w0 = x0 - (v0 / 2 + mp.mpf(1) / 4) * mp.pi
        rec = mp.sqrt(2 / (mp.pi * x0)) * (P0 * mp.cos(w0) - Q0 * mp.sin(w0))
        assert abs(rec - mp.besselj(v0, x0)) < mp.mpf(10) ** (-(DPS - 5))

        prows, qrows = [], []
        for v in _vnodes(mp):
            Ps, Qs = [], []
            for t in nodes(mp, NT_OUT):
                s = mp.sqrt((t + 1) / 2)
                P, Q = _pq(mp, v, xs / s)
                Ps.append(P)
                Qs.append(Q / s)
            prows.append(dct(mp, Ps))
            qrows.append(dct(mp, Qs))
        ptab = param_fit(mp, prows)
        qtab = param_fit(mp, qrows)
    return to_f64(ptab), to_f64(qtab)


def main(out_dir=None):
    write_table_module(
        table_path(out_dir, __file__, "besselj_table.py"),
        "chebax._src.recipes.besselj_gen",
        "TABLE[k, j] = j-th v-coefficient (v in [0, VMAX]) of the k-th "
        "z-coefficient (z in [0, ZMAX])",
        {"nz": NZ, "nv": NV, "zmax": ZMAX, "vmax": VMAX},
        {"ZMAX": ZMAX, "VMAX": VMAX},
        {"TABLE": generate_table()},
    )
    mid = generate_mid_table()
    ptab, qtab = generate_outer_tables()
    write_table_module(
        table_path(out_dir, __file__, "besselj_table_ext.py"),
        "chebax._src.recipes.besselj_gen",
        "T[k, j] = j-th v-coefficient (v in [0, 10]) of the k-th argument "
        "coefficient (x in [MID_X0, MID_X1] for TABLE_MID; t = (XS/x)^2 in "
        "[0, 1] for TABLE_P and TABLE_QS)",
        {
            "nx_mid": NX_MID,
            "nt_out": NT_OUT,
            "nv": NV,
            "mid": (MID_X0, MID_X1),
            "xs": XS,
            "vmax": VMAX,
        },
        {"MID_X0": MID_X0, "MID_X1": MID_X1, "XS": XS},
        {"TABLE_MID": mid, "TABLE_P": ptab, "TABLE_QS": qtab},
    )


if __name__ == "__main__":
    main()
