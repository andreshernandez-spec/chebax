"""Generators for the besselj tables. Build-time only; mpmath imported inside.

Inner table (besselj_table.py): the experiments/02 construction. Fit the
degree-24 Chebyshev coefficients of g_v(z) = 0F1(; v+1; -z/4) on z in [0, 64]
at 64 Chebyshev nodes in v over [0, 10], then fit each coefficient as a
Chebyshev series in v.

M3 tables (besselj_table_ext.py), two regions:

- Mid, x in [8, 30]: J_v(x) fitted directly in x. There is no x = 0 branch
  point inside the interval, so the factorization is unnecessary here. This is
  NOT the unfactored dead end of ../bessel/PROJECT.md 2.5 (stated explicitly
  per the house rule): that verdict is about intervals containing 0. Direct
  fitting is also what keeps the table sup-accurate for large v, where g_v
  spans six orders of magnitude by x = 30 and its f64 Clenshaw floor would
  destroy envelope-relative accuracy.

- Outer, x >= 30: the exact modulus functions, t = s^2 with s = 30/x,

      P(v,x) = sqrt(pi x/2) (J cos w + Y sin w),   w = x - (v/2 + 1/4) pi
      Q(v,x) = sqrt(pi x/2) (Y cos w - J sin w)

  so J = sqrt(2/(pi x)) (P cos w - Q sin w) exactly, with P and Q/s fitted in
  t on [0, 1]. Both are numerically analytic there: the asymptotic
  non-analyticity at t = 0 is O(e^{-2x}) <= e^{-60}. The runtime never forms
  w (that subtraction would cost eps*x of phase); it expands the cosine by
  the angle addition identity and lets sincos(x) do its own reduction.

Every stage of every fit stays in mpmath and rounds to float64 only at the
end; rounding between stages would put eps-level sample noise under the
second fit (the M1 floor). Regenerate with:

    python -m chebax._src.recipes.besselj_gen

The output must reproduce the checked-in tables bit for bit (tested).
"""

import numpy as np

# inner table
NZ = 25      # z-degree 24, covers the worst case (v=0) with margin
NV = 64      # v-nodes, shared by all tables
ZMAX = 64.0  # z = x^2, x in [0, 8]
VMAX = 10.0
DPS = 40

# M3 tables
MID_X0 = 8.0
MID_X1 = 30.0
NX_MID = 42  # x-degree 41; measured tail at degree 40 is ~1e-19
XS = 30.0    # outer switch point; also the seam with the mid region
NT_OUT = 12  # t-degree 11; measured tail is below 1e-16 from degree ~8


def _nodes(mp, n):
    return [mp.cos(mp.pi * (2 * i + 1) / (2 * n)) for i in range(n)]


def _dct(mp, samples):
    n = len(samples)
    c = []
    for j in range(n):
        s = mp.fsum(samples[i] * mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n))
        c.append(2 * s / n)
    c[0] /= 2
    return c


def _to_f64(rows):
    return np.array([[float(x) for x in row] for row in rows])


def _vnodes(mp):
    return [(t + 1) / 2 * mp.mpf(VMAX) for t in _nodes(mp, NV)]


def _nu_fit(mp, rows):
    """Second stage: fit each coefficient across the v-nodes, still in mp."""
    nc = len(rows[0])
    return [_dct(mp, [rows[i][k] for i in range(len(rows))]) for k in range(nc)]


def generate_table():
    """Inner table: coefficients of g_v on z in [0, ZMAX], as series in v."""
    import mpmath as mp

    with mp.workdps(DPS):
        zmax = mp.mpf(ZMAX)
        tz = _nodes(mp, NZ)
        rows = [_dct(mp, [mp.hyp0f1(v + 1, -((t + 1) / 2 * zmax) / 4) for t in tz])
                for v in _vnodes(mp)]
        table = _nu_fit(mp, rows)
    return _to_f64(table)


def generate_mid_table():
    """Mid table: coefficients of J_v on x in [MID_X0, MID_X1], as series in v."""
    import mpmath as mp

    with mp.workdps(DPS):
        x0, x1 = mp.mpf(MID_X0), mp.mpf(MID_X1)
        tx = _nodes(mp, NX_MID)
        rows = [_dct(mp, [mp.besselj(v, (t + 1) / 2 * (x1 - x0) + x0) for t in tx])
                for v in _vnodes(mp)]
        table = _nu_fit(mp, rows)
    return _to_f64(table)


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
            for t in _nodes(mp, NT_OUT):
                s = mp.sqrt((t + 1) / 2)
                P, Q = _pq(mp, v, xs / s)
                Ps.append(P)
                Qs.append(Q / s)
            prows.append(_dct(mp, Ps))
            qrows.append(_dct(mp, Qs))
        ptab = _nu_fit(mp, prows)
        qtab = _nu_fit(mp, qrows)
    return _to_f64(ptab), _to_f64(qtab)


def _format_array(name, arr):
    lines = [f"{name} = np.array(["]
    for row in arr:
        lines.append("    [" + ", ".join(repr(x) for x in row) + "],")
    lines.append("])")
    return lines


def main():
    import pathlib

    import mpmath

    here = pathlib.Path(__file__).parent

    table = generate_table()
    lines = [
        '"""Baked besselj nu-table. Generated by chebax._src.recipes.besselj_gen;',
        'do not edit. Regenerate with:  python -m chebax._src.recipes.besselj_gen',
        '"""',
        "",
        "import numpy as np",
        "",
        f"META = {{'nz': {NZ}, 'nv': {NV}, 'zmax': {ZMAX!r}, 'vmax': {VMAX!r}, "
        f"'dps': {DPS}, 'mpmath': {mpmath.__version__!r}}}",
        f"ZMAX = {ZMAX!r}",
        f"VMAX = {VMAX!r}",
        "",
    ] + _format_array("TABLE", table)
    (here / "besselj_table.py").write_text("\n".join(lines) + "\n")
    print(f"wrote besselj_table.py ({table.shape[0]}x{table.shape[1]})")

    mid = generate_mid_table()
    ptab, qtab = generate_outer_tables()
    lines = [
        '"""Baked besselj mid/outer tables (M3). Generated by',
        'chebax._src.recipes.besselj_gen; do not edit. Regenerate with:',
        'python -m chebax._src.recipes.besselj_gen',
        '"""',
        "",
        "import numpy as np",
        "",
        f"META = {{'nx_mid': {NX_MID}, 'nt_out': {NT_OUT}, 'nv': {NV}, "
        f"'mid': ({MID_X0!r}, {MID_X1!r}), 'xs': {XS!r}, 'vmax': {VMAX!r}, "
        f"'dps': {DPS}, 'mpmath': {mpmath.__version__!r}}}",
        f"MID_X0 = {MID_X0!r}",
        f"MID_X1 = {MID_X1!r}",
        f"XS = {XS!r}",
        "",
    ] + _format_array("TABLE_MID", mid) + [""] \
      + _format_array("TABLE_P", ptab) + [""] \
      + _format_array("TABLE_QS", qtab)
    (here / "besselj_table_ext.py").write_text("\n".join(lines) + "\n")
    print(f"wrote besselj_table_ext.py (mid {mid.shape[0]}x{mid.shape[1]}, "
          f"P {ptab.shape[0]}x{ptab.shape[1]}, Qs {qtab.shape[0]}x{qtab.shape[1]})")


if __name__ == "__main__":
    main()
