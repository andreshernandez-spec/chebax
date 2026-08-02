"""Generator for the hyp1f1 log-tables. Build-time only; mpmath inside.

Kummer's M(a, b, x) = 1F1(a; b; x) for (a, b) in [0.1, 10]^2,
x in [0, inf). Two regions split at x = XS = 30, both tabulating a
logarithm so the singular parts live in exactly-computed prefactors:

- Inner, x in [0, XS]:
      ln R(a, b, x),  where  M = 1 + (a/b) x R,  R(a, b, 0) = 1.
  R = sum_m (a+1)_m / ((b+1)_m (m+1)!) x^m has positive terms, its
  nearest x-zero sits at ~ -2(b+1)/(a+1), and its b-poles start at
  b = -1. Tabulating ln M directly fails on both counts: M's zero at
  x ~ -b/a hugs the origin when b << a (x-degree past 150 at the
  (10, 0.1) corner; ln R needs 112), and 1F1's denominator poles
  start at b = 0, only 0.1 from the box edge (raw b-degree 137-172).
  Runtime: ln M = log1p((a/b) x exp(ln R)), full relative accuracy
  (identity checked in mpmath to 1e-40, experiments/13).

- Tail, x >= XS, t = XS/x:
      T(a, b, t) = ln[ M(a, b, x) Gamma(a)/Gamma(b) e^-x x^(b-a) ]
  the log-remainder of the DLMF 13.7.1 asymptotic; T -> 0 as t -> 0
  (C-infinity endpoint, the besselk/gammainc tail shape) and
  T(a, a, t) = 0 identically (M(a, a, x) = e^x), a built-in sanity
  row. XS = 8 is not asymptotic for |b - a| ~ 10 (t-degree ~100);
  XS = 30 brings the worst t-degree to 33.
  Runtime: ln M = lgamma(b) - lgamma(a) + x + (a - b) ln x + T.

Axes (measured in experiments/13, ~20% node margin): the parameter
axes are log-transformed except the tail's b (raw 20 beats log 44
there); worst measured degrees inner (x 112, ln-a 26, ln-b 36), tail
(t 33, ln-a 26, b 20).

Regenerate with:

    python -m chebax._src.recipes.hyp1f1_gen
"""

from chebax._src.recipes._gen_common import (DPS, dct, nodes, table_path,
                                             write_table_module)

XS_K = 30.0
ALO, AHI = 0.1, 10.0
NX_IN = 136   # x-nodes (worst measured 112, the (10, 0.1) corner)
NA_IN = 32    # ln-a nodes (worst measured 26)
NB_IN = 44    # ln-b nodes (worst measured 36)
NT_TAIL = 40  # t-nodes (worst measured 33; near-diagonal strips report
              # up to 38 on |T| ~ 1e-13, covered too)
NA_TAIL = 32  # ln-a nodes (worst measured 26)
NB_TAIL = 25  # b-nodes, RAW axis (worst measured 20)


def _lnr(mp, a, b, x):
    if x == 0:
        return mp.mpf(0)
    return mp.log((mp.hyp1f1(a, b, x) - 1) * b / (a * x))


def _ltail(mp, a, b, t):
    x = mp.mpf(XS_K) / t
    return (mp.log(mp.hyp1f1(a, b, x)) + mp.loggamma(a) - mp.loggamma(b)
            - x + (b - a) * mp.log(x))


def _param(mp, tnode, lo, hi, log):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    if log:
        lo, hi = mp.log(lo), mp.log(hi)
    v = lo + (tnode + 1) / 2 * (hi - lo)
    return mp.exp(v) if log else v


def inner_rows(mp, a_index):
    """The (a_index)-th ln-a node's (b, x)-coefficient sheet, mp exact.

    Split out so the CI regeneration check can rebuild single sheets
    bit-for-bit without paying for the full tensor."""
    a = _param(mp, nodes(mp, NA_IN)[a_index], ALO, AHI, log=True)
    tx = nodes(mp, NX_IN)
    rows = []
    for tb in nodes(mp, NB_IN):
        b = _param(mp, tb, ALO, AHI, log=True)
        rows.append(dct(mp, [_lnr(mp, a, b, (t + 1) / 2 * mp.mpf(XS_K))
                             for t in tx]))
    return rows


def tail_rows(mp, a_index):
    """The (a_index)-th ln-a node's (b, t)-coefficient sheet, mp exact."""
    a = _param(mp, nodes(mp, NA_TAIL)[a_index], ALO, AHI, log=True)
    tt = nodes(mp, NT_TAIL)
    rows = []
    for tb in nodes(mp, NB_TAIL):
        b = _param(mp, tb, ALO, AHI, log=False)
        rows.append(dct(mp, [_ltail(mp, a, b, (t + 1) / 2) for t in tt]))
    return rows


def generate_table(mp, rows_fn, nx, na, nb):
    """Full (x, a, b) tensor: T[k, i, j] = (a-node i, b-node j) fit of the
    k-th argument coefficient, both parameter fits in mp."""
    import numpy as np

    sheets = [rows_fn(mp, i) for i in range(na)]
    # sheets[i][j][k]: a-node i, b-node j, argument-coefficient k. Fit b
    # per (i, k), then a per (k, j'): both stages stay in mp.
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

    with mp.workdps(DPS):
        inner = generate_table(mp, inner_rows, NX_IN, NA_IN, NB_IN)
        tail = generate_table(mp, tail_rows, NT_TAIL, NA_TAIL, NB_TAIL)
    write_table_module(
        table_path(out_dir, __file__, "hyp1f1_table.py"),
        "chebax._src.recipes.hyp1f1_gen",
        "T[k, i, j] = (i-th a-coefficient, j-th b-coefficient) of the k-th "
        "argument coefficient; TABLE_IN holds ln R (M = 1 + (a/b) x R) with "
        "argument x in [0, XS], TABLE_TAIL the asymptotic log-remainder T "
        "with argument t = XS/x in [0, 1]; a and b in [ALO, AHI] for both, "
        "the a-axes and TABLE_IN's b-axis mapped in ln(parameter), "
        "TABLE_TAIL's b-axis raw",
        {"nx_in": NX_IN, "na_in": NA_IN, "nb_in": NB_IN,
         "nt_tail": NT_TAIL, "na_tail": NA_TAIL, "nb_tail": NB_TAIL,
         "xs": XS_K, "alo": ALO, "ahi": AHI},
        {"XS": XS_K, "ALO": ALO, "AHI": AHI},
        {"TABLE_IN": inner, "TABLE_TAIL": tail})


if __name__ == "__main__":
    main()
