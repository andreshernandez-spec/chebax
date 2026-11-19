"""Generator for the per-element hyp1f1 table. Build-time only; mpmath inside.

hyp1f1_array takes a and b per element, so each element reads one cell of a
3-D table instead of contracting hyp1f1_table's (x, a, b) tensors per call.
Two regions, split at x = X_S, a power of two per base column (XS_COL):

    x <= X_S:  ln M = log1p((a/b) x R), table L = ln R, R = 2F2(a+1, 1; b+1, 2; x)
    x >  X_S:  ln M = x - (a - b) ln t + U, t = X_S/x, table U

R(a, b, 0) = 1 and every term of R is positive, so the inner form has no
cancellation. U holds lnGamma(b) - lnGamma(a) + (a - b) ln X_S and the
log-remainder of DLMF 13.7.1, so no lgamma per element. U is C-infinity but
not analytic at t = 0 (its series has k! coefficients). Each cell's constant
coefficient is kept as hi + lo (C0LO the low part) and added last: where b > a,
x and U cancel next to the seam (to ln M ~ 6 from ~30 at a = 0.1, b = 10), and
the rounding of U's sum cost ~5 eps there. ln M tabulated directly instead of
ln R took 1.6-2.9x the cells where b > a. XS_COL is the seam with the fewest
cells of both regions per column; it changes the total by under 5%.

Cells are boxes in (ln a, ln b, s), s = x or t. Each of NB0 x NB0 base columns
on [ln ALO, ln AHI]^2 starts as one box over [0, X_S] (inner) or [0, 1] (tail)
and is halved, along the direction holding most of its coefficients outside
the total-degree set i + j + k <= DMAX, until those sum to at most TOL w, w the
smallest over the box's nodes of what the test's metric allows: max(1, |ln
M|), times (1 + q)/q, q = M - 1, capped at CAP for ln R (d ln M = q/(1 + q)
dL). A uniform grid does not work: at small a and large b the b-direction
needs cells 4x narrower near x ~ b than elsewhere, and about half the cells
are in the columns with b > 3.16.

One ordering of that set by the largest |coefficient|/(TOL w) over all cells
is shared by every cell, the constant first: float64 keeps its shortest prefix
within TOL w of every cell, float32 the shortest within TOL32 w. At cell
corners, where every T_k is +-1, that bound is nearly reached; shorter
prefixes chosen from random points failed there. Each fit uses NODES^3
Chebyshev nodes, values and DCTs in mpmath, base columns in a process pool
(the output does not depend on its size). Regenerate with:

    python -m chebax._src.recipes.hyp1f1_array_gen      # ~26 min on seven cores

CI rebuilds one base column (column_cells); the whole module is compared with
CHEBAX_FULL_REGEN=1, before releases.
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chebax._src.recipes._gen_common import (
    DPS,
    nodes,
    table_path,
    write_table_module,
)

ALO, AHI = 0.1, 10.0
NB0 = 4  # base columns per log axis
# X_S per base column [ia][ib], ln a and ln b ascending
XS_COL = (
    (8.0, 16.0, 32.0, 32.0),
    (16.0, 16.0, 16.0, 32.0),
    (16.0, 16.0, 16.0, 16.0),
    (16.0, 16.0, 16.0, 16.0),
)
NODES = 16  # per cell and direction
DMAX = 14  # total degree of the candidate set
TOL = 3e-16
TOL32 = 4e-7
CAP = 16.0  # largest relaxation (1 + q)/q of ln R's tolerance
MAXDEPTH = (5, 5, 14)  # halvings allowed along ln a, ln b, s
REGIONS = ("inner", "tail")


def _base_box(i):
    lo, hi = math.log(ALO), math.log(AHI)
    w = (hi - lo) / NB0
    return lo + i * w, lo + (i + 1) * w


def table_fn(mp, region, xs, a, b, s):
    """(tabulated value, metric weight) at a, b and s (x inner, t tail)."""
    if region == "inner":
        lr = mp.log(mp.hyp2f2(a + 1, 1, b + 1, 2, s))
        q = a / b * s * mp.exp(lr)
        lm = mp.log1p(q)
        return lr, max(1, abs(lm)) * min(mp.mpf(CAP), (1 + q) / q)
    x = mp.mpf(xs) / s
    lm = mp.log(mp.hyp1f1(a, b, x))
    return lm - x + (a - b) * mp.log(s), max(1, abs(lm))


def _cos_matrix(mp, n):
    return [
        [mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n)] for j in range(n)
    ]


def _dct1(mp, cm, v):
    n = len(v)
    out = [2 * mp.fsum(v[i] * cm[j][i] for i in range(n)) / n for j in range(n)]
    out[0] /= 2
    return out


def cell_coefs(mp, region, xs, box):
    """(C, w): C[i][j][k] the Chebyshev coefficients of the region's function on
    box = ((ln a), (ln b), (s)) intervals, i in ln a, j in ln b, k in s, and w
    the smallest metric weight over the nodes; values and DCTs in mp."""
    n = NODES
    z = nodes(mp, n)
    pts = [
        [mp.mpf(lo) + (mp.mpf(hi) - mp.mpf(lo)) * (t + 1) / 2 for t in z]
        for lo, hi in box
    ]
    av = [mp.exp(u) for u in pts[0]]
    bv = [mp.exp(u) for u in pts[1]]
    cm = _cos_matrix(mp, n)
    v, w = [], mp.inf
    for a in av:
        rows = []
        for b in bv:
            row = []
            for s in pts[2]:
                f, wk = table_fn(mp, region, xs, a, b, s)
                row.append(f)
                w = min(w, wk)
            rows.append(_dct1(mp, cm, row))  # along s
        v.append(rows)
    v = [
        [_dct1(mp, cm, [v[i][j][k] for j in range(n)]) for k in range(n)]
        for i in range(n)
    ]  # v[i][k][j]
    c = [[[0] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for j in range(n):
            col = _dct1(mp, cm, [v[i][k][j] for i in range(n)])
            for i in range(n):
                c[i][j][k] = col[i]
    return c, w


def _simplex():
    return [
        (i, j, k)
        for i in range(NODES)
        for j in range(NODES)
        for k in range(NODES)
        if i + j + k <= DMAX
    ]


def _outside(mp, c):
    """Sum of |c| outside the simplex, and per direction the part of it
    weighted by that direction's share of the degree."""
    tot, mass = [], [[], [], []]
    for i in range(NODES):
        for j in range(NODES):
            for k in range(NODES):
                d = i + j + k
                if d > DMAX:
                    v = abs(c[i][j][k])
                    tot.append(v)
                    for q, e in enumerate((i, j, k)):
                        mass[q].append(v * e / d)
    return mp.fsum(tot), [mp.fsum(m) for m in mass]


def column_cells(region, ia, ib):
    """Cells of base column (ia, ib) of region in depth-first order (left half
    first), and the column's k-d tree: nodes as (dim, split, left, right) with
    leaves (-1, 0, cell, cell), cell the index into the returned list. Each
    cell is (box, |c| outside the simplex over TOL w, w, simplex coefficients,
    low part of the constant coefficient)."""
    import mpmath as mp

    xs = XS_COL[ia][ib]
    top = xs if region == "inner" else 1.0
    simp = _simplex()
    cells, tree = [], []

    def visit(box, depth):
        me = len(tree)
        tree.append(None)
        c, w = cell_coefs(mp, region, xs, box)
        out, mass = _outside(mp, c)
        ratio = float(out / (TOL * w))
        if ratio > 1.0:
            for d in sorted(range(3), key=lambda q: (-mass[q], q)):
                if depth[d] < MAXDEPTH[d]:
                    break
            else:
                raise RuntimeError(f"{region} cell {box} misses TOL at full depth")
            lo, hi = box[d]
            mid = 0.5 * (lo + hi)
            nd = tuple(depth[q] + (q == d) for q in range(3))
            halves = []
            for half in ((lo, mid), (mid, hi)):
                nb = list(box)
                nb[d] = half
                halves.append(visit(tuple(nb), nd))
            tree[me] = (d, mid, halves[0], halves[1])
            return me
        c0 = c[0][0][0]
        cells.append(
            (
                box,
                ratio,
                float(w),
                [float(c[i][j][k]) for i, j, k in simp],
                float(c0 - mp.mpf(float(c0))),
            )
        )
        tree[me] = (-1, 0.0, len(cells) - 1, len(cells) - 1)
        return me

    with mp.workdps(DPS):
        visit((_base_box(ia), _base_box(ib), (0.0, top)), (0, 0, 0))
    return cells, tree


def _column_job(args):
    return column_cells(*args)


def build(procs=None):
    """Every base column's (cells, tree), regions then (ia, ib) row-major, the
    slowest columns first; the result does not depend on procs."""
    cols = [(r, ia, ib) for r in REGIONS for ia in range(NB0) for ib in range(NB0)]
    # large b is slowest (the transition x ~ b): submit those first
    first = sorted(
        range(len(cols)),
        key=lambda n: (cols[n][0] != "inner", -cols[n][2], -cols[n][1]),
    )
    procs = procs or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(procs) as ex:
        done = dict(zip(first, ex.map(_column_job, [cols[n] for n in first])))
    return cols, [done[n] for n in range(len(cols))]


def order_and_prefix(cells):
    """(order of the simplex indices, NKEEP, NKEEP32): the constant first, then
    by the largest |coefficient|/(TOL w) over all cells, descending (ties by
    index); NKEEP the shortest prefix whose dropped coefficients, plus those
    outside the simplex, sum to at most TOL w on every cell, NKEEP32 the same
    within TOL32 w."""
    simp = _simplex()
    w = np.array([c[2] for c in cells])
    A = np.abs(np.array([c[3] for c in cells])) / w[:, None]
    out = np.array([c[1] for c in cells]) * TOL  # outside / w
    big = A.max(axis=0)
    order = sorted(
        range(len(simp)), key=lambda m: (simp[m] != (0, 0, 0), -big[m], simp[m])
    )
    A = A[:, order]
    tail = np.zeros((A.shape[0], A.shape[1] + 1))
    for m in range(A.shape[1] - 1, -1, -1):
        tail[:, m] = tail[:, m + 1] + A[:, m]
    tail += out[:, None]
    ok = np.all(tail <= TOL, axis=0)
    ok32 = np.all(tail <= TOL32, axis=0)
    nkeep = int(np.nonzero(ok)[0].min()) if ok.any() else len(simp)
    nkeep32 = int(np.nonzero(ok32)[0].min()) if ok32.any() else len(simp)
    return order, max(nkeep, 1), max(nkeep32, 1)


def flatten(cols, results):
    """(cells, tree nodes, roots) over all columns: cell and node indices made
    global, roots[region][ia][ib] the root node of each column's tree."""
    cells, nodes_, roots = [], [], {}
    for (region, ia, ib), (cs, tree) in zip(cols, results):
        c0, n0 = len(cells), len(nodes_)
        cells += cs
        for d, split, lo, hi in tree:
            if d < 0:
                nodes_.append((-1, 0.0, lo + c0, hi + c0))
            else:
                nodes_.append((d, split, lo + n0, hi + n0))
        roots[(region, ia, ib)] = n0
    root = [
        [[roots[(r, ia, ib)] for ib in range(NB0)] for ia in range(NB0)]
        for r in REGIONS
    ]
    return cells, nodes_, root


def tree_depth(tree, root):
    depth = 0
    for r in root:
        for row in r:
            for n in row:
                stack = [(n, 0)]
                while stack:
                    k, d = stack.pop()
                    depth = max(depth, d)
                    if tree[k][0] >= 0:
                        stack += [(tree[k][2], d + 1), (tree[k][3], d + 1)]
    return depth


def write(cols, results, out_dir=None):
    """Emit the table module from build()'s output."""
    import mpmath as mp

    cells, tree, root = flatten(cols, results)
    order, nkeep, nkeep32 = order_and_prefix(cells)
    simp = _simplex()
    idx = tuple(simp[m] for m in order[:nkeep])
    table = np.array([[c[3][m] for m in order[:nkeep]] for c in cells])
    box = np.array([[v for lohi in c[0] for v in lohi] for c in cells])
    depth = tree_depth(tree, root)
    per = {
        r: sum(len(cs) for (rr, _, _), (cs, _) in zip(cols, results) if rr == r)
        for r in REGIONS
    }
    print(
        f"{len(cells)} cells ({', '.join(f'{r} {n}' for r, n in per.items())}), "
        f"tree depth {depth}; kept {nkeep} of {len(simp)} (float32 {nkeep32})",
        flush=True,
    )
    with mp.workdps(DPS):
        write_table_module(
            table_path(out_dir, __file__, "hyp1f1_array_table.py"),
            "chebax._src.recipes.hyp1f1_array_gen",
            "TABLE[n, m] = coefficient IDX[m] = (i, j, k) of T_i(ln a) T_j(ln b) "
            "T_k(s) on cell n, IDX[0] = (0, 0, 0) with low part C0LO[n]; BOX[n] = "
            "(ln a lo, hi, ln b lo, hi, s lo, hi), s = x in [0, X_S] (inner, ln R) "
            "or t = X_S/x in [0, 1] (tail, U), X_S = XS_COL[ia][ib]; a column's "
            "cells are the leaves of a k-d tree from ROOT[region][ia][ib] over "
            "TREE[k] = (dim, split, left, right), leaves (-1, 0, n, n), ln a and "
            "ln b on NB0 uniform base intervals of [ln ALO, ln AHI]; float32 reads "
            "TABLE[:, :NKEEP32]",
            {
                "nodes": NODES,
                "dmax": DMAX,
                "tol": TOL,
                "tol32": TOL32,
                "cap": CAP,
                "maxdepth": MAXDEPTH,
                "ab_domain": (ALO, AHI),
            },
            {
                "ALO": ALO,
                "AHI": AHI,
                "NB0": NB0,
                "XS_COL": XS_COL,
                "NKEEP32": nkeep32,
                "DEPTH": depth,
                "ROOT": root,
                "IDX": idx,
            },
            {
                "TABLE": table,
                "C0LO": np.array([c[4] for c in cells]),
                "BOX": box,
                "TREE": np.array(tree, dtype=float),
            },
        )


def main(out_dir=None, procs=None):
    write(*build(procs), out_dir)


if __name__ == "__main__":
    main()
