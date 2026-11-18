"""Generator for the per-element betainc table. Build-time only; mpmath inside.

betainc_array takes a and b per element, so each element reads one cell of a
3-D table instead of contracting the (x, a, b) tensors of betainc_table per
call. With (p, q, t) = (a, b, x) in the lower orientation, or (b, a, 1 - x)
for the reflection I = 1 - I_t(p, q):

    ln I_t(p, q) = -(p phi(u) + q phi(v)) + T(ln p, ln q, s)

nu = p + q, u = (t nu - p)/p, v = -(t nu - p)/q (so p u + q v = 0 and the
first term is Temme's -nu eta^2/2 with no cancellation), phi(z) = z -
log1p(z), x_s = (p + 1)/(p + q + 2), s = ln(1 - t)/ln(1 - x_s), and

    T = ln 2F1(nu, 1; p + 1; t) - mu(p) - mu(q) + mu(nu) + ln(q/(2 pi p nu))/2

with mu the remainder of Stirling's lnGamma. T holds every (p, q)-only term
(no lgamma or psi per element) and is O(1) where I is not small; written as
a ln x + b ln(1 - x) - ln a - ln B + L, those terms cancel to O(1) from size
a + b. s moves the branch point of I at t = 1 (exponent q) to s = inf: 9-33
s-coefficients across the box against 63+ in t.

Along ln p and ln q the coefficients of T decay like rho = 14.5/w per degree
on intervals of width w, the same everywhere: a singularity at distance ~3.6
from the real ln a axis, as I's poles at a + b = -n are. So the cells are
uniform in ln p and ln q, NAB per axis on [ln ALO, ln AHI], and each column
halves its s-interval [0, S_SPLIT] until every cell's coefficients outside
the total-degree set i + j + k <= DMAX (i in ln p, j in ln q, k in s) sum to
at most TOL. One ordering of that set by the largest |coefficient| over all
cells is shared by every cell: float64 keeps its shortest prefix whose
dropped coefficients sum to at most TOL on every cell, float32 the shortest
within TOL32. That sum is a loose bound: float32's error stays its own
arithmetic (~3 ulps) down to 65 coefficients and passes 5e-7 at 60; TOL32
keeps 73, the fastest float32 value_and_grad on the GPU of those under 5e-7
(experiments/55).

A lane stays in the lower orientation up to s = S_SPLIT of its column rather
than x_s (s = 1): where I(x_s) is small (q below ~3), the reflected side's
I = 1 - I_t multiplied I_t's rounding by I_t/I, up to ~30 at b = 0.11.
S_SPLIT is the largest s with I = I_SPLIT over a 5 x 5 sample of the column,
plus 10%, up to a quarter, and at least 1, so the reflected side reports
I >= ~I_SPLIT. float32 turns at S_SPLIT32 instead, the largest s with I =
I_SPLIT32 over a 9 x 9 sample, within [1, S_SPLIT]: at large p, q the
quarter steps put lanes in the lower orientation up to I ~ 0.95, where T
cancels a first term of O(1) and float32 lost ~6 ulps (experiments/55).
Any split in [1, S_SPLIT] is covered: past x_s the reflected s = ln x/ln x_s
is below 1. Each fit uses NODES^3 Chebyshev nodes, values and DCTs in
mpmath, columns in a process pool (the output does not depend on its size).
Regenerate with:

    python -m chebax._src.recipes.betainc_array_gen      # ~18 min on six cores

CI rebuilds one column (column_cells); the whole module is compared with
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

ALO, AHI = 0.1, 100.0
NAB = 16  # cells per log axis
NODES = 16  # per cell and direction
DMAX = 14  # total degree of the candidate set
TOL = 2e-16
TOL32 = 4e-7
SDEPTH = 7  # deepest s halving, relative to the column's [0, S_SPLIT]
I_SPLIT = 0.45  # I at a column's split, at least
I_SPLIT32 = 0.5  # float32's split


def _lnab_box(i):
    lo, hi = math.log(ALO), math.log(AHI)
    w = (hi - lo) / NAB
    return lo + i * w, lo + (i + 1) * w


def _mu(mp, z):
    """lnGamma(z) minus Stirling's (z - 1/2) ln z - z + ln(2 pi)/2."""
    return mp.loggamma(z) - (z - mp.mpf(0.5)) * mp.log(z) + z - mp.log(2 * mp.pi) / 2


def table_fn(mp, la, lb, s):
    p, q = mp.exp(la), mp.exp(lb)
    nu = p + q
    t = -mp.expm1(s * mp.log((q + 1) / (nu + 2)))
    return (
        mp.log(mp.hyp2f1(nu, 1, p + 1, t))
        - _mu(mp, p)
        - _mu(mp, q)
        + _mu(mp, nu)
        + mp.log(q / (2 * mp.pi * p * nu)) / 2
    )


def _cos_matrix(mp, n):
    return [
        [mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n)] for j in range(n)
    ]


def _dct1(mp, cm, v):
    n = len(v)
    out = [2 * mp.fsum(v[i] * cm[j][i] for i in range(n)) / n for j in range(n)]
    out[0] /= 2
    return out


def cell_coefs(mp, box):
    """C[i][j][k]: Chebyshev coefficients of T on box = ((ln p), (ln q), (s))
    intervals, i in ln p, j in ln q, k in s; values and DCTs in mp."""
    n = NODES
    z = nodes(mp, n)
    pts = [
        [mp.mpf(lo) + (mp.mpf(hi) - mp.mpf(lo)) * (t + 1) / 2 for t in z]
        for lo, hi in box
    ]
    cm = _cos_matrix(mp, n)
    v = [[[table_fn(mp, la, lb, s) for s in pts[2]] for lb in pts[1]] for la in pts[0]]
    v = [[_dct1(mp, cm, v[i][j]) for j in range(n)] for i in range(n)]  # along s
    v = [
        [_dct1(mp, cm, [v[i][j][k] for j in range(n)]) for k in range(n)]
        for i in range(n)
    ]
    # v[i][k][j] now; along ln p last
    c = [[[0] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for j in range(n):
            col = _dct1(mp, cm, [v[i][k][j] for i in range(n)])
            for i in range(n):
                c[i][j][k] = col[i]
    return c


def split_s(mp, ia, ib):
    """S_SPLIT of column (ia, ib): see the module docstring."""
    worst = mp.mpf(1)
    for ua in np.linspace(*_lnab_box(ia), 5):
        for ub in np.linspace(*_lnab_box(ib), 5):
            p, q = mp.exp(mp.mpf(float(ua))), mp.exp(mp.mpf(float(ub)))
            ls = mp.log((q + 1) / (p + q + 2))
            f = (
                lambda s_: mp.betainc(p, q, 0, -mp.expm1(s_ * ls), regularized=True)
                - I_SPLIT
            )
            if f(mp.mpf(1)) >= 0:
                continue
            hi = mp.mpf(2)
            while f(hi) < 0:
                hi *= 2
            worst = max(
                worst, mp.findroot(f, (mp.mpf(1), hi), solver="bisect", tol=1e-6)
            )
    if worst <= 1:
        return 1.0
    return math.ceil(float(worst) * 1.1 * 4) / 4


def split_s32(ia, ib, top):
    """S_SPLIT32 of column (ia, ib), whose S_SPLIT is top: see the module
    docstring."""
    if top <= 1.0:
        return 1.0
    import mpmath as mp

    with mp.workdps(DPS):
        worst, hi = mp.mpf(1), mp.mpf(top)
        for ua in np.linspace(*_lnab_box(ia), 9):
            for ub in np.linspace(*_lnab_box(ib), 9):
                p, q = mp.exp(mp.mpf(float(ua))), mp.exp(mp.mpf(float(ub)))
                ls = mp.log((q + 1) / (p + q + 2))
                f = (
                    lambda s_: mp.betainc(p, q, 0, -mp.expm1(s_ * ls), regularized=True)
                    - I_SPLIT32
                )
                if f(mp.mpf(1)) >= 0:
                    continue
                if f(hi) < 0:
                    return float(top)
                root = mp.findroot(f, (mp.mpf(1), hi), solver="bisect", tol=1e-6)
                worst = max(worst, root)
    return min(math.ceil(float(worst) * 1024) / 1024, float(top))


def _split32_job(args):
    return split_s32(*args)


def _simplex():
    return [
        (i, j, k)
        for i in range(NODES)
        for j in range(NODES)
        for k in range(NODES)
        if i + j + k <= DMAX
    ]


def _outside_simplex(mp, c):
    return mp.fsum(
        abs(c[i][j][k])
        for i in range(NODES)
        for j in range(NODES)
        for k in range(NODES)
        if i + j + k > DMAX
    )


def column_cells(ia, ib):
    """(s cuts, cells) of column (ia, ib), each cell C as nested float lists of
    |c| outside the simplex summed (in mp) and the simplex coefficients."""
    import mpmath as mp

    with mp.workdps(DPS):
        ab = (_lnab_box(ia), _lnab_box(ib))
        top = split_s(mp, ia, ib)
        cuts = [0.0, top]
        memo = {}
        while True:
            bad = None
            for k in range(len(cuts) - 1):
                box = ab + ((cuts[k], cuts[k + 1]),)
                if box not in memo:
                    c = cell_coefs(mp, box)
                    memo[box] = (
                        float(_outside_simplex(mp, c)),
                        [float(c[i][j][k_]) for i, j, k_ in _simplex()],
                    )
                if memo[box][0] > TOL:
                    if cuts[k + 1] - cuts[k] <= top * 2.0**-SDEPTH:
                        raise RuntimeError(f"cell {box} misses TOL at depth {SDEPTH}")
                    bad = k
                    break
            if bad is None:
                break
            cuts.insert(bad + 1, 0.5 * (cuts[bad] + cuts[bad + 1]))
        cells = [memo[ab + ((cuts[k], cuts[k + 1]),)] for k in range(len(cuts) - 1)]
    return cuts, cells


def _column_job(ij):
    return column_cells(*ij)


def build(procs=None):
    """Every column's (cuts, cells) and S_SPLIT32, columns in (ia, ib)
    row-major order; the result does not depend on procs."""
    cols = [(ia, ib) for ia in range(NAB) for ib in range(NAB)]
    procs = procs or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(procs) as ex:
        out = list(ex.map(_column_job, cols))
        tops = [(ia, ib, cuts[-1]) for (ia, ib), (cuts, _) in zip(cols, out)]
        return out, list(ex.map(_split32_job, tops))


def order_and_prefix(cells):
    """(order of the simplex indices, NKEEP, NKEEP32): order by the largest
    |coefficient| over all cells, descending (ties by index); NKEEP the shortest
    prefix whose dropped coefficients, plus those outside the simplex, sum to
    at most TOL on every cell, NKEEP32 the same within TOL32."""
    simp = _simplex()
    A = np.abs(np.array([c for _, c in cells]))  # cells x simplex
    out = np.array([o for o, _ in cells])
    big = A.max(axis=0)
    order = sorted(range(len(simp)), key=lambda m: (-big[m], simp[m]))
    A = A[:, order]
    # dropped sum of the suffix after each prefix length, summed from the small end
    tail = np.zeros((A.shape[0], A.shape[1] + 1))
    for m in range(A.shape[1] - 1, -1, -1):
        tail[:, m] = tail[:, m + 1] + A[:, m]
    tail += out[:, None]
    ok = np.all(tail <= TOL, axis=0)
    ok32 = np.all(tail <= TOL32, axis=0)
    nkeep = int(np.nonzero(ok)[0].min()) if ok.any() else len(simp)
    nkeep32 = int(np.nonzero(ok32)[0].min()) if ok32.any() else len(simp)
    return order, nkeep, nkeep32


def main(out_dir=None, procs=None):
    import mpmath as mp

    cols, s_split32 = build(procs)
    cells = [cell for _, cs in cols for cell in cs]
    order, nkeep, nkeep32 = order_and_prefix(cells)
    simp = _simplex()
    idx = tuple(simp[m] for m in order[:nkeep])
    table = np.array([[c[m] for m in order[:nkeep]] for _, c in cells])
    ns = tuple(len(cuts) - 1 for cuts, _ in cols)
    s_split = tuple(cuts[-1] for cuts, _ in cols)
    base = tuple(int(v) for v in np.concatenate([[0], np.cumsum(ns)[:-1]]))
    pad = max(ns) + 1
    scuts = np.array([cuts + [cuts[-1]] * (pad - len(cuts)) for cuts, _ in cols])
    print(
        f"{len(cells)} cells, s-cells per column {min(ns)}-{max(ns)}; "
        f"kept {nkeep} of {len(simp)} (float32 {nkeep32})",
        flush=True,
    )
    with mp.workdps(DPS):
        write_table_module(
            table_path(out_dir, __file__, "betainc_array_table.py"),
            "chebax._src.recipes.betainc_array_gen",
            "TABLE[n, m] = coefficient IDX[m] = (i, j, k) of T_i(ln p) T_j(ln q) "
            "T_k(s) on cell n; column (ia, ib) = ia NAB + ib holds cells "
            "BASE[col] + l over s interval l (SCUTS[col, l:l+2]), ln p and ln q "
            "on NAB uniform intervals of [ln ALO, ln AHI], s in [0, S_SPLIT[col]]; "
            "float32 reads TABLE[:, :NKEEP32] and turns at S_SPLIT32[col]",
            {
                "nodes": NODES,
                "dmax": DMAX,
                "tol": TOL,
                "tol32": TOL32,
                "i_split": I_SPLIT,
                "i_split32": I_SPLIT32,
                "ab_domain": (ALO, AHI),
            },
            {
                "ALO": ALO,
                "AHI": AHI,
                "NAB": NAB,
                "NKEEP32": nkeep32,
                "NS": ns,
                "S_SPLIT": s_split,
                "S_SPLIT32": tuple(s_split32),
                "BASE": base,
                "IDX": idx,
            },
            {"TABLE": table, "SCUTS": scuts},
        )


if __name__ == "__main__":
    main()
