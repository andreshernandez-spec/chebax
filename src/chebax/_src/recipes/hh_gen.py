"""Generator for the Hh table. Build-time only; mpmath inside.

Hh_nu(z) = (1/nu!) int_0^inf u^nu e^(-(u + z)^2/2) du = e^(-z^2/4) D_{-nu-1}(z)
for real nu >= NU_LO and real z, nu and z per element. The integrand's saddle
point, with Stirling's lnGamma folded in, gives

    ln Hh = nu (1 - ln v) - v^2/2 - ln(v R)/2 + T(nu, z),
    R = sqrt(z^2 + 4 nu),  v = (z + R)/2 (= 2 nu/(R - z) below z = 0),

which has the exact limits -z^2/2 - (nu + 1) ln z as z -> inf and nu ln|z| -
lnGamma(nu + 1) + ln(2 pi)/2 as z -> -inf, so T is bounded: -0.75 at nu = 0.1,
z = 0, and ~g(s)/nu at large nu (-1/(6 nu) at z = 0). The table holds T in
sigma = z/sqrt(z^2 + 4(nu + 1)) in [-1, 1] and in ln nu on [NU_LO, NU1] (zone
L) or tau = NU1/nu in [0, 1] (zone H, so nu reaches infinity). The 1 in sigma's
scale keeps O(1) structure in z resolved at small nu, where z/sqrt(z^2 + 4 nu)
squeezed z in [1.3, 14] into [0.9, 0.999] at nu = 0.1 (1-D degrees summed over
eight intervals 206 there against 152).

Each zone's NP x NS base boxes are halved, along the axis holding the larger
part of the coefficients outside the leading BLOCK, until those sum to at most
TOL w, w the smallest max(1, |ln Hh|) over the patch (the test's metric), and
bound the slopes' error (|T_k'| <= k^2, times each coordinate's slope) to
TOL_D max(1, |slope|) in nu and in z: without it d/dnu was 4.6e-12 off at a
patch corner at nu = 0.13, the value's TOL w (w ~ 13 there) times ~k^2/nu.
float32 reads the leading K32 x K32 block, the smallest within TOL32 w on
every patch. Each fit uses NODES x NODES Chebyshev nodes, values and DCTs in
mpmath; references are mpmath's pcfd up to nu = 150 and a quadrature in the
saddle's scaled variable above (they agree to 1e-38 on nu in [20, 150]).
Regenerate with:

    python -m chebax._src.recipes.hh_gen      # ~16 min on six cores

CI rebuilds one base box (box_cells); the whole module is compared with
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

NU_LO = 0.1
NU1 = 10.0  # zone L in ln nu below, zone H in tau = NU1/nu above
ZONES = ("L", "H")
NP = {"L": 4, "H": 1}  # base boxes along the nu coordinate
SCUTS = (-1.0, -0.5, 0.0, 0.5, 1.0)  # base boxes along sigma
NODES = 16
BLOCK = 8
TOL = 3e-16
TOL_D = 2e-14  # the dropped coefficients' bound on the slopes' relative error
TOL32 = 4e-7
MAXDEPTH = (10, 24)  # halvings along the nu coordinate and sigma
PCFD_NU = 150  # pcfd below, the saddle quadrature above
LNC_J0, LNC_J1 = 45, 91  # ln(j/64) covers mantissas in [45/64, 91/64]


def _pbox(zone, i):
    if zone == "H":
        return (0.0, 1.0)
    lo, hi = math.log(NU_LO), math.log(NU1)
    w = (hi - lo) / NP["L"]
    return (lo + i * w, lo + (i + 1) * w)


def _v_r(mp, nu, z):
    r = mp.sqrt(z * z + 4 * nu)
    return ((z + r) / 2 if z >= 0 else 2 * nu / (r - z)), r


def first(mp, nu, z):
    v, r = _v_r(mp, nu, z)
    return nu * (1 - mp.log(v)) - v * v / 2 - mp.log(v * r) / 2


def _log_j_saddle(mp, nu, z):
    """ln int_0^inf u^nu e^(-(u + z)^2/2) du for large nu, by quadrature in the
    saddle's scaled variable."""
    v, _ = _v_r(mp, nu, z)
    us = nu / v
    sk = mp.sqrt(1 + nu / (us * us))
    f = lambda y: mp.exp(
        nu * mp.log1p(y / (sk * us)) - (y / sk + v) ** 2 / 2 + v * v / 2
    )
    lo = -sk * us
    pts = [lo] + [p for p in (-12, -6, -3, -1, 0, 1, 3, 6, 12) if p > lo] + [mp.inf]
    return nu * mp.log(us) - v * v / 2 + mp.log(mp.quad(f, pts) / sk)


def log_hh(mp, nu, z):
    if nu <= PCFD_NU:
        return mp.re(mp.log(mp.pcfd(-nu - 1, z))) - z * z / 4
    return mp.re(_log_j_saddle(mp, nu, z)) - mp.loggamma(nu + 1)


def point(mp, zone, p, sg):
    """(nu, z) at zone coordinate p and sigma."""
    nu = mp.exp(p) if zone == "L" else mp.mpf(NU1) / p
    return nu, 2 * mp.sqrt(nu + 1) * sg / mp.sqrt(1 - sg * sg)


def _dct1(mp, cm, v):
    n = len(v)
    out = [2 * mp.fsum(v[i] * cm[j][i] for i in range(n)) / n for j in range(n)]
    out[0] /= 2
    return out


def _slope_scales(mp, zone, nu, z, sg):
    """(|dp/dnu|, |dsigma/dnu|, |dsigma/dz|) at a node, each over max(1,
    |slope|) of the first term (T moves the slopes by under 1%)."""
    v, r = _v_r(mp, nu, z)
    q2 = z * z + 4 * (nu + 1)
    s_nu = max(1, abs(-mp.log(v) - 1 / (2 * v * r) - 1 / (r * r)))
    s_z = max(1, abs(v * (1 + 1 / (r * r))))
    dp = 1 / nu if zone == "L" else mp.mpf(NU1) / (nu * nu)
    return dp / s_nu, 2 * abs(sg) / q2 / s_nu, (1 - sg * sg) / mp.sqrt(q2) / s_z


def patch_coefs(mp, zone, box):
    """(C, w, d): C[i][j] the Chebyshev coefficients of T on box = ((p lo,
    hi), (sigma lo, hi)), i along p and j along sigma, w = max(1, min |ln Hh|)
    over the nodes, and d the largest _slope_scales over them; values and DCTs
    in mp."""
    n = NODES
    z = nodes(mp, n)
    pts = [
        [mp.mpf(lo) + (mp.mpf(hi) - mp.mpf(lo)) * (t + 1) / 2 for t in z]
        for lo, hi in box
    ]
    cm = [
        [mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n)] for j in range(n)
    ]
    rows, w, dsc = [], mp.inf, [mp.mpf(0)] * 3
    for p in pts[0]:
        row = []
        for sg in pts[1]:
            nu, x = point(mp, zone, p, sg)
            lh = log_hh(mp, nu, x)
            row.append(lh - first(mp, nu, x))
            w = min(w, abs(lh))
            dsc = [max(a, b) for a, b in zip(dsc, _slope_scales(mp, zone, nu, x, sg))]
        rows.append(_dct1(mp, cm, row))  # along sigma
    c = [[0] * n for _ in range(n)]
    for j in range(n):
        col = _dct1(mp, cm, [rows[i][j] for i in range(n)])
        for i in range(n):
            c[i][j] = col[i]
    return c, max(mp.mpf(1), w), dsc


def _tails(mp, c):
    """|c| outside the leading BLOCK x BLOCK: in all, past BLOCK along p, past
    BLOCK along sigma; and bounds on the dropped part's slopes along p and
    sigma on [-1, 1] (|T_k'| <= k^2)."""
    k = BLOCK
    n = len(c)
    out = [(i, j) for i in range(n) for j in range(n) if i >= k or j >= k]
    tp = mp.fsum(abs(c[i][j]) for i in range(k, n) for j in range(n))
    ts = mp.fsum(abs(c[i][j]) for i in range(n) for j in range(k, n))
    both = mp.fsum(abs(c[i][j]) for i in range(k, n) for j in range(k, n))
    gp = mp.fsum(abs(c[i][j]) * i * i for i, j in out)
    gs = mp.fsum(abs(c[i][j]) * j * j for i, j in out)
    return tp + ts - both, (tp, ts), (gp, gs)


def box_cells(zone, ip, isg):
    """Patches of base box (ip, isg) of zone in depth-first order (lower half
    first), and its k-d tree: nodes as (dim, split, left, right), leaves (-1,
    0, patch, patch). Each patch is (box, |c| outside BLOCK over TOL w, w, the
    full NODES x NODES coefficients as floats)."""
    import mpmath as mp

    cells, tree = [], []

    def visit(box, depth):
        me = len(tree)
        tree.append(None)
        c, w, (dp, dsn, dsz) = patch_coefs(mp, zone, box)
        out, (tp, ts), (gp, gs) = _tails(mp, c)
        ratio = float(out / (TOL * w))
        # the dropped part's error in d ln Hh/dnu and d/dz, relative
        sp, ss = 2 / (box[0][1] - box[0][0]), 2 / (box[1][1] - box[1][0])
        e_p, e_s = gp * sp * dp, gs * ss * max(dsn, dsz)
        dratio = float((e_p + e_s) / TOL_D)
        if ratio > 1.0 or dratio > 1.0:
            if ratio > dratio:
                order = (0, 1) if tp >= ts else (1, 0)
            else:
                order = (0, 1) if e_p >= e_s else (1, 0)
            for d in order:
                if depth[d] < MAXDEPTH[d]:
                    break
            else:
                raise RuntimeError(f"{zone} patch {box} misses TOL at full depth")
            lo, hi = box[d]
            mid = 0.5 * (lo + hi)
            nd = tuple(depth[q] + (q == d) for q in range(2))
            halves = []
            for half in ((lo, mid), (mid, hi)):
                nb = list(box)
                nb[d] = half
                halves.append(visit(tuple(nb), nd))
            tree[me] = (d, mid, halves[0], halves[1])
            return me
        cells.append((box, ratio, float(w), [[float(v) for v in r] for r in c]))
        tree[me] = (-1, 0.0, len(cells) - 1, len(cells) - 1)
        return me

    with mp.workdps(DPS):
        visit((_pbox(zone, ip), (SCUTS[isg], SCUTS[isg + 1])), (0, 0))
    return cells, tree


def _box_job(args):
    return box_cells(*args)


def base_boxes():
    return [
        (z, ip, isg)
        for z in ZONES
        for ip in range(NP[z])
        for isg in range(len(SCUTS) - 1)
    ]


def build(procs=None):
    """Every base box's (patches, tree), in base_boxes() order; the result does
    not depend on procs."""
    boxes = base_boxes()
    procs = procs or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(procs) as ex:
        return boxes, list(ex.map(_box_job, boxes))


def block32(cells):
    """The smallest K32 whose leading K32 x K32 block leaves at most TOL32 w
    (with what lies outside BLOCK) on every patch."""
    for k in range(1, BLOCK + 1):
        ok = True
        for _, ratio, w, c in cells:
            a = np.abs(np.array(c))
            drop = a[:BLOCK, :BLOCK].sum() - a[:k, :k].sum() + ratio * TOL * w
            if drop > TOL32 * w:
                ok = False
                break
        if ok:
            return k
    return BLOCK


def flatten(boxes, results):
    """(patches, tree nodes, roots[zone][ip][isg]) with global indices."""
    cells, nodes_, roots = [], [], {}
    for key, (cs, tree) in zip(boxes, results):
        c0, n0 = len(cells), len(nodes_)
        cells += cs
        for d, split, lo, hi in tree:
            if d < 0:
                nodes_.append((-1, 0.0, lo + c0, hi + c0))
            else:
                nodes_.append((d, split, lo + n0, hi + n0))
        roots[key] = n0
    ns = len(SCUTS) - 1
    root = [
        [[roots[(z, ip, isg)] for isg in range(ns)] for ip in range(NP[z])]
        for z in ZONES
    ]
    return cells, nodes_, root


def tree_depth(tree, root):
    depth = 0
    for zone in root:
        for row in zone:
            for n in row:
                stack = [(n, 0)]
                while stack:
                    k, d = stack.pop()
                    depth = max(depth, d)
                    if tree[k][0] >= 0:
                        stack += [(tree[k][2], d + 1), (tree[k][3], d + 1)]
    return depth


def log_constants(mp):
    """ln 2 as (hi with its low 11 bits clear, so k hi is exact for |k| < 2^11,
    lo), and ln(j/64), j = 45..91, as (hi, lo): the runtime's double-double
    log of v, whose rounding ln Hh otherwise carries times nu."""
    ln2 = mp.log(2)
    hi = float(ln2)
    bits = np.array([hi]).view(np.int64)[0] & ~np.int64((1 << 11) - 1)
    ln2_hi = float(np.array([bits]).view(np.float64)[0])
    ln2_lo = float(ln2 - mp.mpf(ln2_hi))
    lnc = [mp.log(mp.mpf(j) / 64) for j in range(LNC_J0, LNC_J1 + 1)]
    lnc_hi = [float(v) for v in lnc]
    lnc_lo = [float(v - mp.mpf(h)) for v, h in zip(lnc, lnc_hi)]
    return (ln2_hi, ln2_lo), np.array(lnc_hi), np.array(lnc_lo)


def write(boxes, results, out_dir=None):
    """Emit the table module from build()'s output."""
    import mpmath as mp

    cells, tree, root = flatten(boxes, results)
    k32 = block32(cells)
    table = np.array([np.array(c)[:BLOCK, :BLOCK] for _, _, _, c in cells])
    box = np.array([[v for lohi in b for v in lohi] for b, _, _, _ in cells])
    depth = tree_depth(tree, root)
    per = {
        z: sum(len(cs) for k, (cs, _) in zip(boxes, results) if k[0] == z)
        for z in ZONES
    }
    print(
        f"{len(cells)} patches ({', '.join(f'{z} {n}' for z, n in per.items())}), "
        f"tree depth {depth}; float32 block {k32} x {k32}",
        flush=True,
    )
    with mp.workdps(DPS):
        ln2, lnc_hi, lnc_lo = log_constants(mp)
        write_table_module(
            table_path(out_dir, __file__, "hh_table.py"),
            "chebax._src.recipes.hh_gen",
            "TABLE[n, i, j] = coefficient of T_i(p) T_j(sigma) on patch n, BOX[n] = "
            "(p lo, hi, sigma lo, hi), p = ln nu (zone L, nu in [NU_LO, NU1]) or tau "
            "= NU1/nu (zone H), sigma = z/sqrt(z^2 + 4(nu + 1)); a base box's "
            "patches are the leaves of a k-d tree from ROOT[zone][ip][isg] over "
            "TREE[k] = (dim, split, left, right), leaves (-1, 0, n, n), zone L's "
            "p on NP_L uniform base intervals, sigma on the base cuts SCUTS; "
            "float32 reads TABLE[:, :K32, :K32]",
            {
                "nodes": NODES,
                "block": BLOCK,
                "tol": TOL,
                "tol_d": TOL_D,
                "tol32": TOL32,
                "maxdepth": MAXDEPTH,
                "pcfd_nu": PCFD_NU,
            },
            {
                "NU_LO": NU_LO,
                "NU1": NU1,
                "NP_L": NP["L"],
                "SCUTS": SCUTS,
                "K32": k32,
                "DEPTH": depth,
                "ROOT": root,
                "LN2": ln2,
                "LNC_J0": LNC_J0,
            },
            {
                "TABLE": table,
                "BOX": box,
                "TREE": np.array(tree, dtype=float),
                "LNC_HI": lnc_hi,
                "LNC_LO": lnc_lo,
            },
        )


def main(out_dir=None, procs=None):
    write(*build(procs), out_dir)


if __name__ == "__main__":
    main()
