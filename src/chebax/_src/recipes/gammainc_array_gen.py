"""Generator for the per-element gammainc table. Build-time only; mpmath inside.

gammainc_array takes a per element, so instead of collapsing a 2-D table's
a-axis per element (~4,300 coefficients) each element evaluates one BLOCK
patch of a tensor grid over its zone. Zones (lambda = x/a, phi = lambda - 1
- ln lambda, eta = sign(lambda - 1) sqrt(2 phi), Gc(a) = lnGamma(a+1) minus
its Stirling form); each table folds the a-only prefactor in, so the runtime
calls no lgamma:

a < 1, split at x = a + 1, parameter u = ln(1 + a):
  A0  r = x/(a+1) in [0, 1]:   ln P = a ln x - x + A0(u, r),
                               A0 = ln 1F1(1; a+1; x) - lnGamma(a+1)
  B0  t = (a+1)/x in [0, 1]:   ln Q = (a-1) ln x - x + ln a + B0(u, t),
                               B0 = ln[Gamma(a, x) x^(1-a) e^x] - lnGamma(a+1)
a >= 1, by lambda:
  L   lambda in [0, LAM_LO]:   ln P = -a phi - ln(2 pi a)/2 + L(p, lambda),
                               L = ln 1F1(1; a+1; x) - Gc(a)
  T   eta in [ELO, EHI]:       Q = erfc(y)/2 + e^(-y^2) T(w, eta)/sqrt(2 pi a),
                               y = eta sqrt(a/2), w = 1/a in [0, 1]
  U   s = a/x in [0, S_HI]:    ln Q = -a phi - ln(2 pi a)/2 - ln lambda + U(p, s),
                               U = ln[Gamma(a, x) x^(1-a) e^x] - Gc(a)
L and U take p = u for a <= 10 (L1, U1) and p = v = 10/a above (L2, U2);
T is gammainc_large's Temme function, refitted in w down to a = 1, and
LAM_LO, LAM_HI are lambda at eta = ELO, EHI.

u = ln(1 + a) moves the pole of 1F1(1; a+1; .) at a = -1 to infinity
(degree 17 in u against 53 in a at x -> 0, experiments/03); the split at
x = a + 1 puts the transition at a fixed coordinate and removes the uniform
recipe's x = 8 seam. Each zone starts as one patch, fitted at NODES x NODES
nodes in mpmath, and a patch whose coefficients outside its leading BLOCK
block sum to more than TOL (a relative bound on P or Q) is halved along the
axis holding the larger tail, until every patch fits: 1088 patches at 8 x 8
(A0 8 x 8, B0 8 x 29, L1 24 x 8, U1 29 x 6, L2 17 x 14, U2 2 x 4, T 10 x 18,
parameter x argument intervals), B0 splitting toward t = 0 (a C-infinity
endpoint) and L1, U1 toward a = 1. A halving cuts the zone's whole grid, so
the count grows fast as the block shrinks (65 patches at 12 x 12); 8 x 8 is
chosen for the runtime, which reads every coefficient of a block per element.
float32 reads the leading KY32 x KP32 block, the smallest within TOL32 on
every patch: 5 x 5. Regenerate with:

    python -m chebax._src.recipes.gammainc_array_gen      # ~17 min

CI rebuilds zone U2 only (zone_table, tests/test_gammainc_array.py); the whole
module is compared with CHEBAX_FULL_REGEN=1, before releases.
"""

import numpy as np

from chebax._src.recipes import gammainc_large_gen as _glg
from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    param_fit,
    table_path,
    write_table_module,
)

ALO = 0.1  # the uniform recipe's lower end, chebax.domains.GAMMAINC.lo
ASPLIT = 10.0  # L1/U1 in u up to here, L2/U2 in v = ASPLIT/a above
ELO, EHI = _glg.ELO, _glg.EHI
NODES = 16  # per patch and direction
# the block marcum_gen and skewnorm_gen take from here: argument x parameter
KY, KP = 12, 12
BLOCK = (8, 8)  # this table's
TOL = 1e-15
TOL32 = 1e-8  # float32 roundoff is 6e-8
NAMES = ("A0", "B0", "L1", "U1", "L2", "U2", "T")


def _l1f1(mp, a, x):
    return mp.log(_glg._hyp1f1_1(mp, a + 1, x))


def _lup(mp, a, x):
    """ln[Gamma(a, x) x^(1-a) e^x] for x > a, from the continued fraction."""
    return _glg._log_q(mp, a, x) + mp.loggamma(a) + x + (1 - a) * mp.log(x)


def _gc(mp, a):
    stirling = (a + mp.mpf(1) / 2) * mp.log(a) - a + mp.log(2 * mp.pi) / 2
    return mp.loggamma(a + 1) - stirling


def _f_a0(mp, a, r):
    return _l1f1(mp, a, r * (a + 1)) - mp.loggamma(a + 1)


def _f_b0(mp, a, t):
    return _lup(mp, a, (a + 1) / t) - mp.loggamma(a + 1)


def _f_low(mp, a, lam):
    return _l1f1(mp, a, a * lam) - _gc(mp, a)


def _f_up(mp, a, s):
    return _lup(mp, a, a / s) - _gc(mp, a)


def _f_temme(mp, a, eta):
    return _glg._temme(mp, _glg.ASPLIT_L / a, eta)


def _a_of(mp, kind, p):
    if kind == "u":
        return mp.expm1(p)
    if kind == "v":
        return ASPLIT / p
    return 1 / p


def _zones(mp):
    """(name, f(a, y), parameter kind, parameter box, argument box), with
    every box end a float so the runtime cuts are exactly the fitted ones."""
    lam_lo = float(_glg._lam_of_eta(mp, mp.mpf(ELO)))
    s_hi = float(1 / _glg._lam_of_eta(mp, mp.mpf(EHI)))
    u_lo = float(mp.log1p(ALO))
    u_1 = float(mp.log(2))
    u_10 = float(mp.log1p(ASPLIT))
    specs = (
        (_f_a0, "u", (u_lo, u_1), (0.0, 1.0)),
        (_f_b0, "u", (u_lo, u_1), (0.0, 1.0)),
        (_f_low, "u", (u_1, u_10), (0.0, lam_lo)),
        (_f_up, "u", (u_1, u_10), (0.0, s_hi)),
        (_f_low, "v", (0.0, 1.0), (0.0, lam_lo)),
        (_f_up, "v", (0.0, 1.0), (0.0, s_hi)),
        (_f_temme, "w", (0.0, 1.0), (ELO, EHI)),
    )
    return lam_lo, s_hi, [(n, *s) for n, s in zip(NAMES, specs)]


def _patch(mp, f, kind, pbox, ybox, n_nodes=None, a_of=_a_of):
    """C[k][j], the j-th parameter coefficient of the k-th argument
    coefficient of f on pbox x ybox; both fit stages in mp."""
    p0, p1 = (mp.mpf(v) for v in pbox)
    y0, y1 = (mp.mpf(v) for v in ybox)
    t = nodes(mp, NODES if n_nodes is None else n_nodes)
    rows = []
    for tp in t:
        a = a_of(mp, kind, p0 + (tp + 1) / 2 * (p1 - p0))
        rows.append(dct(mp, [f(mp, a, y0 + (ty + 1) / 2 * (y1 - y0)) for ty in t]))
    return param_fit(mp, rows)


def _outside(mp, c, ky, kp):
    """Sum of |c| outside the leading ky x kp block, summed directly: total
    minus kept cannot resolve 1e-15 of a total near 10 in float64."""
    n = len(c)
    return mp.fsum(
        abs(c[k][j]) for k in range(n) for j in range(n) if k >= ky or j >= kp
    )


def _refine(mp, f, kind, pbox, ybox, *, block=None, tol=None, n_nodes=None, a_of=_a_of):
    """Split pbox x ybox until every patch's coefficients outside its leading
    block (KY x KP by default) sum to at most tol (TOL); returns the
    parameter cuts, the argument cuts and the patches. The quantile generator
    passes its own block, tolerance, node count and parameter map."""
    ky, kp = (KY, KP) if block is None else block
    tol = TOL if tol is None else tol
    pc, yc, memo = list(pbox), list(ybox), {}

    def coefs(i, j):
        key = (pc[i], pc[i + 1], yc[j], yc[j + 1])
        if key not in memo:
            memo[key] = _patch(mp, f, kind, key[:2], key[2:], n_nodes, a_of)
        return memo[key]

    while True:
        worst = None
        for i in range(len(pc) - 1):
            for j in range(len(yc) - 1):
                drop = _outside(mp, coefs(i, j), ky, kp)
                if drop > tol and (worst is None or drop > worst[0]):
                    worst = (drop, i, j)
        if worst is None:
            break
        _, i, j = worst
        c = coefs(i, j)
        n = len(c)
        arg_tail = mp.fsum(abs(c[k][q]) for k in range(ky, n) for q in range(n))
        par_tail = mp.fsum(abs(c[k][q]) for k in range(n) for q in range(kp, n))
        if arg_tail >= par_tail:
            yc.insert(j + 1, (yc[j] + yc[j + 1]) / 2)
        else:
            pc.insert(i + 1, (pc[i] + pc[i + 1]) / 2)
    patches = [coefs(i, j) for i in range(len(pc) - 1) for j in range(len(yc) - 1)]
    return pc, yc, patches


def _block32(absc, tol=TOL32, block=None):
    """Smallest (ky, kp) within block (KY x KP by default) whose outside sum is
    <= tol (TOL32) on every patch."""
    best = (KY, KP) if block is None else block
    for ky in range(1, best[0] + 1):
        for kp in range(1, best[1] + 1):
            out = absc.sum(axis=(1, 2)) - absc[:, :ky, :kp].sum(axis=(1, 2))
            if np.all(out <= tol) and ky * kp < best[0] * best[1]:
                best = (ky, kp)
    return best


def zone_table(mp, name):
    """(parameter cuts, argument cuts, BLOCK blocks, |coefficients|) of one zone
    as main writes it, inside mp.workdps(DPS); the CI canary rebuilds U2 with it."""
    _, _, zones = _zones(mp)
    _, f, kind, pbox, ybox = next(z for z in zones if z[0] == name)
    pc, yc, patches = _refine(mp, f, kind, pbox, ybox, block=BLOCK)
    ky, kp = BLOCK
    blocks = [[[float(c[k][j]) for j in range(kp)] for k in range(ky)] for c in patches]
    absc = [[[float(abs(v)) for v in row] for row in c] for c in patches]
    return pc, yc, blocks, absc


def main(out_dir=None):
    import mpmath as mp

    with mp.workdps(DPS):
        lam_lo, s_hi, _ = _zones(mp)
        blocks, absc, pcuts, ycuts, npar, narg, offset = [], [], [], [], [], [], []
        for name in NAMES:
            pc, yc, zb, za = zone_table(mp, name)
            offset.append(len(blocks))
            npar.append(len(pc) - 1)
            narg.append(len(yc) - 1)
            pcuts.append(pc)
            ycuts.append(yc)
            blocks += zb
            absc += za
            print(f"{name}: {npar[-1]} x {narg[-1]} patches", flush=True)
    # float32 decides on 1e-8, far above float64's resolution of the sums
    ky32, kp32 = _block32(np.array(absc), block=BLOCK)
    # the runtime locates with the interior cuts and the last; pad by
    # repeating the last so every zone's row has the same length
    pad = max(map(len, pcuts)), max(map(len, ycuts))
    pcuts = [c + [c[-1]] * (pad[0] - len(c)) for c in pcuts]
    ycuts = [c + [c[-1]] * (pad[1] - len(c)) for c in ycuts]
    write_table_module(
        table_path(out_dir, __file__, "gammainc_array_table.py"),
        "chebax._src.recipes.gammainc_array_gen",
        "TABLE[n, k, j] = j-th parameter coefficient of the k-th argument "
        "coefficient of patch n; zone z holds patches OFFSET[z] + i NARG[z] + j "
        "over parameter interval i (PCUTS[z, i:i+2]) and argument interval j "
        "(YCUTS[z, j:j+2]), zones in NAMES order; float32 reads "
        "TABLE[:, :KY32, :KP32]",
        {
            "nodes": NODES,
            "tol": TOL,
            "tol32": TOL32,
            "alo": ALO,
            "asplit": ASPLIT,
            "elo": ELO,
            "ehi": EHI,
        },
        {
            "NAMES": NAMES,
            "ALO": ALO,
            "ASPLIT": ASPLIT,
            "LAM_LO": lam_lo,
            "LAM_HI": float(1 / mp.mpf(s_hi)),
            "NPAR": tuple(npar),
            "NARG": tuple(narg),
            "OFFSET": tuple(offset),
            "KY32": ky32,
            "KP32": kp32,
        },
        {"TABLE": np.array(blocks), "PCUTS": np.array(pcuts), "YCUTS": np.array(ycuts)},
    )


if __name__ == "__main__":
    main()
