"""Generator for the Marcum Q_1 table. Build-time only; mpmath inside.

Q = Q_1(a, b), the Marcum Q-function of order 1 at a, b >= 0, and P = 1 - Q
(the standard Rice distribution's CDF at b, shape a). With xi = a b and
d = b - a, every term below is positive:

  b >= a, rho = a/b:  Q = e^(-d^2/2) S_Q,  S_Q = e^(-xi) sum_{k>=0} rho^k I_k(xi)
  b <= a, rho = b/a:  P = e^(-d^2/2) S_P,  S_P = e^(-xi) sum_{k>=1} rho^k I_k(xi)

so the smaller tail is formed on each side (on b <= a P <= 1/2 always; on
b >= a Q is the smaller except next to b = a and at small b, see CORNER). The
generating function of e^(-xi) I_k(xi) with v = sqrt(xi) 2 sin(theta/2) gives

  S_Q = (1/pi) int_0^(2/eps) e^(-v^2/2) (eta + rho eps v^2/2)/(eta^2 + rho v^2)
        (1 - eps^2 v^2/4)^(-1/2) dv,
  S_P/rho: the same with (eta - eps v^2/2) in the numerator,

eps = xi^(-1/2), eta = (1 - rho)/eps = |d| sqrt(min(a, b)/max(a, b)). At
eps = 0 both are erfcx(eta/sqrt 2)/2. Each side has three charts, the P and Q
functions of a chart sharing its cuts:

  BULK    xi <= E1^2           (rho, xi)            ln S_Q,  ln(2 S_P/(rho xi))
  MEDIAN  xi > E1^2, eta < E1  (eps, eta)           ln(2 S/erfcx(eta/sqrt 2))
  TAIL    eta >= E1            (rho, t = 1/(1+eta)) the same as MEDIAN

with S = S_Q or S_P/rho in MEDIAN and TAIL.

eta <= sqrt(xi), so eta >= E1 implies xi >= E1^2 and the charts partition
each side. TAIL's corner (eps, t) -> (0, 0) holds every rho in [0, 1] (eta
enters at O(1), rho at O(eps^2)); in (rho, t) eps = (1 - rho) t/(1 - t) is a
smooth function of the coordinates there. MEDIAN's eta <= E1 keeps eps eta
<= 1. CORNER is the P side of b >= a at small b, where P is small and Q near
1: ln(2 P e^((a^2 + b^2)/2)/b^2), an entire function of (rho^2, b^2), on b <=
B_C = sqrt(2 ln 2), below which P <= 1/2. Above B_C P >= 0.31 on b >= a, so
1 - Q loses at most a bit or two there.

The sums use Miller's backward recurrence up to xi = 400 and the v-integral
above. They agree to 6e-48 at xi = 26 to 900 and rho = 0.05 to 1, and match
Q(a, a) = (1 + e^(-a^2) I_0(a^2))/2 to 8e-51.

Patches as in gammainc_array_gen, 12 x 12 blocks at TOL, except that a
chart's P and Q functions are refined together: a patch is split while
either function's coefficients outside the block sum above TOL. The cost is
at moderate xi, where the Bessel functions turn from series to asymptotic
behaviour (ln I_0 has branch points at xi = +-2.405i): at E1 = 2 the Q side
alone took 14 (BULK) + 18 (MEDIAN) + 36 (TAIL) patches, at E1 = 3 44 + 12 +
15 and at E1 = 5 85 + 12 + 4. A sqrt(xi) bulk axis did not help (90 at
E1 = 5). Regenerate with:

    python -m chebax._src.recipes.marcum_gen
"""

import math

import numpy as np

from chebax._src.recipes import gammainc_array_gen as _gag
from chebax._src.recipes._gen_common import DPS, table_path, write_table_module

E1 = 2.0  # seams: xi = E1^2 (BULK), eta = E1 (MEDIAN | TAIL)
B_C = math.sqrt(2.0 * math.log(2.0))  # CORNER: b <= B_C on b >= a
TOL32 = 5e-8  # float32 block: under float32 rounding (6e-8)
XI_SERIES = 400  # series below, v-integral above
NAMES = ("BULK", "MEDIAN", "TAIL", "CORNER")
SIDES = ("Q", "P")  # CORNER has the P function only
# (parameter cuts, argument cuts) found by refining from each chart's box;
# the refinement starts from them and still splits any patch above TOL
BULK_CUTS = ((0.0, 0.5, 1.0), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, E1 * E1))
MEDIAN_CUTS = (
    (0.0, 0.125, 0.25, 0.3125, 0.375, 0.4375, 1.0 / E1),
    (0.0, 0.5, 1.0, E1),
)
TAIL_CUTS = (
    (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
    (0.0, 1 / 12, 1 / 6, 5 / 24, 1 / 4, 7 / 24, 1.0 / (1.0 + E1)),
)
CORNER_CUTS = ((0.0, 1.0), (0.0, B_C * B_C / 2, B_C * B_C))


def _ie(mp, xi, n):
    """e^(-xi) I_k(xi) for k = 0..n, Miller's backward recurrence normalized by
    e^xi = I_0 + 2 sum I_k (to every k with e^(-k^2/(2 xi)) above 1e-60)."""
    top = int(n + 60 + 17 * mp.sqrt(xi))
    y = [mp.mpf(0)] * (top + 2)
    y[top] = mp.mpf(10) ** -30
    for k in range(top, 0, -1):
        y[k - 1] = y[k + 1] + 2 * k / xi * y[k]
        if y[k - 1] > mp.mpf(10) ** 30:
            y = [v * mp.mpf(10) ** -60 for v in y]
    norm = y[0] + 2 * mp.fsum(y[1:])
    return [v / norm for v in y[: n + 1]]


def _s_series(mp, side, xi, rho):
    """S_Q, or S_P/rho, from the series."""
    k = 10
    while k * -mp.log(rho) + k * k / (2 * (xi + 1)) < 150:
        k += 10
    ie = _ie(mp, xi, k + 1)
    if side == "Q":
        return mp.fsum(rho**j * v for j, v in enumerate(ie))
    return mp.fsum(rho ** (j - 1) * v for j, v in enumerate(ie) if j >= 1)


def _s_v(mp, side, eps, eta, rho):
    """S_Q, or S_P/rho, from the v-integral; past v = 15 e^(-v^2/2) < 1e-48."""
    c = rho if side == "Q" else -1

    def g(v):
        return (
            mp.exp(-v * v / 2)
            * (eta + c * eps * v * v / 2)
            / (eta**2 + rho * v * v)
            / mp.sqrt(1 - (eps * v / 2) ** 2)
        )

    top = min(2 / eps, mp.mpf(15))
    if eta == 0:
        # rho = 1, and eta/(eta^2 + v^2) is a half delta: S = 1/2 +- the eps/2 part
        def g0(v):
            return mp.exp(-v * v / 2) * eps / 2 / mp.sqrt(1 - (eps * v / 2) ** 2)

        part = mp.quad(g0, [0, 1, 4, top]) / mp.pi
        return mp.mpf(1) / 2 + (part if side == "Q" else -part)
    pts = {mp.mpf(p) for p in (eta / 8, eta / 2, eta, 2 * eta, 0.5, 1, 2, 4, 8)}
    return mp.quad(g, [0] + sorted(p for p in pts if 0 < p < top) + [top]) / mp.pi


def _s(mp, side, eps, eta, rho):
    xi = 1 / (eps * eps)
    if xi <= XI_SERIES:
        return _s_series(mp, side, xi, rho)
    return _s_v(mp, side, eps, eta, rho)


def _ln_erfcx(mp, z):
    return mp.log(mp.erfc(z)) + z * z


def _f_bulk(mp, side, rho, xi):
    if side == "Q":
        return mp.log(_s_series(mp, side, xi, rho))
    return mp.log(2 * _s_series(mp, side, xi, rho) / xi)


def _f_median(mp, side, eps, eta):
    s = _s(mp, side, eps, eta, 1 - eps * eta)
    return mp.log(2 * s) - _ln_erfcx(mp, eta / mp.sqrt(2))


def _f_tail(mp, side, rho, t):
    eta = (1 - t) / t
    s = _s(mp, side, (1 - rho) / eta, eta, rho)
    return mp.log(2 * s) - _ln_erfcx(mp, eta / mp.sqrt(2))


def _f_corner(mp, side, r2, b2):
    """sum_{k>=1, m>=0} b^(2(k+2m-1)) rho^(2m)/(2^(k+2m-1) m! (k+m)!)."""
    tiny = mp.mpf(10) ** -60
    s, k = mp.mpf(0), 1
    while True:
        sk, m = mp.mpf(0), 0
        while True:
            e = k + 2 * m - 1
            term = b2**e * r2**m / (2**e * mp.factorial(m) * mp.factorial(k + m))
            sk += term
            if term < tiny * (sk + s):
                break
            m += 1
        s += sk
        if sk < tiny * s:
            return mp.log(s)
        k += 1


def _charts():
    """(name, f(mp, side, p, y), sides, parameter cuts, argument cuts)."""
    return (
        ("BULK", _f_bulk, SIDES, *BULK_CUTS),
        ("MEDIAN", _f_median, SIDES, *MEDIAN_CUTS),
        ("TAIL", _f_tail, SIDES, *TAIL_CUTS),
        ("CORNER", _f_corner, ("P",), *CORNER_CUTS),
    )


def _refine_shared(mp, f, sides, pbox, ybox):
    """gammainc_array_gen._refine for the functions f(mp, side, p, y) of every
    side on one set of cuts: the worst patch over all sides is split, along
    the axis where its coefficient tail is larger."""
    pc, yc, memo = list(pbox), list(ybox), {}

    def coefs(side, i, j):
        key = (side, pc[i], pc[i + 1], yc[j], yc[j + 1])
        if key not in memo:
            memo[key] = _gag._patch(
                mp,
                lambda mp_, p, y: f(mp_, side, p, y),
                "id",
                key[1:3],
                key[3:],
                a_of=lambda mp_, kind, p: p,
            )
        return memo[key]

    ky, kp = _gag.KY, _gag.KP
    while True:
        worst = None
        for side in sides:
            for i in range(len(pc) - 1):
                for j in range(len(yc) - 1):
                    drop = _gag._outside(mp, coefs(side, i, j), ky, kp)
                    if drop > _gag.TOL and (worst is None or drop > worst[0]):
                        worst = (drop, side, i, j)
        if worst is None:
            break
        _, side, i, j = worst
        c = coefs(side, i, j)
        n = len(c)
        arg_tail = mp.fsum(abs(c[k][q]) for k in range(ky, n) for q in range(n))
        par_tail = mp.fsum(abs(c[k][q]) for k in range(n) for q in range(kp, n))
        if arg_tail >= par_tail:
            yc.insert(j + 1, (yc[j] + yc[j + 1]) / 2)
        else:
            pc.insert(i + 1, (pc[i] + pc[i + 1]) / 2)
    patches = {
        s: [coefs(s, i, j) for i in range(len(pc) - 1) for j in range(len(yc) - 1)]
        for s in sides
    }
    return pc, yc, patches


def main(out_dir=None, charts=None):
    import mpmath as mp

    with mp.workdps(DPS):
        blocks, absc, pcuts, ycuts, npar, narg, offset = [], [], [], [], [], [], []
        for name, f, sides, pbox, ybox in _charts():
            if charts is not None and name not in charts:
                continue
            pc, yc, patches = _refine_shared(mp, f, sides, pbox, ybox)
            offset.append(len(blocks))
            npar.append(len(pc) - 1)
            narg.append(len(yc) - 1)
            pcuts.append(pc)
            ycuts.append(yc)
            for side in sides:
                for c in patches[side]:
                    blocks.append(
                        [
                            [float(c[k][j]) for j in range(_gag.KP)]
                            for k in range(_gag.KY)
                        ]
                    )
                    absc.append([[float(abs(v)) for v in row] for row in c])
            print(f"{name}: {npar[-1]} x {narg[-1]} patches per side", flush=True)
    if charts is not None:
        return
    ky32, kp32 = _gag._block32(np.array(absc), TOL32)
    pad = max(map(len, pcuts)), max(map(len, ycuts))
    pcuts = [c + [c[-1]] * (pad[0] - len(c)) for c in pcuts]
    ycuts = [c + [c[-1]] * (pad[1] - len(c)) for c in ycuts]
    write_table_module(
        table_path(out_dir, __file__, "marcum_table.py"),
        "chebax._src.recipes.marcum_gen",
        "TABLE[n, k, j] = j-th parameter coefficient of the k-th argument "
        "coefficient of patch n; chart z, side s (0 Q, 1 P) holds patches "
        "OFFSET[z] + s NPAR[z] NARG[z] + i NARG[z] + j over parameter interval i "
        "(PCUTS[z, i:i+2]) and argument interval j (YCUTS[z, j:j+2]), charts in "
        "NAMES order: BULK over (rho, xi), MEDIAN over (eps, eta), TAIL over "
        "(rho, t), CORNER (P only) over (rho^2, b^2); float32 reads "
        "TABLE[:, :KY32, :KP32]",
        {
            "nodes": _gag.NODES,
            "tol": _gag.TOL,
            "tol32": TOL32,
            "e1": E1,
            "b_c": B_C,
            "xi_series": XI_SERIES,
        },
        {
            "NAMES": NAMES,
            "E1": E1,
            "B_C": B_C,
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
