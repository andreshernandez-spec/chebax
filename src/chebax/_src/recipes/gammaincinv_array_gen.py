"""Generator for the per-element Gamma quantile table. Build-time only;
mpmath inside.

gammaincinv_array used to run 8 Halley steps on gammainc_array's ln P and
ln Q, one table patch per step. This table holds the quantile itself to
1e-8, found by root finding in mpmath at the nodes, so the runtime reads one
patch and takes one Newton step in float64 (none in float32). The zones
follow the quantile's structure; the tabulated function is a correction to
a closed form in each (experiments/25 chose the coordinates, 26 proved the
start), with the parameter panel in u = ln(1 + a) below a = ASPLIT, v =
ASPLIT/a above, and w = 1/a for the centre, every box ending at a = inf:

  lower   s = (p Gamma(a+1))^(1/a) / (a+1) <= S (S_SMALL below a = 1,
          S_BIG above): x = (a+1) s (1 + rho(s, a)); P = x^a/Gamma(a+1)
          (1 - a x/(a+1) + ...) makes rho analytic with rho(0, a) = 0,
          and for large a, s -> lambda e^-lambda, singular only at
          lambda = 1 (s = 1/e), so S_BIG keeps clear of it
  centre  a >= 1 and eta0 = ndtri(p)/sqrt(a) <= EHI: Temme's inversion
          eta = eta0 + w eps(eta0, w), the table holds eps = (eta - eta0)/w,
          O(1) down to w = 0; then ln lambda(eta) = eta g(eta) from the
          1-D series LAM on LAM_BAND, g = 1 - eta/3 + ..., relatively
          accurate at eta -> 0, where x = a (1 + eta + ...) at large a
  upper   the rest: x_W from x - (a-1) ln x = -ln q - lnGamma(a), the
          Lambert-W-type leading part (four Newton steps on that elementary
          equation at run time), tau = (a+1)/x_W <= TAU_HI per panel, and
          x = x_W (1 + delta(tau, a))

Each zone starts as one patch, fitted at NODES x NODES nodes, and is halved
along the axis holding the larger tail until the coefficients outside its
leading KY x KP block sum to under TOL (gammainc_array_gen's refinement
with this block and tolerance). The runtime locates one patch per element
from the zone's cuts. Regenerate with:

    python -m chebax._src.recipes.gammaincinv_array_gen      # ~2.5 min
"""

import math

import numpy as np

from chebax._src.recipes import gammainc_array_gen as _gag
from chebax._src.recipes import gammainc_large_gen as _glg
from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    table_path,
    write_table_module,
)

ALO = _gag.ALO
ASPLIT = _gag.ASPLIT
ELO, EHI = _glg.ELO, _glg.EHI
NODES = 14
KY, KP = 10, 10
TOL = 1e-8
S_SMALL, S_BIG = 0.4, 0.3
# tau = (a+1)/x_W per upper panel: at a >= 10 tau = 0.5 is lambda ~ 2, past
# the centre's EHI end; below a = 1 the lower and upper zones meet at 1.5
TAU_HI = (1.5, 1.0, 0.5)
# eta band of the ln lambda(eta) series: the centre's eta0 plus w eps
LAM_BAND = (-1.1, 3.1)
LAM_NODES = 40
LAM_TOL = 1e-12
NAMES = ("LO0", "LO1", "LO2", "CE1", "CE2", "UP0", "UP1", "UP2")


def _log_p(mp, a, x):
    if x > a + 1:
        return mp.log1p(-mp.exp(_glg._log_q(mp, a, x)))
    return a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_glg._hyp1f1_1(mp, a + 1, x))


def _log_q(mp, a, x):
    if x > a + 1:
        return _glg._log_q(mp, a, x)
    lp = a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_glg._hyp1f1_1(mp, a + 1, x))
    return mp.log1p(-mp.exp(lp))


def _solve(mp, f, target, x0):
    """x with f(x) = target, secant in ln x from x0, bisection if it strays."""
    g = lambda t: f(mp.exp(t)) - target
    t0 = mp.log(x0)
    try:
        t = mp.findroot(
            g, (t0, t0 + mp.mpf("1e-6")), solver="secant", tol=1e-30, maxsteps=60
        )
        if abs(g(t)) < mp.mpf("1e-25") * max(1, abs(target)):
            return mp.exp(t)
    except (ValueError, ZeroDivisionError):
        pass
    lo, hi = t0 - 1, t0 + 1
    for _ in range(12):
        if g(lo) * g(hi) <= 0:
            break
        lo, hi = lo - 2 * (t0 - lo), hi + 2 * (hi - t0)
    else:
        raise ValueError(f"no bracket around ln x = {t0}")
    for _ in range(140):
        m = (lo + hi) / 2
        if g(m) * g(lo) > 0:
            lo = m
        else:
            hi = m
    return mp.exp((lo + hi) / 2)


def _f_lower(mp, a, s):
    y = s * (a + 1)
    ltp = a * mp.log(y) - mp.loggamma(a + 1)
    return _solve(mp, lambda x: _log_p(mp, a, x), ltp, y) / y - 1


def _f_centre(mp, a, eta0):
    z = eta0 * mp.sqrt(a)
    c = z / (3 * mp.sqrt(a)) - 1 / (9 * a)
    x0 = a * c**3 if c > 0 else a * mp.exp(3 * c)
    if z < 0:
        x = _solve(mp, lambda x: _log_p(mp, a, x), mp.log(mp.ncdf(z)), x0)
    else:
        x = _solve(mp, lambda x: _log_q(mp, a, x), mp.log(mp.ncdf(-z)), x0)
    lam = x / a
    eta = mp.sign(lam - 1) * mp.sqrt(2 * (lam - 1 - mp.log(lam)))
    return (eta - eta0) * a


def _f_upper(mp, a, tau):
    xw = (a + 1) / tau
    r = xw - (a - 1) * mp.log(xw)
    ltq = -r - mp.loggamma(a)
    return _solve(mp, lambda x: _log_q(mp, a, x), ltq, xw) / xw - 1


def _a_of(mp, kind, p):
    if kind == "u":
        return mp.expm1(p)
    if kind == "v":
        return ASPLIT / p
    return 1 / p


def _zones(mp):
    """(name, f(a, y), parameter kind, parameter box, argument box), box
    ends as floats so the runtime's cuts are the fitted ones."""
    u_lo = float(mp.log1p(ALO))
    u_1 = float(mp.log(2))
    u_10 = float(mp.log1p(ASPLIT))
    specs = (
        (_f_lower, "u", (u_lo, u_1), (0.0, S_SMALL)),
        (_f_lower, "u", (u_1, u_10), (0.0, S_BIG)),
        (_f_lower, "v", (0.0, 1.0), (0.0, S_BIG)),
        (_f_centre, "w", (1.0 / ASPLIT, 1.0), (ELO, EHI)),
        (_f_centre, "w", (0.0, 1.0 / ASPLIT), (ELO, EHI)),
        (_f_upper, "u", (u_lo, u_1), (0.0, TAU_HI[0])),
        (_f_upper, "u", (u_1, u_10), (0.0, TAU_HI[1])),
        (_f_upper, "v", (0.0, 1.0), (0.0, TAU_HI[2])),
    )
    return [(n, *s) for n, s in zip(NAMES, specs)]


def _lam_series(mp):
    """Coefficients of g(eta) = ln lambda(eta) / eta on LAM_BAND, trimmed."""
    lo, hi = (mp.mpf(v) for v in LAM_BAND)
    vals = []
    for t in nodes(mp, LAM_NODES):
        eta = lo + (t + 1) / 2 * (hi - lo)
        vals.append(mp.log(_glg._lam_of_eta(mp, eta)) / eta)
    c = np.array([float(v) for v in dct(mp, vals)])
    keep = int(np.nonzero(np.abs(c) > LAM_TOL)[0].max()) + 1
    return c[:keep]


def main(out_dir=None):
    import time

    import mpmath as mp

    t0 = time.time()
    with mp.workdps(DPS):
        blocks, pcuts, ycuts, npar, narg, offset = [], [], [], [], [], []
        for name, f, kind, pbox, ybox in _zones(mp):
            pc, yc, patches = _gag._refine(
                mp,
                f,
                kind,
                pbox,
                ybox,
                block=(KY, KP),
                tol=TOL,
                n_nodes=NODES,
                a_of=_a_of,
            )
            offset.append(len(blocks))
            npar.append(len(pc) - 1)
            narg.append(len(yc) - 1)
            pcuts.append(pc)
            ycuts.append(yc)
            for c in patches:
                blocks.append([[float(c[k][j]) for j in range(KP)] for k in range(KY)])
            print(
                f"{name}: {npar[-1]} x {narg[-1]} patches  [{time.time() - t0:.0f} s]",
                flush=True,
            )
        lam = _lam_series(mp)
    pad = max(map(len, pcuts)), max(map(len, ycuts))
    pcuts = [c + [c[-1]] * (pad[0] - len(c)) for c in pcuts]
    ycuts = [c + [c[-1]] * (pad[1] - len(c)) for c in ycuts]
    write_table_module(
        table_path(out_dir, __file__, "gammaincinv_array_table.py"),
        "chebax._src.recipes.gammaincinv_array_gen",
        "TABLE[n, k, j] = j-th parameter coefficient of the k-th argument "
        "coefficient of patch n; zone z holds patches OFFSET[z] + i NARG[z] + j "
        "over parameter interval i (PCUTS[z, i:i+2]) and argument interval j "
        "(YCUTS[z, j:j+2]), zones in NAMES order; LAM[k] = k-th coefficient of "
        "ln lambda(eta) / eta on LAM_BAND",
        {
            "nodes": NODES,
            "tol": TOL,
            "alo": ALO,
            "asplit": ASPLIT,
            "elo": ELO,
            "ehi": EHI,
            "s_small": S_SMALL,
            "s_big": S_BIG,
            "tau_hi": TAU_HI,
            "lam_band": LAM_BAND,
            "lam_tol": LAM_TOL,
        },
        {
            "NAMES": NAMES,
            "ALO": ALO,
            "ASPLIT": ASPLIT,
            "ELO": ELO,
            "EHI": EHI,
            "S_SMALL": S_SMALL,
            "S_BIG": S_BIG,
            "TAU_HI": TAU_HI,
            "LAM_BAND": LAM_BAND,
            "NPAR": tuple(npar),
            "NARG": tuple(narg),
            "OFFSET": tuple(offset),
        },
        {
            "TABLE": np.array(blocks),
            "PCUTS": np.array(pcuts),
            "YCUTS": np.array(ycuts),
            "LAM": lam,
        },
    )


if __name__ == "__main__":
    main()
