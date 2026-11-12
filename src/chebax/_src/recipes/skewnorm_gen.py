"""Generator for the skew-normal log-CDF table. Build-time only; mpmath inside.

Every case of skewnorm.py reduces to the lower tail at a shape b >= 0,

  K(s, b) = F(-s; b) = (1/pi) int_b^inf exp(-s^2 (1 + t^2)/2) / (1 + t^2) dt,

s >= 0, which is Phi(-s) - 2 T(s, b) without the cancellation. With sigma = s
sqrt(1 + b^2)/sqrt(2) (sigma^2 is K's exponent), delta = b/sqrt(1 + b^2) and c0
= 1/sqrt(1 + b^2), two charts:

  bulk  sigma <= S1:  K = e^(-sigma^2) (c0/pi) N(sigma, delta),
        N = int_0^1 exp(-sigma^2 (1/y^2 - 1)) / sqrt(1 - c0^2 y^2) dy
  tail  sigma > S1:   N = sqrt(pi) erfcx(mu) R(mu, eps) / (2 sigma),
        mu = delta sigma = b s/sqrt(2), eps = 1/sigma^2, R = J(mu^2, eps)/J(mu^2, 0),
        J(lam, eps) = int_0^inf e^(-w) (lam + w)^(-1/2) (1 + eps w)^(-1) dw

(N from t = tan(theta), y = cos(theta)/c0; the tail from w = s^2 (t^2 - b^2)/2,
whose J(mu^2, 0) is sqrt(pi) erfcx(mu)). The table holds ln N on (delta, sigma)
in [0, 1] x [0, S1] and ln R on (eps, t) in [0, 1/S1^2] x [0, 1], t = 1/(1 +
mu). N is sampled in theta, N = (1/c0) int_theta0^(pi/2) exp(-sigma^2
(c0^2/cos^2 theta - 1)) dtheta, sin(theta0) = delta, smooth on the whole
interval and twice as fast to integrate as the y-form, whose 1/sqrt(1 - c0^2
y^2) turns at y ~ 1 - delta^2/2.

N is analytic in sigma, not sigma^2: e^(-sigma^2/y^2) near y = 0 gives odd
powers of sigma. In delta its nearest singularity is delta = -1 (N(0, delta) =
arccos(delta)/sqrt(1 - delta^2)): half of [0, 1] converges 12-17x per term at
sigma = 0 and 9-13x at sigma = 3, as the delta-dependence sharpens like
erfcx(delta sigma), which is what ends the bulk at S1. R -> 1 as eps -> 0 and is
C-infinity but not analytic there (its eps-series has E[W^k] ~ k! coefficients),
so the tail's patches split toward eps = 0; mu -> inf is analytic in t, R(inf,
eps) = e^(1/eps) E1(1/eps)/eps, and J's integrand turns at v = mu (w = v^2),
where it has break points. Both quadratures agree with 70 dps to 2e-38, worst
next to delta = 1. experiments/03 (skewnorm) measures these.

Patches as in gammainc_array_gen: each chart starts as one patch fitted at NODES
x NODES nodes and is halved along the axis with the larger tail until the
coefficients outside the leading KY x KP block sum to at most TOL, an absolute
bound on ln N and ln R, so a relative bound on K. S1 = 3 needs the fewest, 46 at
12 x 12 (BULK 5 x 5, TAIL 3 x 7, parameter x argument intervals), against 51 at
S1 = 2 (16 + 35; experiments/03 skewnorm --seams); a lower seam moves patches
from bulk to tail. Halving from one box took 16 min, half of it on patches later
split, so the refinement now starts from the cuts it found (BULK_CUTS,
TAIL_CUTS, dyadic in the box ends): it checks every patch against TOL and would
still split one that fails. float32 reads the smallest leading block within
TOL32 on every patch. Regenerate with:

    python -m chebax._src.recipes.skewnorm_gen      # ~5 min
"""

import numpy as np

from chebax._src.recipes import gammainc_array_gen as _gag
from chebax._src.recipes._gen_common import DPS, table_path, write_table_module

S1 = 3.0  # bulk/tail seam in sigma
NAMES = ("BULK", "TAIL")
_E1 = 1.0 / (S1 * S1)
# (parameter cuts, argument cuts): BULK (delta, sigma), TAIL (eps, t)
BULK_CUTS = ((0.0, 0.125, 0.25, 0.5, 0.75, 1.0), (0.0, 0.375, 0.75, 1.5, 2.25, S1))
TAIL_CUTS = (
    (0.0, _E1 / 4, _E1 / 2, _E1),
    (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0),
)


def _break_points(mp, top, *cands):
    pts = sorted({mp.mpf(c) for c in cands if 0 < c < top})
    return [mp.mpf(0)] + pts + [top]


def _log_n(mp, dl, sig):
    """ln N(sigma, delta) in theta, c0^2/cos^2 - 1 as sin(t - t0) sin(t + t0)/cos^2
    so it does not cancel next to theta0."""
    th0 = mp.asin(dl)
    c0 = mp.sqrt((1 - dl) * (1 + dl))

    def g(t):
        return mp.exp(-sig * sig * mp.sin(t - th0) * mp.sin(t + th0) / mp.cos(t) ** 2)

    return mp.log(mp.quad(g, [th0, mp.pi / 2]) / c0)


def _log_r(mp, eps, t):
    """ln R(mu, eps) at mu = (1 - t)/t, J with w = v^2."""
    mu = (1 - t) / t
    lam = mu * mu

    def g(v):
        return mp.exp(-v * v) * 2 * v / mp.sqrt(lam + v * v) / (1 + eps * v * v)

    pts = _break_points(mp, mp.mpf(6), mu / 2, mu, 2 * mu, 2) + [mp.inf]
    j0 = mp.sqrt(mp.pi) * mp.erfc(mu) * mp.exp(lam)
    return mp.log(mp.quad(g, pts) / j0)


def _ident(mp, kind, p):
    return p


def _charts():
    """(name, f(p, y), parameter cuts, argument cuts), cuts floats."""
    return (("BULK", _log_n, *BULK_CUTS), ("TAIL", _log_r, *TAIL_CUTS))


def main(out_dir=None):
    import mpmath as mp

    with mp.workdps(DPS):
        blocks, absc, pcuts, ycuts, npar, narg, offset = [], [], [], [], [], [], []
        for name, f, pbox, ybox in _charts():
            pc, yc, patches = _gag._refine(mp, f, "id", pbox, ybox, a_of=_ident)
            offset.append(len(blocks))
            npar.append(len(pc) - 1)
            narg.append(len(yc) - 1)
            pcuts.append(pc)
            ycuts.append(yc)
            for c in patches:
                blocks.append(
                    [[float(c[k][j]) for j in range(_gag.KP)] for k in range(_gag.KY)]
                )
                absc.append([[float(abs(v)) for v in row] for row in c])
            print(f"{name}: {npar[-1]} x {narg[-1]} patches", flush=True)
    ky32, kp32 = _gag._block32(np.array(absc))
    pad = max(map(len, pcuts)), max(map(len, ycuts))
    pcuts = [c + [c[-1]] * (pad[0] - len(c)) for c in pcuts]
    ycuts = [c + [c[-1]] * (pad[1] - len(c)) for c in ycuts]
    write_table_module(
        table_path(out_dir, __file__, "skewnorm_table.py"),
        "chebax._src.recipes.skewnorm_gen",
        "TABLE[n, k, j] = j-th parameter coefficient of the k-th argument "
        "coefficient of patch n; chart z holds patches OFFSET[z] + i NARG[z] + j "
        "over parameter interval i (PCUTS[z, i:i+2]) and argument interval j "
        "(YCUTS[z, j:j+2]), charts in NAMES order: BULK ln N over (delta, sigma), "
        "TAIL ln R over (eps, t); float32 reads TABLE[:, :KY32, :KP32]",
        {
            "nodes": _gag.NODES,
            "tol": _gag.TOL,
            "tol32": _gag.TOL32,
            "s1": S1,
        },
        {
            "NAMES": NAMES,
            "S1": S1,
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
