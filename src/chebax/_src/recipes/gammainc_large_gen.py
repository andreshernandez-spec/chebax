"""Generator for the large-a gammainc tables. Build-time only; mpmath inside.

Extends the gammainc recipe from a in [0.1, 10] to (10, 1000] with
three small 2-D tables on v = 10/a in [0.01, 1], zoned by
lambda = x/a (degrees measured in experiments/14):

- Temme transition zone, eta in [-0.7, 2.6]:
      eta(lambda) = sign(lambda - 1) sqrt(2 (lambda - 1 - ln lambda))
      Q = 1/2 erfc(eta sqrt(a/2)) + e^(-a eta^2/2) / sqrt(2 pi a) * T
  with T(v, eta) tabulated (T -> -1/3 at eta = 0, a -> inf). The
  complement is computed the same way on its own erfc side,
  P = 1/2 erfc(-eta sqrt(a/2)) - correction, so BOTH tails keep
  relative accuracy. eta-degree 20, v-degree 7: the whole transition
  of the incomplete gamma for a up to 1000 fits in a 26x10 table.
- Lower zone, lambda in [0, 1/2] (eta(0.5) = -0.63 > -0.7):
      D(v, lambda) = ln 1F1(1; a+1; a lambda)
  the existing inner kernel in scaled coordinates;
  P = exp(a ln x - x - lnGamma(a+1) + D).
- Upper zone, s = (a-1)/x in [0, 1/6] (eta(5.4) = 2.42 < 2.6):
      U(v, s) = ln[ Gamma(a, x) x^(1-a) e^x ]
  the existing tail kernel in the uniform variable s;
  Q = exp((a-1) ln x - x - lnGamma(a) + U).

The box stops at a = 1000 (chi-squared: 2000 dof) because the 40-dps
reference evaluation does, not the representation: mpmath's gammainc
diverges near lambda ~ 1 for a >~ 1e5, and references must be built on
the small side of the erfc split (see experiments/14; the naive
subtraction is catastrophic past ~92 nats). All ~1.3k coefficients
regenerate in seconds, so the bit-for-bit test covers the whole module
with no canary split. Regenerate with:

    python -m chebax._src.recipes.gammainc_large_gen
"""

from chebax._src.recipes._gen_common import (DPS, dct, nodes, table_path,
                                             write_table_module)

ASPLIT_L, AHI_L = 10.0, 1000.0
VLO, VHI = 0.01, 1.0    # v = ASPLIT_L / a
ELO, EHI = -0.7, 2.6
LHI = 0.5               # lower zone: lambda in [0, LHI]
SHI = 1.0 / 6.0         # upper zone: s = (a-1)/x in [0, SHI]
NE = 26   # eta-nodes (worst measured 20)
NV_T = 10  # v-nodes, Temme (worst measured 7)
NL = 24   # lambda-nodes (worst measured 18)
NV_D = 34  # v-nodes, lower (worst measured 28)
NS = 16   # s-nodes (worst measured 11)
NV_U = 12  # v-nodes, upper (worst measured 8)


def _lam_of_eta(mp, eta):
    if eta == 0:
        return mp.mpf(1)
    z = -mp.exp(-1 - eta * eta / 2)
    lam = -mp.lambertw(z, 0 if eta < 0 else -1)
    return mp.mpf(lam.real)


def _hyp1f1_1(mp, a1, x):
    # 1F1(1; a1; x) by direct series (geometric ratio x/(a1+n));
    # mpf exponents keep the surrounding tiny magnitudes harmless
    s = t = mp.mpf(1)
    n = 0
    while abs(t) > mp.mpf(10) ** (-DPS - 10) * abs(s):
        t *= x / (a1 + n)
        s += t
        n += 1
    return s


def _temme(mp, v, eta):
    a = ASPLIT_L / v
    lam = _lam_of_eta(mp, eta)
    x = a * lam
    y = eta * mp.sqrt(a / 2)
    if eta < 0:
        # small side: R = erfc(-y)/2 - P, both ~ e^(-a eta^2/2)
        p = (mp.exp(a * mp.log(x) - x - mp.loggamma(a + 1))
             * _hyp1f1_1(mp, a + 1, x))
        r = mp.erfc(-y) / 2 - p
    else:
        q = mp.gammainc(a, x, mp.inf, regularized=True)
        r = q - mp.erfc(y) / 2
    return r * mp.sqrt(2 * mp.pi * a) * mp.exp(a * eta * eta / 2)


def _lower(mp, v, lam):
    a = ASPLIT_L / v
    return mp.log(_hyp1f1_1(mp, a + 1, a * lam))


def _upper(mp, v, s):
    a = ASPLIT_L / v
    x = (a - 1) / s
    return (mp.log(mp.gammainc(a, x, mp.inf)) + x + (1 - a) * mp.log(x))


def _fit2d(mp, f, narg, nv, arg_lo, arg_hi):
    """T[k, j] = j-th v-coefficient of the k-th argument coefficient
    (the gammainc_table convention); both fit stages in mp."""
    lo, hi = mp.mpf(arg_lo), mp.mpf(arg_hi)
    vlo, vhi = mp.mpf(VLO), mp.mpf(VHI)
    rows = []
    for tv in nodes(mp, nv):
        v = vlo + (tv + 1) / 2 * (vhi - vlo)
        rows.append(dct(mp, [f(mp, v, lo + (t + 1) / 2 * (hi - lo))
                             for t in nodes(mp, narg)]))
    import numpy as np
    out = np.empty((narg, nv))
    for k in range(narg):
        col = dct(mp, [rows[j][k] for j in range(nv)])
        for j in range(nv):
            out[k, j] = float(col[j])
    return out


def main(out_dir=None):
    import mpmath as mp

    with mp.workdps(DPS):
        temme = _fit2d(mp, _temme, NE, NV_T, ELO, EHI)
        lower = _fit2d(mp, _lower, NL, NV_D, 0.0, LHI)
        upper = _fit2d(mp, _upper, NS, NV_U, 0.0, SHI)
    write_table_module(
        table_path(out_dir, __file__, "gammainc_large_table.py"),
        "chebax._src.recipes.gammainc_large_gen",
        "T[k, j] = j-th v-coefficient of the k-th argument coefficient, "
        "v = ASPLIT/a in [VLO, 1]; TABLE_TEMME has argument eta in "
        "[ELO, EHI], TABLE_LOW argument lambda = x/a in [0, LHI], "
        "TABLE_UP argument s = (a-1)/x in [0, SHI]",
        {"ne": NE, "nv_t": NV_T, "nl": NL, "nv_d": NV_D, "ns": NS,
         "nv_u": NV_U, "asplit": ASPLIT_L, "ahi": AHI_L},
        {"ASPLIT": ASPLIT_L, "AHI": AHI_L, "VLO": VLO, "ELO": ELO,
         "EHI": EHI, "LHI": LHI, "SHI": SHI},
        {"TABLE_TEMME": temme, "TABLE_LOW": lower, "TABLE_UP": upper})


if __name__ == "__main__":
    main()
