"""Generator for the large-a gammainc tables. Build-time only; mpmath inside.

a > 10 through three 2-D tables on v = 10/a in [0, 1], zoned by lambda =
x/a (experiments/14 and 15). Temme's zone, eta = sign(lambda - 1) sqrt(2
(lambda - 1 - ln lambda)) in [-0.7, 2.6]: Q = erfc(eta sqrt(a/2))/2 +
e^(-a eta^2/2) T(v, eta)/sqrt(2 pi a), and P the same on its own erfc side
(eta-degree 20, v-degree 7). Lower, lambda in [0, 1/2]: D(v, lambda) =
ln 1F1(1; a+1; a lambda). Upper, s = (a-1)/x in [0, 1/6]: U(v, s) =
ln[Gamma(a, x) x^(1-a) e^x]. v = 0 is a regular point of all three, so the
box has no upper end. References: Legendre's continued fraction for Q and
the 1F1 series for P, each on its converging side, with the erfc split on
its small side (mpmath's gammainc stalls near lambda = 1 past a ~ 1e5; the
naive subtraction fails past ~92 nats). Regenerate with:

    python -m chebax._src.recipes.gammainc_large_gen
"""

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    table_path,
    write_table_module,
)

ASPLIT_L = 10.0
VLO, VHI = 0.0, 1.0  # v = ASPLIT_L / a, v = 0 being a = inf
ELO, EHI = -0.7, 2.6
LHI = 0.5  # lower zone: lambda in [0, LHI]
SHI = 1.0 / 6.0  # upper zone: s = (a-1)/x in [0, SHI]
NE = 26  # eta-nodes (worst measured 20)
NV_T = 10  # v-nodes, Temme (worst measured 7)
NL = 24  # lambda-nodes (worst measured 18)
NV_D = 36  # v-nodes, lower (worst measured 29 on [0, 1])
NS = 16  # s-nodes (worst measured 11)
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


def _log_q(mp, a, x):
    """ln Q(a, x) by Legendre's continued fraction (modified Lentz), for
    x > a. mpmath's own gammainc gives up out here; see experiments/16."""
    tiny = mp.mpf(10) ** (-2 * DPS)
    eps = mp.mpf(10) ** (-DPS - 5)
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    else:
        raise RuntimeError(f"continued fraction stalled at a={a}, x={x}")
    return a * mp.log(x) - x - mp.loggamma(a) + mp.log(h)


def _temme(mp, v, eta):
    a = ASPLIT_L / v
    lam = _lam_of_eta(mp, eta)
    x = a * lam
    y = eta * mp.sqrt(a / 2)
    if eta < 0:
        # small side: R = erfc(-y)/2 - P, both ~ e^(-a eta^2/2)
        p = mp.exp(a * mp.log(x) - x - mp.loggamma(a + 1)) * _hyp1f1_1(mp, a + 1, x)
        r = mp.erfc(-y) / 2 - p
    else:
        r = mp.exp(_log_q(mp, a, x)) - mp.erfc(y) / 2
    return r * mp.sqrt(2 * mp.pi * a) * mp.exp(a * eta * eta / 2)


def _lower(mp, v, lam):
    a = ASPLIT_L / v
    return mp.log(_hyp1f1_1(mp, a + 1, a * lam))


def _upper(mp, v, s):
    a = ASPLIT_L / v
    x = (a - 1) / s
    return _log_q(mp, a, x) + x + (1 - a) * mp.log(x) + mp.loggamma(a)


def _fit2d(mp, f, narg, nv, arg_lo, arg_hi):
    """T[k, j] = j-th v-coefficient of the k-th argument coefficient
    (the gammainc_table convention); both fit stages in mp."""
    lo, hi = mp.mpf(arg_lo), mp.mpf(arg_hi)
    vlo, vhi = mp.mpf(VLO), mp.mpf(VHI)
    rows = []
    for tv in nodes(mp, nv):
        v = vlo + (tv + 1) / 2 * (vhi - vlo)
        rows.append(
            dct(mp, [f(mp, v, lo + (t + 1) / 2 * (hi - lo)) for t in nodes(mp, narg)])
        )
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
        {
            "ne": NE,
            "nv_t": NV_T,
            "nl": NL,
            "nv_d": NV_D,
            "ns": NS,
            "nv_u": NV_U,
            "asplit": ASPLIT_L,
        },
        {
            "ASPLIT": ASPLIT_L,
            "VLO": VLO,
            "ELO": ELO,
            "EHI": EHI,
            "LHI": LHI,
            "SHI": SHI,
        },
        {"TABLE_TEMME": temme, "TABLE_LOW": lower, "TABLE_UP": upper},
    )


if __name__ == "__main__":
    main()
