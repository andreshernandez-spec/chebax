"""Generator for the ndtri_exp tables. Build-time only; mpmath inside.

z(y) = ndtri(e^y) for y <= 0, from two regions whose panels are laid end
to end in one table, each panel fitted on NN nodes:

- centre, YC <= y <= Y_HI = log(1 - e^YC): h(d) = z / d, d = y + log 2,
  on NC panels of width 1/WC from D_LO. h is smooth through the median
  d = 0, so d h(d) stays relative next to z's zero there; the runtime
  splits log 2, and y + LN2_HI is exact across the centre.
- tail: g(s) = a(s) / s, a(s) = -ndtri(e^(-s^2/2)), on K equal panels of
  each octave [2^e, 2^(e+1)) below 2^OCTAVES, from panel J0 of the first,
  which holds sqrt(-2 YC). Below YC s = sqrt(-2 y) and z = -a; above Y_HI
  s = sqrt(-2 log(1 - e^y)) and z = a. From 2^OCTAVES up, a = s: a/s =
  1 - (log s + log(2 pi)/2)/s^2 + ..., under 0.2 ulp from 1 there.

The seams sit at p = e^YC = e^-1.5 because near the median a is
ill-conditioned in s, and the rounding of s and of log(1 - e^y) grows by
s a'/a: 5.8 at p = e^-1, 2.9 at e^-1.5. Tail panels follow the octave
because a panel series converges at a rate set by its width over s (a's
singularity is at s = 0). The centre's convergence is set by its distance
from y = 0 (d = log 2), which 1/64-wide panels keep within the tail's
terms up to the seam.

The derivative has its own table on the same panels, fitted like the
values: dz/dy in the centre and a'(s) = s Phi(-a)/phi(a) in the tail, so
dz/dy = a'(s)/s below YC and a'(s)/s * e^y/(1 - e^y) above Y_HI. Measured
(experiments/03): NTP = 10 terms hold the values, 9 in the centre, and
NTP_D = 10 the derivative, both to eps/4 of the panel's leading
coefficient; float32 keeps 5 of each. Regenerate with:

    python -m chebax._src.recipes.ndtri_exp_gen
"""

import numpy as np

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    table_path,
    write_table_module,
)

YC = -1.5  # centre seam: p = e^YC and 1 - e^YC
WC = 64  # centre panels per unit of d = y + log 2
K = 16  # tail panels per octave
OCTAVES = 30  # tail octaves [1, 2^30)
NN = 24  # nodes per panel fit
NTP = 10  # terms kept per value panel
NTP_D = 10  # and per derivative panel


def _erfcx(mp, x):
    """e^(x^2) erfc(x); the asymptotic series past 50, where e^(x^2) would
    need an integer of x^2 bits."""
    if x < 50:
        return mp.exp(x * x) * mp.erfc(x)
    t, term, total, n = 1 / (2 * x * x), mp.mpf(1), mp.mpf(1), 1
    while abs(term) > mp.eps:
        term *= -(2 * n - 1) * t
        total += term
        n += 1
    return total / (x * mp.sqrt(mp.pi))


def _mills(mp, z):
    """Phi(z) / phi(z)."""
    return mp.sqrt(mp.pi / 2) * _erfcx(mp, -z / mp.sqrt(2))


def _tail_a(mp, s):
    """a with log Phi(-a) = -s^2/2, by Newton in log Phi from the asymptote."""
    y = -s * s / 2
    if s < 3:
        z = mp.sqrt(2) * mp.erfinv(2 * mp.exp(y) - 1)
    else:
        z = -(s - (mp.log(s) + mp.log(2 * mp.pi) / 2) / s)
    for _ in range(60):
        x = -z / mp.sqrt(2)
        step = (mp.log(_erfcx(mp, x) / 2) - x * x - y) * _mills(mp, z)
        z -= step
        if abs(step) <= abs(z) * mp.eps * 16:
            return -z
    raise RuntimeError(f"no convergence at s = {s}")


def generate_tables():
    import mpmath as mp

    with mp.workdps(DPS):
        ln2, yc = mp.log(2), mp.mpf(YC)
        y_hi = mp.log(1 - mp.exp(yc))
        c_lo = int(mp.floor((yc + ln2) * WC))
        c_hi = int(mp.ceil((y_hi + ln2) * WC))
        j0 = int(mp.floor((mp.sqrt(-2 * yc) - 1) * K))

        def centre(d):
            z = mp.sqrt(2) * mp.erfinv(mp.expm1(d))
            return z / d, _mills(mp, z)

        def tail(s):
            a = _tail_a(mp, s)
            return a / s, s * _mills(mp, -a)

        def fit(f, lo, w):
            fs = [f(lo + w * (u + 1) / 2) for u in nodes(mp, NN)]
            return dct(mp, [v[0] for v in fs]), dct(mp, [v[1] for v in fs])

        fits = [fit(centre, mp.mpf(j) / WC, mp.mpf(1) / WC) for j in range(c_lo, c_hi)]
        for e in range(OCTAVES):
            w = mp.mpf(2) ** e / K
            for j in range(j0 if e == 0 else 0, K):
                fits.append(fit(tail, 2**e + j * w, w))
        scalars = {
            "NC": c_hi - c_lo,
            "D_LO": c_lo / WC,
            "J0": j0,
            "Y_HI": float(y_hi),
            "LN2_HI": float(ln2),
            "LN2_LO": float(ln2 - mp.mpf(float(ln2))),
        }
    return [f[0] for f in fits], [f[1] for f in fits], scalars


def main(out_dir=None):
    rows, ders, scalars = generate_tables()
    tof = lambda rs, n: np.array([[float(v) for v in r[:n]] for r in rs])
    write_table_module(
        table_path(out_dir, __file__, "ndtri_exp_table.py"),
        "chebax._src.recipes.ndtri_exp_gen",
        "P[i, k] the k-th coefficient of panel i: for i < NC, of "
        "ndtri(e^y) / d on d = y + log 2 in [D_LO + i/WC, D_LO + (i+1)/WC]; "
        "from NC on, of a(s)/s, a = -ndtri(e^(-s^2/2)), on the panel "
        "2^e [1 + m/K, 1 + (m+1)/K] with e K + m = i - NC + J0. DP[i, k] the "
        "same for dz/dy in the centre and a'(s) in the tail",
        {
            "yc": YC,
            "wc": WC,
            "k": K,
            "octaves": OCTAVES,
            "nn": NN,
            "ntp": NTP,
            "ntp_d": NTP_D,
        },
        {
            "YC": YC,
            "WC": float(WC),
            "K": K,
            "S_BIG": float(2**OCTAVES),
            **scalars,
        },
        {"P": tof(rows, NTP), "DP": tof(ders, NTP_D)},
    )


if __name__ == "__main__":
    main()
