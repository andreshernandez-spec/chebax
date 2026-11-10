"""Generator for the gammaln tables. Build-time only; mpmath inside.

Below x = 64 the tables are panels: each octave [2^e, 2^(e+1)) of [1, 64)
splits into K equal panels, each fitted on NN nodes. G and LG hold the
value, NTP terms a panel: on [1, 4), g = lgamma(x) / ((x - 1)(x - 2)),
smooth through lgamma's zeros at 1 and 2 (g(1) = -psi(1) = Euler's
gamma, g(2) = psi(2)), so the product keeps relative accuracy next to
them; on [4, 64), lgamma. PSI holds digamma on the same panels, NTP_PSI
terms, and on [1, 2) r = psi(x) / (x - X0), smooth through digamma's zero
X0 = 1.4616... for the same reason; X0 is split into X0_HI + X0_LO so
x - X0_HI is exact. gammaln's derivative is read from PSI, not from the
value's series (gammaln.py says why). Below 1 the runtime uses
lgamma(x + 1) - log x and psi(x + 1) - 1/x. From 64 up, Stirling's series
(x - 1/2)(log x - 1) + log(2 pi)/2 - 1/2 + sum_k A_k / x^(2k+1),
A_k = B_(2k+2) / ((2k+2)(2k+1)); NA terms reach float64 there for the
value and for digamma, whose terms come from the same A_k.

Panels are octave fractions because a panel series converges at a rate
set by the panel's width over x (lgamma's singularities are at 0, -1,
-2, ...), and its derivative shows it first. Equal panels of 1/4 on
[3, 13) and of 1 on [13, 64), nine terms, held lgamma to 3.5e-16 but its
derivative only to 9.0e-14 (at x = 3). At 1/32 of an octave, 8 terms
hold lgamma and 9 hold digamma and its derivative (trigamma, which jax
takes from PSI by differentiating it); 8 leave trigamma at 6e-14
(experiments/44, tails in experiments/03). Float32 keeps 4 and 5 terms,
and 2 of Stirling's. The octave and the panel come from the exponent
bits, with no log. Regenerate with:

    python -m chebax._src.recipes.gammaln_gen
"""

import numpy as np

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    table_path,
    write_table_module,
)

OCTAVES = 6  # [1, 64)
G_OCTAVES = 2  # g on [1, 4)
K = 32  # panels per octave
NN = 24  # nodes per panel fit
NTP = 8  # terms kept per lgamma panel
NTP_PSI = 9  # and per digamma panel
NA = 4  # Stirling terms


def generate_tables():
    import mpmath as mp

    with mp.workdps(DPS):

        def g(x):
            if x == 1:
                return -mp.digamma(1)
            if x == 2:
                return mp.digamma(2)
            return mp.loggamma(x) / ((x - 1) * (x - 2))

        x0 = mp.findroot(mp.digamma, mp.mpf("1.4616"))

        def r(x):
            return mp.digamma(x) / (x - x0)

        rows, psi = [], []
        for e in range(OCTAVES):
            f = g if e < G_OCTAVES else mp.loggamma
            fp = r if e == 0 else mp.digamma
            h = mp.mpf(2) ** e / K
            for j in range(K):
                xs = [2**e + h * (j + (u + 1) / 2) for u in nodes(mp, NN)]
                rows.append([float(v) for v in dct(mp, [f(x) for x in xs])[:NTP]])
                psi.append([float(v) for v in dct(mp, [fp(x) for x in xs])[:NTP_PSI]])
        a = [
            float(mp.bernoulli(2 * k + 2) / ((2 * k + 2) * (2 * k + 1)))
            for k in range(NA)
        ]
        c = float(mp.log(2 * mp.pi) / 2 - mp.mpf(1) / 2)
        x0_hi = float(x0)
        x0_lo = float(x0 - x0_hi)
    n = G_OCTAVES * K
    tables = np.array(rows[:n]), np.array(rows[n:]), np.array(psi), np.array(a)
    return tables, (c, x0_hi, x0_lo)


def main(out_dir=None):
    (g, lg, psi, a), (c, x0_hi, x0_lo) = generate_tables()
    write_table_module(
        table_path(out_dir, __file__, "gammaln_table.py"),
        "chebax._src.recipes.gammaln_gen",
        "G[j, k] the k-th coefficient of lgamma(x) / ((x - 1)(x - 2)) on "
        "panel j of [1, 4) and LG[j, k] that of lgamma on panel j of [4, 64), "
        "where panel j is 2^e [1 + m/K, 1 + (m+1)/K], e = j // K (+ 2 for LG), "
        "m = j % K; PSI[j, k] that of digamma on panel j of [1, 64), of "
        "digamma(x) / (x - X0) on [1, 2); STIRLING[k] = B_(2k+2) / "
        "((2k+2)(2k+1)), the coefficient of x^-(2k+1) from XB up",
        {
            "octaves": OCTAVES,
            "g_octaves": G_OCTAVES,
            "k": K,
            "nn": NN,
            "ntp": NTP,
            "ntp_psi": NTP_PSI,
            "na": NA,
        },
        {
            "K": K,
            "XB": float(2**OCTAVES),
            "STIRLING_C": c,
            "X0_HI": x0_hi,
            "X0_LO": x0_lo,
        },
        {"G": g, "LG": lg, "PSI": psi, "STIRLING": a},
    )


if __name__ == "__main__":
    main()
