"""Generator for the Tweedie slice of Wright's function. Build-time only.

    W(rho, z) = Phi(rho, 0; z) = sum_{n>=1} z^n / (n! Gamma(rho n))

is what the Tweedie compound Poisson-gamma density needs for 1 < p < 2
(rho = (2-p)/(p-1), see tweedie.py), and only at beta = 0, so this is a
2-D table rather than the 3-D (rho, beta, z) tensor experiments/17 sized
for the general function. Design measured in experiments/20:

- Large z: a saddle point of the Hankel integral gives
      ln W ~ Z + ln(t0)/2 - ln(2 pi (1+rho))/2 + R,
      t0 = (rho z)^(1/(1+rho)),  Z = (1+rho) t0 / rho,
  with R -> 0 like c1(rho)/Z. The table holds T = Z R (T -> c1(rho), so the
  1/Z variation stays out of the rho direction) in s = Z1(rho)/Z in (0, 1].
- Below the split Z1(rho) = ZA + S1 (1+rho)^2 the runtime sums the series
  directly over n = 1..N_TERMS; only rho n appears inside Gamma, and rho is
  uniform per call, so lgamma(rho n) is per call and the per-point work is
  N_TERMS exp and multiply-adds. No table there.
- The split is in Z, not z, and holds two things at once. The terms peak at
  n0 = Z/(1+rho) with variance Z/(1+rho)^2; below a variance of ~2 the sum
  is a few discrete terms and differs from its smooth saddle form by a
  ripple of size exp(-2 pi^2 variance), one period per term, which no
  degree resolves (rho = 19 saturates 64 nodes at a fixed n0 split). That is
  the S1 (1+rho)^2 term. ZA keeps small rho off the poorly converged
  Z ~ 2 end of the expansion. Smooth in rho, as the table needs.
- Parameter axis ln rho = -logit(p - 1) on [-ln 19, ln 19], p in
  [1.05, 1.95]: degree 51 in ln rho against 63 in p, and c1 is symmetric
  under rho -> 1/rho. Degree 24 in s at rho = 1/19, 5 at rho = 19.
- The reference sums a window around the peak term (terms are log-concave in
  n) to 1e-52 of the peak, so z ~ 1e9 costs O(sqrt(n0)) terms.

Both fit stages stay in mpmath (the M1 floor rule). Regenerate with:

    python -m chebax._src.recipes.tweedie_gen
"""

import math

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    param_fit,
    table_path,
    to_f64,
    write_table_module,
)

P_LO, P_HI = 1.05, 1.95
RHO_MAX = 19.0  # (2 - P_LO) / (P_LO - 1)
RHO_MIN = 1.0 / 19.0  # (2 - P_HI) / (P_HI - 1)
ZA = 12.0
S1 = 2.0
N_TERMS = 64  # direct sum below the split; 54 measured worst need
NS = 32  # s-nodes (worst measured degree 24)
NA = 64  # ln-rho nodes (worst measured degree 51)


def _log_w(mp, rho, z):
    lz = mp.log(z)

    def lt(n):
        return n * lz - mp.loggamma(n + 1) - mp.loggamma(rho * n)

    n0 = max(1, int((rho * z) ** (1 / (1 + rho)) / rho))
    while n0 > 1 and lt(n0 - 1) > lt(n0):
        n0 -= 1
    while lt(n0 + 1) > lt(n0):
        n0 += 1
    peak = lt(n0)
    s = mp.mpf(0)
    for step, start in ((-1, n0 - 1), (1, n0)):
        n = start
        while n >= 1:
            d = lt(n) - peak
            if d < -120:
                break
            s += mp.exp(d)
            n += step
    return peak + mp.log(s)


def _t(mp, rho, s):
    """T = Z (ln W - leading saddle term) at Z = Z1(rho) / s."""
    big_z = (ZA + S1 * (1 + rho) ** 2) / s
    t0 = rho * big_z / (1 + rho)
    z = t0 ** (1 + rho) / rho
    lead = big_z + mp.log(t0) / 2 - mp.log(2 * mp.pi * (1 + rho)) / 2
    return big_z * (_log_w(mp, rho, z) - lead)


def generate_tail_table():
    import mpmath as mp

    with mp.workdps(DPS):
        a1 = mp.log(mp.mpf(19))
        rows = []
        for ta in nodes(mp, NA):
            rho = mp.e ** (-a1 + (ta + 1) / 2 * 2 * a1)
            rows.append(dct(mp, [_t(mp, rho, (ts + 1) / 2) for ts in nodes(mp, NS)]))
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    tail = generate_tail_table()
    write_table_module(
        table_path(out_dir, __file__, "tweedie_table.py"),
        "chebax._src.recipes.tweedie_gen",
        "T[k, j] = j-th ln-rho coefficient of the k-th s coefficient; "
        "ln rho in [-A1, A1], s = Z1(rho)/Z in [0, 1], Z1 = ZA + S1 (1+rho)^2",
        {
            "ns": NS,
            "na": NA,
            "za": ZA,
            "s1": S1,
            "n_terms": N_TERMS,
            "p_lo": P_LO,
            "p_hi": P_HI,
        },
        {
            "P_LO": P_LO,
            "P_HI": P_HI,
            "RHO_MIN": RHO_MIN,
            "RHO_MAX": RHO_MAX,
            "A1": math.log(RHO_MAX),
            "ZA": ZA,
            "S1": S1,
            "N_TERMS": N_TERMS,
        },
        {"TABLE_TAIL": tail},
    )


if __name__ == "__main__":
    main()
