"""Generator for the gammainc log-tables. Build-time only; mpmath inside.

The regularized incomplete gamma pair P(a, x) = gamma(a, x)/Gamma(a),
Q = 1 - P, for a in [0.1, 10], x in [0, inf). Two regions, both tabulating
a logarithm so the singular parts live in exactly-computed prefactors
(the besselk move):

- Inner, x in [0, XS]:
      L(a, x) = ln 1F1(1; a+1; x)
  DLMF 8.5.1: P = exp(a ln x - x - lnGamma(a+1) + L). 1F1(1; a+1; x) is
  entire in x and >= 1 (all series terms positive), so L is analytic and
  O(x); the x^a branch point and the full dynamic range of P sit in the
  prefactor. Measured worst tails (experiments/03): degree 23 in x. The
  a-direction is the hard axis, degree 53, worst at the x -> 0 edge where
  L ~ x/(a+1): the pole at a = -1 sits 1.1 from the domain edge and sets
  the Bernstein ellipse. No a-panels needed (nothing cancels; the betainc
  a-axis is the same shape).

- Tail, x >= XS, t = XS/x:
      T(a, t) = ln[ Gamma(a, x) x^(1-a) e^x ]
  Q = exp((a-1) ln x - x - lnGamma(a) + T). T -> 0 as t -> 0 (the
  asymptotic series is divergent, so t = 0 is a C-infinity endpoint
  singularity, same as besselk's tail; Chebyshev still converges past
  1e-15 by degree 20). T(1, t) = 0 identically (Gamma(1, x) = e^-x), a
  built-in sanity row. Measured worst tails: degree 20 in t, 20 in a.

Q comes out at relative accuracy for x >= XS and P at relative accuracy
below the transition; each is 1-minus-the-other on its far side, which is
where it is ~1 anyway. Both fit stages stay in mpmath (the M1 floor rule).
Regenerate with:

    python -m chebax._src.recipes.gammainc_gen
"""

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, table_path,
                                             write_table_module)

XS_G = 8.0
ALO, AHI = 0.1, 10.0
NX_IN = 32   # x-nodes (degree 31; worst measured tail at 23)
NA_IN = 68   # a-nodes (degree 67; worst measured 53, the x -> 0 edge)
NT_TAIL = 26  # t-nodes (worst measured 20)
NA_TAIL = 30  # a-nodes (worst measured 20)


def _lser(mp, a, x):
    return mp.log(mp.hyp1f1(1, a + 1, x))


def _ltail(mp, a, t):
    x = mp.mpf(XS_G) / t
    return mp.log(mp.gammainc(a, x, mp.inf)) + x + (1 - a) * mp.log(x)


def generate_inner_table():
    import mpmath as mp

    with mp.workdps(DPS):
        tx = nodes(mp, NX_IN)
        lo, hi = mp.mpf(ALO), mp.mpf(AHI)
        rows = []
        for ta in nodes(mp, NA_IN):
            a = lo + (ta + 1) / 2 * (hi - lo)
            rows.append(dct(mp, [_lser(mp, a, (t + 1) / 2 * mp.mpf(XS_G))
                                 for t in tx]))
        return to_f64(param_fit(mp, rows))


def generate_tail_table():
    import mpmath as mp

    with mp.workdps(DPS):
        tt = nodes(mp, NT_TAIL)
        lo, hi = mp.mpf(ALO), mp.mpf(AHI)
        rows = []
        for ta in nodes(mp, NA_TAIL):
            a = lo + (ta + 1) / 2 * (hi - lo)
            rows.append(dct(mp, [_ltail(mp, a, (t + 1) / 2) for t in tt]))
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    inner = generate_inner_table()
    tail = generate_tail_table()
    write_table_module(
        table_path(out_dir, __file__, "gammainc_table.py"),
        "chebax._src.recipes.gammainc_gen",
        "T[k, j] = j-th a-coefficient of the k-th argument coefficient; "
        "TABLE_IN has argument x in [0, XS], TABLE_TAIL argument t = XS/x "
        "in [0, 1]; a in [ALO, AHI] for both",
        {"nx_in": NX_IN, "na_in": NA_IN, "nt_tail": NT_TAIL,
         "na_tail": NA_TAIL, "xs": XS_G, "alo": ALO, "ahi": AHI},
        {"XS": XS_G, "ALO": ALO, "AHI": AHI},
        {"TABLE_IN": inner, "TABLE_TAIL": tail})


if __name__ == "__main__":
    main()
