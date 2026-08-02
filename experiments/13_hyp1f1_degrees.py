#!/usr/bin/env python3
"""Degree measurement for a hyp1f1 recipe (Kummer M, jax#21503).

WHAT THIS MEASURES
------------------
Chebyshev degree requirements for tabulating Kummer's M(a, b, x) on
(a, b) in [0.1, 10]^2, x >= 0, with a gammainc-style split at x = XS.
Two candidate inner kernels are compared:

- ln M directly: M >= 1 on the positive box, but M's nearest zero
  sits at x ~ -b/a, which hugs the origin when b << a (0.01 at the
  (10, 0.1) corner) and drives the x-degree past 150.
- ln R with M = 1 + (a/b) x R (the generalization of gammainc's
  1F1(1; a+1; x) trick): R(a, b, 0) = 1, R > 0, its nearest zero
  moves out to x ~ -2(b+1)/(a+1) and its b-poles start at b = -1,
  restoring the 1.1 clearance the raw b-axis lacks (1F1's
  denominator poles start at 0, only 0.1 from the box edge).
  Runtime reassembles ln M = log1p((a/b) x exp(ln R)) at full
  relative accuracy (identity checked in mpmath to 1e-40).

Tail, x >= XS, t = XS/x: T = ln[M Gamma(a)/Gamma(b) e^-x x^(b-a)],
the log-remainder of DLMF 13.7.1; T -> 0 as t -> 0 and T(a, a, t) = 0
identically (M(a, a, x) = e^x).

FINDINGS (2026-08-01, reproduced by this script)
------------------------------------------------
- ln R beats ln M in x: worst corner (10, 0.1) drops 155 -> 61 on
  [0, 8], (saturated) -> 112 on [0, 30].
- Parameter axes want log for the inner kernel: a raw 43 -> log 26,
  b raw 50 -> log 36. The tail wants log-a (133 raw -> 26) but RAW b
  (20 raw vs 44 log).
- The tail needs XS ~ 30: at XS = 8 the asymptotic is not yet
  asymptotic for |b - a| ~ 10 and t needs degree ~100; at XS = 30 the
  worst real t-degree is 33. Per-corner fits at a = b report full
  degree, but that is the zero function measured relative to its own
  noise floor (max |T| = 7e-40); near-diagonal strips report degrees
  on values that are absolutely invisible in ln M ~ x >= 30 the same
  way ((1, 2) says 38 at XS = 30 with max |T| ~ 9e-14, and 115 at
  XS = 50 with max |T| ~ 1e-22).
- Chosen architecture: XS = 30; inner (x, a, b) tensor deg
  (112, 26, 36) with log-a, log-b; tail (t, a, b) deg (33, 26, 20)
  with log-a, raw-b; ~20% node margin on each. About 200k
  coefficients total, comparable to one wide betainc panel.

WHAT IT DOES NOT MEASURE
------------------------
Runtime accuracy of any table (none is built here); negative x (the
Kummer transform M(a,b,x) = e^x M(b-a,b,-x) is a scope decision, not
a measurement); a or b above 10.

Run:  python experiments/13_hyp1f1_degrees.py  (~1 min, CPU)
"""

import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, "src")
from chebax._src.recipes._gen_common import DPS, dct, nodes  # noqa: E402

mp.mp.dps = DPS


def tail(coeffs):
    a = np.abs([float(x) for x in coeffs])
    a /= max(a.max(), 1e-300)
    idx = np.nonzero(a > 1e-15)[0]
    return int(idx.max()) if idx.size else -1


def fit_axis(f, n, lo, hi):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    return dct(mp, [f(lo + (t + 1) / 2 * (hi - lo)) for t in nodes(mp, n)])


def L(a, b, x):
    return mp.log(mp.hyp1f1(a, b, x))


def lnR(a, b, x):
    if x == 0:
        return mp.mpf(0)
    return mp.log((mp.hyp1f1(a, b, x) - 1) * b / (a * x))


def T(a, b, t, xs):
    x = mp.mpf(xs) / t
    return (mp.log(mp.hyp1f1(a, b, x)) + mp.loggamma(a) - mp.loggamma(b)
            - x + (b - a) * mp.log(x))


CORNERS = [(0.1, 0.1), (0.1, 10.0), (10.0, 0.1), (10.0, 10.0),
           (5.0, 0.15), (0.15, 5.0)]


def main():
    print("inner x-direction, ln M vs ln R (160 nodes), per corner:")
    for xs in (8.0, 30.0):
        lm = {c: tail(fit_axis(lambda x: L(mp.mpf(c[0]), mp.mpf(c[1]), x),
                               160, 0, xs)) for c in CORNERS}
        lr = {c: tail(fit_axis(lambda x: lnR(mp.mpf(c[0]), mp.mpf(c[1]), x),
                               160, 0, xs)) for c in CORNERS}
        print(f"  XS={xs}: ln M worst {max(lm.values())}, "
              f"ln R worst {max(lr.values())}")
        print("    ln R per corner: " +
              " ".join(f"({a},{b})={d}" for (a, b), d in lr.items()))

    print("ln R a-direction raw (128) vs log (128):")
    for b in (0.1, 10.0):
        for x in (1.0, 30.0):
            raw = tail(fit_axis(lambda a: lnR(a, mp.mpf(b), mp.mpf(x)), 128,
                                0.1, 10))
            lg = tail(fit_axis(lambda u: lnR(mp.exp(u), mp.mpf(b), mp.mpf(x)),
                               128, mp.log(mp.mpf(0.1)), mp.log(mp.mpf(10))))
            print(f"  b={b}, x={x}: raw {raw}, log {lg}")

    print("ln R b-direction raw (128) vs log (128):")
    for a in (0.1, 10.0):
        for x in (1.0, 30.0):
            raw = tail(fit_axis(lambda b: lnR(mp.mpf(a), b, mp.mpf(x)), 128,
                                0.1, 10))
            lg = tail(fit_axis(lambda v: lnR(mp.mpf(a), mp.exp(v), mp.mpf(x)),
                               128, mp.log(mp.mpf(0.1)), mp.log(mp.mpf(10))))
            print(f"  a={a}, x={x}: raw {raw}, log {lg}")

    print("tail t-direction (128 nodes; a = b rows are the zero function")
    print("measured against its own noise floor -- ignore them):")
    pts = CORNERS + [(1.0, 2.0)]
    for xs in (8.0, 30.0, 50.0):
        per = {c: tail(fit_axis(lambda t: T(mp.mpf(c[0]), mp.mpf(c[1]), t,
                                            xs), 128, 1e-8, 1)) for c in pts}
        off = max(d for (a, b), d in per.items() if a != b)
        print(f"  XS={xs}: worst off-diagonal {off}   " +
              " ".join(f"({a},{b})={d}" for (a, b), d in per.items()))
    print("  diagonal artifact: max |T(a, a, t)| = "
          + str(float(max(abs(T(mp.mpf(v), mp.mpf(v), mp.mpf(t), 30.0))
                          for v in (0.1, 10.0) for t in (0.1, 0.5, 1.0)))))

    print("tail a-direction at t=1, XS=30, raw (128) vs log (128):")
    for b in (0.1, 10.0):
        raw = tail(fit_axis(lambda a: T(a, mp.mpf(b), mp.mpf(1), 30.0), 128,
                            0.1, 10))
        lg = tail(fit_axis(lambda u: T(mp.exp(u), mp.mpf(b), mp.mpf(1), 30.0),
                           128, mp.log(mp.mpf(0.1)), mp.log(mp.mpf(10))))
        print(f"  b={b}: raw {raw}, log {lg}")

    print("tail b-direction at t=1, XS=30, raw (128) vs log (128):")
    for a in (0.1, 10.0):
        raw = tail(fit_axis(lambda b: T(mp.mpf(a), b, mp.mpf(1), 30.0), 128,
                            0.1, 10))
        lg = tail(fit_axis(lambda v: T(mp.mpf(a), mp.exp(v), mp.mpf(1), 30.0),
                           128, mp.log(mp.mpf(0.1)), mp.log(mp.mpf(10))))
        print(f"  a={a}: raw {raw}, log {lg}")


if __name__ == "__main__":
    main()
