#!/usr/bin/env python3
"""Feasibility of widening betainc's (a, b) box beyond [0.1, 10]^2.

WHAT THIS MEASURES
------------------
Chebyshev degree growth of L = ln 2F1(a+b, 1; a+1; x) (the tabulated
betainc kernel, x in [0, 1/2]) when the parameter box widens to 50 or
100, under candidate axis treatments. Everything downstream inherits
the box: betainc a/b gradients, betaincinv, stdtr/stdtrit's nu = 2a
range, TruncatedBeta, the pytensor plugin's nan box.

Findings (2026-08-01, reproduced by this script):
- The axes are asymmetric: raw a over [0.1, 100] needs degree 129-175
  (the a-pole structure), log-a tames it to 40-95; raw b needs only
  26-63 and LOG-b is WORSE (97). Panels in a beat global transforms.
- x-degree grows from 19 to 62, worst where the CDF transition
  x* = a/(a+b) is interior and sharp (a = 10, b = 100).
- A 2x2 panel scheme at [0.1, 100]^2 (raw axes per panel) needs, with
  ~20% margin, roughly: LOxLO 24x72x28 (the existing table), LOxHI
  76x64x60, HIxLO 24x60x20, HIxHI 76x62x60 - about 650k coefficients
  (5 MB) vs 48k today, and ~13x the mpmath generation time (~1 h; the
  bit-for-bit regen test at that size no longer fits the CI budget).
- A single [0.1, 50]^2 tensor (log-a, raw b) needs ~56x90x50 = 252k:
  half the reach for ~40% of the cost.
- The cheapest useful extension is ONE off-diagonal strip (a in
  [0.1, 10] x b in [10, 100], with the swapped orientation reusing it
  via the reflection): ~292k, and it already unlocks stdtr nu up to
  200 (nu = 2a needs a large with b = 1/2 fixed - served by the
  HIxLO corner, degrees 19/49/15, only ~29k!).

WHAT IT DOES NOT MEASURE
------------------------
Runtime accuracy of any widened table (none is built); generation
wall-time (estimated from the current generator's samples/s); whether
the regen-test convention should change for large tables (a policy
decision, not a measurement).

Run:  python experiments/11_betainc_widening_feasibility.py  (~25 min, CPU)
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
    return mp.log(mp.hyp2f1(a + b, 1, a + 1, x))


def main():
    print("x-direction on [0, 1/2] (96 nodes) at wide corners:")
    for a, b in [(0.1, 100.0), (100.0, 0.1), (100.0, 100.0), (50.0, 50.0),
                 (10.0, 100.0), (0.1, 10.0)]:
        d = tail(fit_axis(lambda x: L(mp.mpf(a), mp.mpf(b), x), 96, 0, 0.5))
        print(f"  a={a:5}, b={b:5}: {d}")

    print("a-direction on [0.1, 100] at x=0.5: raw (192 nodes) vs log (128):")
    for b in [0.1, 10.0, 100.0]:
        raw = tail(fit_axis(lambda a: L(a, mp.mpf(b), mp.mpf(0.5)), 192, 0.1, 100))
        log = tail(fit_axis(lambda u: L(mp.exp(u), mp.mpf(b), mp.mpf(0.5)), 128,
                            mp.log(mp.mpf(0.1)), mp.log(mp.mpf(100))))
        print(f"  b={b:5}: raw {raw}, log {log}")

    print("b-direction on [0.1, 100] at x=0.5: raw (192) vs log (128):")
    for a in [0.1, 10.0, 100.0]:
        raw = tail(fit_axis(lambda b: L(mp.mpf(a), b, mp.mpf(0.5)), 192, 0.1, 100))
        log = tail(fit_axis(lambda v: L(mp.mpf(a), mp.exp(v), mp.mpf(0.5)), 128,
                            mp.log(mp.mpf(0.1)), mp.log(mp.mpf(100))))
        print(f"  a={a:5}: raw {raw}, log {log}")

    panels = [("LO", 0.1, 10.0), ("HI", 10.0, 100.0)]
    print("per-panel worst degrees (x on [0,1/2]; a, b raw within panel):")
    print(f"  {'panel a x b':12s} {'x-deg':>6s} {'a-deg':>6s} {'b-deg':>6s}")
    for na, alo, ahi in panels:
        for nb, blo, bhi in panels:
            xw = max(tail(fit_axis(lambda x: L(mp.mpf(a), mp.mpf(b), x),
                                   96, 0, 0.5))
                     for a in (alo, ahi) for b in (blo, bhi))
            aw = max(tail(fit_axis(lambda a: L(a, mp.mpf(b), mp.mpf(x)),
                                   96, alo, ahi))
                     for b in (blo, bhi) for x in (0.01, 0.5))
            bw = max(tail(fit_axis(lambda b: L(mp.mpf(a), b, mp.mpf(x)),
                                   96, blo, bhi))
                     for a in (alo, ahi) for x in (0.01, 0.5))
            print(f"  {na}x{nb:10s} {xw:6d} {aw:6d} {bw:6d}")

    print("single-tensor [0.1, 50]^2, log-a, raw b:")
    xw = max(tail(fit_axis(lambda x: L(mp.mpf(a), mp.mpf(b), x), 96, 0, 0.5))
             for a, b in [(0.1, 50), (10, 50), (50, 50), (50, 0.1)])
    aw = max(tail(fit_axis(lambda u: L(mp.exp(u), mp.mpf(b), mp.mpf(0.5)), 128,
                           mp.log(mp.mpf(0.1)), mp.log(mp.mpf(50))))
             for b in (0.1, 50.0))
    bw = max(tail(fit_axis(lambda b: L(mp.mpf(a), b, mp.mpf(0.5)), 128, 0.1, 50))
             for a in (0.1, 50.0))
    print(f"  x-deg {xw}, a-deg(log) {aw}, b-deg(raw) {bw}")


if __name__ == "__main__":
    main()
