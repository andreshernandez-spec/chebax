#!/usr/bin/env python3
"""Experiment 03 — reproduce every measured-degree claim in the generators.

WHAT IS MEASURED
----------------
For each recipe's tabulated function, the Chebyshev coefficient tail in each
table axis, at the axis positions the generator docstrings cite as worst
cases. "last>1e-15" is the highest coefficient index still above 1e-15
relative to the largest — the number a generator's node count must exceed
with margin. These are the measurements behind every NW/NK/NZ/NV constant;
they were originally made in scratch prototypes, which violated the
project's own regenerability rule — this script fixes that.

Run everything:      python experiments/03_degree_measurement.py
One family:          python experiments/03_degree_measurement.py besselk
Families: besselj besselk besseli bessely betainc vonmises erf

WHAT THIS DOES NOT ESTABLISH
----------------------------
Runtime accuracy (that is the test suite, against mpmath references) or the
besselj INNER degrees, which are the bessel project's experiments/02.
The full run takes ~10 minutes; vonmises and betainc dominate (mp.quad and
mp.hyp2f1 sampling).
"""

import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, "src")
from chebax._src.recipes._gen_common import DPS, dct, nodes  # noqa: E402

mp.mp.dps = DPS


def report(tag, coeffs, marks=()):
    a = np.abs([float(x) for x in coeffs])
    a /= max(a.max(), 1e-300)
    idx = np.nonzero(a > 1e-15)[0]
    extra = "  ".join(f"a[{m}]={a[m]:.1e}" for m in marks if m < len(a))
    print(f"  {tag}: last>1e-15 at k={idx.max() if idx.size else -1}  {extra}")


def fit_axis(f, n, lo, hi):
    """Chebyshev tail of f sampled at n nodes on [lo, hi]."""
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    return dct(mp, [f(lo + (t + 1) / 2 * (hi - lo)) for t in nodes(mp, n)])


def besselj():
    print("besselj mid: J_v direct in x on [8, 30] (claims: degree <= 43)")
    for v in [0.0, 5.0, 9.97]:
        report(f"x-dir v={v}", fit_axis(lambda x: mp.besselj(v, x), 48, 8, 30), [40, 44])
    print("besselj outer: P and Q/s in t = (30/x)^2 (claims: degree <= 8)")

    def pq(v, x):
        w = x - (mp.mpf(v) / 2 + mp.mpf(1) / 4) * mp.pi
        r = mp.sqrt(mp.pi * x / 2)
        J, Y = mp.besselj(v, x), mp.bessely(v, x)
        return r * (J * mp.cos(w) + Y * mp.sin(w)), r * (Y * mp.cos(w) - J * mp.sin(w))

    for v in [0.5, 9.97]:
        report(f"P t-dir v={v}", fit_axis(lambda t: pq(v, 30 / mp.sqrt(t))[0], 16, 1e-8, 1), [10, 14])
        report(f"Qs t-dir v={v}",
               fit_axis(lambda t: pq(v, 30 / mp.sqrt(t))[1] / mp.sqrt(t), 16, 1e-8, 1), [10, 14])


def besselk():
    ltil = lambda v, u: mp.log(mp.besselk(v, mp.exp(u))) + v * (u - mp.log(2))
    u0, u1 = mp.log(mp.mpf("1e-6")), mp.log(mp.mpf(8))
    print("besselk inner: Ltil u-direction (claims: worst degree 66)")
    for v in [0.05, 5.0, 10.0]:
        report(f"u-dir v={v}", fit_axis(lambda u: ltil(mp.mpf(v), u), 80, u0, u1), [64, 72])
    print("besselk inner: Ltil v-direction per panel (claims: worst degree 45)")
    for lo, hi in [(0, 1), (1, 10)]:
        report(f"v-dir [{lo},{hi}] u=ln 1e-6",
               fit_axis(lambda v: ltil(v, u0), 56, lo, hi), [44, 50])
    print("besselk tail: Lt in t = 8/x (claims: degree 19)")
    lt = lambda v, t: (lambda x: mp.log(mp.besselk(v, x)) + x + mp.log(mp.sqrt(2 * x / mp.pi)))(8 / t)
    for v in [0.0, 10.0]:
        report(f"t-dir v={v}", fit_axis(lambda t: lt(mp.mpf(v), t), 24, 1e-8, 1), [18, 22])


def besseli():
    lh = lambda v, z: (lambda x: mp.log(mp.besseli(v, x)) - v * mp.log(x / 2)
                       + mp.loggamma(v + 1))(mp.sqrt(z))
    print("besseli inner: Lh z-direction (claims: worst degree ~64 at v=0)")
    for v in [0.0, 10.0]:
        report(f"z-dir v={v}", fit_axis(lambda z: lh(mp.mpf(v), z), 72, 1e-10, 64), [56, 64])
    print("besseli inner: v-direction (claims: degree 52)")
    report("v-dir z=64", fit_axis(lambda v: lh(v, mp.mpf(64)), 64, 0, 10), [48, 56])
    print("besseli tail: Lt in t = 8/x (claims: worst degree 39 at v=1/2)")
    lt = lambda v, t: (lambda x: mp.log(mp.besseli(v, x)) - x
                       + mp.log(mp.sqrt(2 * mp.pi * x)))(8 / t)
    for v in [0.5, 10.0]:
        report(f"t-dir v={v}", fit_axis(lambda t: lt(mp.mpf(v), t), 48, 1e-8, 1), [40, 44])


def bessely():
    u0, u1 = mp.log(mp.mpf("1e-6")), mp.log(mp.mpf(5))
    ta = lambda m, u: (lambda x: (x / 2) ** m * mp.bessely(m, x))(mp.exp(u))
    tb = lambda m, u: (lambda x: (x / 2) ** (m + 1) * mp.bessely(m + 1, x))(mp.exp(u))
    print("bessely inner: u-direction (claims: worst degree 82, T_b near mu=1)")
    for m in [0.0, 1.0]:
        report(f"Ta u-dir mu={m}", fit_axis(lambda u: ta(mp.mpf(m), u), 96, u0, u1), [80, 88])
        report(f"Tb u-dir mu={m}", fit_axis(lambda u: tb(mp.mpf(m), u), 96, u0, u1), [80, 88])
    print("bessely inner: mu-direction on [0, 1] (claims: worst degree 33)")
    report("Ta mu-dir u=ln 1e-6", fit_axis(lambda m: ta(m, u0), 48, 0, 1), [36, 44])
    print("bessely mid: Y_v direct in x on [5, 30] (claims: degree 60 at v=9.97)")
    report("x-dir v=9.97", fit_axis(lambda x: mp.bessely(mp.mpf("9.97"), x), 68, 5, 30), [60, 64])


def betainc():
    lnF = lambda a, b, x: mp.log(mp.hyp2f1(a + b, 1, a + 1, x))
    print("betainc: ln F degrees (claims: x <= 19, a <= 58, b <= 20)")
    for a, b in [(0.1, 10), (10, 10)]:
        report(f"x-dir a={a} b={b}",
               fit_axis(lambda x: lnF(mp.mpf(a), mp.mpf(b), x), 24, 0, 0.5), [20, 22])
    for b in [0.1, 10]:
        report(f"a-dir b={b} x=0.5",
               fit_axis(lambda a: lnF(a, mp.mpf(b), mp.mpf("0.5")), 72, 0.1, 10), [58, 64])
    report("b-dir a=0.1 x=0.5",
           fit_axis(lambda b: lnF(mp.mpf("0.1"), b, mp.mpf("0.5")), 28, 0.1, 10), [20, 24])


def vonmises():
    def H(kappa, w):
        th = mp.sqrt(w)
        i0 = mp.besseli(0, kappa)
        F = mp.quad(lambda u: mp.exp(kappa * mp.cos(u)), [-mp.pi, th]) / (2 * mp.pi * i0)
        return (F - mp.mpf(1) / 2 - th / (2 * mp.pi)) / th

    W = float(mp.pi ** 2)
    print("vonmises: H degrees (claims: w ~92 at kappa=50, r ~68)")
    for k in [10.0, 50.0]:
        report(f"w-dir kappa={k}", fit_axis(lambda w: H(mp.mpf(k), w), 104, 1e-10, W), [92, 100])
    report("r-dir w=0.01",
           fit_axis(lambda r: H(r * r, mp.mpf("0.01")), 80, 0, float(mp.sqrt(50))), [68, 76])


def erf():
    dawsn = lambda x: mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)
    erfcx = lambda x: mp.erfc(x) * mp.exp(x * x)
    print("erf family (claims: dawsn E 37 / G 10, erfcx C 32 / H 9; seam 6)")
    report("dawsn E z-dir", fit_axis(lambda z: dawsn(mp.sqrt(z)) / mp.sqrt(z), 44, 1e-10, 36), [38, 42])
    report("dawsn G t-dir", fit_axis(lambda t: (lambda x: x * dawsn(x))(6 / mp.sqrt(t)), 16, 1e-8, 1), [12, 14])
    report("erfcx C x-dir", fit_axis(erfcx, 40, 0, 6), [33, 38])
    report("erfcx H t-dir", fit_axis(lambda t: (lambda x: x * erfcx(x))(6 / mp.sqrt(t)), 16, 1e-8, 1), [10, 14])


FAMILIES = {"besselj": besselj, "besselk": besselk, "besseli": besseli,
            "bessely": bessely, "betainc": betainc, "vonmises": vonmises,
            "erf": erf}

if __name__ == "__main__":
    picks = sys.argv[1:] or list(FAMILIES)
    for name in picks:
        print(f"== {name} ==")
        FAMILIES[name]()
