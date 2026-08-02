#!/usr/bin/env python3
"""Degree measurement for extending gammainc past a = 10 (Temme zone).

WHAT THIS MEASURES
------------------
Chebyshev degree requirements for a large-a gammainc extension on
v = 10/a in [0.01, 1] (a in [10, 1000]), in three zones of
lambda = x/a:

- Temme transition zone, parametrized by eta:
      eta(lambda) = sign(lambda - 1) sqrt(2 (lambda - 1 - ln lambda))
      Q(a, x) = 1/2 erfc(eta sqrt(a/2)) + R
      T(v, eta) = R sqrt(2 pi a) e^(a eta^2 / 2)      <- the tabuland
  T is O(1) (T -> -1/3 at eta = 0, a -> inf) and carries the whole
  correction; the erfc term and the e^(-a eta^2/2) prefactor are
  exactly computed at runtime.
- Lower zone, lambda in [0, 1/2]:
      D(v, lambda) = ln 1F1(1; a+1; a lambda)
  the existing inner kernel in scaled coordinates (at a = inf it is
  -ln(1 - lambda); the b-pole clearance is a+1 ~ 11+).
- Upper zone, s = (a-1)/x in [0, 1/6] (lambda >= ~5.4):
      U(v, s) = ln[ Gamma(a, x) x^(1-a) e^x ]
  the existing tail kernel in the uniform variable s.

lambda(eta) is recovered through the two real branches of Lambert W:
lambda = -W_0(-e^(-1-eta^2/2)) for eta <= 0, -W_{-1} for eta > 0.

FINDINGS (2026-08-01, reproduced by this script)
------------------------------------------------
- eta-direction degree 20 on [-0.7, 2.6] (21 on [-1, 3]), UNIFORM in
  v; v-direction degree 7 at every eta. The Temme table is ~26x10.
- Lower zone: lambda 18, v 28 on [0, 0.5]. Upper: s 11, v 8.
- Zone edges: eta(0.5) = -0.63 and eta(lambda = 5.4) = 2.42 both sit
  inside [-0.7, 2.6], so three hard lambda-selects cover a > 10
  completely. Total ~1.3k coefficients.
- The box stops at a = 1000 because the 40-dps REFERENCE evaluation
  does, not the representation: mpmath's gammainc fails to converge
  near lambda ~ 1 for a >~ 1e5, and the naive R = Q - erfc/2
  subtraction is catastrophic once a eta^2/2 exceeds ~92 nats (40
  digits), so references must be computed on the SMALL side of the
  split: for eta < 0, R = erfc(-y)/2 - P with P built from an
  explicit 1F1(1; a+1; x) series (geometric ratio lambda < 1; mpf
  exponents make the ~e^-500 magnitudes harmless). Both artifacts
  produced saturated fake degrees before the fix.
- a in [10, 1000] means chi-squared to 2000 dof; extend later only
  with a reference evaluator that survives larger a.

WHAT IT DOES NOT MEASURE
------------------------
Runtime accuracy (no table is built here); the f64 conditioning of
the runtime eta(lambda) evaluation near lambda = 1 (a build-time
concern: lambda - 1 - ln(lambda) needs its small-delta series).

Run:  python experiments/14_gammainc_large_a.py  (~10 s, CPU)
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


def lam_of_eta(eta):
    if eta == 0:
        return mp.mpf(1)
    z = -mp.exp(-1 - eta * eta / 2)
    lam = -mp.lambertw(z, 0 if eta < 0 else -1)
    return mp.mpf(lam.real)


def hyp1f1_1(a1, x):
    s = t = mp.mpf(1)
    n = 0
    while abs(t) > mp.mpf(10) ** (-DPS - 10) * abs(s):
        t *= x / (a1 + n)
        s += t
        n += 1
    return s


def T(v, eta):
    a = 10 / v
    lam = lam_of_eta(eta)
    x = a * lam
    y = eta * mp.sqrt(a / 2)
    if eta < 0:
        p = mp.exp(a * mp.log(x) - x - mp.loggamma(a + 1)) * hyp1f1_1(a + 1, x)
        r = mp.erfc(-y) / 2 - p
    else:
        q = mp.gammainc(a, x, mp.inf, regularized=True)
        r = q - mp.erfc(y) / 2
    return r * mp.sqrt(2 * mp.pi * a) * mp.exp(a * eta * eta / 2)


def D(v, lam):
    a = 10 / v
    return mp.log(hyp1f1_1(a + 1, a * lam))


def U(v, s):
    a = 10 / v
    x = (a - 1) / s
    return (mp.log(mp.gammainc(a, x, mp.inf)) + x + (1 - a) * mp.log(x))


def main():
    print("sanity: T(v, 0) vs -1/3 as v -> 0:")
    for v in (1.0, 0.1, 0.01):
        print(f"  v={v}: T = {float(T(mp.mpf(v), mp.mpf(0))):.6f}")

    for elo, ehi in ((-0.7, 2.6), (-1.0, 3.0)):
        per = {v: tail(fit_axis(lambda e: T(mp.mpf(v), e), 96, elo, ehi))
               for v in (1.0, 0.5, 0.1, 0.01)}
        print(f"Temme eta-direction on [{elo}, {ehi}]: " +
              " ".join(f"v={v}:{d}" for v, d in per.items()))

    print("Temme v-direction on [0.01, 1] (96 nodes):")
    for eta in (-0.65, -0.3, 0.0, 1.0, 2.0, 2.55):
        d = tail(fit_axis(lambda v: T(v, mp.mpf(eta)), 96, 0.01, 1))
        print(f"  eta={eta}: {d}")

    print("outer zones, v in [0.01, 1]:")
    for lhi in (0.5, 0.6):
        d = max(tail(fit_axis(lambda l: D(mp.mpf(v), l), 96, 0, lhi))
                for v in (1.0, 0.01))
        dv = max(tail(fit_axis(lambda v: D(v, mp.mpf(lam)), 96, 0.01, 1))
                 for lam in (0.25, lhi))
        print(f"  D lambda on [0, {lhi}]: worst {d}; v-direction worst {dv}")
    for shi in (1.0 / 6, 0.2):
        d = max(tail(fit_axis(lambda s: U(mp.mpf(v), s), 96, 1e-6, shi))
                for v in (1.0, 0.01))
        dv = max(tail(fit_axis(lambda v: U(v, mp.mpf(s)), 96, 0.01, 1))
                 for s in (shi / 2, shi))
        print(f"  U s on [0, {shi:.3f}]: worst {d}; v-direction worst {dv}")


if __name__ == "__main__":
    main()
