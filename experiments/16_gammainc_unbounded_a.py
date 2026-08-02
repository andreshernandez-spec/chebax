#!/usr/bin/env python3
"""Can the Temme-zone tables cover a > 1000, up to a -> inf?

WHAT THIS MEASURES
------------------
The same three kernels as experiments/14, on the FULL parameter
interval v = 10/a in [0, 1] rather than [0.01, 1]:

    T(v, eta) = R sqrt(2 pi a) e^(a eta^2 / 2),  R the Temme correction
    D(v, lambda) = ln 1F1(1; a+1; a lambda)
    U(v, s) = ln[ Gamma(a, x) x^(1-a) e^x ],  s = (a-1)/x

v = 0 is a -> inf, and it is a regular point of all three: the
kernels tend to their asymptotic limits, D -> -ln(1 - lambda) and
U -> -ln(1 - s) (both are the geometric sums their series become
once (a+1)_n / a^n -> 1), T -> Temme's leading coefficient. So the
question is only whether the extra sliver of v costs degree, and
what reference survives out there.

The 2026-08-01 cap at a = 1000 was a REFERENCE limit, not a
representation one (experiments/14 FINDINGS): mpmath's gammainc stops
converging near lambda ~ 1 around a ~ 1e5. This script replaces every
mpmath gammainc call with a side-aware pair,

    P from the 1F1(1; a+1; a lambda) series      (converges for x < a)
    Q from Legendre's continued fraction         (converges for x > a)

each used only where it converges and only for the SMALL side of the
split, which is the rule experiments/14 arrived at. Chebyshev nodes
are first-kind, so v = 0 itself is never sampled: at 34 nodes the
smallest is v = 5.3e-4, i.e. a = 1.9e4, well inside what the pair
handles.

FINDINGS (2026-08-02, reproduced by this script)
------------------------------------------------
See results/16_gammainc_unbounded_a.txt.

WHAT IT DOES NOT MEASURE
------------------------
Runtime accuracy (no table is built here). The f64 conditioning of
the runtime assembly at large a is a separate matter: a ln x - x -
lnGamma(a+1) is a cancelling bracket whose terms grow like a ln a, so
the LOG forms are what stay accurate out there while P and Q
themselves underflow, as they must.

Run:  python experiments/16_gammainc_unbounded_a.py  (~40 s, CPU)
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


def log_p(a, x):
    """ln P(a, x) from the series. x < a, or the ratio stops being small."""
    return (a * mp.log(x) - x - mp.loggamma(a + 1)
            + mp.log(hyp1f1_1(a + 1, x)))


def log_q(a, x):
    """ln Q(a, x) from Legendre's continued fraction (modified Lentz).
    x > a, where the fraction converges; this is what mpmath's own
    gammainc gives up on for large a."""
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
        raise RuntimeError(f"cf stalled at a={a}, x={x}")
    return a * mp.log(x) - x - mp.loggamma(a) + mp.log(h)


def T(v, eta):
    a = 10 / v
    lam = lam_of_eta(eta)
    x = a * lam
    y = eta * mp.sqrt(a / 2)
    if eta < 0:
        # small side is P, and erfc(-y)/2 is its own leading term
        r = mp.erfc(-y) / 2 - mp.exp(log_p(a, x))
    else:
        r = mp.exp(log_q(a, x)) - mp.erfc(y) / 2
    return r * mp.sqrt(2 * mp.pi * a) * mp.exp(a * eta * eta / 2)


def D(v, lam):
    a = 10 / v
    return mp.log(hyp1f1_1(a + 1, a * lam))


def U(v, s):
    a = 10 / v
    x = (a - 1) / s
    return log_q(a, x) + x + (1 - a) * mp.log(x) + mp.loggamma(a)


ELO, EHI, LHI, SHI = -0.7, 2.6, 0.5, 1.0 / 6.0


def main():
    print("reference cross-check against mpmath where mpmath still works")
    print("  (ln P and ln Q, absolute difference)")
    for a, lam in ((50, 0.6), (300, 0.9), (900, 1.2), (900, 3.0)):
        x = mp.mpf(a) * lam
        dp = log_p(mp.mpf(a), x) - mp.log(
            mp.gammainc(mp.mpf(a), 0, x, regularized=True))
        dq = (log_q(mp.mpf(a), x) if x > a else mp.mpf(0)) - (
            mp.log(mp.gammainc(mp.mpf(a), x, mp.inf, regularized=True))
            if x > a else mp.mpf(0))
        print(f"  a={a:5d} lambda={lam}: dlnP={float(dp):+.2e} "
              f"dlnQ={float(dq):+.2e}")

    print("\nkernel limits as v -> 0 (a -> inf)")
    print("  T(v, 0) -> -1/3;  D(v, .5) -> -ln(1-.5) = 0.693147;"
          "  U(v, 1/6) -> -ln(1-1/6) = 0.182322")
    for v in (1.0, 0.1, 0.01, 1e-3, 5.3e-4):
        tv = float(T(mp.mpf(v), mp.mpf(0)))
        dv = float(D(mp.mpf(v), mp.mpf(LHI)))
        uv = float(U(mp.mpf(v), mp.mpf(SHI)))
        print(f"  v={v:8.2e} (a={10/v:9.1f}): T={tv:+.9f} D={dv:.9f} "
              f"U={uv:.9f}")

    for vlo, label in ((0.01, "current [0.01, 1]"), (0.0, "extended [0, 1]")):
        print(f"\n=== v on {label}")
        per = {v: tail(fit_axis(lambda e: T(mp.mpf(v), e), 96, ELO, EHI))
               for v in (1.0, 0.1, 0.01, 1e-3)}
        if vlo == 0.0:
            per[1e-4] = tail(fit_axis(lambda e: T(mp.mpf(1e-4), e), 96,
                                      ELO, EHI))
        print("  Temme eta-direction: " +
              " ".join(f"v={v:g}:{d}" for v, d in per.items()))
        dv = {eta: tail(fit_axis(lambda v: T(v, mp.mpf(eta)), 96, vlo, 1))
              for eta in (-0.65, 0.0, 1.0, 2.55)}
        print("  Temme v-direction:   " +
              " ".join(f"eta={e}:{d}" for e, d in dv.items()))

        dl = max(tail(fit_axis(lambda l: D(mp.mpf(v), l), 96, 0, LHI))
                 for v in (1.0, 0.01, 1e-3))
        dvl = max(tail(fit_axis(lambda v: D(v, mp.mpf(lam)), 96, vlo, 1))
                  for lam in (0.25, LHI))
        print(f"  D: lambda {dl}, v {dvl}")

        ds = max(tail(fit_axis(lambda s: U(mp.mpf(v), s), 96, 1e-9, SHI))
                 for v in (1.0, 0.01, 1e-3))
        dvu = max(tail(fit_axis(lambda v: U(v, mp.mpf(s)), 96, vlo, 1))
                  for s in (SHI / 2, SHI))
        print(f"  U: s {ds}, v {dvu}")


if __name__ == "__main__":
    main()
