#!/usr/bin/env python3
"""Degree measurement for Wright's generalized Bessel function.

    Phi(rho, beta; z) = sum_k z^k / (k! Gamma(rho k + beta))

jax has no wright_bessel and scipy does (scipy.special.wright_bessel),
so it is on the "still absent from jax" list in docs/adoption-map.md.
It is a THREE-argument family, which puts it in betainc's shape: two
instantiation parameters (rho, beta) and one evaluation argument (z),
i.e. a 3-D tensor table, not the 2-D tables of the one-parameter
recipes.

WHAT THIS MEASURES
------------------
Chebyshev degrees in all three axes, for two candidate tabulands:

    L(rho, beta; z) = ln Phi                 (the usual move)
    G(rho, beta; z) = Gamma(beta) Phi        (what the numbers pick)

The log is what besselk, besseli and betainc all tabulate, and it is
WRONG here; the measurement below is what says so. Gamma(beta) Phi is
the besselj move instead: pull the pole-carrying factor out
analytically and tabulate the entire remainder, which is 1 at z = 0.

Axes: z on panels of [0, 30], rho on [0.1, 2], beta on [0.1, 5] both
directly and through ln beta.

The reference is a direct mpmath series at 40 dps. Every term is
positive for z >= 0, so there is no cancellation to fight and no
side-aware treatment, unlike the incomplete gamma.

FINDINGS (2026-08-02, reproduced by this script)
------------------------------------------------
- THE LOG IS THE WRONG TABULAND, which is the finding worth keeping.
  Every other positive family here tabulates its log; ln Phi needs
  degree 128 in z at (rho, beta) = (1, 0.1) on [0, 8], against 20 for
  Gamma(beta) Phi on the same panel. Phi is ENTIRE with complex zeros,
  so ln Phi carries a branch point at each of them and the Bernstein
  ellipse collapses; the log helps only where the function it is
  applied to has no zeros near the interval (besselk, besseli). The
  degree blows up exactly where beta is small, i.e. where the nearest
  zero comes closest.
- Gamma(beta) Phi is the right object: entire, equal to 1 at z = 0, and
  it carries the Gamma pole analytically the way besselj's (x/2)^nu
  carries the branch point. z-degrees 12 / 20 / 30 on the panels
  [0, 1], [1, 8], [8, 30].
- beta MUST go through ln beta: 121-123 direct against 25-27 in ln beta,
  the Gamma pole crowding the small end.
- rho is the expensive axis whole (28 to 64 as z grows) and cheap in
  panels: 30 / 25 / 27 on [0.1, 0.5], [0.5, 1], [1, 2].
- So the shape is a 3-D tensor per (rho panel, z panel) pair, in the
  betainc mould: 9 tensors of about 36 x 34 x 32, order 3e5
  coefficients all told, a few times betainc's 48k. Buildable, and the
  size is the thing to decide before building.
- G spans 2e9 across [8, 30], so that panel needs its scale factored per
  (rho, beta) (a separate small ln G table) or splitting further, or its
  relative accuracy at the low end goes. Panels [0, 1] and [1, 8] span
  16 and 4e3 and are fine as they stand.

WHAT IT DOES NOT MEASURE
------------------------
Negative z (the series alternates there and the function has zeros: a
different, sup-normalized problem, and the reason scipy's own
implementation splits). Runtime accuracy or cost; no table is built
here. Whether a 3-D tensor of the measured size is worth shipping is a
decision for the FINDINGS below, not for this script.

Run:  python experiments/17_wright_bessel_degrees.py   (~3 min, CPU)
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


def phi(rho, beta, z):
    """Phi(rho, beta; z) by its defining series, all terms positive."""
    rho, beta, z = mp.mpf(rho), mp.mpf(beta), mp.mpf(z)
    s = mp.mpf(0)
    k = 0
    while True:
        t = z ** k / (mp.factorial(k) * mp.gamma(rho * k + beta))
        s += t
        if k > 5 and t < s * mp.mpf(10) ** (-DPS - 5):
            return s
        k += 1
        if k > 20000:
            raise RuntimeError(f"series stalled at rho={rho} beta={beta} z={z}")


def log_phi(rho, beta, z):
    return mp.log(phi(rho, beta, z))


RHOS = (0.1, 0.5, 1.0, 2.0)
BETAS = (0.1, 0.5, 1.0, 2.5, 5.0)
PANELS = ((0.0, 1.0), (1.0, 8.0), (8.0, 30.0))
NODES = 200   # readings saturate at n-1; 80 nodes read 79 everywhere


def gphi(rho, beta, z):
    """Gamma(beta) Phi: entire in z, equal to 1 at z = 0, and with the
    Gamma pole at small beta carried analytically rather than fitted."""
    return mp.gamma(mp.mpf(beta)) * phi(rho, beta, z)


def main():
    print("cross-check against scipy where available")
    try:
        from scipy.special import wright_bessel
        for r, b, z in ((1.0, 1.0, 2.0), (0.5, 1.5, 3.0), (2.0, 0.5, 10.0)):
            got = float(phi(r, b, z))
            ref = float(wright_bessel(r, b, z))
            print(f"  rho={r} beta={b} z={z}: mp {got:.12g} scipy {ref:.12g} "
                  f"rel {abs(got / ref - 1):.1e}")
    except ImportError:
        print("  scipy not installed, skipped")

    print("\nz-axis: ln Phi vs Gamma(beta) Phi, per (rho, beta) corner")
    print("  (the log is the standard move and it loses badly here)")
    for lo, hi in PANELS:
        dl = max(tail(fit_axis(lambda z: log_phi(r, b, z), NODES, lo, hi))
                 for r in RHOS for b in BETAS)
        dg = max(tail(fit_axis(lambda z: gphi(r, b, z), NODES, lo, hi))
                 for r in RHOS for b in BETAS)
        rng = max(float(gphi(r, b, hi) / gphi(r, b, max(lo, 1e-30)))
                  for r in RHOS for b in BETAS)
        print(f"  z in [{lo:4g}, {hi:4g}]: ln Phi {dl:4d}   Gamma(beta) Phi "
              f"{dg:3d}   (G spans {rng:.3g} over the panel)")

    print("\n  the ln Phi degrees by beta, on [0, 8] (the pole end is the bad one)")
    for r in RHOS:
        ds = [tail(fit_axis(lambda z: log_phi(r, b, z), NODES, 0.0, 8.0))
              for b in BETAS]
        print(f"    rho={r:4}: " + " ".join(f"beta={b}:{d}"
                                            for b, d in zip(BETAS, ds)))

    print("\nparameter axes for G, beta direct vs through ln beta")
    for z in (0.5, 4.0, 20.0):
        dr = max(tail(fit_axis(lambda r: gphi(r, b, z), NODES, 0.1, 2.0))
                 for b in BETAS)
        db = max(tail(fit_axis(lambda b: gphi(r, b, z), NODES, 0.1, 5.0))
                 for r in RHOS)
        dlb = max(tail(fit_axis(lambda lb: gphi(r, mp.e ** lb, z), NODES,
                                float(mp.log(0.1)), float(mp.log(5.0))))
                  for r in RHOS)
        print(f"  z={z:5g}: rho {dr}, beta {db}, ln beta {dlb}")

    print("\nrho panels, since rho is the expensive axis")
    for rlo, rhi in ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0)):
        d = max(tail(fit_axis(lambda r: gphi(r, b, z), NODES, rlo, rhi))
                for b in BETAS for z in (0.5, 4.0, 20.0))
        print(f"  rho on [{rlo}, {rhi}]: worst degree {d}")


if __name__ == "__main__":
    main()
