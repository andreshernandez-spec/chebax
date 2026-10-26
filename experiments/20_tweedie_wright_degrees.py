#!/usr/bin/env python3
"""Design measurements for the Tweedie slice of Wright's function.

    W(rho, z) = Phi(rho, 0; z) = sum_{n>=1} z^n / (n! Gamma(rho n))

The Tweedie compound Poisson-gamma density for 1 < p < 2 is

    f(y) = exp(-lambda - y/theta) / y * W(rho, z),   y > 0,
    lambda = mu^(2-p) / (phi (2-p)),  theta = phi (p-1) mu^(p-1),
    rho = (2-p)/(p-1),  z = lambda (y/theta)^rho,

and f(0) = exp(-lambda). So the density needs only beta = 0 of the
three-argument function that experiments/17 measured, and it needs z far
past that experiment's [0, 30]: z = lambda (y/theta)^rho reaches 1e9 at
y = 1 for p = 1.1, mu = phi = 1.

WHAT THIS MEASURES
------------------
1. The identity above, against the Poisson-gamma mixture it comes from.
2. The large-z remainder. A saddle point of the Hankel integral gives

       ln W ~ Z + ln(t0)/2 - ln(2 pi (1+rho))/2,
       t0 = (rho z)^(1/(1+rho)),  Z = (1+rho) t0 / rho,

   and R = ln W - (that) must go to 0. It does, like c1(rho)/Z, with
   c1(rho) = c1(1/rho) (the table below shows the symmetry).
3. Where to split. The series' terms peak at n0 = Z/(1+rho) with variance
   sigma^2 = Z/(1+rho)^2. Two things go wrong at the split if it is
   placed carelessly:
   - large rho (p near 1): when sigma < ~1 the sum is a few discrete
     terms, and it differs from its smooth saddle approximation by a
     ripple of size ~exp(-2 pi^2 sigma^2), one period per term (Poisson
     summation). A split at fixed n0 leaves that ripple in the tail and
     no degree resolves it (63 of 64 nodes at rho = 19 even at n0 = 16).
     The split must hold sigma^2 fixed instead: Z1 >= S1 (1+rho)^2.
   - small rho (p near 2): sigma^2 = Z there, so a sigma^2-only split
     lands at Z ~ 2 where the large-Z expansion is still poor
     (R ~ -0.5) and the tail needs degree 57.
   Z1(rho) = Za + S1 (1+rho)^2 handles both, and is smooth in rho, which
   the 2-D table needs (a max() would put a kink across the rho axis).
4. Degrees of the tail tabuland in s = Z1(rho)/Z in (0, 1] and in the
   parameter axis, for R and for T = Z R (T -> c1(rho) as Z -> inf, which
   takes the 1/Z1(rho) variation out of the rho direction).
5. How many terms an exact direct sum needs below the split, which is the
   inner region: n = 1..N with lgamma(rho n) per CALL (rho is uniform), so
   the per-point work is N exp and multiply-adds.

FINDINGS (2026-10-26, reproduced by this script)
------------------------------------------------
- The identity holds to 1e-40.
- Parameter axis: ln rho, not p (50 against 63 over p in [1.05, 1.95]).
  ln rho = -logit(p - 1), and the c1 symmetry is rho <-> 1/rho.
- Chosen: Za = 12, S1 = 2, T = Z R, p in [1.05, 1.95]. Tail degree 24 in s
  (at rho = 0.053, falling to 5 at rho = 19) and 51 in ln rho (at the split
  edge, s = 1). The direct sum needs at most 54 terms there.
- S1 = 3 needs 76 terms at p = 1.05 for no gain in s.

WHAT IT DOES NOT MEASURE
------------------------
Runtime accuracy or speed; no table is built here (the generator,
chebax._src.recipes.tweedie_gen, does that with node counts taken from
these degrees). p outside [1.05, 1.95]: the direct sum grows like 2(1+rho)
terms as p -> 1 and the rho axis widens.

Run:  python experiments/20_tweedie_wright_degrees.py   (~30 s, CPU)
"""

import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, "src")
from chebax._src.recipes._gen_common import DPS, dct, nodes  # noqa: E402

mp.mp.dps = DPS
NODES = 64


def tail(coeffs):
    a = np.abs([float(x) for x in coeffs])
    a /= max(a.max(), 1e-300)
    idx = np.nonzero(a > 1e-15)[0]
    return int(idx.max()) if idx.size else -1


def log_w(rho, z):
    """ln W by summing a window around the peak term; the terms are
    log-concave in n, so everything outside the window is below 1e-52 of
    the peak and the sum costs O(sqrt(n0)) terms even at z ~ 1e9."""
    rho, z = mp.mpf(rho), mp.mpf(z)
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


def lead(rho, z):
    rho, z = mp.mpf(rho), mp.mpf(z)
    t0 = (rho * z) ** (1 / (1 + rho))
    big_z = (1 + rho) * t0 / rho
    return big_z + mp.log(t0) / 2 - mp.log(2 * mp.pi * (1 + rho)) / 2


def z_of(rho, big_z):
    rho, big_z = mp.mpf(rho), mp.mpf(big_z)
    return (rho * big_z / (1 + rho)) ** (1 + rho) / rho


def rem(rho, big_z):
    z = z_of(rho, big_z)
    return log_w(rho, z) - lead(rho, z)


def rho_of_p(p):
    p = mp.mpf(p)
    return (2 - p) / (p - 1)


def fit(f, lo, hi):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    return dct(mp, [f(lo + (t + 1) / 2 * (hi - lo)) for t in nodes(mp, NODES)])


def terms_needed(rho, z):
    rho, z = mp.mpf(rho), mp.mpf(z)
    lz = mp.log(z)
    lt = [n * lz - mp.loggamma(n + 1) - mp.loggamma(rho * n) for n in range(1, 400)]
    m = max(lt)
    w = [mp.exp(v - m) for v in lt]
    total, acc = mp.fsum(w), mp.mpf(0)
    for n in range(len(w), 0, -1):
        acc += w[n - 1]
        if acc > mp.mpf(10) ** -17 * total:
            return n
    return 0


def main():
    print("1) Tweedie density two ways (mu=1.3, phi=0.7, p=1.4, y=2.1)")
    mu, phi, p, y = mp.mpf("1.3"), mp.mpf("0.7"), mp.mpf("1.4"), mp.mpf("2.1")
    lam = mu ** (2 - p) / (phi * (2 - p))
    rho, theta = rho_of_p(p), phi * (p - 1) * mu ** (p - 1)
    mix = mp.nsum(lambda n: mp.exp(-lam) * lam ** n / mp.factorial(n)
                  * y ** (n * rho - 1) * mp.exp(-y / theta)
                  / (mp.gamma(n * rho) * theta ** (n * rho)), [1, mp.inf])
    via = mp.exp(-lam - y / theta + log_w(rho, lam * (y / theta) ** rho)) / y
    print(f"   mixture {mp.nstr(mix, 16)}   via W {mp.nstr(via, 16)}   "
          f"rel {mp.nstr(abs(mix / via - 1), 2)}")

    rhos = (0.0526, 0.111, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 19.0)
    print("\n2) R = ln W - leading saddle term, by Z (rows) and rho (columns)")
    print("        Z" + "".join(f"{r:>10}" for r in rhos))
    for big_z in (1, 4, 16, 64, 256, 1024):
        print(f"   {big_z:6g}" + "".join(f"{float(rem(r, big_z)):10.2e}" for r in rhos))

    print("\n3) split placement: tail degree in s = Z1/Z, by rho "
          f"{(0.0526, 0.25, 1.0, 4.0, 19.0)}")
    for label, z1 in (("n0 = 16 (Z1 = 16(1+rho))", lambda r: 16 * (1 + r)),
                      ("sigma^2 = 2 (Z1 = 2(1+rho)^2)", lambda r: 2 * (1 + r) ** 2),
                      ("Z1 = 12 + 2(1+rho)^2", lambda r: 12 + 2 * (1 + r) ** 2)):
        degs = [tail(fit(lambda s: rem(r, z1(mp.mpf(r)) / s), 0, 1))
                for r in (0.0526, 0.25, 1.0, 4.0, 19.0)]
        print(f"   {label:30s} {degs}")

    print("\n4) chosen split Z1 = 12 + 2(1+rho)^2, p in [1.05, 1.95]")
    a0, a1 = mp.log(rho_of_p("1.95")), mp.log(rho_of_p("1.05"))

    def z1(r):
        return 12 + 2 * (1 + r) ** 2

    for name, g in (("R", lambda r, s: rem(r, z1(r) / s)),
                    ("T = Z R", lambda r, s: z1(r) / s * rem(r, z1(r) / s))):
        by_lnrho = [tail(fit(lambda a: g(mp.e ** a, mp.mpf(s)), a0, a1))
                    for s in ("0.01", "0.5", "1")]
        by_p = [tail(fit(lambda pp: g(rho_of_p(pp), mp.mpf(s)), "1.05", "1.95"))
                for s in ("0.01", "0.5", "1")]
        by_s = [tail(fit(lambda s: g(mp.mpf(r), s), 0, 1))
                for r in (float(mp.e ** a0), 0.25, 1.0, 4.0, float(mp.e ** a1))]
        print(f"   {name:8s} ln-rho axis at s = (0.01, 0.5, 1): {by_lnrho}   "
              f"p axis: {by_p}   s axis at rho = (0.053, 0.25, 1, 4, 19): {by_s}")

    print("\n5) direct-sum terms for 1e-17 truncation at the split, by p")
    for za, s1 in ((12, 3), (12, 2)):
        need = {pp: terms_needed(rho_of_p(pp), z_of(rho_of_p(pp), za + s1 * (1 + rho_of_p(pp)) ** 2))
                for pp in ("1.05", "1.1", "1.2", "1.5", "1.8", "1.9", "1.95")}
        print(f"   Z1 = {za} + {s1}(1+rho)^2: {need}")


if __name__ == "__main__":
    main()
