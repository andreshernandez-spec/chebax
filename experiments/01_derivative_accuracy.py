#!/usr/bin/env python3
"""
Experiment 01 — Is the derivative of the Chebyshev approximant usable as THE gradient?

WHAT IS MEASURED
----------------
Fit g_v(z) = 0F1(; v+1; -z/4) on z in [0, 64] (i.e. x in [0, 8], z = x^2) at degree 20,
apply the exact Chebyshev derivative recurrence (chebder) to the coefficients, and
assemble both

    J_v(x)  = (x/2)^v / Gamma(v+1) * g_v(z)
    J_v'(x) = (x/2)^v / Gamma(v+1) * [ (v/x) g_v(z) + 2x g_v'(z) ]

from the two coefficient vectors. Compare against mpmath (40 dps) on a dense grid.
Errors are sup-normalized (|err| / max|f| over the grid), the absolute-accuracy
contract of ../../bessel/PROJECT.md section 4 q4. Two pipelines: f64 coefficients with
f64 Clenshaw, and f32 coefficients with f32 Clenshaw. The prefactor is computed in f64
in BOTH cases, so the numbers isolate the polynomial pipeline.

WHY IT MATTERS
--------------
If the differentiated series is machine-accurate at the same degree, then gradient
support costs one extra small coefficient vector (computed once at build time) plus one
extra Clenshaw per evaluation. No autodiff tape, no finite differences. jax.grad
through Clenshaw computes the same polynomial, so binding custom_jvp to the chebder
series is exactly consistent with what optimizers see.

WHAT THIS DOES NOT ESTABLISH
----------------------------
GPU cost (bessel B3 gates all speed claims); behavior for x > 8 or orders outside
{0.5, 2.5}; Y/I/K; derivatives beyond the first (chebder iterates, but not measured
here); the error contribution of an f32 prefactor.

Run:  python 01_derivative_accuracy.py     (~5 s)
"""
import numpy as np
import mpmath as mp

mp.mp.dps = 40

ZMAX = mp.mpf(64)  # z = x^2, x in [0, 8]


def g(v, z):
    return mp.hyp0f1(v + 1, -z / 4)


def dct_fit(samples):
    """Plain Chebyshev coefficients from samples at n Chebyshev-Gauss nodes."""
    n = len(samples)
    c = []
    for j in range(n):
        s = mp.fsum(samples[i] * mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n))
        c.append(2 * s / n)
    c[0] /= 2
    return c


def fit_g(v, n):
    tk = [mp.cos(mp.pi * (2 * i + 1) / (2 * n)) for i in range(n)]
    return dct_fit([g(v, (t + 1) / 2 * ZMAX) for t in tk])


def chebder_mp(c):
    """Derivative of a plain Chebyshev series on [-1,1] (numpy's algorithm, in mp)."""
    c = [mp.mpf(x) for x in c]
    n = len(c) - 1
    der = [mp.mpf(0)] * n
    for j in range(n, 2, -1):
        der[j - 1] = 2 * j * c[j]
        c[j - 2] += j * c[j] / (j - 2)
    if n > 1:
        der[1] = 4 * c[2]
    der[0] = c[1]
    return der


def clenshaw(c, t):
    """Clenshaw in whatever dtype c and t carry (exercises the f32 pipeline)."""
    b1 = t * 0
    b2 = t * 0
    for cj in c[:0:-1]:
        b1, b2 = 2 * t * b1 - b2 + cj, b1
    return t * b1 - b2 + c[0]


def main():
    print("Derivative of the Chebyshev approximant vs mpmath, x in [0.05, 8], 160 pts")
    print("deg=20 fit of g_v(z)=0F1(;v+1;-z/4) on z in [0,64]; errors sup-normalized.")
    print("Prefactor (x/2)^v/Gamma(v+1) computed in f64 in all rows.\n")
    zmax = float(ZMAX)
    for v in (0.5, 2.5):
        deg = 20
        c = fit_g(v, deg + 1)
        dc = chebder_mp(c)
        c64 = np.array([float(x) for x in c])
        dc64 = np.array([float(x) for x in dc])
        xs = np.linspace(0.05, 8.0, 160)
        J = np.array([float(mp.besselj(v, mp.mpf(x))) for x in xs])
        Jp = np.array([float(mp.besselj(v, mp.mpf(x), 1)) for x in xs])
        pref = np.array([float((mp.mpf(x) / 2) ** v / mp.gamma(v + 1)) for x in xs])
        t = 2 * xs**2 / zmax - 1
        for tag, cc, dcc, tt in (
            ("f64", c64, dc64, t),
            ("f32", c64.astype(np.float32), dc64.astype(np.float32), t.astype(np.float32)),
        ):
            pv = clenshaw(cc, tt).astype(np.float64)
            pdv = clenshaw(dcc, tt).astype(np.float64)
            Jh = pref * pv
            Jph = pref * ((v / xs) * pv + (4 * xs / zmax) * pdv)
            ev = np.max(np.abs(Jh - J)) / np.max(np.abs(J))
            ep = np.max(np.abs(Jph - Jp)) / np.max(np.abs(Jp))
            print(f"  v={v:3.1f}  deg={deg}  [{tag}]   J err: {ev:8.2e}   J' err: {ep:8.2e}")
    print("\nVerdict: no degree penalty for the first derivative; f32 floor ~1e-6.")


if __name__ == "__main__":
    main()
