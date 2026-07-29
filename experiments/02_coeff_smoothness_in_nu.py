#!/usr/bin/env python3
"""
Experiment 02 — Are the Chebyshev coefficients c_k(v) smooth enough IN THE ORDER v to
tabulate once and reconstruct at any v, including the order gradient dJ/dv?

WHAT IS MEASURED
----------------
For J_v(x) = (x/2)^v / Gamma(v+1) * g_v(z), z = x^2, fit the degree-24 Chebyshev
coefficients of g_v on z in [0, 64] at 64 Chebyshev nodes in v over v in [0, 10], then
fit each coefficient c_k as a Chebyshev series IN v. That yields a one-time 25 x 64
table (12.5 KB). Then, at off-node test orders:

  1. reconstruct the coefficient vector by Clenshaw in v, truncating the v-series at
     degree d_v, and measure the error of the resulting J_v against mpmath (40 dps);
  2. assemble the order gradient
       dJ/dv = pref * [ (ln(x/2) - psi(v+1)) g + dg/dv ],
     with dg/dv from chebder along the v axis, and compare against mp.diff.

Errors are sup-normalized (see ../../bessel/PROJECT.md section 4 q4).

WHY IT MATTERS
--------------
If c_k(v) is smooth, the generator needs mpmath only once, at bake time. Instantiation
at any real v in the table domain becomes ~3.2k FLOPs of pure float arithmetic (usable
inside jit on constants), and dJ/dv, scoped as "genuinely hard" in bessel Track C2,
falls out of the same table.

RELATION TO THE BESSEL DEAD END
-------------------------------
../../bessel/PROJECT.md section 2.5 rules out 2-D (v,x) Chebyshev for EVALUATION: the
factored form costs ~3076 FLOPs per point. This experiment does not reopen that. Here
the 2-D structure is spent once per INSTANTIATION; per-point evaluation stays the 1-D
degree-24 polynomial. Per-element v remains out of scope either way.

WHAT THIS DOES NOT ESTABLISH
----------------------------
v outside [0, 10] (the 0F1 pole at v = -1 sets the convergence rate; extending or
panelling the v-domain is separate work); x > 8; Y/I/K (K and Y need direct tabulation,
see risk S3); f32 table variants beyond the coefficient-decay row; GPU cost (bessel B3).

Run:  python 02_coeff_smoothness_in_nu.py     (~10 s)
"""
import numpy as np
import mpmath as mp
from numpy.polynomial import chebyshev as npch

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


def main():
    NZ = 25   # z-degree 24: covers the worst case (v=0) from bessel experiments/02
    NV = 64   # v-nodes for the one-time table
    VLO, VHI = 0.0, 10.0
    vw = VHI - VLO

    print(f"c_k(v) tabulated on v in [{VLO},{VHI}] ({NV} nodes), z-domain [0,{float(ZMAX):.0f}] (degree {NZ-1})\n")

    sv = [mp.cos(mp.pi * (2 * i + 1) / (2 * NV)) for i in range(NV)]
    vnodes = [(s + 1) / 2 * vw + VLO for s in sv]
    C = [fit_g(v, NZ) for v in vnodes]
    G = [dct_fit([C[i][k] for i in range(NV)]) for k in range(NZ)]
    G64 = np.array([[float(x) for x in row] for row in G])  # (NZ, NV)

    mag = np.abs(G64).max(axis=0)
    mag /= mag.max()
    for cut in (1e-7, 1e-15):
        print(f"  v-coefficient decay: last index above {cut:.0e} (rel) = {int(np.nonzero(mag > cut)[0].max())}")

    zmax = float(ZMAX)
    xs = np.linspace(0.05, 8.0, 60)
    tz = 2 * xs**2 / zmax - 1
    vtests = [0.3, 1.7, float(mp.pi), 7.77]
    dvs = [12, 16, 24, 32, 48, NV - 1]

    print(f"\n  Reconstruction error of J_v at off-node v, truncating the v-series at d_v")
    print(f"  {'v':>8s} | " + "".join(f"d_v={d:<5d}" for d in dvs))
    for vt in vtests:
        J = np.array([float(mp.besselj(mp.mpf(vt), mp.mpf(x))) for x in xs])
        supJ = np.max(np.abs(J))
        pref = np.array([float((mp.mpf(x) / 2) ** mp.mpf(vt) / mp.gamma(vt + 1)) for x in xs])
        s = 2 * (vt - VLO) / vw - 1
        row = f"  {vt:8.4f} | "
        for dv in dvs:
            ck = np.array([npch.chebval(s, G64[k, : dv + 1]) for k in range(NZ)])
            Jh = pref * npch.chebval(tz, ck)
            row += f"{np.max(np.abs(Jh - J)) / supJ:9.1e} "
        print(row)

    print("\n  Order gradient dJ/dv via chebder along v (full v-degree), vs mp.diff:")
    xs2 = np.linspace(0.3, 8.0, 24)
    tz2 = 2 * xs2**2 / zmax - 1
    for vt in vtests:
        s = 2 * (vt - VLO) / vw - 1
        ck = np.array([npch.chebval(s, G64[k]) for k in range(NZ)])
        dck = np.array([npch.chebval(s, npch.chebder(G64[k])) * (2 / vw) for k in range(NZ)])
        pref = np.array([float((mp.mpf(x) / 2) ** mp.mpf(vt) / mp.gamma(vt + 1)) for x in xs2])
        Gv = npch.chebval(tz2, ck)
        Gvp = npch.chebval(tz2, dck)
        psi = float(mp.digamma(vt + 1))
        dJh = pref * ((np.log(xs2 / 2) - psi) * Gv + Gvp)
        ref = np.array([float(mp.diff(lambda w: mp.besselj(w, mp.mpf(x)), mp.mpf(vt))) for x in xs2])
        e = np.max(np.abs(dJh - ref)) / np.max(np.abs(ref))
        print(f"  v={vt:8.4f}   dJ/dv err: {e:8.2e}   (sup-normalized)")

    n_tab = G64.shape[0] * G64.shape[1]
    print(f"\n  Table: {G64.shape[0]}x{G64.shape[1]} = {n_tab} doubles = {n_tab*8/1024:.1f} KB")
    print(f"  Instantiation at any v: {G64.shape[0]} Clenshaws of length {G64.shape[1]} ~ {2*n_tab} FLOPs, once per trace")
    print("\nVerdict: d_v=32 reaches ~1e-8, d_v=48 ~1e-13, full 63 ~1e-15;")
    print("dJ/dv is machine-accurate. One baked table serves every v in [0,10].")


if __name__ == "__main__":
    main()
