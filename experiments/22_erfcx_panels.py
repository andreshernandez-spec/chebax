#!/usr/bin/env python3
"""erfcx in t = K / (K + x): one series or t-panels, against the two-table erfcx.

WHAT THIS MEASURES
------------------
chebax.erfcx up to 0.4.1: a series on [0, 6] and a tail series in (6/x)^2,
a hard select between them, and the reflection 2 e^{x^2} - erfcx(-x) for
x < 0. Since 2026-10-31 it maps all of [0, inf) onto t in (0, 1]
(Schonfelder, Math. Comp. 32, 1978):

    erfcx(x) = t h(t),  t = K / (K + x),  h(t) = erfcx(x) (K + x) / K

h is smooth on [0, 1] (h(1) = 1, h(0) = 1 / (K sqrt(pi))), so no select on
x is needed. One series in t takes 26 terms; chebax uses 32 equal t-panels
of 8 terms, picked by floor(32 t). The reflection splits x^2 exactly before
the exp.

1. Degrees: Chebyshev terms of h above 2^-56 min h, for one series by K,
   and per panel (the worst panel) by the number of panels at K = 3.
2. Values and derivative against mpmath (40 dps; 60 for the derivative,
   whose identity 2 x erfcx - 2/sqrt(pi) cancels at large x), pointwise
   relative and in ulps of the reference: the two-table erfcx (refitted
   here as its generator did), one series with x^2 rounded and with x^2
   split, chebax.erfcx (32 panels, x^2 split), jax.scipy.special.erfcx.
3. erfc from the panels, erfc(x) = t h(t) e^{-x^2} for x >= 0 and
   2 - erfc(-x) below, x^2 split exactly (Dekker): what a compiled kernel
   built on the table computes before any compensated arithmetic. Against
   XLA's erfc. CPU only.
4. Speed under jit, values and the gradient of the sum: GPU (N = 2^22) and
   one CPU thread (N = 2^20), x ~ normal(0, 3).

FINDINGS (2026-11-01, RTX 3080 Laptop, jax 0.11.0)
--------------------------------------------------
results/22_erfcx_panels.txt. One series takes 26 terms at K = 3 (25-26
for K in [2.5, 5]); the worst of 32 panels takes 8 for values, and the
table keeps 9 so the derivative's worst case is 7.0e-16 rather than
6.8e-15 at a panel edge. Values: 4.4e-16 for x < 0 and 5.8e-16 for x >= 0
against the two tables' 5.7e-14 and 4.8e-16; the x < 0 gain is the split
x^2 (the rounded x^2 costs 5.7e-14 at x = -22.75 with any series).
Derivative 5.9e-16 / 7.0e-16 against 5.7e-14 / 1.4e-14. Under jit XLA
folds e^hi e^lo into e^(hi + lo), which undoes the split, so the library
writes e^hi (1 + expm1(lo)), expm1 by three terms of its series. jax
0.11's own erfcx returns 0 for x in [26.543, 26.642] (erfc underflows
before its asymptotic branch starts at 26.64; jax has since rewritten
erfcx). Speed against the two tables, values / gradient: CPU 0.52x /
0.46x the time, GPU 0.42x / 0.52x; faster than jax's own erfcx on the GPU.

WHAT IT DOES NOT ESTABLISH
--------------------------
float32, other hardware than this laptop (consumer fp64 at 1:64), jax
releases after 0.11, and anything about compiled kernels outside XLA (the
erfc of section 3 is the formula, not a fast kernel).
"""

import os

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import math
import sys
import time

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

jax.config.update("jax_enable_x64", True)

import chebax
from chebax._src.recipes import erf_family
from chebax._src.recipes import erf_table as et
from chebax._src.recipes._gen_common import DPS, dct, nodes
from chebax._src.series import ChebSeries

TOL = 2.0**-56
XMAX = 27.5  # erfc(27.5) rounds to 0


def erfcx_mp(x):
    # mpmath's erfc overflows a float internally near x ~ 1e300; past 1e5 the
    # asymptotic series is exact far beyond f64
    x = mp.mpf(x)
    if x > 1e5:
        y = 1 / (2 * x * x)
        return (1 - y + 3 * y * y - 15 * y**3) / (x * mp.sqrt(mp.pi))
    return mp.erfc(x) * mp.exp(x * x)


def h_mp(K, t):
    a = K * (1 - t) / t
    return mp.erfc(a) * mp.exp(a * a) * (K + a) / K


def terms(c, hmin):
    m = len(c)
    while m > 1 and abs(c[m - 1]) < TOL * hmin:
        m -= 1
    return m


def h_terms(K):
    with mp.workdps(DPS):
        K = mp.mpf(K)
        c = [float(v) for v in dct(mp, [h_mp(K, (u + 1) / 2) for u in nodes(mp, 64)])]
    return terms(c, 1 / (float(K) * math.sqrt(math.pi)))


def panel_terms(P, K=3):
    """Terms of the worst panel among P equal t-panels, each fitted on 24 nodes."""
    with mp.workdps(DPS):
        K = mp.mpf(K)
        worst = 0
        for j in range(P):
            lo, hi = mp.mpf(j) / P, mp.mpf(j + 1) / P
            c = [float(v) for v in dct(mp, [h_mp(K, lo + (u + 1) / 2 * (hi - lo)) for u in nodes(mp, 24)])]
            worst = max(worst, terms(c, float(min(h_mp(K, hi), h_mp(K, lo) if lo else 1 / (K * mp.sqrt(mp.pi))))))
    return worst


def one_series_fit(K=3.0, n=32):
    with mp.workdps(DPS):
        return np.array([float(v) for v in dct(mp, [h_mp(mp.mpf(K), (u + 1) / 2) for u in nodes(mp, n)])])


def two_tables():
    """chebax.erfcx up to 0.4.1, refitted as erf_gen did then."""
    with mp.workdps(DPS):
        xs = mp.mpf(6)
        c = dct(mp, [erfcx_mp((t + 1) / 2 * xs) for t in nodes(mp, 40)])
        h = dct(mp, [xs / mp.sqrt((t + 1) / 2) * erfcx_mp(xs / mp.sqrt((t + 1) / 2)) for t in nodes(mp, 16)])
    C = ChebSeries(np.array([float(v) for v in c]), (0.0, 6.0))
    H = ChebSeries(np.array([float(v) for v in h]), (0.0, 1.0))

    def erfcx(x):
        ax = jnp.where(x >= 0, x, -x)
        inner = C(jnp.where(ax <= 6.0, ax, 6.0))
        xo = jnp.where(ax <= 6.0, 6.0, ax)
        pos = jnp.where(ax <= 6.0, inner, H((6.0 / xo) ** 2) / xo)
        xn = jnp.where(x < 0, x, 0.0)
        return jnp.where(x >= 0, pos, 2.0 * jnp.exp(xn * xn) - pos)

    return erfcx


T = ChebSeries(one_series_fit(), (0.0, 1.0))


def one_pos(a):
    t = 3.0 / (3.0 + a)
    return t * T(t)


def one_rounded(x):
    ax = jnp.where(x >= 0, x, -x)
    xn = jnp.where(x < 0, x, 0.0)
    return jnp.where(x >= 0, one_pos(ax), 2.0 * jnp.exp(xn * xn) - one_pos(ax))


def one_split(x):
    ax = jnp.where(x >= 0, x, -x)
    xn = jnp.where(x < 0, x, 0.0)
    return jnp.where(x >= 0, one_pos(ax), 2.0 * erf_family._exp_sq(xn) - one_pos(ax))


def panel_erfc(x):
    """erfc from chebax.erfcx's panels (x >= 0 lanes) times e^{-x^2}, x^2 split."""
    a = jnp.minimum(jnp.where(x >= 0, x, -x), XMAX)
    c = 134217729.0 * a
    ah = c - (c - a)
    al = a - ah
    hi = a * a
    lo = ((ah * ah - hi) + 2.0 * ah * al) + al * al
    y = chebax.erfcx(a) * (jnp.exp(-hi) * (1.0 - lo))
    return jnp.where(x >= 0, y, 2.0 - y)


def grid():
    pos = np.concatenate([np.linspace(0.0, 30.0, 6001), np.geomspace(1e-300, 1.0, 200),
                          np.geomspace(1.0, 1e300, 400), [6.0, 6.0001, np.nextafter(6.0, 0)]])
    neg = -np.concatenate([np.linspace(1e-3, 26.5, 2001), np.geomspace(1e-300, 1.0, 100)])
    return np.unique(np.concatenate([pos, neg]))


XG = grid()


def report(name, got, ref):
    got = np.asarray(got)
    for side, m in (("x < 0", XG < 0), ("x >= 0", XG >= 0)):
        ok = m & np.isfinite(ref) & (ref != 0)
        rel = np.abs(got[ok] - ref[ok]) / np.abs(ref[ok])
        ulp = np.abs(got[ok] - ref[ok]) / np.spacing(np.abs(ref[ok]))
        print(f"   {name:30s} {side:6s} max rel {rel.max():.1e} ({ulp.max():.0f} ulp) at x = "
              f"{XG[ok][np.argmax(rel)]:.6g}; mean {ulp.mean():.2f} ulp")


def time_fn(f, x, reps=20):
    f(x).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(x).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    print(f"jax {jax.__version__}, mpmath {mp.__version__}, devices {jax.devices()}")
    print("\n1. terms of h above 2^-56 min h (fit on 64 nodes)")
    print("   one series: " + "; ".join(f"K = {K}: {h_terms(K)}" for K in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)))
    print("   worst panel, K = 3: " + "; ".join(f"{P} panels: {panel_terms(P)}" for P in (8, 16, 32, 64)))
    print(f"   the table: K = {et.K}, {et.NP} panels x {et.ERFCX_P.shape[1]} terms")

    two = two_tables()
    fns = {"two tables (<= 0.4.1)": two, "one series, x^2 rounded": one_rounded,
           "one series, x^2 split": one_split, "chebax.erfcx (32 panels)": chebax.erfcx,
           "jax.scipy.special.erfcx": jax.scipy.special.erfcx}
    with mp.workdps(40):
        ref = np.array([float(erfcx_mp(x)) for x in XG])
        ref_c = np.array([float(mp.erfc(mp.mpf(x))) if x < 1e3 else 0.0 for x in XG])
    xd = XG[np.abs(XG) <= 1e6]
    with mp.workdps(60):
        ref_d = np.array([float(2 * mp.mpf(x) * erfcx_mp(x) - 2 / mp.sqrt(mp.pi)) for x in xd])

    print(f"\n2. erfcx on {XG.size} points in [-26.5, 1e300], against mpmath")
    with jax.default_device(jax.devices("cpu")[0]):
        xs = jnp.asarray(XG)
        for name, f in fns.items():
            report(name, jax.jit(f)(xs), ref)
        print(f"   derivative on {xd.size} points with |x| <= 1e6, against 2 x erfcx - 2/sqrt(pi)")
        for name, f in fns.items():
            g = np.asarray(jax.jit(jax.vmap(jax.grad(f)))(jnp.asarray(xd)))
            rel = np.abs(g - ref_d) / np.abs(ref_d)
            neg, pos_ = xd < 0, xd >= 0
            print(f"   {name:30s} x < 0 max rel {rel[neg].max():.1e} at x = {xd[neg][np.argmax(rel[neg])]:.6g}; "
                  f"x >= 0 {rel[pos_].max():.1e} at x = {xd[pos_][np.argmax(rel[pos_])]:.6g}")
        jz = np.asarray(jax.jit(jax.scipy.special.erfcx)(jnp.asarray(np.linspace(26.5, 26.7, 2001))))
        zx = np.linspace(26.5, 26.7, 2001)[jz == 0]
        print(f"   jax.scipy.special.erfcx == 0 on [26.5, 26.7] (step 1e-4): {zx.size} points, "
              f"x in [{zx.min():.4f}, {zx.max():.4f}]" if zx.size else "   jax erfcx: no zeros on [26.5, 26.7]")

        print("\n3. erfc from the panels, against mpmath")
        normal = ref_c >= np.finfo(float).tiny
        for name, f in (("panels, x^2 split", panel_erfc), ("XLA erfc", jax.scipy.special.erfc)):
            v = np.asarray(jax.jit(f)(xs))
            rel = np.abs(v[normal] - ref_c[normal]) / ref_c[normal]
            ulp = np.abs(v[normal] - ref_c[normal]) / np.spacing(ref_c[normal])
            print(f"   {name:30s} normal results: max rel {rel.max():.1e} ({ulp.max():.0f} ulp) at x = "
                  f"{XG[normal][np.argmax(rel)]:.6g}; mean {ulp.mean():.2f} ulp")
        print("   (subnormal results are flushed to 0 by XLA on the CPU, in both)")

    print("\n4. ms per call under jit, values / gradient of the sum, x ~ normal(0, 3), median of 20")
    fns = {**fns, "panels erfc": panel_erfc, "XLA erfc": jax.scipy.special.erfc}
    for dev, n in ((jax.devices("cpu")[0], 1 << 20), *[(d, 1 << 22) for d in jax.devices() if d.platform == "gpu"]):
        with jax.default_device(dev):
            x = jax.device_put(jnp.asarray(np.random.default_rng(0).normal(0.0, 3.0, n)), dev)
            print(f"   {dev.platform}, N = {n}")
            for name, f in fns.items():
                tv = time_fn(jax.jit(f), x)
                tg = time_fn(jax.jit(jax.grad(lambda z, f=f: jnp.sum(f(z)))), x)
                print(f"   {name:30s} {tv * 1e3:8.3f} / {tg * 1e3:8.3f}")


if __name__ == "__main__":
    sys.exit(main())
