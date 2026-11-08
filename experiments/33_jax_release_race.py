#!/usr/bin/env python3
"""chebax against jax's own special functions, jax 0.11.0 (July 2026) and
0.12.0.1 (October 2026, the latest release on 2026-11-06): accuracy and
speed of values and gradients, float64 and float32.

WHAT THIS MEASURES
------------------
Every chebax function that has a counterpart in jax.scipy:

    erfcx, dawsn          jax.scipy.special.erfcx, dawsn
    besseli, orders 0, 1  i0, i1, i0e, i1e (scaled=True for the e forms)
    besselj, n = 0..10    bessel_jn(x, v=n)[n] (default n_iter=50)
    betainc               betainc (jax has no gradient in a or b)
    gammainc, gammaincc   gammainc, gammaincc; chebax's one-a path (a
                          traced scalar) and its per-element path
                          (gammainc_array, gammaincc_array)
    hyp1f1                hyp1f1
    stdtr                 jax has no t.cdf: numpyro's StudentT.cdf formula
                          over jax's betainc, so no gradient in nu

1. Accuracy against mpmath at 40 dps (60 for the erfcx and dawsn
   derivatives, whose identities cancel). Grids are rounded to float32
   first, so one reference serves both dtypes. Values and every gradient
   the implementation supports, per point (vmap of grad), on the CPU and
   the GPU; the tables show the worse of the two. Metrics follow the test
   files: relative for erfcx, dawsn, I and their derivatives (a window of
   1% around an interior zero of the reference is skipped); sup-normalized
   per order for J and J'; absolute for CDF values (relative also shown)
   and their parameter derivatives; densities relative where they are at
   least 1e-4 of the CDF, absolute in the saturated wedge (F ~ 1);
   hyp1f1 values relative / max(1, |ln M|), its derivatives as errors in
   d ln M / dq, absolute below 1. Points where the reference or the
   function value is outside the dtype's normal range are skipped. A nan
   or inf where the reference is finite is counted, not hidden.
2. Speed under jit: the sum of values, and value_and_grad of the sum in
   the arguments jax can differentiate (chebax also in all of its
   arguments, where that is more), ns per element (median), one pinned CPU
   core at N = 2^16 and the GPU at N = 2^20. Parameters are traced 0-d
   arrays (a latent parameter), orders are constants. Compile time of
   each program is recorded. Every cell is timed in two passes over all
   cells, minutes apart, and the faster is kept (other jobs on the laptop
   slowed single cells by up to 20x in a one-pass run of experiments/40);
   cells whose passes differ by more than 1.25x are listed in the results.
3. The log forms (log_betainc, log_gammainc, log_gammaincc, log_hyp1f1,
   log_stdtr) against the log of jax's function, which is what a jax user
   writes: ln F and d ln F / dq, error / max(1, |ref|). Accuracy only.
4. hyp1f1's value_and_grad on the GPU split by argument at N = 2^16 and
   2^20, to tell a per-element cost from a per-call one.

FINDINGS (2026-11-07, RTX 3080 Laptop, chebax main at 1aab896)
--------------------------------------------------------------
results/33_jax_release_race.txt. float64 unless said; speed is value /
value_and_grad.

jax 0.11.0 -> 0.12.0.1, for these functions:
- betainc is 5.5-13x faster: it now stops each element's continued
  fraction when that element converges (2.5-2.6 us -> 0.23-0.46 us per
  value on one CPU core, 300 -> 24-48 ns on the GPU). numpyro's t CDF,
  built on it, gains the same.
- erfcx: the float32 zeros near x = 9.3 are gone (relative error 1.0 ->
  1.8e-7) and the derivative at large x is right (2.6e-6 -> 3.8e-16;
  float32 2.2e3 -> 2.4e-7). Its value_and_grad got slower (GPU 1.3 -> 2.8
  ns, CPU 7.0 -> 12 ns).
- i0's derivative near 0: 2.7e-10 -> 7.4e-16. The value_and_grad of i0
  and i1 on one CPU core is 1.4-1.7x faster (32-33 -> 22-24 ns; float32
  8.2-8.4 -> 4.8-5.2 ns).
- bessel_jn is 1.2x faster on the GPU (19 -> 15.5 ns per value), hyp1f1's
  values 1.3x on one CPU core (1.7 -> 1.35 us); both keep their accuracy.
- Nothing else here moved: gammainc, gammaincc, dawsn, i0e and i1e keep
  their accuracy and speed, i1 its accuracy.
- chebax computes the same on both; under 0.12 a few of its errors drop
  (stdtr 9.2e-14 -> 3.6e-15) and gammainc's CPU gradient is faster
  (130 -> 92 ns at a = 2.5).

jax defects present in both releases:
- bessel_jn (default n_iter=50): nan for x <= 1e-6, derivative wrong near
  0 (-2.4e6 at x = 2e-5, truth -1.05e-5), values wrong from x ~ 45
  (J0(200) = -1.8).
- hyp1f1: gradients in a and b wrong at large x (100% at x = 500, the
  b-gradient off by 35 decades); float32 nan near x = 100 where M fits.
- betainc has no gradient in a or b; the t CDF formula's t-gradient is
  nan at t = 0.
- Derivatives through identities lose relative accuracy far out: dawsn'
  2.8e-8 (float32 12), i0e' / i1e' 3e-12 / 5e-12 (float32 2e-3).
- log of a CDF is -inf where the CDF underflows (betainc 255 points,
  gammainc 292 of the grid; hyp1f1 in float32).

chebax against jax 0.12.0.1, speed:
- Faster, CPU / GPU: J 4-11x / 6-7x; betainc 9-9.6x / 16-21x; gammainc
  with one a 10-26x / 3.5-5.3x, a per element 2.6-4.8x / 7.3-10x; hyp1f1
  16-24x / 22-33x; t CDF 3.2-3.9x / 4.6-4.9x; erfcx on the GPU 2.4-3.2x
  (on par on the CPU in float64, 1.7-2x slower in float32). float32 on
  the GPU: J 89-99x, betainc 12-59x, gammainc 5.5-58x, hyp1f1 30-190x,
  t CDF 35-41x.
- Slower: I0, I1, i0e, i1e 1.9-2.7x on the CPU and 1.4-1.6x on the GPU
  (float32 CPU 2.2-2.9x; float32 GPU values on par, gradients 1.2-1.5x);
  dawsn 1.4-1.8x on the CPU (on the GPU its gradient is on par and its
  value 1.5x faster).
- The gradient cost of 0.5.0's two density fixes (experiments/35) is
  taken back (#74, experiments/36): betainc's x-gradient here is 9x
  jax's on one CPU core, where it was 4.9x on 0.5.0 (this experiment at
  353ea37).
- hyp1f1's float32 gradient on the GPU cost a fixed ~2.6 s per call (~5 s
  under jax 0.12) whatever N, from one fused kernel (experiments/34). With
  an optimization barrier (#61) it is 1.6 ns per element at 2^20; the
  (a, b) part is 4.5-5.2 ns at 2^16 (was 40-76 us).

chebax accuracy:
- Values: 3e-16 to 4e-13 where jax is 3e-16 to 4e-12 (bessel_jn aside).
  chebax is better on gammainc (3e-15 vs 3e-13 absolute) and the t CDF
  (3.6e-15 vs 3.7e-12), jax on hyp1f1 (6e-16 vs 3.6e-13); dawsn and I
  are close (4e-16 to 1.3e-15 against jax's 4e-16 to 8e-16).
- float32 I0 and I1 are finite up to x = 91.9, where jax (its 2 nan)
  and 0.5.0 overflow above x = 88.7. But from x = 80 they go through one
  exponential of the whole argument: 3.2e-6 and 5.0e-6 relative at
  x = 86.3 and 82.4, against jax's worst of 3.6e-7 and 7.1e-7. i0e and
  i1e are not affected (2.3e-7, 2.6e-7).
- Derivatives: better than jax wherever jax goes through an identity, and
  it has the ones jax lacks (betainc in a, b: 2.8e-13; t CDF in nu).
- betainc's x-derivative was 3.3 relative off at (a, b, x) = (100, 0.1,
  0.7287), float32 1.1e8 at (100, 0.5, 0.7287): it was I times
  d ln I/dt, which carries I's absolute error. Since #63 it is the
  density, 2.8e-13 (float32 4.5e-5), as jax.
- log_betainc does not resolve the lower tail for a >= 20 with small b at
  x ~ 0.7-0.99: ln I(100, 0.1, 0.7287) = -30.23 against -36.90 (I off
  800x, error 0.18 of |ln I|); log_stdtr has the same limit at nu = 200
  (0.40). Both docstrings and the README say so (above x = 1/2, and
  between -sqrt(nu) and the median). In that band log_stdtr's nu-gradient
  is off by up to 4.3e-2 (float32 0.1), nan where ln F rounds to -inf.
- The nu-gradients of stdtr (float32, nu > ~47), besselk_scalar and
  matern (float32, nu > 3.3) and log_stdtr (large nu, both dtypes) were
  nan; fixed in #59 and #62. stdtr's float32 d/dnu is now 2-4e-7 absolute.
- hyp1f1's parameter gradients: ~1e-15 inside the box, up to 1.4e-10 in
  d ln M / db at the a, b = 0.1 corner (test_hyp1f1.py records 2.9e-15
  on its own points).
- hyp1f1's float32 derivatives are inf at three (a, b, x) points, the nan
  counts in the report. At (0.1, 2.5, 100) and (2.5, 0.1, 75) the
  derivative is past float32's max or within 5% of it. At (10, 5, 75)
  M = 1.0e38 and its derivatives (1.1-3.0e38) fit, but reverse mode
  overflows in its terms: ln x times M is 4.4e38 before -psi(a) times M
  brings the sum back. Forward mode and log_hyp1f1 are finite there.
- gammaincc with one a is only absolutely accurate below x = 8: 3.5e-11
  relative at Q ~ 1e-5 (float32 8e-2); gammaincc_array keeps 7.6e-14.

WHAT IT DOES NOT ESTABLISH
--------------------------
Anything about other hardware (consumer fp64 at 1:64 on this GPU), TPUs,
multi-core CPU scaling or whole models. jax's looped functions cost what
their slowest lane costs, so their times move with the input
distribution; the distributions used are listed with each row. chebax
families without a jax counterpart (besselk, bessely, the quantiles,
vonmises, lambertw, spherical, matern, tweedie, the log forms) are not
covered.

Run (each stage writes into results/33_cache/, not checked in):

  python experiments/33_jax_release_race.py refs                (~5 s)
  JAX_PLATFORMS=cuda,cpu taskset -c 3 \\
      python experiments/33_jax_release_race.py run f64         (~20 min)
  ... run f32 (~21 min), runlog f64, runlog f32 (~8 min each), then all
  four again with jax 0.12.0.1 first on PYTHONPATH (a pip --target install)
  python experiments/33_jax_release_race.py report > results/33_jax_release_race.txt
  JAX_PLATFORMS=cuda,cpu python experiments/33_jax_release_race.py hyp1f1grad
      (once per jax; ~2 min, mostly compiling), appended to the results
      file under a "=== HYP1F1 GRADIENT ON THE GPU" header line
"""

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "33_cache")
SIZES = {"cpu": 2**16, "gpu": 2**20}


def r32(v):
    return np.unique(np.asarray(v, np.float64).astype(np.float32).astype(np.float64))


# ---------------------------------------------------------------- grids


def _x_gamma(a):
    lo = np.logspace(-4, np.log10(3 * a + 60), 120)
    mid = np.clip(a + np.sqrt(a) * np.linspace(-8, 12, 80), 1e-6, None)
    return r32(np.concatenate([lo, mid]))


def grids():
    """family -> (argument names, [(params, x)])."""
    g = {}
    x = np.concatenate(
        [
            -np.logspace(-3, np.log10(26.5), 250),
            np.linspace(-1, 1, 201),
            np.logspace(-3, 5, 400),
        ]
    )
    g["erfcx"] = (("x",), [((), r32(x))])
    x = np.concatenate([-np.logspace(-4, 4, 300), [0.0], np.logspace(-4, 4, 300)])
    g["dawsn"] = (("x",), [((), r32(x))])
    x = r32(np.logspace(-6, 4, 500))
    for name in ("i0", "i1", "i0e", "i1e"):
        g[name] = (("x",), [((), x)])
    x = r32(
        np.concatenate(
            [[0.0], np.logspace(-6, 0, 60), np.linspace(0, 100, 1001)[1:]]
            + [np.linspace(100, 1000, 451)[1:]]
        )
    )
    for n in (0, 1, 2, 5, 10):
        g[f"j{n}"] = (("x",), [((), x)])
    ab = [float(r32(v)[0]) for v in (0.1, 0.5, 1, 2.5, 7, 20, 100)]
    t = np.logspace(-8, np.log10(0.5), 30)
    x = r32(np.concatenate([t, 1 - t]))
    x = x[(x > 0) & (x < 1)]
    g["betainc"] = (("a", "b", "x"), [((a, b), x) for a in ab for b in ab])
    av = [float(r32(v)[0]) for v in (0.1, 0.3, 1, 2.5, 7, 10, 30, 100, 1000)]
    for name in ("gammainc", "gammaincc"):
        g[name] = (("a", "x"), [((a,), _x_gamma(a)) for a in av])
    hv = [float(r32(v)[0]) for v in (0.1, 0.5, 1, 2.5, 5, 10)]
    x = r32(
        np.concatenate(
            [
                [0.0],
                np.logspace(-4, 0, 15),
                np.linspace(1, 50, 30)[1:],
                [75, 100, 200, 500],
            ]
        )
    )
    g["hyp1f1"] = (("a", "b", "x"), [((a, b), x) for a in hv for b in hv])
    nv = [float(r32(v)[0]) for v in (0.5, 1, 2.5, 4.5, 10, 30, 200)]
    t = np.logspace(-3, 3, 60)
    x = r32(np.concatenate([-t, [0.0], t]))
    g["stdtr"] = (("nu", "t"), [((nu,), x) for nu in nv])
    return g


# ----------------------------------------------------------- references

LOG_FAMILIES = ("betainc", "gammainc", "gammaincc", "hyp1f1", "stdtr")


def _ref_chunk(task):
    import mpmath as mp

    fam, params, xs = task
    mp.mp.dps = 40
    rows = []
    for xf in xs:
        X = mp.mpf(float(xf))
        r = {}
        if fam == "erfcx":
            with mp.workdps(60):
                v = mp.exp(X * X) * mp.erfc(X)
                r = {"v": v, "d_x": 2 * X * v - 2 / mp.sqrt(mp.pi)}
        elif fam == "dawsn":
            with mp.workdps(60):
                v = mp.sqrt(mp.pi) / 2 * mp.exp(-X * X) * mp.erfi(X)
                r = {"v": v, "d_x": 1 - 2 * X * v}
        elif fam in ("i0", "i1", "i0e", "i1e"):
            n = int(fam[1])
            v = mp.besseli(n, X)
            d = (
                mp.besseli(1, X)
                if n == 0
                else (mp.besseli(0, X) + mp.besseli(2, X)) / 2
            )
            if fam.endswith("e"):
                v, d = mp.exp(-X) * v, mp.exp(-X) * (d - v)
            r = {"v": v, "d_x": d}
        elif fam.startswith("j"):
            n = int(fam[1:])
            lo = -mp.besselj(1, X) if n == 0 else mp.besselj(n - 1, X)
            r = {"v": mp.besselj(n, X), "d_x": (lo - mp.besselj(n + 1, X)) / 2}
        elif fam == "betainc":
            a, b = (mp.mpf(p) for p in params)
            lo = mp.betainc(a, b, 0, X, regularized=True)
            # differentiate the smaller tail, so P ~ 1 does not cancel
            if lo < 0.5:
                da = mp.diff(lambda s: mp.betainc(s, b, 0, X, regularized=True), a)
                db = mp.diff(lambda s: mp.betainc(a, s, 0, X, regularized=True), b)
            else:
                da = -mp.diff(lambda s: mp.betainc(s, b, X, 1, regularized=True), a)
                db = -mp.diff(lambda s: mp.betainc(a, s, X, 1, regularized=True), b)
            dens = mp.exp(
                (a - 1) * mp.log(X) + (b - 1) * mp.log1p(-X) - mp.log(mp.beta(a, b))
            )
            r = {"v": lo, "d_x": dens, "d_a": da, "d_b": db}
        elif fam in ("gammainc", "gammaincc"):
            a = mp.mpf(params[0])
            p = mp.gammainc(a, 0, X, regularized=True)
            q = mp.gammainc(a, X, mp.inf, regularized=True)
            if p < 0.5:
                dp = mp.diff(lambda s: mp.gammainc(s, 0, X, regularized=True), a)
            else:
                dp = -mp.diff(lambda s: mp.gammainc(s, X, mp.inf, regularized=True), a)
            dens = (
                mp.exp((a - 1) * mp.log(X) - X - mp.loggamma(a)) if X > 0 else mp.mpf(0)
            )
            if fam == "gammainc":
                r = {"v": p, "d_x": dens, "d_a": dp}
            else:
                r = {"v": q, "d_x": -dens, "d_a": -dp}
        elif fam == "hyp1f1":
            a, b = (mp.mpf(p) for p in params)
            r = {
                "v": mp.hyp1f1(a, b, X),
                "d_x": a / b * mp.hyp1f1(a + 1, b + 1, X),
                "d_a": mp.diff(lambda s: mp.hyp1f1(s, b, X), a),
                "d_b": mp.diff(lambda s: mp.hyp1f1(a, s, X), b),
            }
        elif fam == "stdtr":
            nu = mp.mpf(params[0])

            def tail(s):
                return (
                    mp.betainc(
                        s / 2, mp.mpf(1) / 2, 0, s / (s + X * X), regularized=True
                    )
                    / 2
                )

            if X == 0:
                v, dnu = mp.mpf(1) / 2, mp.mpf(0)
            elif X > 0:
                v, dnu = 1 - tail(nu), -mp.diff(tail, nu)
            else:
                v, dnu = tail(nu), mp.diff(tail, nu)
            pdf = mp.exp(
                mp.loggamma((nu + 1) / 2)
                - mp.loggamma(nu / 2)
                - mp.log(nu * mp.pi) / 2
                - (nu + 1) / 2 * mp.log1p(X * X / nu)
            )
            r = {"v": v, "d_t": pdf, "d_nu": dnu}
        if fam in LOG_FAMILIES:
            # log forms: ln F and d ln F / dq, resolved where F underflows
            r.update(
                {
                    "l" + k: mp.log(r["v"]) if k == "v" else r[k] / r["v"]
                    for k in list(r)
                }
            )
        rows.append({k: float(val) for k, val in r.items()})
    return fam, params, xs, rows


def refs():
    from multiprocessing import Pool

    import mpmath

    tasks = []
    for fam, (_, cases) in grids().items():
        for params, x in cases:
            for i in range(0, len(x), 64):
                tasks.append((fam, params, x[i : i + 64]))
    t0 = time.time()
    with Pool(8) as pool:
        done = pool.map(_ref_chunk, tasks, chunksize=1)
    out = {}
    for fam, params, xs, rows in done:
        acc = out.setdefault(fam, {"x": [], "case": [], "params": []})
        acc["x"].append(xs)
        acc["params"].append(np.tile(np.asarray(params, np.float64), (len(xs), 1)))
        for q in rows[0]:
            acc.setdefault(q, []).append(np.array([r[q] for r in rows]))
    flat = {}
    for fam, acc in out.items():
        for k, v in acc.items():
            if v:
                flat[f"{fam}|{k}"] = np.concatenate(v)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "refs.npz"), **flat)
    print(
        f"refs: {len(tasks)} chunks, mpmath {mpmath.__version__}, {time.time() - t0:.0f} s"
    )


# --------------------------------------------------------------- run


def implementations(jax, jnp, jsp, chebax):
    """family -> {impl: (f(*params, x), grad argnums, per_element)}."""
    imp = {}
    for name in ("erfcx", "dawsn"):
        imp[name] = {
            "jax": (getattr(jsp, name), (0,), False),
            "chebax": (getattr(chebax, name), (0,), False),
        }
    for name in ("i0", "i1", "i0e", "i1e"):
        nu, scaled = float(name[1]), name.endswith("e")
        imp[name] = {
            "jax": (getattr(jsp, name), (0,), False),
            "chebax": (
                lambda x, nu=nu, s=scaled: chebax.besseli(nu, x, scaled=s),
                (0,),
                False,
            ),
        }
    for n in (0, 1, 2, 5, 10):
        imp[f"j{n}"] = {
            "jax": (lambda x, n=n: jsp.bessel_jn(x, v=n)[n], (0,), False),
            # built outside jit: a first besselj(n, x) call inside a trace
            # caches tracers and breaks every later call (chebax 77ccbd5)
            "chebax": (lambda x, f=chebax.besselj_frozen(n): f(x), (0,), False),
        }
    imp["betainc"] = {
        "jax": (jsp.betainc, (2,), False),
        "chebax": (chebax.betainc, (2,), False),
        "chebax_all": (chebax.betainc, (0, 1, 2), False),
    }
    imp["gammainc"] = {
        "jax": (jsp.gammainc, (0, 1), False),
        "chebax": (chebax.gammainc_scalar, (0, 1), False),
        "chebax_array": (chebax.gammainc_array, (0, 1), True),
    }
    imp["gammaincc"] = {
        "jax": (jsp.gammaincc, (0, 1), False),
        "chebax": (chebax.gammaincc_scalar, (0, 1), False),
        "chebax_array": (chebax.gammaincc_array, (0, 1), True),
    }
    imp["hyp1f1"] = {
        "jax": (jsp.hyp1f1, (0, 1, 2), False),
        "chebax": (chebax.hyp1f1, (0, 1, 2), False),
    }

    def numpyro_t_cdf(nu, t):
        # numpyro.distributions.StudentT.cdf with loc 0, scale 1
        half_sign = 0.5 * jnp.sign(t)
        z = nu / (nu + t * t)
        return 0.5 + half_sign - 0.5 * jnp.sign(t) * jsp.betainc(0.5 * nu, 0.5, z)

    imp["stdtr"] = {
        "jax": (numpyro_t_cdf, (1,), False),
        "chebax": (chebax.stdtr, (1,), False),
        "chebax_all": (chebax.stdtr, (0, 1), False),
    }
    return imp


def log_implementations(jnp, jsp, chebax, imp):
    """The log forms against the log of jax's function (what a jax user
    writes); jax's gradients are those of its plain function."""

    def logged(f):
        return lambda *a: jnp.log(f(*a))

    out = {
        "betainc": {"chebax": (chebax.log_betainc, (0, 1, 2), False)},
        "gammainc": {
            "chebax": (chebax.log_gammainc_scalar, (0, 1), False),
            "chebax_array": (chebax.log_gammainc_array, (0, 1), True),
        },
        "gammaincc": {
            "chebax": (chebax.log_gammaincc_scalar, (0, 1), False),
            "chebax_array": (chebax.log_gammaincc_array, (0, 1), True),
        },
        "hyp1f1": {"chebax": (chebax.log_hyp1f1, (0, 1, 2), False)},
        "stdtr": {"chebax": (chebax.log_stdtr, (0, 1), False)},
    }
    for fam in out:
        f, argnums, per_el = imp[fam]["jax"]
        out[fam] = {"jax": (logged(f), argnums, per_el), **out[fam]}
    return out


def timing_settings():
    """(label, family, params or 'spread', x sampler, impls)."""

    def gam(rng, n, a):
        return rng.gamma(a, size=n)

    return [
        ("erfcx, x ~ N(0, 3)", "erfcx", (), lambda r, n: r.normal(0, 3, n)),
        ("dawsn, x ~ N(0, 5)", "dawsn", (), lambda r, n: r.normal(0, 5, n)),
        ("i0, x ~ U(0, 50)", "i0", (), lambda r, n: r.uniform(0, 50, n)),
        ("i1, x ~ U(0, 50)", "i1", (), lambda r, n: r.uniform(0, 50, n)),
        ("i0e, x ~ U(0, 50)", "i0e", (), lambda r, n: r.uniform(0, 50, n)),
        ("i1e, x ~ U(0, 50)", "i1e", (), lambda r, n: r.uniform(0, 50, n)),
        ("J1, x ~ U(0, 50)", "j1", (), lambda r, n: r.uniform(0, 50, n)),
        ("J10, x ~ U(0, 50)", "j10", (), lambda r, n: r.uniform(0, 50, n)),
        (
            "betainc a=2.5 b=7, x ~ Beta",
            "betainc",
            (2.5, 7.0),
            lambda r, n: r.beta(2.5, 7.0, n),
        ),
        (
            "betainc a=0.5 b=50, x ~ Beta",
            "betainc",
            (0.5, 50.0),
            lambda r, n: r.beta(0.5, 50.0, n),
        ),
        ("gammainc a=2.5, x ~ Gamma", "gammainc", (2.5,), lambda r, n: gam(r, n, 2.5)),
        ("gammainc a=30, x ~ Gamma", "gammainc", (30.0,), lambda r, n: gam(r, n, 30.0)),
        ("gammainc a per element, x ~ Gamma", "gammainc", "spread", None),
        (
            "gammaincc a=2.5, x ~ Gamma",
            "gammaincc",
            (2.5,),
            lambda r, n: gam(r, n, 2.5),
        ),
        (
            "hyp1f1 a=1.5 b=4, x ~ U(0, 20)",
            "hyp1f1",
            (1.5, 4.0),
            lambda r, n: r.uniform(0, 20, n),
        ),
        (
            "hyp1f1 a=0.5 b=8, x ~ U(0, 20)",
            "hyp1f1",
            (0.5, 8.0),
            lambda r, n: r.uniform(0, 20, n),
        ),
        (
            "stdtr nu=4.5, t ~ t(4.5)",
            "stdtr",
            (4.5,),
            lambda r, n: r.standard_t(4.5, n),
        ),
    ]


# every speed cell is timed in this many passes over all cells, the fastest
# kept; cells whose passes differ by more than DISAGREE are listed
PASSES = 2
DISAGREE = 1.25


def timed(call, n, reps=5, budget=1.0):
    """Median ns per element; a warm-up spin first (the GPU clock drops
    while compiling). Same as experiment 22."""
    import jax

    jax.block_until_ready(call())
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        jax.block_until_ready(call())
    t0 = time.perf_counter()
    jax.block_until_ready(call())
    one = time.perf_counter() - t0
    inner = max(1, int(budget / reps / max(one, 1e-6)))
    ts = []
    for _ in range(reps if one < budget else 3):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = call()
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts)) / n * 1e9


def run(dname_dtype, logs=False):
    wide = dname_dtype == "f64"
    import jax

    jax.config.update("jax_enable_x64", wide)
    import jax.numpy as jnp
    import jax.scipy.special as jsp

    import chebax

    dt = np.float64 if wide else np.float32
    imp = implementations(jax, jnp, jsp, chebax)
    g = grids()
    pre = ""
    if logs:  # accuracy of the log forms only
        imp = log_implementations(jnp, jsp, chebax, imp)
        g = {f: g[f] for f in imp}
        pre = "l"
    devs = {"cpu": jax.devices("cpu")[0], "gpu": jax.devices("gpu")[0]}
    chebax_dir = os.path.dirname(os.path.dirname(os.path.dirname(chebax.__file__)))
    commit = subprocess.run(
        ["git", "-C", chebax_dir, "log", "-1", "--format=%h %cd", "--date=short"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    meta = {
        "jax": jax.__version__,
        "dtype": dname_dtype,
        "chebax": f"{chebax_dir} {commit}",
        "devices": {k: d.device_kind for k, d in devs.items()},
        "load_start": os.getloadavg(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(json.dumps(meta), flush=True)
    outs, notes = {}, {}

    def put(v, dev):
        return jax.device_put(jnp.asarray(v, dt), dev)

    # 1. accuracy
    for dname, dev in devs.items():
        for fam, (argn, cases) in g.items():
            for iname, (f, argnums, per_el) in imp[fam].items():
                argnums_q = argnums
                key = f"{fam}|{iname}|{dname}"
                t0 = time.time()
                try:
                    vf = jax.jit(f)
                    if per_el:
                        gf = jax.jit(jax.grad(lambda *a: jnp.sum(f(*a)), argnums_q))
                    else:
                        axes = (None,) * (len(argn) - 1) + (0,)
                        gf = jax.jit(jax.vmap(jax.grad(f, argnums_q), in_axes=axes))
                    vals = {pre + "v": []}
                    vals.update({f"{pre}d_{argn[i]}": [] for i in argnums_q})
                    for params, x in cases:
                        xx = put(x, dev)
                        ps = [put(p, dev) for p in params]
                        if per_el:
                            ps = [jnp.broadcast_to(p, xx.shape) for p in ps]
                        vals[pre + "v"].append(np.asarray(vf(*ps, xx), np.float64))
                        for i, d in zip(argnums_q, gf(*ps, xx)):
                            vals[f"{pre}d_{argn[i]}"].append(np.asarray(d, np.float64))
                    for q, v in vals.items():
                        outs[f"{key}|{q}"] = np.concatenate(v)
                except Exception as e:  # recorded, not fatal
                    notes[key] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                print(
                    f"acc {key} {time.time() - t0:.1f} s {notes.get(key, '')}",
                    flush=True,
                )
    if logs:
        tag = f"{jax.__version__}_{dname_dtype}"
        np.savez_compressed(os.path.join(CACHE, f"logacc_{tag}.npz"), **outs)
        with open(os.path.join(CACHE, f"lognotes_{tag}.json"), "w") as fh:
            json.dump({"meta": meta, "notes": notes}, fh, indent=1)
        return

    # 2. speed: compile every cell, then time all of them PASSES times
    rows = []
    rng = np.random.default_rng(33)
    for dname, dev in devs.items():
        n = SIZES[dname]
        progs = []
        for label, fam, params, sampler in timing_settings():
            if params == "spread":
                mu = np.clip(np.exp(rng.normal(1.0, 0.5, n)), 0.6, 20.0)
                a = mu**2 / 1.5**2
                args = [put(a, dev), put(rng.gamma(a), dev)]
                which = ("jax", "chebax_array")
            else:
                args = [put(p, dev) for p in params] + [put(sampler(rng, n), dev)]
                which = [k for k in imp[fam] if k != "chebax_array"]
            for iname in which:
                f, argnums, _ = imp[fam][iname]
                total = lambda *a, f=f: jnp.sum(f(*a))  # noqa: E731
                rec = {"setting": label, "impl": iname, "device": dname, "n": n}
                rows.append(rec)
                try:
                    t0 = time.perf_counter()
                    cv = jax.jit(total).lower(*args).compile()
                    rec["compile_v"] = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    cg = (
                        jax.jit(jax.value_and_grad(total, argnums))
                        .lower(*args)
                        .compile()
                    )
                    rec["compile_g"] = time.perf_counter() - t0
                    calls = {"t_g": cg}
                    if iname != "chebax_all":
                        calls = {"t_v": cv, "t_g": cg}
                    progs.append((rec, calls, args))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        for _ in range(PASSES):
            for rec, calls, args in progs:
                if "error" in rec:
                    continue
                try:
                    for k, c in calls.items():
                        rec.setdefault(k + "_passes", []).append(
                            timed(lambda c=c, args=args: c(*args), n)
                        )
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        for rec in rows:
            if rec["device"] != dname:
                continue
            for k in ("t_v", "t_g"):
                if k + "_passes" in rec:
                    rec[k] = min(rec[k + "_passes"])
            print("time", json.dumps(rec), flush=True)
    meta["load_end"] = os.getloadavg()
    tag = f"{jax.__version__}_{dname_dtype}"
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, f"acc_{tag}.npz"), **outs)
    with open(os.path.join(CACHE, f"run_{tag}.json"), "w") as fh:
        json.dump({"meta": meta, "notes": notes, "timing": rows}, fh, indent=1)


def hyp1f1_grad():
    """Split hyp1f1's value_and_grad on the GPU by argument, at two N, to
    tell a per-element cost from a per-call one."""
    import jax
    import jax.numpy as jnp

    import chebax

    def tmed(f, *a):
        jax.block_until_ready(f(*a))
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < 1.0:
            out = f(*a)
            n += 1
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n

    print("jax", jax.__version__)
    for dtn, x64 in (("f32", False), ("f64", True)):
        jax.config.update("jax_enable_x64", x64)
        dt = jnp.float64 if x64 else jnp.float32
        for n in (2**16, 2**20):
            x = np.random.default_rng(0).uniform(0, 20, n)
            x = jax.device_put(jnp.asarray(x, dt), jax.devices("gpu")[0])
            a, b = jnp.asarray(1.5, dt), jnp.asarray(4.0, dt)

            def tot(a, b, x):
                return jnp.sum(chebax.hyp1f1(a, b, x))

            res = {}
            for name, argn in (
                ("val", None),
                ("dx", (2,)),
                ("dab", (0, 1)),
                ("all", (0, 1, 2)),
            ):
                f = (
                    jax.jit(tot)
                    if argn is None
                    else jax.jit(jax.value_and_grad(tot, argn))
                )
                res[name] = tmed(f, a, b, x) / n * 1e9
            hlo = (
                jax.jit(jax.value_and_grad(tot, (0, 1, 2)))
                .lower(a, b, x)
                .compile()
                .as_text()
            )
            print(
                dtn,
                "N",
                n,
                " ".join(f"{k} {v:.3g}" for k, v in res.items()),
                "ns/el | scatter",
                hlo.count(" scatter("),
                "while",
                hlo.count(" while("),
                "sort",
                hlo.count(" sort("),
            )


# ------------------------------------------------------------- report

# family -> quantity -> metrics
# densities: relative where not negligible against the CDF (>= 1e-4 F),
# absolute in the saturated wedge (F ~ 1), as in test_gammainc.py
CDF = {"v": ("abs", "rel"), "d_x": ("split",), "d_t": ("split",), "d_a": ("abs",)}
CDF.update({"d_b": ("abs",), "d_nu": ("abs",)})
METRICS = {
    **{
        f: {"v": ("rel",), "d_x": ("rel",)}
        for f in ("erfcx", "dawsn", "i0", "i1", "i0e", "i1e")
    },
    **{f"j{n}": {"v": ("sup",), "d_x": ("sup",)} for n in (0, 1, 2, 5, 10)},
    "betainc": CDF,
    "gammainc": CDF,
    "gammaincc": CDF,
    "stdtr": CDF,
    # derivatives as errors in d ln M / dq, absolute below 1 (test_hyp1f1.py)
    "hyp1f1": {"v": ("rel_lnM",), **{q: ("dlog",) for q in ("d_x", "d_a", "d_b")}},
}
# J slices: jax's bessel_jn recursion starts at n_iter = 50
SLICES = {
    f"j{n}": (("x<=50", lambda x: x <= 50), ("x>50", lambda x: x > 50))
    for n in (0, 1, 2, 5, 10)
}


def _zero_windows(x, r, case):
    """Points within 1% of an interior sign change of r, per case."""
    skip = np.zeros(len(x), bool)
    for c in np.unique(case):
        idx = np.where(case == c)[0]
        idx = idx[np.argsort(x[idx])]
        xs, rs = x[idx], r[idx]
        for i in np.where(np.sign(rs[:-1]) * np.sign(rs[1:]) < 0)[0]:
            x0 = xs[i] - rs[i] * (xs[i + 1] - xs[i]) / (rs[i + 1] - rs[i])
            skip |= np.abs(x - x0) <= 0.01 * max(abs(x0), 1e-3)
    return skip


def score(metric, y, r, x, case, vref, dt):
    fi = np.finfo(dt)
    ok = np.isfinite(r) & (np.abs(r) <= fi.max)
    if metric != "abs":
        # the reference and the function value both in the normal range
        ok &= (r == 0) | (np.abs(r) >= fi.tiny)
        ok &= (vref == 0) | ((np.abs(vref) >= fi.tiny) & (np.abs(vref) <= fi.max))
    if metric in ("rel", "rel_lnM"):
        ok &= r != 0
        ok &= ~_zero_windows(x, r, case)
    if metric == "dlog":
        ok &= vref != 0
    bad = ok & ~np.isfinite(y)
    good = ok & np.isfinite(y)
    with np.errstate(all="ignore"):
        err = np.abs(y - r)
        if metric == "rel":
            err = err / np.abs(r)
        elif metric == "rel_lnM":
            err = err / (np.abs(r) * np.maximum(1.0, np.abs(np.log(np.abs(vref)))))
        elif metric == "split":
            dense = (r != 0) & (np.abs(r) >= 1e-4 * np.abs(vref))
            err = np.where(dense, err / np.abs(r), err)
        elif metric == "dlog":
            err = err / np.maximum(np.abs(vref), np.abs(r))
        elif metric == "sup":
            err = err / np.max(np.abs(r[ok]))
    assert not np.isnan(err[good]).any(), metric
    err = np.where(good, err, -1.0)
    i = int(np.argmax(err))
    return float(err[i]), int(bad.sum()), int(ok.sum()), i


def report_logs(ref, versions, new):
    """ln F and d ln F / dq: error / max(1, |ref|). jax is the log of its
    plain function, so it goes -inf where F underflows."""
    accs = {}
    for path in glob.glob(os.path.join(CACHE, "logacc_*.npz")):
        accs[os.path.basename(path)[7:-4]] = np.load(path)
    for dtn, dt in (("f64", np.float64), ("f32", np.float32)):
        fi = np.finfo(dt)
        print(
            f"\n=== LOG FORMS {dtn}: chebax log_* against log(jax f); error / max(1, |ref|), worst over CPU and GPU"
        )
        cols = [("jax", v) for v in versions] + [
            ("chebax", new),
            ("chebax", versions[0]),
            ("chebax_array", new),
        ]
        hdr = ["family", "quantity"] + [f"{i} {v}" for i, v in cols]
        lines, where = [hdr], []
        for fam in LOG_FAMILIES:
            x, pr = ref[f"{fam}|x"], ref[f"{fam}|params"]
            for q in [k.split("|")[1] for k in ref.keys() if k.startswith(fam + "|l")]:
                r = ref[f"{fam}|{q}"]
                ok = np.isfinite(r) & (np.abs(r) <= fi.max)
                row = [fam, q]
                for iname, ver in cols:
                    acc = accs.get(f"{ver}_{dtn}")
                    worst, nbad, at, found = -1.0, 0, None, False
                    for dname in ("cpu", "gpu"):
                        key = f"{fam}|{iname}|{dname}|{q}"
                        if acc is None or key not in acc:
                            continue
                        found = True
                        y = acc[key]
                        good = ok & np.isfinite(y)
                        nbad += int((ok & ~np.isfinite(y)).sum())
                        with np.errstate(all="ignore"):
                            err = np.where(
                                good, np.abs(y - r) / np.maximum(1.0, np.abs(r)), -1.0
                            )
                        i = int(np.argmax(err))
                        if err[i] > worst:
                            worst, at = float(err[i]), (
                                dname,
                                pr[i].tolist(),
                                float(x[i]),
                            )
                    if not found:
                        row.append("-")
                        continue
                    cell = f"{worst:.1e}" if worst >= 0 else "all bad"
                    if nbad:
                        cell += f" ({nbad} nan)"
                    row.append(cell)
                    if not (iname == "jax" and ver != new):
                        where.append(f"  {fam} {q} {iname} {ver}: {cell} at {at}")
                lines.append(row)
        widths = [max(len(str(l[i])) for l in lines) for i in range(len(hdr))]
        for l in lines:
            print("  ".join(str(c).ljust(w) for c, w in zip(l, widths)))
        print(f"\n  worst points (device, params, x), log forms {dtn}:")
        print("\n".join(where))


def report():
    ref = np.load(os.path.join(CACHE, "refs.npz"))
    runs = {}
    for path in sorted(glob.glob(os.path.join(CACHE, "run_*.json"))):
        tag = os.path.basename(path)[4:-5]
        with open(path) as fh:
            runs[tag] = json.load(fh)
        runs[tag]["acc"] = np.load(os.path.join(CACHE, f"acc_{tag}.npz"))
    versions = sorted(
        {r["meta"]["jax"] for r in runs.values()},
        key=lambda v: [int(p) for p in v.split(".")],
    )
    new = versions[-1]
    print("chebax against jax's special functions, jax " + " and ".join(versions))
    for tag, r in runs.items():
        m = r["meta"]
        print(
            f"  run {tag}: {m['date']}, {m['devices']}, chebax {m['chebax'].split(' ', 1)[1]}, load {m['load_start'][0]:.1f} -> {m['load_end'][0]:.1f}"
        )
        for k, v in r["notes"].items():
            print(f"    error {k}: {v}")
    g = grids()
    for dtn, dt in (("f64", np.float64), ("f32", np.float32)):
        print(
            f"\n=== ACCURACY {dtn} (worst over CPU and GPU; '(k nan)' = k points nan/inf where the reference is finite)"
        )
        cols = [("jax", v) for v in versions] + [
            ("chebax", new),
            ("chebax", versions[0]),
        ]
        hdr = (
            ["family", "quantity", "metric"]
            + [f"{i} {v}" for i, v in cols]
            + ["chebax array " + new]
        )
        lines, where = [hdr], []
        for fam in g:
            x = ref[f"{fam}|x"]
            pr = ref[f"{fam}|params"]
            case = (
                np.unique(pr, axis=0, return_inverse=True)[1].ravel()
                if pr.shape[1]
                else np.zeros(len(x), int)
            )
            for q, metrics in METRICS[fam].items():
                if f"{fam}|{q}" not in ref:
                    continue
                r = ref[f"{fam}|{q}"]
                for metric in metrics:
                    for sname, sel in SLICES.get(
                        fam, (("", lambda x: np.ones(len(x), bool)),)
                    ):
                        m = sel(x)
                        row = [fam, q + (f" [{sname}]" if sname else ""), metric]
                        cells = cols + [("chebax_array", new)]
                        for iname, ver in cells:
                            run = runs.get(f"{ver}_{dtn}")
                            worst, nbad, at = -1.0, 0, None
                            found = False
                            for dname in ("cpu", "gpu"):
                                key = f"{fam}|{iname}|{dname}|{q}"
                                if run is not None and key not in run["acc"]:
                                    key = f"{fam}|{iname}_all|{dname}|{q}"
                                if run is None or key not in run["acc"]:
                                    continue
                                found = True
                                y = run["acc"][key][m]
                                e, b, n_ok, i = score(
                                    metric,
                                    y,
                                    r[m],
                                    x[m],
                                    case[m],
                                    ref[f"{fam}|v"][m],
                                    dt,
                                )
                                nbad += b
                                if e > worst:
                                    worst, at = e, (
                                        dname,
                                        pr[m][i].tolist(),
                                        float(x[m][i]),
                                    )
                            if not found:
                                row.append("-")
                                continue
                            cell = f"{worst:.1e}" if worst >= 0 else "all bad"
                            if nbad:
                                cell += f" ({nbad} nan)"
                            row.append(cell)
                            if iname == "jax" and ver != new:
                                continue
                            where.append(
                                f"  {fam} {row[1]} {metric} {iname} {ver}: {cell} at {at}"
                            )
                        if any(c != "-" for c in row[3:]):
                            lines.append(row)
        widths = [max(len(str(l[i])) for l in lines) for i in range(len(hdr))]
        for l in lines:
            print("  ".join(str(c).ljust(w) for c, w in zip(l, widths)))
        print(f"\n  worst points (device, params, x), {dtn}:")
        print("\n".join(where))

    report_logs(ref, versions, new)

    print(
        "\n=== SPEED, ns per element (median): value / value_and_grad; compile s value + grad"
    )
    for dtn in ("f64", "f32"):
        for dname in ("cpu", "gpu"):
            print(f"\n--- {dtn} {dname} (N = {SIZES[dname]})")
            hdr = (
                ["setting"]
                + [f"jax {v}" for v in versions]
                + [f"chebax {v}" for v in versions]
                + [f"speedup vs jax {new}"]
            )
            lines = [hdr]
            settings = [s[0] for s in timing_settings()]
            for label in settings:
                cells = {}
                for v in versions:
                    run = runs.get(f"{v}_{dtn}")
                    if run is None:
                        continue
                    for rec in run["timing"]:
                        if rec["setting"] == label and rec["device"] == dname:
                            cells[(rec["impl"], v)] = rec
                row = [label]

                def fmt(rec):
                    if rec is None:
                        return "-"
                    if "error" in rec:
                        return "ERR " + rec["error"][:40]
                    tv = f"{rec['t_v']:.3g}" if "t_v" in rec else "-"
                    return f"{tv} / {rec['t_g']:.3g} ({rec['compile_v']:.1f}+{rec['compile_g']:.1f}s)"

                cimpl = "chebax_array" if "per element" in label else "chebax"
                for v in versions:
                    row.append(fmt(cells.get(("jax", v))))
                for v in versions:
                    row.append(fmt(cells.get((cimpl, v))))
                j, c = cells.get(("jax", new)), cells.get((cimpl, new))
                if j and c and "t_v" in j and "t_v" in c:
                    row.append(
                        f"{j['t_v'] / c['t_v']:.2g}x / {j['t_g'] / c['t_g']:.2g}x"
                    )
                else:
                    row.append("-")
                lines.append(row)
                ca = cells.get(("chebax_all", new))
                if ca and "t_g" in ca:
                    extra = ["  chebax, grad in all args"] + ["-"] * len(versions)
                    extra += [fmt(cells.get(("chebax_all", v))) for v in versions] + [
                        "-"
                    ]
                    lines.append(extra)
            widths = [max(len(str(l[i])) for l in lines) for i in range(len(hdr))]
            for l in lines:
                print("  ".join(str(c).ljust(w) for c, w in zip(l, widths)))

    print(
        f"\n=== TIMING PASSES: cells whose {PASSES} passes differ by more than "
        f"{DISAGREE:.2f}x (the faster is shown above)"
    )
    worst, listed = 1.0, 0
    for tag in sorted(runs):
        for rec in runs[tag]["timing"]:
            for k in ("t_v", "t_g"):
                ps = rec.get(k + "_passes")
                if not ps:
                    continue
                r = max(ps) / min(ps)
                worst = max(worst, r)
                if r > DISAGREE:
                    listed += 1
                    print(
                        f"  {tag} {rec['device']} {rec['setting']} {rec['impl']} {k}: "
                        + " / ".join(f"{t:.3g}" for t in ps)
                    )
    print(f"  {listed} listed; largest ratio over all cells {worst:.2f}x")


if __name__ == "__main__":
    {
        "refs": refs,
        "run": lambda: run(sys.argv[2]),
        "runlog": lambda: run(sys.argv[2], logs=True),
        "hyp1f1grad": hyp1f1_grad,
        "report": report,
    }[sys.argv[1]]()
