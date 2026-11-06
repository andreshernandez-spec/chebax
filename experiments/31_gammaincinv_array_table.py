#!/usr/bin/env python3
"""Experiment 31: the shipped per-element Gamma quantile, table start plus
Newton steps, against the solve it replaces.

WHAT IS MEASURED
----------------
gammaincinv_array and gammainccinv_array as shipped (gammaincinv_array_gen's
table, two Newton steps in float64, none in float32) against the previous
array path, rebuilt here from the kernel it still shares with the scalar
path: 8 Halley steps on gammainc_array's ln P and ln Q from the kernel's
asymptotic start, then the residual check on x and its float neighbours.

Accuracy: the round trip |F(x)/target - 1| on the target's own side against
the 40-dps series and continued fraction, over experiment 26's grid (a from
0.1 to 1000, both tails to 1e-300), for the table alone (what float32
ships), the shipped float64 path and the old solve. Speed: ns per element
at N = 2^14 on one CPU core and 2^20 on the GPU, float64 and float32, for
the shipped path and the old solve, with a log-uniform in [0.1, 1000] and
p uniform.

WHAT IT MEANS
-------------
The table start costs one patch evaluation and lets two Newton steps (one
plus its check) do what 8 Halley steps and a three-point check did. The
float64 accuracy must stay at the array path's floor (experiment 26:
5.9e-13 for the prototype, 3.8e-13 for the solve); the float32 table alone
is held by test_gammaincinv_array's child-process bar.

WHAT THIS DOES NOT ESTABLISH
----------------------------
a above 1000 (covered by tests against the scalar path, not by mpmath,
whose series stall there), derivatives (unchanged: the implicit function
theorem on the same ln P and ln Q), and the scalar path, which still runs
the 40-step solve on the uniform tables.

Run:  taskset -c 0 python experiments/31_gammaincinv_array_table.py   # ~5 min
"""

import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

sys.path.insert(0, "src")
if "--speed-only" not in sys.argv:
    jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402
from chebax._src.recipes import gammainc_array as _ga  # noqa: E402
from chebax._src.recipes import gammaincinv_array as _gi  # noqa: E402
from chebax._src.recipes import gammaincinv_kernel as _k  # noqa: E402
from chebax._src.recipes._gen_common import DPS  # noqa: E402
from chebax._src.recipes.gammainc_large_gen import _hyp1f1_1, _log_q  # noqa: E402

mp.mp.dps = DPS


# the previous array path: main before this change, gammaincinv_array._solve


def _old_solve(a, targets, ceiling=False):
    def log_pq(x):
        return _ga._log_pq(a + 0.0 * jnp.where(jnp.isfinite(x), x, 0.0), x)

    ltp, ltq, _, upper = targets
    eps = jnp.finfo(a.dtype).eps
    bar = 256.0 * eps * jnp.maximum(1.0, jnp.abs(jnp.where(upper, ltq, ltp)))
    iters = {"float64": 8, "float32": 6}[a.dtype.name]
    return _k.solve(a, *targets, log_pq, iters, ceiling=ceiling, rtol=bar, halley=True)


def old_inv(a, p):
    bad, a = _gi._domain(a)
    pc = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
    x, ok, below, _ = _old_solve(a, _k.inv_targets(pc))
    flat = a > _gi._FLAT[a.dtype.name]
    x = jnp.where(flat, a, x)
    return _k.finish_inv(p, x, ok | flat, below & ~flat, bad)


def old_cinv(a, y):
    bad, a = _gi._domain(a)
    yc = jnp.where((y > 0.0) & (y < 1.0), y, 0.5)
    x, ok, below, above = _old_solve(a, _k.cinv_targets(yc), ceiling=True)
    flat = a > _gi._FLAT[a.dtype.name]
    x = jnp.where(flat, a, x)
    return _k.finish_cinv(y, x, ok | flat, below & ~flat, above & ~flat, bad)


# references, as in experiment 26


def log_p(a, x):
    if x > a + 1:
        return mp.log1p(-mp.exp(_log_q(mp, a, x)))
    return a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))


def log_q(a, x):
    if x > a + 1:
        return _log_q(mp, a, x)
    lp = a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))
    return mp.log1p(-mp.exp(lp))


A_GRID = [0.1, 0.17, 0.3, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0, 9.0, 10.0, 17.0, 50.0]
A_GRID += [180.0, 640.0, 1000.0]
P_GRID = [1e-300, 1e-100, 1e-20, 1e-6, 1e-3, 0.02, 0.1, 0.3, 0.5]
Q_GRID = [0.4, 0.2, 0.05, 1e-3, 1e-6, 1e-20, 1e-100, 1e-300]


def grid():
    rows = []
    for a in A_GRID:
        am = mp.mpf(a)
        for p in P_GRID:
            if (mp.log(mp.mpf(p)) + mp.loggamma(am + 1)) / am > -700:
                rows.append((a, p, np.nan))
        for q in Q_GRID:
            rows.append((a, np.nan, q))
    return (np.array(c) for c in zip(*rows))


def round_trip(a, p, q, x):
    out = np.empty(len(a))
    for i in range(len(a)):
        am, xm = mp.mpf(float(a[i])), mp.mpf(float(x[i]))
        if not np.isfinite(x[i]) or x[i] <= 0:
            out[i] = np.inf
            continue
        if np.isnan(q[i]):
            out[i] = float(abs(mp.expm1(log_p(am, xm) - mp.log(mp.mpf(float(p[i]))))))
        else:
            out[i] = float(abs(mp.expm1(log_q(am, xm) - mp.log(mp.mpf(float(q[i]))))))
    return out


def accuracy():
    a, p, q = grid()
    lower = np.isnan(q)
    print(
        f"round trip |F(x)/target - 1| on {len(a)} points, a in [0.1, 1000], "
        "both tails, float64"
    )
    print(f"  {'method':30s} {'worst':>9s} {'median':>9s}  worst at")

    def run(name, fn_p, fn_q):
        x = np.empty(len(a))
        x[lower] = np.asarray(fn_p(jnp.asarray(a[lower]), jnp.asarray(p[lower])))
        x[~lower] = np.asarray(fn_q(jnp.asarray(a[~lower]), jnp.asarray(q[~lower])))
        err = round_trip(a, p, q, x)
        k = int(np.argmax(err))
        tgt = f"p={p[k]:.0e}" if lower[k] else f"q={q[k]:.0e}"
        print(f"  {name:30s} {err.max():9.2e} {np.median(err):9.2e}  a={a[k]:g}, {tgt}")

    inv, cinv = chebax.gammaincinv_array, chebax.gammainccinv_array
    steps = _gi._STEPS["float64"]
    for n in range(steps + 1):
        _gi._STEPS["float64"] = n
        jax.clear_caches()
        label = "table alone" if n == 0 else f"table + {n} Newton step(s)"
        run(label + (" (shipped)" if n == steps else ""), jax.jit(inv), jax.jit(cinv))
    _gi._STEPS["float64"] = steps
    jax.clear_caches()
    run("old solve (8 Halley + check)", jax.jit(old_inv), jax.jit(old_cinv))


def timed(call, reps=7):
    jax.block_until_ready(call())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def speed():
    """The canonical dtype's rows: float64 here, float32 in a child with x64
    off (the public names promote to the canonical dtype, so float32 under
    x64 would silently time float64)."""
    rng = np.random.default_rng(0)
    dtype = jnp.empty(()).dtype
    devs = jax.devices("cpu")[:1] + [d for d in jax.devices() if d.platform == "gpu"]
    for dev in devs:
        n = 2**14 if dev.platform == "cpu" else 2**20
        a = jax.device_put(
            jnp.asarray(np.exp(rng.uniform(np.log(0.1), np.log(1e3), n)), dtype), dev
        )
        p = jax.device_put(jnp.asarray(rng.uniform(1e-6, 1 - 1e-6, n), dtype), dev)
        print(
            f"\n{dev.platform}, N = {n}, {dtype.name}, ns per element (median of 7); "
            f"shipped ({_gi._STEPS[dtype.name]} steps) / old solve, ratio",
            flush=True,
        )
        for name, new, old in (
            ("gammaincinv_array", chebax.gammaincinv_array, old_inv),
            ("gammainccinv_array", chebax.gammainccinv_array, old_cinv),
        ):
            with jax.default_device(dev):
                fn, fo = jax.jit(new), jax.jit(old)
                tn = timed(lambda: fn(a, p))
                to = timed(lambda: fo(a, p))
                tn = min(tn, timed(lambda: fn(a, p)))
                # the unjudged start plus one step: what the check costs
                steps = _gi._STEPS[dtype.name]
                _gi._STEPS[dtype.name] = 1
                jax.clear_caches()
                t1 = timed(lambda: jax.jit(new)(a, p))
                _gi._STEPS[dtype.name] = steps
                jax.clear_caches()
            print(
                f"  {name:20s} {1e9 * tn / n:8.1f} / {1e9 * to / n:8.1f}  "
                f"{to / tn:4.1f}x   (one step, unjudged: {1e9 * t1 / n:6.1f})",
                flush=True,
            )


def main():
    if "--speed-only" in sys.argv:
        speed()
        return
    print(f"jax {jax.__version__}, mpmath {mp.__version__}, devices {jax.devices()}")
    accuracy()
    speed()
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    subprocess.run([sys.executable, __file__, "--speed-only"], env=env, check=True)


if __name__ == "__main__":
    main()
