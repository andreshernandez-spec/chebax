#!/usr/bin/env python3
"""Experiment 32: the one-a Gamma quantile from the quantile table's start,
against the 40-step solve it replaces.

WHAT IS MEASURED
----------------
gammaincinv_scalar and gammainccinv_scalar inside the tables (a >= 0.1):
the quantile table's start (gammaincinv_start, shared with the array path
since increment 44) and two Newton steps on the uniform tables' ln P and
ln Q, the second step's residual being the check (gammaincinv_kernel
.table_solve), against the kernel's full solve on the same residual: the
asymptotic start, 40 safeguarded Newton steps (24 in float32) and the
three-point residual check, rebuilt here from the same pieces.

Accuracy: the round trip |F(x)/target - 1| on the target's own side at 40
dps, a from 0.1 to 1e9 (the small tables to 10, the Temme zone above),
both tails to 1e-300, for the shipped path and the old solve. Speed: ms
per call at N = 2^20 on the GPU and N = 2^14 on one CPU core, one a per
call as the path takes it, float64; float32 rows in a child with x64 off.

WHAT IT MEANS
-------------
The scalar path paid 40 residual evaluations per call for a start it did
not trust; from a start within 1e-8 two are enough, so the table removes
most of its cost at the same accuracy. The dP/da term of the derivative
and the a < 0.1 and a > 1e10 boxes are unchanged (experiments/09 for the
former).

WHAT THIS DOES NOT ESTABLISH
----------------------------
a < 0.1 (jax's gammainc, the old solve still), a > 1e10 (Temme's variable,
8 steps as before), and derivatives.

Run:  taskset -c 0 python experiments/32_gammaincinv_scalar_table.py   # ~5 min
"""

import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")
if "--speed-only" not in sys.argv:
    jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402
from chebax._src.recipes import gammaincinv_kernel as _k  # noqa: E402
from chebax._src.recipes import gammaincinv_scalar as _gs  # noqa: E402
from chebax._src.recipes._common import canon_tag, solver_limits  # noqa: E402


def _mode(a):
    return "small" if a <= 10.0 else "large"


def old_solve(a, targets, mode, ceiling=False):
    """The kernel's full solve on the box's residual: the scalar path before
    this change (gammaincinv_scalar._gammainv_core, increment 37). mode is
    the box, a python constant as the path's lax.cond makes it."""
    lim = solver_limits(canon_tag())
    log_pq, seam, small = _gs._residual(a, mode)
    return _k.solve(
        a, *targets, log_pq, lim.n_gamma, ceiling=ceiling, seam=seam, small=small
    )


def old_inv(a, p, mode):
    pc = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
    x, ok, below, _ = old_solve(a, _k.inv_targets(pc), mode)
    return _k.finish_inv(p, x, ok, below, False)


def old_cinv(a, y, mode):
    yc = jnp.where((y > 0.0) & (y < 1.0), y, 0.5)
    x, ok, below, above = old_solve(a, _k.cinv_targets(yc), mode, ceiling=True)
    return _k.finish_cinv(y, x, ok, below, above, False)


A_GRID = [0.1, 0.17, 0.3, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0, 9.0, 10.0, 17.0, 50.0]
A_GRID += [180.0, 640.0, 1000.0, 1e4, 1e6, 1e9]
P_GRID = [1e-300, 1e-100, 1e-20, 1e-6, 1e-3, 0.02, 0.1, 0.3, 0.5]
Q_GRID = [0.4, 0.2, 0.05, 1e-3, 1e-6, 1e-20, 1e-100, 1e-300]


def accuracy():
    import mpmath as mp

    from chebax._src.recipes._gen_common import DPS
    from chebax._src.recipes.gammainc_large_gen import _hyp1f1_1, _log_q

    mp.mp.dps = DPS

    def log_p(a, x):
        if x > a + 1:
            return mp.log1p(-mp.exp(_log_q(mp, a, x)))
        return a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))

    def log_q(a, x):
        if x > a + 1:
            return _log_q(mp, a, x)
        lp = a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))
        return mp.log1p(-mp.exp(lp))

    rows = []
    for a in A_GRID:
        am = mp.mpf(a)
        for p in P_GRID:
            # the lower target must leave the quantile above the float floor,
            # and mpmath's series stall past a ~ 1e6 away from the mode
            if (mp.log(mp.mpf(p)) + mp.loggamma(am + 1)) / am > -700 and (
                a <= 1e4 or p >= 1e-6
            ):
                rows.append((a, p, np.nan))
        for q in Q_GRID:
            if a <= 1e4 or q >= 1e-6:
                rows.append((a, np.nan, q))
    print(
        f"round trip |F(x)/target - 1| on {len(rows)} points, a in [0.1, 1e9], "
        "both tails, float64"
    )
    print(f"  {'method':30s} {'worst':>9s} {'median':>9s}  worst at")
    new_p, new_q = jax.jit(chebax.gammaincinv_scalar), jax.jit(
        chebax.gammainccinv_scalar
    )
    oi, oc = jax.jit(old_inv, static_argnums=2), jax.jit(old_cinv, static_argnums=2)
    paths = {
        "table + 2 steps (shipped)": (
            lambda a, p: new_p(a, p),
            lambda a, q: new_q(a, q),
        ),
        "old solve (40 steps + check)": (
            lambda a, p: oi(a, p, _mode(a)),
            lambda a, q: oc(a, q, _mode(a)),
        ),
    }
    for name, (fp, fq) in paths.items():
        errs = []
        for a, p, q in rows:
            if np.isnan(q):
                x = float(fp(a, p))
                e = abs(mp.expm1(log_p(mp.mpf(a), mp.mpf(x)) - mp.log(mp.mpf(p))))
            else:
                x = float(fq(a, q))
                e = abs(mp.expm1(log_q(mp.mpf(a), mp.mpf(x)) - mp.log(mp.mpf(q))))
            errs.append(float(e) if np.isfinite(x) and x > 0 else np.inf)
        errs = np.array(errs)
        k = int(np.argmax(errs))
        a, p, q = rows[k]
        tgt = f"p={p:.0e}" if np.isnan(q) else f"q={q:.0e}"
        print(f"  {name:30s} {errs.max():9.2e} {np.median(errs):9.2e}  a={a:g}, {tgt}")


def timed(call, reps=7):
    jax.block_until_ready(call())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def speed():
    rng = np.random.default_rng(0)
    dtype = jnp.empty(()).dtype
    devs = jax.devices("cpu")[:1] + [d for d in jax.devices() if d.platform == "gpu"]
    cases = [
        ("a=0.5,  p~U(1e-6,1-..)", 0.5, lambda n: rng.uniform(1e-6, 1 - 1e-6, n)),
        ("a=3.5,  p~U(1e-6,1-..)", 3.5, lambda n: rng.uniform(1e-6, 1 - 1e-6, n)),
        ("a=9.9,  p~U(1e-6,1-..)", 9.9, lambda n: rng.uniform(1e-6, 1 - 1e-6, n)),
        ("a=3.5,  p~U(1e-12,1e-2)", 3.5, lambda n: 10 ** rng.uniform(-12, -2, n)),
        ("a=50,   p~U(1e-6,1-..)", 50.0, lambda n: rng.uniform(1e-6, 1 - 1e-6, n)),
        ("a=500,  p~U(1e-6,1-..)", 500.0, lambda n: rng.uniform(1e-6, 1 - 1e-6, n)),
        ("a=500,  p~U(1e-12,1e-2)", 500.0, lambda n: 10 ** rng.uniform(-12, -2, n)),
    ]
    for dev in devs:
        n = 2**14 if dev.platform == "cpu" else 2**20
        print(
            f"\n{dev.platform}, N = {n}, {dtype.name}, ms per call (median of 7); "
            "gammaincinv_scalar shipped / old solve, ratio",
            flush=True,
        )
        for name, a, draw in cases:
            p = jax.device_put(jnp.asarray(draw(n), dtype), dev)
            av = jnp.asarray(a, dtype)
            with jax.default_device(dev):
                fn = jax.jit(chebax.gammaincinv_scalar)
                fo = jax.jit(old_inv, static_argnums=2)
                tn = timed(lambda: fn(av, p))
                to = timed(lambda: fo(av, p, _mode(a)))
                tn = min(tn, timed(lambda: fn(av, p)))
            print(
                f"  {name:26s} {1e3 * tn:8.2f} / {1e3 * to:8.2f}  {to / tn:5.1f}x",
                flush=True,
            )


def main():
    if "--speed-only" in sys.argv:
        speed()
        return
    print(f"jax {jax.__version__}, devices {jax.devices()}")
    accuracy()
    speed()
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    subprocess.run([sys.executable, __file__, "--speed-only"], env=env, check=True)


if __name__ == "__main__":
    main()
