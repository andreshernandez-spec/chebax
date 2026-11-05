#!/usr/bin/env python3
"""Experiment 29: betainc gradients with derivatives computed inside the
panel switch.

WHAT IS MEASURED
----------------
betainc_scalar, log_betainc_scalar and betaincinv pick their (a, b) panel
with one lax.switch. Reverse mode through that switch cost ~120 ms per
differentiated parameter on the GPU at N = 2^20 (chebax-perf-leads, "235
ms per call", cause found 2026-11-05: the switch itself, two identical
branches cost the same, jax.checkpoint on the branches changes nothing).
The custom_jvp rules now compute I and dI/da, dI/db, dI/dx as values
inside the switch and apply the tangents outside, so reverse mode never
transposes a switch branch (symbolic_zeros skips the derivatives nobody
asked for). This times, per (a, b) across the four panels, values, grad
in x (or p), grad in (a, b) and value_and_grad in all three, for the new
rules against the old path (plain AD through with_panel_series, rebuilt
from the module's helpers), on the GPU at N = 2^20 and one CPU thread at
N = 2^16, float64, medians of interleaved reps in alternating order.

WHAT IT MEANS
-------------
old / new is the gradient speedup a traced-parameter user gets; the values
row should be 1.0 (the primal path is unchanged).

WHAT THIS DOES NOT ESTABLISH
----------------------------
Accuracy (tests/test_betainc.py and test_betaincinv.py, against mp.diff);
the per-group path (vmap turns the switch into a select; not timed here).

Run:  python experiments/29_betainc_grad_switch.py   (~5 min; CPU rows
      pinned with taskset -c 3)
"""

import time

import numpy as np

PAIRS = ((2.5, 3.5), (0.5, 0.5), (9.5, 0.2), (30.0, 70.0), (0.3, 50.0))
REPS = 9


def bench(fns, warmup=1, reps=REPS):
    for launch, sync in fns.values():
        for _ in range(warmup):
            sync(launch())
    out = {k: [] for k in fns}
    for rep in range(reps):
        # alternate the order so neither side always follows the other's largest program
        for k in list(fns) if rep % 2 == 0 else reversed(list(fns)):
            launch, sync = fns[k]
            t0 = time.perf_counter()
            sync(launch())
            out[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax
    from chebax._src.recipes import betainc as bi
    from chebax._src.recipes.betainc_kernel import (
        betaincinv_core,
        eval_betainc,
        ln_beta,
    )

    sync = lambda y: jax.tree_util.tree_map(lambda a: a.block_until_ready(), y)

    # the old path: plain AD through the panel switch
    def old_betainc(a, b, x):
        out = bi.with_panel_series(
            a, b, lambda cab, cba: eval_betainc(a, b, x, cab, cba)
        )
        return bi.with_edge_slopes(a, b, x, out)

    def old_log_betainc(a, b, x):
        return bi.with_panel_series(
            a, b, lambda cab, cba: bi._log_eval(a, b, x, cab, cba)
        )

    @jax.custom_jvp
    def old_betaincinv(a, b, p):
        return bi.with_panel_series(
            a, b, lambda cab, cba: betaincinv_core(a, b, p, cab, cba)
        )

    @old_betaincinv.defjvp
    def _old_inv_jvp(primals, tangents):
        a, b, p = primals
        da, db, dp = tangents
        x = old_betaincinv(a, b, p)
        interior = (x > 0.0) & (x < 1.0)
        xs = jnp.where(interior, x, 0.5)
        pdf = jnp.exp((a - 1) * jnp.log(xs) + (b - 1) * jnp.log1p(-xs) - ln_beta(a, b))
        _, d_cdf = jax.jvp(lambda aa, bb: old_betainc(aa, bb, xs), (a, b), (da, db))
        dx = (dp - d_cdf) / pdf
        return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))

    def rows(f):
        s = lambda a, b, x: f(a, b, x).sum()
        return {
            "values": jax.jit(f),
            "grad x": jax.jit(jax.grad(s, argnums=2)),
            "grad a,b": jax.jit(jax.grad(s, argnums=(0, 1))),
            "v&g a,b,x": jax.jit(jax.value_and_grad(s, argnums=(0, 1, 2))),
        }

    fams = {
        "betainc_scalar": (chebax.betainc_scalar, old_betainc),
        "log_betainc_scalar": (chebax.log_betainc_scalar, old_log_betainc),
        "betaincinv": (chebax.betaincinv, old_betaincinv),
    }
    devs = [d for d in jax.devices() if d.platform == "gpu"] + jax.devices("cpu")[:1]
    print(f"jax {jax.__version__}, devices {devs}")
    for dev in devs:
        n = 1 << (20 if dev.platform == "gpu" else 16)
        x = jax.device_put(jnp.asarray(np.random.default_rng(0).uniform(0, 1, n)), dev)
        print(
            f"\n{dev.platform}, N = {n}, float64, ms (median of {REPS}); new / old, ratio"
        )
        for fam, (new, old) in fams.items():
            jn, jo = rows(new), rows(old)
            print(f"  {fam}")
            print(f"  {'a, b':>10s} | " + " | ".join(f"{k:>24s}" for k in jn))
            for a, b in PAIRS:
                a_, b_ = jax.device_put(jnp.asarray(a), dev), jax.device_put(
                    jnp.asarray(b), dev
                )
                fns = {}
                for k in jn:
                    fns[k + " n"] = (lambda f=jn[k]: f(a_, b_, x), sync)
                    fns[k + " o"] = (lambda f=jo[k]: f(a_, b_, x), sync)
                t = bench(fns)
                cells = [
                    f"{t[k + ' n'] * 1e3:8.2f} / {t[k + ' o'] * 1e3:8.2f} {t[k + ' o'] / t[k + ' n']:4.1f}x"
                    for k in jn
                ]
                print(f"  {a:4g}, {b:4g} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
