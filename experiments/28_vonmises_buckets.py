#!/usr/bin/env python3
"""Experiment 28: von Mises kappa buckets against the full 112-term table.

WHAT IS MEASURED
----------------
vonmises_cdf and vonmises_icdf collapse the table's kappa axis once per
call and then run a Clenshaw recurrence of the table's w-length per
element. The bucketed runtime keeps, for kappa <= 2 and kappa <= 10, only
the w-coefficients whose dropped tail is under 1e-15 in H, dH/dw and
dH/dkappa over that range (31 and 53 of 112, experiments/24's bounds on
the sub-range), picked at trace time for a concrete kappa and by
lax.switch for a traced one. This races, per kappa in {0.5, 2, 5, 10, 20,
50}: the bucketed cdf and icdf (values and value_and_grad in kappa and
the point) against the same functions on the full table, under jit with
kappa traced, on the GPU at N = 2^22 and one CPU thread at N = 2^18,
float64, interleaved reps, medians. One more row vmaps the cdf over 8
groups of kappa (per-group use): under vmap the switch runs every bucket,
so that row is the cost of the buckets where they cannot pay.

WHAT IT MEANS
-------------
The bucketed/full ratio is what a user with a concentration in a bucket
gains; kappa > 10 runs the full table and should show 1.0. The vmapped
row is the price of a batched kappa.

WHAT THIS DOES NOT ESTABLISH
----------------------------
Accuracy (tests/test_vonmises.py, with the bucket edges in KAPPAS and the
bucketed-vs-full comparison); float32; what the buckets do for a caller
whose kappa changes between calls (each bucket compiles once).

Run:  python experiments/28_vonmises_buckets.py   (~3 min, GPU; CPU rows
      pinned with taskset -c 3)
"""

import os
import time

import numpy as np

KAPPAS = (0.5, 2.0, 5.0, 10.0, 20.0, 50.0)
REPS = 15


def bench(fns, warmup=2, reps=REPS):
    for launch, sync in fns.values():
        for _ in range(warmup):
            sync(launch())
    out = {k: [] for k in fns}
    for _ in range(reps):
        for k, (launch, sync) in fns.items():
            t0 = time.perf_counter()
            sync(launch())
            out[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax
    from chebax._src.recipes import vonmises as vm

    full_cdf = lambda k, th: vm._cdf_cj(k, th, None)
    full_icdf = lambda k, p: vm._icdf_cj(k, p, None)
    sync = lambda y: jax.tree_util.tree_map(lambda a: a.block_until_ready(), y)

    def vg(f):
        return jax.jit(jax.value_and_grad(lambda k, x: f(k, x).sum(), argnums=(0, 1)))

    devs = [d for d in jax.devices() if d.platform == "gpu"] + jax.devices("cpu")[:1]
    print(f"jax {jax.__version__}, devices {devs}")
    for dev in devs:
        n = 1 << (22 if dev.platform == "gpu" else 18)
        print(
            f"\n{dev.platform}, N = {n}, float64, ms (median of {REPS}); "
            "bucketed / full, ratio"
        )
        th = jax.device_put(
            jnp.asarray(np.random.default_rng(0).uniform(-np.pi, np.pi, n)), dev
        )
        p = jax.device_put(
            jnp.asarray(np.random.default_rng(1).uniform(1e-3, 1 - 1e-3, n)), dev
        )
        print(
            f"  {'kappa':>6s} | {'cdf':>22s} | {'cdf v&g':>22s} | {'icdf':>22s} | {'icdf v&g':>22s}"
        )
        jitted = {
            "cdf b": jax.jit(chebax.vonmises_cdf),
            "cdf f": jax.jit(full_cdf),
            "cdfg b": vg(chebax.vonmises_cdf),
            "cdfg f": vg(full_cdf),
            "icdf b": jax.jit(chebax.vonmises_icdf),
            "icdf f": jax.jit(full_icdf),
            "icdfg b": vg(chebax.vonmises_icdf),
            "icdfg f": vg(full_icdf),
        }
        for kappa in KAPPAS:
            k = jax.device_put(jnp.asarray(kappa), dev)
            fns = {
                name: (
                    lambda f=f, x=(p if name.startswith("icdf") else th): f(k, x),
                    sync,
                )
                for name, f in jitted.items()
            }
            t = bench(fns)
            cells = []
            for name in ("cdf", "cdfg", "icdf", "icdfg"):
                b, f = t[name + " b"] * 1e3, t[name + " f"] * 1e3
                cells.append(f"{b:7.2f} / {f:7.2f} {f / b:4.2f}x")
            print(f"  {kappa:6g} | " + " | ".join(cells))
        ks = jax.device_put(
            jnp.asarray([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]), dev
        )
        thg = th[: n // 8 * 8].reshape(8, -1)
        vb = jax.jit(jax.vmap(chebax.vonmises_cdf))
        vf = jax.jit(jax.vmap(full_cdf))
        t = bench({"b": (lambda: vb(ks, thg), sync), "f": (lambda: vf(ks, thg), sync)})
        print(
            f"  vmap over 8 kappas, cdf: {t['b'] * 1e3:7.2f} / {t['f'] * 1e3:7.2f} "
            f"{t['f'] / t['b']:4.2f}x"
        )


if __name__ == "__main__":
    main()
