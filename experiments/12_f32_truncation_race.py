#!/usr/bin/env python3
"""Does dropping the converged f64 tail speed up f32 GPU evaluation?

WHAT THIS MEASURES
------------------
The gate PROJECT.md put on truncate(tol): "benchmark under B3 before
bothering, the f32 roofline says the extra terms are free." astype(f32)
keeps the full f64 degree, roughly 2x the terms f32 accuracy needs;
truncate(1e-7) halves or thirds them (besselk inner 79 -> 26, betainc
x 23 -> 9). This script races full-degree f32 instances against their
truncated twins on GPU, N = 2^24, device-resident, interleaved reps,
medians, with an agreement column at each family's own metric
(sup-normalized for oscillatory J, where a relative metric misfires at
the zeros; pointwise-relative for positive K; absolute for the betainc
CDF): truncation error must sit at the f32 grade, not above it. f64
rows ride along as context.

WHAT IT DOES NOT MEASURE
------------------------
Baked-artifact size (truncation always shrinks that, no benchmark
needed); C++/CUDA kernels (the header path; this is the jax runtime);
accuracy contracts (tests own them).

Run:  python experiments/12_f32_truncation_race.py   (~2 min, needs GPU)
"""

import time

import numpy as np

N = 1 << 24
REPS = 25


def bench(fns, warmup=3, reps=REPS):
    for launch, sync in fns.values():
        for _ in range(warmup):
            launch()
        sync()
    out = {k: [] for k in fns}
    for _ in range(reps):
        for k, (launch, sync) in fns.items():
            t0 = time.perf_counter()
            launch()
            sync()
            out[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax

    dev = jax.devices()[0]
    print(f"jax {jax.__version__} on {dev}, N = {N}, reps = {REPS} (medians)")
    rng = np.random.default_rng(0)

    cases = [
        ("besselj(2.5)", chebax.besselj(2.5), rng.uniform(0.05, 30.0, N), "sup"),
        ("besselk(1.5)", chebax.besselk(1.5),
         np.exp(rng.uniform(np.log(1e-4), np.log(80.0), N)), "rel"),
        ("betainc(2,3)", chebax.betainc(2.0, 3.0),
         rng.uniform(1e-6, 1 - 1e-6, N), "abs"),
    ]

    hdr = (f"{'instance':14s} {'degrees':>12s} {'full64':>8s} {'full32':>8s}"
           f" {'trunc32':>8s} {'32 ratio':>8s}  agree (family metric)")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, inst, x_host, metric in cases:
        t7 = inst.truncate(1e-7)
        degs = "/".join(str(getattr(inst, f).degree) for f in inst._series_fields[:2])
        degs += " ->" + "/".join(str(getattr(t7, f).degree) for f in t7._series_fields[:2])
        x64 = jax.device_put(jnp.asarray(x_host, jnp.float64), dev)
        x32 = jax.device_put(jnp.asarray(x_host, jnp.float32), dev)
        f64 = jax.jit(inst)
        f32 = jax.jit(inst.astype(jnp.float32))
        f32t = jax.jit(t7.astype(jnp.float32))
        ys = {k: [None] for k in ("a", "b", "c")}

        def mk(key, fn, x):
            def launch():
                ys[key][0] = fn(x)
            return launch, lambda: ys[key][0].block_until_ready()

        t = bench({"f64": mk("a", f64, x64), "f32": mk("b", f32, x32),
                   "f32t": mk("c", f32t, x32)})
        ref = np.asarray(ys["a"][0], dtype=np.float64)
        got = np.asarray(ys["c"][0], dtype=np.float64)
        err = np.abs(got - ref)
        if metric == "sup":
            agree = float(err.max() / np.abs(ref).max())
        elif metric == "rel":
            agree = float(np.max(err / np.abs(ref)))
        else:
            agree = float(err.max())
        print(f"{name:14s} {degs:>12s} {t['f64']*1e3:6.2f}ms {t['f32']*1e3:6.2f}ms"
              f" {t['f32t']*1e3:6.2f}ms {t['f32']/t['f32t']:7.2f}x  {agree:.1e}")

    print("\n'32 ratio' is full-degree f32 over truncated f32: the truncation"
          "\nspeedup, if the roofline is wrong and there is one.")


if __name__ == "__main__":
    main()
