#!/usr/bin/env python3
"""Race the full three-region besselj against a single-region narrow fit.

WHAT THIS MEASURES
------------------
The queued PROJECT.md benchmark: every full-domain besselj eval executes all
three regions plus two selects, branchlessly. A user whose inputs live in a
declared window (the Matern kernel case) could run one short polynomial and
zero selects, about a third of the arithmetic. This times both on the same
device-resident x ~ U(0, 8) arrays (GPU via jax; f64 and f32).

Two narrow kernels are timed, and they answer different questions.
`inner` calls the instance's own inner-region evaluation, the arithmetic
floor. `domain` is the SHIPPED besselj(v, domain=(0, 8)), which is that
same evaluation plus the out-of-domain nan guard, so the gap between
them is what the guard costs and the ratio against `full` is what a user
actually gets.

WHAT IT DOES NOT MEASURE
------------------------
Nothing about accuracy (identical inner table -> identical values on the
window; asserted). Laptop-GPU absolute numbers; the fp64 side is
compute-bound on consumer cards (bessel experiments/07), which is exactly
why arithmetic reduction shows up ~proportionally there and matters less
where bandwidth-bound.

Run:  python experiments/04_narrow_vs_three_region.py   (~30 s, needs GPU)
"""

import time

import numpy as np

N = 1 << 24
REPS = 25


def med(ts):
    return float(np.median(np.asarray(ts)))


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
    return {k: med(v) for k, v in out.items()}


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax

    print(f"jax {jax.__version__} on {jax.devices()[0]}, N = {N}")
    rng = np.random.default_rng(0)
    x_host = rng.uniform(1e-6, 8.0, N)

    for dtype, tag in [(jnp.float64, "f64"), (jnp.float32, "f32")]:
        inst = chebax.besselj(2.5).astype(dtype)
        trimmed = chebax.besselj(2.5, domain=(0.0, 8.0)).astype(dtype)
        x = jnp.asarray(x_host, dtype)
        full = jax.jit(inst)
        inner = jax.jit(inst._inner)   # same table, no mid/outer, no selects
        dom = jax.jit(trimmed)         # the shipped narrow instance
        y = {}

        def mk(name, fn):
            def launch():
                y[name] = fn(x)
            return launch, lambda: y[name].block_until_ready()

        t = bench({"full": mk("full", full), "inner": mk("inner", inner),
                   "domain": mk("domain", dom)})
        for k in ("inner", "domain"):
            diff = float(jnp.max(jnp.abs(y["full"] - y[k])))
            assert diff == 0.0, (k, diff)   # same table on the window
        print(f"{tag}: full {t['full'] * 1e3:7.2f} ms   "
              f"inner {t['inner'] * 1e3:7.2f} ms ({t['full'] / t['inner']:.2f}x)   "
              f"domain= {t['domain'] * 1e3:7.2f} ms "
              f"({t['full'] / t['domain']:.2f}x)")


if __name__ == "__main__":
    main()
