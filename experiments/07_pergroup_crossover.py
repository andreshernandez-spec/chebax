#!/usr/bin/env python3
"""Measure the per-group crossover: when does G x table reconstruction
stop mattering?

WHAT THIS MEASURES
------------------
The queued per-group parameter API (PROJECT.md) would serve G unique
parameter values per call: build G coefficient tables, evaluate each
element against its group's table. Modeled cost is
G x reconstruction + N x polynomial. This script measures the real thing
with the mechanism available today, vmap over equal-size groups: reshape
x to (G, N/G) and vmap the traced function over per-group scalar
parameters. Per row:

  uniform   chebax with ONE parameter set for all N elements (the floor;
            the contract the library ships today)
  pergroup  chebax vmapped over G groups; the G reconstructions run
            inside jit, inside the timed region
  jax       jax.scipy.special.betainc with the same per-group parameters
            broadcast per element (the incumbent's cost in the
            hierarchical regime; betainc only, jax has no bessel K)

Two recipes bracket the reconstruction cost: betainc_fn (3-D tensor,
the most expensive instantiation) and besselk_fn (2-D, typical).
G sweeps at fixed total N (2^20 and 2^24), GPU f64, plus an MCMC-scale
block (G = 4 chains, small per-group n) on GPU and CPU, which is where
the hierarchical-model regime actually lives. The crossover reads off
the overhead column: the smallest n per group where pergroup/uniform
is ~1 is the size above which reconstruction is free.

WHAT IT DOES NOT MEASURE
------------------------
Gradient passes (NUTS pays forward + backward; reconstruction is
differentiated through, so its share roughly doubles); the group-index
gather variant an API would ship for ragged groups (equal-size reshape
isolates reconstruction cost, a gather adds bandwidth, not FLOPs);
accuracy (one f64 agreement column guards wiring, tests own the
contracts); gammaincinv and the other solve wrappers (they rebuild no
tables, so per-group costs them nothing beyond the solve itself).

Run:  python experiments/07_pergroup_crossover.py   (~4 min, needs GPU)
"""

import time

import numpy as np

REPS = 15


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


def run_row(fns, device, jax):
    jitted = {k: jax.jit(f) for k, f in fns.items()}
    ys = {k: [None] for k in fns}

    def make(k):
        def launch():
            ys[k][0] = jitted[k]()
        return launch, lambda: ys[k][0].block_until_ready()

    t = bench({k: make(k) for k in fns})
    return t, ys


def sweep(name, uniform_fn, pergroup_fn, jax_fn, n, gs, params, x_host,
          device, jax, jnp):
    has_jax = jax_fn is not None
    hdr = (f"{name}, N = 2^{int(np.log2(n))}, f64, {device.platform.upper()}"
           f"  (overhead = pergroup/uniform)")
    print(hdr)
    cols = (f"  {'G':>6s} {'n/group':>8s} {'uniform':>9s} {'pergroup':>9s}"
            f" {'overhead':>8s}")
    if has_jax:
        cols += f" {'jax':>9s} {'jax/pg':>7s}"
    print(cols)
    x = jax.device_put(jnp.asarray(x_host, jnp.float64), device)
    agree = None
    for g in gs:
        m = n // g
        xg = x.reshape(g, m)
        pg = {k: jax.device_put(jnp.asarray(v[:g], jnp.float64), device)
              for k, v in params.items()}
        fns = {
            "uniform": lambda x=x: uniform_fn(x),
            "pergroup": lambda pg=pg, xg=xg: pergroup_fn(pg, xg),
        }
        if has_jax:
            fns["jax"] = lambda pg=pg, xg=xg: jax_fn(pg, xg)
        t, ys = run_row(fns, device, jax)
        line = (f"  {g:6d} {m:8d} {t['uniform']*1e3:7.3f}ms"
                f" {t['pergroup']*1e3:7.3f}ms"
                f" {t['pergroup']/t['uniform']:7.2f}x")
        if has_jax:
            line += (f" {t['jax']*1e3:7.3f}ms"
                     f" {t['jax']/t['pergroup']:6.1f}x")
            if agree is None:
                agree = float(jnp.max(jnp.abs(ys['jax'][0] - ys['pergroup'][0])))
        print(line)
    if agree is not None:
        print(f"  max|jax - pergroup| (f64, first row): {agree:.1e}")
    print()


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax

    gpu = jax.devices()[0]
    cpu = jax.devices("cpu")[0]
    print(f"jax {jax.__version__} on {gpu}, reps = {REPS} (medians)\n")
    rng = np.random.default_rng(0)

    gmax = 16384
    ab = {"a": rng.uniform(0.5, 9.5, gmax), "b": rng.uniform(0.5, 9.5, gmax)}
    nus = {"nu": rng.uniform(0.2, 9.8, gmax)}

    def bi_uniform(x):
        return chebax.betainc_fn(2.5, 3.5, x)

    def bi_pergroup(pg, xg):
        return jax.vmap(chebax.betainc_fn)(pg["a"], pg["b"], xg)

    def bi_jax(pg, xg):
        return jax.scipy.special.betainc(pg["a"][:, None], pg["b"][:, None], xg)

    def bk_uniform(x):
        return chebax.besselk_fn(2.5, x)

    def bk_pergroup(pg, xg):
        return jax.vmap(chebax.besselk_fn)(pg["nu"], xg)

    gs = (1, 4, 16, 64, 256, 1024, 4096, 16384)
    for n in (1 << 20, 1 << 24):
        xb = np.clip(rng.uniform(0.0, 1.0, n), 1e-6, 1 - 1e-6)
        sweep("betainc", bi_uniform, bi_pergroup, bi_jax, n, gs, ab, xb,
              gpu, jax, jnp)
        xk = np.exp(rng.uniform(np.log(0.1), np.log(100.0), n))
        sweep("besselk", bk_uniform, bk_pergroup, None, n, gs, nus, xk,
              gpu, jax, jnp)

    # MCMC scale: 4 chains, small per-group n; launch overhead territory,
    # but this is the regime the hierarchical pitches live in
    for dev in (gpu, cpu):
        for m in (256, 2048, 16384):
            n = 4 * m
            xb = np.clip(rng.uniform(0.0, 1.0, n), 1e-6, 1 - 1e-6)
            sweep("betainc MCMC scale", bi_uniform, bi_pergroup, bi_jax,
                  n, (4,), ab, xb, dev, jax, jnp)


if __name__ == "__main__":
    main()
