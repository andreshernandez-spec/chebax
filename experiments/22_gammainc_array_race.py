#!/usr/bin/env python3
"""Per-element gammainc: gammainc_elem against jax's gammainc and against
the uniform-a path vmapped over a.

WHAT THIS MEASURES
------------------
P(a, x) for N elements, each with its own a, x drawn from Gamma(a) (the
arguments a Gamma likelihood meets), three layouts of a:

- "a = 2.5 each":  a = 2.5 on every element, but passed as an array, so
                   no implementation can use that it is uniform;
- "spread":        a = mu^2 / 1.5^2 with mu lognormal, 0.16 to 178 (a
                   Gamma regression with a fixed coefficient of variation);
- "wide":          a log-uniform on [0.1, 1e4].

Three implementations, values and value_and_grad in (a, x) of the sum:

- jax:   jax.scipy.special.gammainc, the per-element op numpyro and
         pytensor's JAX backend lower to (series and continued fraction
         in while loops run to the worst lane's count);
- elem:  chebax.gammainc_elem;
- vmap:  jax.vmap(chebax.gammainc_fn), the uniform-a path with a per
         element (its table collapse then runs per element).

Times are ns per element (median), on one CPU core (run under taskset) at
N = 2^16 and on the GPU at N = 2^20; vmap runs N = 2^12 on both, since
its gradient costs ~0.2 ms per element on the CPU and its GPU gradient
program did not finish compiling in 20 min at 2^20 (ptxas; 25 s at 2^12).
float64, or float32 with x64 off.
Also: max |P_elem - P_jax| and the number of while loops in elem's
compiled program.

FINDINGS (2026-11-01, RTX 3080 Laptop, reproduced by this script)
-----------------------------------------------------------------
- elem costs the same on every layout: float64 0.74-0.78 us per value and
  1.1-1.2 us per value_and_grad on one CPU core, 12.5-12.8 / 22.8-23.3 ns
  on the GPU; float32 0.20-0.21 / 0.21-0.23 us and 1.1 / 1.7 ns.
- Against jax's gammainc, float64: 1.0-8.9x on values and 1.6-15x on
  value_and_grad on the CPU, 2.3-18x and 2.8-24x on the GPU, growing with
  the spread of a. float32 on the GPU: 4.5-67x and 6.8-96x. float32 on the
  CPU: jax's value is 1.8x faster at a = 2.5 each, elem 1.4-8.0x faster on
  the other two; value_and_grad 1.4-19x.
- Against vmap of gammainc_fn, float64: 1.5x on values and 71-78x on
  value_and_grad on the CPU, 9x and 53-57x on the GPU.
- On "wide", jax's gammainc is off by up to 1.4e-11 (float64) and 7e-3
  (float32) near x ~ a ~ 1e4, against mpmath; elem is within 2e-16 and
  5e-8 at those points, so "agree" there is jax's error.
- No while loops in elem's compiled program.
- The load average was 1.8-2.4 during the run (other work on the laptop);
  the CPU numbers are on one pinned core.

WHAT IT DOES NOT ESTABLISH
--------------------------
Anything about other hardware (consumer fp64 at 1:64), multi-core CPU
scaling, or whole models. jax's gammainc cost depends on the worst lane's
loop count, so its numbers move with the layout; elem's do not. Compile
time is not raced.

Run:  taskset -c 3 python experiments/22_gammainc_elem_race.py f64   (~10 min)
      taskset -c 3 python experiments/22_gammainc_elem_race.py f32
"""

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

WIDE = sys.argv[1] == "f64"
jax.config.update("jax_enable_x64", WIDE)

import chebax  # noqa: E402

DT = np.float64 if WIDE else np.float32
SIZES = {"cpu": 2**16, "gpu": 2**20}
VMAP_N = 2**12


def layouts(n):
    rng = np.random.default_rng(22)
    mu = np.clip(np.exp(rng.normal(1.0, 0.5, n)), 0.6, 20.0)
    out = {
        "a = 2.5 each": np.full(n, 2.5),
        "spread": mu**2 / 1.5**2,
        "wide": np.exp(rng.uniform(np.log(0.1), np.log(1e4), n)),
    }
    return {k: (a, rng.gamma(a)) for k, a in out.items()}


def methods():
    def total(f):
        return lambda a, x: jnp.sum(f(a, x))

    return {
        "jax": total(jax.scipy.special.gammainc),
        "elem": total(chebax.gammainc_elem),
        "vmap": total(jax.vmap(chebax.gammainc_fn)),
    }


def timed(call, n, reps=5, budget=1.0):
    """Median ns per element; a warm-up spin first (the GPU clock drops
    while compiling)."""
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


def main():
    devs = {"cpu": jax.devices("cpu")[0]}
    try:
        devs["gpu"] = jax.devices("gpu")[0]
    except RuntimeError:
        pass
    print(
        f"{DT.__name__}, ns per element (value / value_and_grad in a and x); "
        + ", ".join(f"{k} N = {SIZES[k]} ({d.device_kind})" for k, d in devs.items()),
        flush=True,
    )
    for dname, dev in devs.items():
        n = SIZES[dname]
        for lname, (a, x) in layouts(n).items():
            a, x = (jax.device_put(jnp.asarray(v, DT), dev) for v in (a, x))
            row = []
            for mname, f in methods().items():
                m = VMAP_N if mname == "vmap" else n
                am, xm = a[:m], x[:m]
                v = jax.jit(f)
                g = jax.jit(jax.value_and_grad(f, (0, 1)))
                tv = timed(lambda: v(am, xm), m)
                tg = timed(lambda: g(am, xm), m)
                row.append(f"{mname} {tv:7.1f} / {tg:8.1f}")
            p_elem = jax.jit(chebax.gammainc_elem)(a, x)
            p_jax = jax.jit(jax.scipy.special.gammainc)(a, x)
            agree = float(jnp.max(jnp.abs(p_elem - p_jax)))
            hlo = jax.jit(chebax.gammainc_elem).lower(a, x).compile().as_text()
            loops = hlo.count(" while(")
            print(
                f"{dname} {lname:13s} | " + " | ".join(row)
                + f" | agree {agree:.1e}, elem while loops {loops}",
                flush=True,
            )


if __name__ == "__main__":
    main()
