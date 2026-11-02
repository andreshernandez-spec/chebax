#!/usr/bin/env python3
"""Per-element Gamma quantiles: gammaincinv_array against TFP's igammainv
and, where a is uniform, against gammaincinv_scalar.

WHAT THIS MEASURES
------------------
x with P(a, x) = p for N elements, each with its own a and p ~ U(0, 1)
(an inverse-CDF sampler, a Gamma copula margin, quantile residuals), the
three layouts of a from experiment 22:

- "a = 2.5 each":  a = 2.5 on every element, passed as an array;
- "spread":        a = mu^2 / 1.5^2 with mu lognormal, 0.16 to 178;
- "wide":          a log-uniform on [0.1, 1e4].

Implementations, values and value_and_grad in (a, p) of the sum:

- array:  chebax.gammaincinv_array;
- tfp:    tensorflow_probability.substrates.jax.math.igammainv, what
          pytensor's JAX backend falls back to for a per element (skipped
          when it does not import; TFP 0.25 does not import under jax
          0.11, tfp-nightly 0.26.0-dev20261101 does);
- scalar: chebax.gammaincinv_scalar(2.5, p), the "a = 2.5 each" row only:
          what the plain name runs when a is passed as one value.

jax.vmap(chebax.gammaincinv_scalar) is not raced: under vmap its lax.conds
on a run every branch, jax's gammainc fallback among them, and a 6-element
call did not finish compiling in 12 minutes on the CPU.

Times are ns per element (median of warm calls), on one CPU core (run
under taskset) at N = 2^14 and on the GPU at N = 2^20, float64, or
float32 with x64 off. Compile seconds are for the value function, then
value_and_grad. Accuracy: the round trip of 200 elements per layout
through mpmath's regularized gamma at 40 digits, relative to the smaller
tail, for array and tfp.

FINDINGS (2026-11-02, RTX 3080 Laptop, reproduced by this script)
-----------------------------------------------------------------
- gammaincinv_array's values cost the same on every layout: float64
  5.6-5.8 us on one CPU core and 188-195 ns on the GPU, float32 2.4-2.5 us
  and 13.3 ns. value_and_grad is 7.4-7.5 us / 229-237 ns until some a
  passes 1e3, where the derivatives find the unrounded root again:
  10.6 us / 358 ns on "wide" (float32 7.4 us / 37.6 ns).
- TFP's cost follows the spread of a. Against it (a = 2.5 each / spread /
  wide): GPU float64 0.93x / 2.3x / 7.3x on values and 0.91x / 2.3x / 4.8x
  on value_and_grad; GPU float32 2.6x / 7.0x / 34x and 2.0x / 5.8x / 14x.
  One CPU core float64 0.61x / 1.6x / 5.2x and 0.57x / 1.6x / 3.4x; CPU
  float32 0.19x / 0.50x / 2.8x and 0.14x / 0.38x / 1.1x.
- Round trips: the array path is within 1.0e-14 (float64) and 8.4e-6
  (float32); TFP is 2.0e-11 and 1.1e-2 on "wide", and 3.0e-5 in float32 on
  "spread".
- When a is one value the scalar path is faster (GPU float64 136 vs
  188 ns, float32 2.7 vs 13.3 ns, CPU float64 4.9 vs 5.7 us); the plain
  name runs it whenever a is passed as one value.
- Compile, one pinned CPU core: 3 s for the array path's values and
  93-99 s for its value_and_grad (TFP 1-2 s); unpinned, value_and_grad
  takes ~42 s, 26 s of it without the large-a root (2^14 elements). GPU
  6-10 s and 14-18 s.

WHAT IT DOES NOT ESTABLISH
--------------------------
Anything about other hardware, multi-core CPU scaling, or whole models.
The CPU compile times depend on XLA's CPU fusion emitter: with
XLA_FLAGS=--xla_cpu_use_fusion_emitters=false gammainc_array's core
compiles in 0.5 s instead of 24 s (jax 0.11); this script runs with the
default.

Run:  taskset -c 3 python experiments/23_gammaincinv_array_race.py f64
      taskset -c 3 python experiments/23_gammaincinv_array_race.py f32
"""

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

WIDE = sys.argv[1] == "f64"
jax.config.update("jax_enable_x64", WIDE)

import chebax  # noqa: E402

try:
    from tensorflow_probability.substrates import jax as tfp  # noqa: E402
except Exception as e:  # noqa: BLE001, any import failure skips it
    tfp = None
    TFP_SKIP = f"{type(e).__name__}: {e}"

DT = np.float64 if WIDE else np.float32
SIZES = {"cpu": 2**14, "gpu": 2**20}
N_REF = 200


def layouts(n):
    rng = np.random.default_rng(23)
    mu = np.clip(np.exp(rng.normal(1.0, 0.5, n)), 0.6, 20.0)
    out = {
        "a = 2.5 each": np.full(n, 2.5),
        "spread": mu**2 / 1.5**2,
        "wide": np.exp(rng.uniform(np.log(0.1), np.log(1e4), n)),
    }
    return {k: (a, rng.uniform(1e-12, 1.0, n)) for k, a in out.items()}


def methods(lname):
    def total(f):
        return lambda a, p: jnp.sum(f(a, p))

    out = {"array": total(chebax.gammaincinv_array)}
    if tfp is not None:
        out["tfp"] = total(tfp.math.igammainv)
    if lname == "a = 2.5 each":
        out["scalar"] = lambda a, p: jnp.sum(chebax.gammaincinv_scalar(a[0], p))
    return out


def timed(call, n, reps=5, budget=1.0):
    """Median ns per element; a warm-up spin first."""
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


def compiled(f, *args):
    t0 = time.perf_counter()
    c = jax.jit(f).lower(*args).compile()
    return c, time.perf_counter() - t0


def round_trip(f, a, p):
    """Worst relative error of the smaller tail at the returned x."""
    import mpmath as mp

    mp.mp.dps = 40
    x = np.asarray(f(jnp.asarray(a, DT), jnp.asarray(p, DT)), np.float64)
    worst = 0.0
    for ai, pi, xi in zip(a, p, x):
        if not np.isfinite(xi):
            return np.inf
        am, xm, pm = mp.mpf(float(DT(ai))), mp.mpf(xi), mp.mpf(float(DT(pi)))
        if xm < am:
            lo = mp.gammainc(am, 0, xm, regularized=True)
            hi = 1 - lo
        else:
            hi = mp.gammainc(am, xm, mp.inf, regularized=True)
            lo = 1 - hi
        got, want = (lo, pm) if pm <= 0.5 else (hi, 1 - pm)
        worst = max(worst, float(abs(got / want - 1)))
    return worst


def main():
    devs = {"cpu": jax.devices("cpu")[0]}
    try:
        devs["gpu"] = jax.devices("gpu")[0]
    except RuntimeError:
        pass
    print(
        f"{DT.__name__}, ns per element (value / value_and_grad in a and p); "
        + ", ".join(f"{k} N = {SIZES[k]} ({d.device_kind})" for k, d in devs.items()),
        flush=True,
    )
    if tfp is None:
        print(f"tfp skipped: {TFP_SKIP}", flush=True)
    for lname, (a, p) in layouts(N_REF).items():
        errs = [f"array {round_trip(chebax.gammaincinv_array, a, p):.1e}"]
        if tfp is not None:
            errs.append(f"tfp {round_trip(tfp.math.igammainv, a, p):.1e}")
        print(f"round trip, {lname:13s}: " + ", ".join(errs), flush=True)
    for dname, dev in devs.items():
        n = SIZES[dname]
        for lname, (a, p) in layouts(n).items():
            a, p = (jax.device_put(jnp.asarray(v, DT), dev) for v in (a, p))
            row = []
            for mname, f in methods(lname).items():
                v, cv = compiled(f, a, p)
                g, cg = compiled(jax.value_and_grad(f, (0, 1)), a, p)
                tv = timed(lambda: v(a, p), n)
                tg = timed(lambda: g(a, p), n)
                row.append(
                    f"{mname} {tv:8.1f} / {tg:8.1f} (compile {cv:.0f} / {cg:.0f} s)"
                )
            print(f"{dname} {lname:13s} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
