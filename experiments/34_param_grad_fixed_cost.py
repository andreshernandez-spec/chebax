#!/usr/bin/env python3
"""Per-call costs in the parameter gradients: value_and_grad in the traced
parameters at N = 16 and N = 2^20, every function with a traced parameter.

WHAT THIS MEASURES
------------------
ms per call of jit(value_and_grad(sum f, params)) and of jit(sum f), on the
GPU and one CPU core, float32 and float64. A per-element cost grows with N;
a cost that is the same at N = 16 and N = 2^20 is per call. It looks for
the case experiments/33 found: hyp1f1's gradient in (a, b) on the GPU in
float32 took ~2.7 s per call whatever N, against 12 ms at N = 2^20 in
float64. XLA fused the derivative of the coefficient reconstruction (a
Clenshaw in a over the (136, 44) slice of the table) into one kernel,
loop_multiply_fusion, that ran for the whole 2.7 s. hyp1f1 is timed twice:
as in the library (an optimization_barrier on the reconstructed
coefficients) and without the barrier, the code before the fix.

FINDINGS (2026-11-06, RTX 3080 Laptop, jax 0.11.0)
--------------------------------------------------
results/34_param_grad_fixed_cost.txt.
- hyp1f1 without the barrier, GPU float32: 2656 ms per call at N = 16 and
  at N = 2^20; with it, 0.14 ms and 1.3 ms. float64 and the CPU run as fast
  with the barrier as without (within 1.2%), and the values and gradients are
  bit-identical in all four configurations.
- No other function has a per-call cost like it. At N = 16 every call is
  under 1.4 ms except the solvers on the GPU in float64 (stdtrit 3.5 ms,
  betaincinv 2.2 ms), whose Newton loops run a fixed number of steps.

WHAT IT DOES NOT ESTABLISH
--------------------------
Why XLA's GPU backend is slow on that fusion in float32 and not in float64
(only the float32 program was profiled); other GPUs; jax releases other than
the one run. The barrier blocks one fusion; a later XLA could fuse elsewhere.

Run:  taskset -c 3 python experiments/34_param_grad_fixed_cost.py   (~90 min,
      mostly compiling)
"""

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

SIZES = (16, 2**20)


def timed_ms(f, *args, budget=1.0):
    jax.block_until_ready(f(*args))
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < budget:
        out = f(*args)
        n += 1
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


def cases(chebax):
    """name -> (f(params..., x), params, x sampler)."""
    u = lambda lo, hi: (lambda r, n: r.uniform(lo, hi, n))  # noqa: E731
    return {
        "besselk_scalar": (chebax.besselk_scalar, (5.5,), u(0.1, 20)),
        "besseli_scalar": (chebax.besseli_scalar, (5.5,), u(0.1, 20)),
        "matern": (lambda n, r: chebax.matern(n, r, 1.0), (5.5,), u(0.0, 3)),
        "betainc_scalar": (chebax.betainc_scalar, (2.5, 30.0), u(0.01, 0.5)),
        "gammainc_scalar": (chebax.gammainc_scalar, (2.5,), u(0.1, 10)),
        "gammaincc_scalar": (chebax.gammaincc_scalar, (30.0,), u(10, 60)),
        "hyp1f1_scalar": (chebax.hyp1f1_scalar, (1.5, 4.0), u(0, 20)),
        "hyp1f1, no barrier": ("nobarrier", (1.5, 4.0), u(0, 20)),
        "log_hyp1f1": (chebax.log_hyp1f1, (1.5, 4.0), u(0, 200)),
        "stdtr": (chebax.stdtr, (4.5,), u(-5, 5)),
        "log_stdtr": (chebax.log_stdtr, (60.0,), u(-5, 5)),
        "stdtrit": (chebax.stdtrit, (4.5,), u(0.01, 0.99)),
        "betaincinv": (chebax.betaincinv, (2.5, 7.0), u(0.01, 0.99)),
        "gammaincinv_scalar": (chebax.gammaincinv_scalar, (2.5,), u(0.01, 0.99)),
        "vonmises_cdf": (chebax.vonmises_cdf, (5.0,), u(-3, 3)),
        "log_wright_bessel0": (chebax.log_wright_bessel0, (2.0,), u(0.1, 50)),
        "tweedie_logpdf": (
            lambda p, y: chebax.tweedie_logpdf(p, 2.0, 1.0, y),
            (1.5,),
            u(0.0, 10),
        ),
    }


def main():
    import chebax
    from chebax._src.recipes import hyp1f1 as _h
    from chebax._src.series import _chebval

    def coefs_no_barrier(tensor, sa, sb):
        # _coefs_traced as it was before the fix
        t = jnp.asarray(tensor)
        va = jax.vmap(jax.vmap(lambda row: _chebval(sa, row, (-1.0, 1.0))))(
            jnp.swapaxes(t, 1, 2)
        )
        return jax.vmap(lambda row: _chebval(sb, row, (-1.0, 1.0)))(va)

    lib_coefs = _h._coefs_traced
    print(f"jax {jax.__version__}; ms per call: value / value_and_grad in the params")
    for dtn, x64 in (("float32", False), ("float64", True)):
        jax.config.update("jax_enable_x64", x64)
        dt = jnp.float64 if x64 else jnp.float32
        for dname in ("gpu", "cpu"):
            dev = jax.devices(dname)[0]
            print(f"\n--- {dtn} {dname}", flush=True)
            outputs = {}
            for name, (f, params, sample) in cases(chebax).items():
                if f == "nobarrier":
                    f = chebax.hyp1f1_scalar
                    _h._coefs_traced = coefs_no_barrier
                row = []
                for n in SIZES:
                    rng = np.random.default_rng(34)
                    x = jax.device_put(jnp.asarray(sample(rng, n), dt), dev)
                    ps = [jax.device_put(jnp.asarray(p, dt), dev) for p in params]
                    argn = tuple(range(len(ps)))

                    def tot(*a, f=f):
                        return jnp.sum(f(*a))

                    tv = timed_ms(jax.jit(tot), *ps, x)
                    vg = jax.jit(jax.value_and_grad(tot, argn))
                    tg = timed_ms(vg, *ps, x)
                    outputs[name, n] = [
                        np.asarray(v) for v in jax.tree.leaves(vg(*ps, x))
                    ]
                    row.append(f"N={n}: {tv:8.3f} / {tg:9.3f}")
                _h._coefs_traced = lib_coefs
                if name == "hyp1f1, no barrier":
                    same = all(
                        np.array_equal(u, v)
                        for n in SIZES
                        for u, v in zip(outputs[name, n], outputs["hyp1f1_scalar", n])
                    )
                    row.append(f"value and grads bit-identical with barrier: {same}")
                print(f"{name:20s} " + "   ".join(row), flush=True)


if __name__ == "__main__":
    sys.exit(main())
