"""betainc_array: speed against jax.scipy's betainc and the per-call path.

Measures ns per element of chebax.betainc_array (value; value_and_grad of
the sum in a, b and x) and log_betainc_array, with a and b drawn per element
in five regimes (log-uniform on [ALO, 0.1], [0.1, 1], [1, 10], [10, 100] and
the whole box [ALO, 100], ALO = 10^-2.125) and x ~ Beta(a, b). Compared with
jax.scipy.special.betainc (value only: jax has no a or b gradient,
jax#38610) and with vmap of chebax.betainc_scalar, which is what a
per-element caller had before (smaller N, it is ~1000x slower; its box is
[0.1, 100], so only those regimes).

Two passes per cell, minimum kept; cells whose passes differ by more than
1.25x are marked *. CPU runs on one core (run under taskset -c N, XLA's
eigen pool off); GPU separately. float32 with x64 off (pass f32).

    taskset -c 5 python experiments/54_betainc_array.py cpu      # ~5 min
    python experiments/54_betainc_array.py gpu
    python experiments/54_betainc_array.py gpu f32

Does NOT establish: speed inside a model (fused loops, gathers), the
Numba port's speed, or accuracy (tests/test_betainc_array.py has that).
"""

import os
import sys
import time

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
f32 = "f32" in sys.argv
if dev == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
    ).strip()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", not f32)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes.betainc_array_table import ALO  # noqa: E402

DT = jnp.float32 if f32 else jnp.float64
N = 1 << (20 if dev == "gpu" else 16)
N_SCALAR = 1 << 8
REGIMES = {
    "a, b in [ALO, 0.1]": (ALO, 0.1),
    "[0.1, 1]": (0.1, 1.0),
    "[1, 10]": (1.0, 10.0),
    "[10, 100]": (10.0, 100.0),
    "box": (ALO, 100.0),
}


def timeit(f, *args, reps=15):
    jax.block_until_ready(f(*args))
    for _ in range(3):  # GPU clock settles after compiling
        jax.block_until_ready(f(*args))
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        best = min(best, time.perf_counter() - t)
    return best / args[0].shape[0] * 1e9


def inputs(lo, hi, n, seed):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    b = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    x = np.clip(rng.beta(a, b), 1e-30, 1 - 1e-16)
    d = jax.devices(dev)[0]
    return tuple(jax.device_put(jnp.asarray(v, DT), d) for v in (a, b, x))


def vg(f):
    return jax.jit(
        jax.value_and_grad(lambda a, b, x: f(a, b, x).sum(), argnums=(0, 1, 2))
    )


ops = {
    "betainc_array": jax.jit(chebax.betainc_array),
    "betainc_array v+g": vg(chebax.betainc_array),
    "log_betainc_array": jax.jit(chebax.log_betainc_array),
    "log_betainc_array v+g": vg(chebax.log_betainc_array),
    "jax.scipy betainc": jax.jit(jax.scipy.special.betainc),
}
if not f32:
    ops["vmap betainc_scalar v+g"] = vg(jax.vmap(chebax.betainc_scalar))

print(
    f"jax {jax.__version__} on {jax.devices(dev)[0]}, {'float32' if f32 else 'float64'}, "
    f"N = {N} (vmap betainc_scalar N = {N_SCALAR}); ns per element, min of 2 passes"
)
print("| op | " + " | ".join(REGIMES) + " |")
print("|---|" + "---|" * len(REGIMES))
res = {}
for p in range(2):
    for r, (lo, hi) in REGIMES.items():
        full = inputs(lo, hi, N, 1)
        small = tuple(v[:N_SCALAR] for v in full)
        for name, f in ops.items():
            if "betainc_scalar" in name and lo < 0.1:
                continue
            args = small if "vmap" in name else full
            res.setdefault((name, r), []).append(
                timeit(f, *args, reps=5 if "vmap" in name else 15)
            )
for name in ops:
    cells = []
    for r in REGIMES:
        v = res.get((name, r))
        if v is None:
            cells.append("-")
            continue
        flag = "*" if max(v) / min(v) > 1.25 else ""
        cells.append(f"{min(v):.1f}{flag}")
    print(f"| {name} | " + " | ".join(cells) + " |")
