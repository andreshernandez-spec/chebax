"""hyp1f1_array: speed against jax.scipy's hyp1f1 and the per-call path.

Measures ns per element of chebax.hyp1f1_array (value; value_and_grad of the
sum in a, b and x) and log_hyp1f1_array, with a and b drawn per element in
four regimes:

    noncentral t   a = nu/2 + 1, nu ~ U(1, 18), b = 3/2, x ~ U(0, 8)
                   (the 1F1 term of the noncentral t density, mu up to 4)
    box, x <= X_S  a, b log-uniform on [0.1, 10], x ~ U(0, X_S), X_S the
                   seam of (a, b)'s column
    box, tail      x log-uniform on [X_S, 1e4]
    box, mixed     x log-uniform on [1e-3, 1e3]

Compared with jax.scipy.special.hyp1f1 (its series is truncated at 1e-8 and
unstable, jax#21503; its gradients come from differentiating that series) and
with vmap of
chebax.hyp1f1_scalar, which is what a per-element caller had before
(smaller N, it is ~1000x slower).

Two passes per cell, minimum kept; cells whose passes differ by more than
1.25x are marked *. CPU runs on one core (run under taskset -c N, XLA's
eigen pool off); GPU separately. float32 with x64 off (pass f32).

    taskset -c 6 python experiments/56_hyp1f1_array.py cpu      # ~5 min (a core whose
                                                               #  sibling is idle)
    python experiments/56_hyp1f1_array.py gpu
    python experiments/56_hyp1f1_array.py gpu f32

Does NOT establish: speed inside a model, a Numba port's speed, or accuracy
(tests/test_hyp1f1_array.py and experiments/57 have that).
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
from chebax._src.recipes import hyp1f1_array_table as _t  # noqa: E402


def seam(a, b):
    """X_S of each (a, b)'s base column."""
    w0 = (np.log(_t.AHI) - np.log(_t.ALO)) / _t.NB0
    i = [
        np.clip(np.floor((np.log(v) - np.log(_t.ALO)) / w0), 0, _t.NB0 - 1).astype(int)
        for v in (a, b)
    ]
    return np.asarray(_t.XS_COL)[i[0], i[1]]

DT = jnp.float32 if f32 else jnp.float64
N = 1 << (20 if dev == "gpu" else 16)
N_SCALAR = 1 << 8
REGIMES = ("noncentral t", "box, x <= X_S", "box, tail", "box, mixed")


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


def inputs(regime, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = np.log(0.1), np.log(10.0)
    if regime == "noncentral t":
        a = rng.uniform(1.0, 18.0, n) / 2.0 + 1.0
        b = np.full(n, 1.5)
        x = rng.uniform(0.0, 8.0, n)
    else:
        a = np.exp(rng.uniform(lo, hi, n))
        b = np.exp(rng.uniform(lo, hi, n))
        xs = seam(a, b)
        x = {
            "box, x <= X_S": lambda: rng.uniform(0.0, xs),
            "box, tail": lambda: np.exp(rng.uniform(np.log(xs), np.log(1e4))),
            "box, mixed": lambda: np.exp(rng.uniform(np.log(1e-3), np.log(1e3), n)),
        }[regime]()
    d = jax.devices(dev)[0]
    return tuple(jax.device_put(jnp.asarray(v, DT), d) for v in (a, b, x))


def vg(f):
    return jax.jit(
        jax.value_and_grad(lambda a, b, x: f(a, b, x).sum(), argnums=(0, 1, 2))
    )


ops = {
    "hyp1f1_array": jax.jit(chebax.hyp1f1_array),
    "hyp1f1_array v+g": vg(chebax.hyp1f1_array),
    "log_hyp1f1_array": jax.jit(chebax.log_hyp1f1_array),
    "log_hyp1f1_array v+g": vg(chebax.log_hyp1f1_array),
    "jax.scipy hyp1f1": jax.jit(jax.scipy.special.hyp1f1),
    "jax.scipy hyp1f1 v+g": vg(jax.scipy.special.hyp1f1),
}
if not f32:
    ops["vmap hyp1f1_scalar v+g"] = vg(jax.vmap(chebax.hyp1f1_scalar))

print(
    f"jax {jax.__version__} on {jax.devices(dev)[0]}, {'float32' if f32 else 'float64'}, "
    f"N = {N} (vmap hyp1f1_scalar N = {N_SCALAR}); ns per element, min of 2 passes"
)
print("| op | " + " | ".join(REGIMES) + " |")
print("|---|" + "---|" * len(REGIMES))
res = {}
for p in range(2):
    for r in REGIMES:
        full = inputs(r, N, 1)
        small = tuple(v[:N_SCALAR] for v in full)
        for name, f in ops.items():
            args = small if "vmap" in name else full
            res.setdefault((name, r), []).append(
                timeit(f, *args, reps=5 if "vmap" in name else 15)
            )
for name in ops:
    cells = []
    for r in REGIMES:
        v = res[(name, r)]
        flag = "*" if max(v) / min(v) > 1.25 else ""
        cells.append(f"{min(v):.1f}{flag}")
    print(f"| {name} | " + " | ".join(cells) + " |")
