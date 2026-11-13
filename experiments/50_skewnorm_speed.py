"""Where log_skewnorm_cdf's time goes, and what the runtime's design choices
are worth.

Each variant is the library's own skewnorm.py with one piece changed: built from
its source text (the substitution must apply) or with one module global swapped,
and the unchanged copy is first checked equal to chebax.log_skewnorm_cdf bit
for bit. Timed: values and value_and_grad of the sum in (a, x), a uniform on
[-5, 5] and x on [-4, 4], 2^16 elements on one CPU core (run under taskset -c
0) or 2^20 on the GPU (--gpu), float64, or float32 with --float32 (a separate
process: x64 is global). Median of 20 calls after warm-up, best of two passes.

Ablations (a cheap stand-in, the result is wrong): no patch sum, no locate (the
coordinate clipped onto [-1, 1] in patch 0), no erfcx, plain jnp.log1p (what
jax >= 0.12 runs; this environment's jax is older). Design variants (the
result is right):

- lax.erf: erf(y) from lax.erf, not the series and 1 - erfc
- log(erf + exp(lk)): ln A without logaddexp
- gathered cuts: the cell found by gathering each element's row of cuts
  (locate_patch, as the first version did), not by compares with constants
- erfcx(mu) in the value: the tail's erfcx computed in the value path too, as
  when the table held ln R without it (added as 0 * ln erfcx, which XLA keeps)
- float32 7 x 7 blocks (TOL32 = 1e-8), not 6 x 6
- barrier: an optimization_barrier after the patch sum

It does NOT establish: costs inside a PyMC model, other XLA versions (fusion
decisions move between releases), or accuracy (tests/test_skewnorm.py).

    taskset -c 0 python experiments/50_skewnorm_speed.py          # ~4 min
    taskset -c 0 python experiments/50_skewnorm_speed.py --float32
    python experiments/50_skewnorm_speed.py --gpu
    python experiments/50_skewnorm_speed.py --gpu --float32
"""

import argparse
import inspect
import os
import time
import types

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--float32", action="store_true")
args = ap.parse_args()
if not args.gpu:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402

if not args.float32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes import _common  # noqa: E402
from chebax._src.recipes import skewnorm as S  # noqa: E402

SOURCE = inspect.getsource(S)
DT = jnp.empty(()).dtype
N = 2**20 if args.gpu else 2**16
rng = np.random.default_rng(0)
A = jnp.asarray(rng.uniform(-5, 5, N), DT)
X = jnp.asarray(rng.uniform(-4, 4, N), DT)


def build(name, subs=(), swaps=None):
    src = SOURCE
    for old, new in subs:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    mod = types.ModuleType(f"skewnorm_{name}")
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    for k, v in (swaps or {}).items():
        setattr(mod, k, v)
    return mod


def gathered_locate(val, cuts, which):
    """locate_patch on a per-element row of cuts, gathered by chart."""
    rows = [np.asarray([float(c) for c in r]) for r in cuts]
    width = max(map(len, rows))
    rows = [np.concatenate([r, np.full(width - len(r), r[-1])]) for r in rows]
    table = jnp.asarray(np.stack(rows), val.dtype)
    n = jnp.asarray([len(r) - 1 for r in cuts])
    zone = which.astype(jnp.int32)
    return _common.locate_patch(val, table[zone], n[zone])


ERF_NEW = "    erf_y = jnp.where(near, y * series, -jnp.expm1(ln_erfcx_y - y2))\n"
LA_NEW = "    l_a = jnp.logaddexp(jnp.log(erf_y), lk)\n"
VARIANTS = [
    ("as shipped", {}),
    ("no patch sum", {"swaps": {"patch_sum": lambda tab, n, sy, sp: 0.0 * sy}}),
    (
        "no locate",
        {
            "swaps": {
                "locate_static": lambda val, cuts, which: (
                    jnp.zeros(jnp.shape(val), jnp.int32),
                    jnp.clip(val, -1.0, 1.0),
                )
            }
        },
    ),
    ("no erfcx", {"swaps": {"erfcx": lambda y: 1.0 / (1.0 + y)}}),
    ("plain jnp.log1p", {"swaps": {"log1p": jnp.log1p}}),
    ("lax.erf", {"subs": [(ERF_NEW, "    erf_y = jax.lax.erf(y)\n")]}),
    ("log(erf + exp(lk))", {"subs": [(LA_NEW, "    l_a = jnp.log(erf_y + jnp.exp(lk))\n")]}),
    ("gathered cuts", {"swaps": {"locate_static": gathered_locate}}),
    (
        "erfcx(mu) in the value",
        {
            "subs": [
                (
                    "    lk = jnp.where(fin_b, lk_s - sig2, -jnp.inf)\n",
                    "    lk = jnp.where(fin_b, lk_s - sig2 + 0.0 * ln_ex, -jnp.inf)\n",
                )
            ]
        },
    ),
    (
        "barrier",
        {
            "subs": [
                (
                    "    f = patch_sum(tab, n, sy, sp)\n",
                    "    f = jax.lax.optimization_barrier(patch_sum(tab, n, sy, sp))\n",
                )
            ]
        },
    ),
]
if DT == jnp.float32:
    VARIANTS.append(
        (
            "float32 7 x 7 blocks",
            {
                "subs": [
                    (
                        "_t.TABLE[:, : _t.KY32, : _t.KP32]",
                        "_t.TABLE[:, :7, :7]",
                    )
                ]
            },
        )
    )


def time_one(f):
    jax.block_until_ready(f(A, X))
    jax.block_until_ready(f(A, X))
    passes = []
    for _ in range(2):
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            jax.block_until_ready(f(A, X))
            ts.append(time.perf_counter() - t0)
        passes.append(float(np.median(ts)))
    return min(passes) / N * 1e9


def both(fn):
    value = jax.jit(lambda a, x: fn(a, x))
    vag = jax.jit(jax.value_and_grad(lambda a, x: jnp.sum(fn(a, x)), (0, 1)))
    return time_one(value), time_one(vag)


def textbook(a, x):
    sp = jax.scipy.special
    return jnp.log(sp.ndtr(x) - 2.0 * sp.owens_t(x, a))


dev = jax.devices()[0]
print(f"jax {jax.__version__}, {dev.platform} ({dev.device_kind}), {DT}, n = {N}")
shipped = build("check").log_skewnorm_cdf
same = np.array_equal(np.asarray(shipped(A, X)), np.asarray(chebax.log_skewnorm_cdf(A, X)))
print(f"unchanged copy equals chebax.log_skewnorm_cdf bit for bit: {same}")
assert same
print(f"{'variant':26} {'value ns/el':>12} {'vag ns/el':>10} {'value vs shipped':>17} {'vag vs shipped':>15}")
tv, tg = both(textbook)
rows = []
for name, spec in VARIANTS:
    mod = build(name.replace(" ", "_"), spec.get("subs", ()), spec.get("swaps"))
    rows.append((name, *both(mod.log_skewnorm_cdf)))
base_v, base_g = rows[0][1], rows[0][2]
for name, v, g in rows:
    print(f"{name:26} {v:12.2f} {g:10.2f} {v / base_v:16.2f}x {g / base_g:14.2f}x")
print(f"{'textbook ln(ndtr - 2 T)':26} {tv:12.2f} {tg:10.2f} {tv / base_v:16.2f}x {tg / base_g:14.2f}x")
print("(vs shipped = this row's time / the shipped time: above 1 is slower than shipped)")
