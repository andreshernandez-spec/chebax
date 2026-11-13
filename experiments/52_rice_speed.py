"""Speed of log_marcum_p1 and log_marcum_q1 (the Rice log CDF and log SF)
against TFP's NoncentralChi2, the other Rice CDF in jax, and where chebax's
time goes.

TFP's log_cdf(x^2) at df = 2, noncentrality a^2 sums Abramowitz and Stegun
26.4.25 in a while_loop that runs until every element has converged, so its
cost grows with the largest a and x in the batch; its gradient in the
noncentrality is a second sum at df = 4. Two input ranges, uniform: moderate,
a on [0, 5] and x on [0, 10]; wide, a on [0, 50] and x on [0, 60]. Timed:
values and value_and_grad of the sum in (a, x), 2^16 elements on one CPU core
(run under taskset -c 0) or 2^20 on the GPU (--gpu), float64, or float32 with
--float32 (a separate process: x64 is global). Median of 20 calls after
warm-up, best of two passes.

Ablations of chebax (cheap stand-ins, the result is wrong), each the shipped
marcum.py with one module global swapped, after the unchanged copy is checked
equal to chebax.log_marcum_p1 bit for bit: no patch sum, no locate (patch 0
of each chart, the coordinate clipped onto [-1, 1]), no erfcx. And one design
variant (the result is right), on the CPU: the slopes reading the value's
patch sum, where the shipped code gives them their own from the negated
table. On other platforms the shipped code shares, and the variant is the
opposite: their own sum there too.

It does NOT establish: accuracy (TFP's is in experiments/51, chebax's in
tests/test_marcum.py), costs inside a PyMC model, or other XLA versions.

    taskset -c 0 python experiments/52_rice_speed.py
    taskset -c 0 python experiments/52_rice_speed.py --float32
    python experiments/52_rice_speed.py --gpu
    python experiments/52_rice_speed.py --gpu --float32
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
from tensorflow_probability.substrates import jax as tfp  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes import marcum as R  # noqa: E402

DT = jnp.empty(()).dtype
N = 2**20 if args.gpu else 2**16
SOURCE = inspect.getsource(R)


def build(name, swaps, subs=()):
    src = SOURCE
    for old, new in subs:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    mod = types.ModuleType(f"marcum_{name}")
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    for k, v in swaps.items():
        setattr(mod, k, v)
    return mod


def time_one(f, a, x):
    jax.block_until_ready(f(a, x))
    jax.block_until_ready(f(a, x))
    passes = []
    for _ in range(2):
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            jax.block_until_ready(f(a, x))
            ts.append(time.perf_counter() - t0)
        passes.append(float(np.median(ts)))
    return min(passes) / N * 1e9


def both(fn, a, x):
    value = jax.jit(lambda a, x: fn(a, x))
    vag = jax.jit(jax.value_and_grad(lambda a, x: jnp.sum(fn(a, x)), (0, 1)))
    return time_one(value, a, x), time_one(vag, a, x)


def tfp_cdf(a, x):
    return tfp.distributions.NoncentralChi2(2.0, a * a).log_cdf(x * x)


def tfp_sf(a, x):
    return tfp.distributions.NoncentralChi2(2.0, a * a).log_survival_function(x * x)


def no_locate(val, cuts, which):
    return jnp.zeros(jnp.shape(val), jnp.int32), jnp.clip(val, -1.0, 1.0)


FLIP = (
    "    ln_s_slope = ln_s_of(_slope_sum(tab, n, sy, sp, f))\n",
    "    ln_s_slope = ln_s_of(%s(tab, n, sy, sp, f))\n"
    % ("_own_sum" if args.gpu else "_shared_sum"),
)
ABLATIONS = [
    ("no patch sum", {"patch_sum": lambda tab, n, sy, sp: 0.0 * sy}, ()),
    ("no locate", {"locate_static": no_locate}, ()),
    ("no erfcx", {"erfcx": lambda y: 1.0 / (1.0 + y)}, ()),
    ("slopes " + ("own patch sum" if args.gpu else "share the patch sum"), {}, (FLIP,)),
]

dev = jax.devices()[0]
print(f"jax {jax.__version__}, {dev.platform} ({dev.device_kind}), {DT}, n = {N}")
rng = np.random.default_rng(52)
for label, a_hi, x_hi in (("moderate", 5.0, 10.0), ("wide", 50.0, 60.0)):
    a = jnp.asarray(rng.uniform(0, a_hi, N), DT)
    x = jnp.asarray(rng.uniform(0, x_hi, N), DT)
    check = build("check", {}).log_marcum_p1
    same = np.array_equal(
        np.asarray(check(a, x)), np.asarray(chebax.log_marcum_p1(a, x))
    )
    assert same, "the unchanged copy differs from chebax.log_marcum_p1"
    print(f"\n{label}: a on [0, {a_hi:g}], x on [0, {x_hi:g}]")
    print(f"  {'':24} {'value ns/el':>12} {'vag ns/el':>10}")
    rows = [
        ("chebax log_marcum_p1", chebax.log_marcum_p1),
        ("chebax log_marcum_q1", chebax.log_marcum_q1),
        ("TFP log_cdf", tfp_cdf),
        ("TFP log_survival_function", tfp_sf),
    ]
    t = {}
    for name, fn in rows:
        t[name] = both(fn, a, x)
        print(f"  {name:24} {t[name][0]:12.1f} {t[name][1]:10.1f}")
    for ours, theirs in (
        ("chebax log_marcum_p1", "TFP log_cdf"),
        ("chebax log_marcum_q1", "TFP log_survival_function"),
    ):
        (cv, cg), (tv, tg) = t[ours], t[theirs]
        print(
            f"  {theirs} / {ours}: values {tv / cv:.2f}x, value_and_grad {tg / cg:.2f}x"
        )
    if label == "moderate":
        base_v, base_g = t["chebax log_marcum_p1"]
        print("  variants of log_marcum_p1 (this row's time / shipped):")
        for name, swaps, subs in ABLATIONS:
            v, g = both(build(name.replace(" ", "_"), swaps, subs).log_marcum_p1, a, x)
            print(
                f"  {name:24} {v:12.1f} {g:10.1f}   {v / base_v:.2f}x / {g / base_g:.2f}x"
            )
