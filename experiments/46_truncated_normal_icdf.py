"""PyMC's TruncatedNormal icdf on the JAX backend, with pytensor's own lowering of
NdtriExp (ndtri(exp(y))) against chebax.pytensor's (chebax.ndtri_exp): accuracy
against mpmath, and speed of the compiled icdf and of its gradient in mu.

Both variants lower the same graph and differ only in NdtriExp's lowering; every
other op is lowered as chebax.pytensor lowers it (pytensor 3.2.3's own Erfcx
lowering needs tfp). The icdf forms log p and log(1 - p) for the truncated
distribution and calls ndtri_exp on both (pt.switch evaluates both branches). Far in a tail, log(1 - p) is
below -745 in float64 (-103 in float32), where exp underflows and pytensor's
lowering returns -inf. PyMC's normal-CDF helpers build constants such as
pt.log(0.5) and pt.sqrt(2.0), which pytensor's default cast policy ("custom") makes
float32; that limits the icdf to ~1e-8 whatever ndtri_exp does. The cast policy
"numpy+floatX" keeps them float64 and shows what ndtri_exp itself leaves.

Sections:
  accuracy  the icdf of TruncatedNormal(0, 1, lower, upper), float64 parameters,
            at u = 1e-12, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12 for each
            truncation; error |x - x_ref| / max(|x_ref|, 1) against an mpmath
            quantile (60 digits, Newton in log Phi on the side of the smaller
            tail), and the count of nonfinite results. Float64 under both cast
            policies, then float32 (floatX = float32, x64 off; u up to 1 - 1e-6).
  speed     ns per element at N = 2^20 (CPU) or 2^22 (GPU), u uniform on (0, 1),
            TruncatedNormal(mu, 1, lower=0): the icdf, and value_and_grad of its
            sum in mu; median over E46_REPS alternations of the best of 7 calls.

Usage:
  [E46_DEVICES=cpu,gpu E46_REPS=5 E46_CORE=12] python experiments/46_truncated_normal_icdf.py

Needs pymc and pytensor >= 3.2.2. One CPU core is taskset's, with eigen's thread
pool off.

Not established: other PyMC distributions that call ndtri_exp, or speed inside a
sampler, where the icdf is one op among many.
"""

import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "src"

CASES = (
    (-1.0, None),
    (5.0, None),
    (10.0, None),
    (38.0, None),
    (40.0, None),
    (1000.0, None),
    (None, 1.0),
    (None, -5.0),
    (None, -40.0),
    (-3.0, 3.0),
    (5.0, 6.0),
    (40.0, 41.0),
)
CASES_F32 = (
    (-1.0, None),
    (5.0, None),
    (10.0, None),
    (13.0, None),
    (20.0, None),
    (None, -13.0),
    (-3.0, 3.0),
    (5.0, 6.0),
)
U64 = (1e-12, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12)
U32 = U64[:-1]


def _quantile(mp, a, b, u):
    """mpmath quantile of N(0, 1) truncated to [a, b] at u."""

    def erfcx(x):
        if x < 50:
            return mp.exp(x * x) * mp.erfc(x)
        t, term, total, n = 1 / (2 * x * x), mp.mpf(1), mp.mpf(1), 1
        while abs(term) > mp.eps:
            term *= -(2 * n - 1) * t
            total += term
            n += 1
        return total / (x * mp.sqrt(mp.pi))

    def log_phi(z):
        x = -z / mp.sqrt(2)
        return mp.log(erfcx(x) / 2) - x * x

    def solve(target_log):  # z <= 0 with log Phi(z) = target_log
        if target_log > -3:
            z = mp.sqrt(2) * mp.erfinv(2 * mp.exp(target_log) - 1)
        else:
            s = mp.sqrt(-2 * target_log)
            z = -(s - (mp.log(s) + mp.log(2 * mp.pi) / 2) / s)
        for _ in range(200):
            x = -z / mp.sqrt(2)
            step = (log_phi(z) - target_log) * mp.sqrt(mp.pi / 2) * erfcx(x)
            z -= step
            if abs(step) <= (abs(z) + mp.mpf(10) ** -50) * mp.eps * 16:
                break
        return z

    a = -mp.inf if a is None else mp.mpf(a)
    b = mp.inf if b is None else mp.mpf(b)
    u = mp.mpf(u)
    if a + b > 0:  # the right tail by symmetry
        return -_quantile(mp, -b, -a, 1 - u)
    pa = mp.mpf(0) if a == -mp.inf else mp.exp(log_phi(a))
    pb = mp.mpf(1) if b == mp.inf else mp.exp(log_phi(b))
    target = pa + u * (pb - pa)
    if target > mp.mpf(1) / 2:
        return -solve(mp.log(1 - target))
    return solve(mp.log(target))


def _per_lowering(build):
    """build() once with pytensor's own NdtriExp lowering and once with
    chebax's; every other op as chebax.pytensor lowers it (pytensor 3.2.3
    lowers Erfcx through tfp). jax_funcify reads the registry when a
    function is lowered, so each build keeps the lowering it saw."""
    from pytensor.link.jax.dispatch import jax_funcify
    from pytensor.link.jax.dispatch.scalar import jax_funcify_NdtriExp
    from pytensor.scalar.math import NdtriExp

    import chebax.pytensor as cp

    assert pathlib.Path(cp.__file__).resolve().is_relative_to(SRC)
    out = {}
    for name, lowering in (
        ("pytensor", jax_funcify_NdtriExp),
        ("chebax", cp._jax_funcify_NdtriExp),
    ):
        jax_funcify.register(NdtriExp)(lowering)
        out[name] = build()
    return out


def _icdf_functions(cases, dt):
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    u = pt.vector("u", dtype=dt.__name__)

    def build():
        out = {}
        for lo, hi in cases:
            lower = None if lo is None else dt(lo)
            upper = None if hi is None else dt(hi)
            d = pm.TruncatedNormal.dist(mu=dt(0), sigma=dt(1), lower=lower, upper=upper)
            out[(lo, hi)] = pytensor.function([u], pm.icdf(d, u), mode="JAX")
        return out

    fns = _per_lowering(build)
    return fns["pytensor"], fns["chebax"]


def accuracy(tag, dt):
    import mpmath as mp
    import numpy as np

    mp.mp.dps = 60
    f32 = dt == np.float32
    cases, us = (CASES_F32, U32) if f32 else (CASES, U64)
    plain, cheb = _icdf_functions(cases, dt)
    uv = np.array(us, dt)
    print(f"{tag} icdf error |x - x_ref| / max(|x_ref|, 1), n nonfinite")
    print(f"  {'lower':>7s} {'upper':>7s} {'pytensor':>18s} {'chebax':>18s}")
    for case in cases:
        ref = np.array([float(_quantile(mp, *case, float(v))) for v in uv])
        cells = []
        for fns in (plain, cheb):
            got = np.asarray(fns[case](uv), np.float64)
            fin = np.isfinite(got)
            err = np.abs(got[fin] - ref[fin]) / np.maximum(np.abs(ref[fin]), 1.0)
            worst = f"{err.max():8.1e}" if err.size else f"{'-':>8s}"
            cells.append(f"{worst} {int((~fin).sum()):4d}")
        lo = "-inf" if case[0] is None else f"{case[0]:g}"
        hi = "inf" if case[1] is None else f"{case[1]:g}"
        print(f"  {lo:>7s} {hi:>7s} {cells[0]:>18s} {cells[1]:>18s}")


def speed(tag, device, dt):
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.fg import FunctionGraph
    from pytensor.link.jax.dispatch import jax_funcify

    import jax
    import jax.numpy as jnp

    n = 2**20 if device == "cpu" else 2**22
    reps = int(os.environ.get("E46_REPS", 5))
    u = pt.vector("u", dtype=dt.__name__)
    mu = pt.scalar("mu", dtype=dt.__name__)
    d = pm.TruncatedNormal.dist(mu=mu, sigma=dt(1), lower=dt(0))
    fg = FunctionGraph([u, mu], [pm.icdf(d, u)], clone=True)

    def lowered():
        fn = jax_funcify(fg)
        return lambda uv, m: fn(uv, m)[0]

    fns = _per_lowering(lowered)
    plain, cheb = fns["pytensor"], fns["chebax"]
    rng = np.random.default_rng(46)
    uv = jax.device_put(jnp.asarray(rng.uniform(0, 1, n), dt))
    m = jnp.asarray(0.3, dt)
    print(f"{tag} ns per element, N = {n}, TruncatedNormal(mu, 1, lower=0)")
    for mode in ("icdf", "value_and_grad in mu"):
        comps = {}
        for name, f in (("pytensor", plain), ("chebax", cheb)):
            g = (
                f
                if mode == "icdf"
                else jax.value_and_grad(lambda m, f=f: jnp.sum(f(uv, m)))
            )
            args = (uv, m) if mode == "icdf" else (m,)
            comps[name] = (jax.jit(g).lower(*args).compile(), args)
            jax.block_until_ready(comps[name][0](*args))
        times = {k: [] for k in comps}
        for _ in range(reps):
            for k, (c, args) in comps.items():
                best = np.inf
                for _ in range(7):
                    t0 = time.perf_counter()
                    jax.block_until_ready(c(*args))
                    best = min(best, time.perf_counter() - t0)
                times[k].append(best)
        tp = np.median(times["pytensor"])
        for k in comps:
            t = np.median(times[k])
            print(
                f"  {mode:21s} {k:9s} {t / n * 1e9:7.3f} ns {tp / t:5.2f}x pytensor's"
            )


def run(section, device, dtype, policy):
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    flags = [f"cast_policy={policy}"]
    if dtype == "f32":
        flags.append("floatX=float32")
    os.environ["PYTENSOR_FLAGS"] = ",".join(flags)
    sys.path.insert(0, str(SRC))
    import warnings

    warnings.filterwarnings("ignore")  # pymc's import-time and dist() warnings
    import jax

    jax.config.update("jax_enable_x64", dtype == "f64")
    import numpy as np

    dt = np.float64 if dtype == "f64" else np.float32
    tag = f"[{device} {dtype} cast_policy={policy}]"
    if section == "accuracy":
        accuracy(tag, dt)
    else:
        speed(tag, device, dt)


def main():
    devices = os.environ.get("E46_DEVICES", "cpu,gpu").split(",")
    core = os.environ.get("E46_CORE", "4")
    jobs = [
        ("accuracy", "cpu", "f64", "custom"),
        ("accuracy", "cpu", "f64", "numpy+floatX"),
        ("accuracy", "cpu", "f32", "custom"),
    ]
    jobs += [("speed", d, t, "custom") for d in devices for t in ("f64", "f32")]
    for section, device, dtype, policy in jobs:
        cmd = [sys.executable, __file__, "run", section, device, dtype, policy]
        if device == "cpu":
            cmd = ["taskset", "-c", core, *cmd]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            sys.exit(res.stderr[-3000:])
        print(res.stdout, end="", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run(*sys.argv[2:6])
    else:
        main()
