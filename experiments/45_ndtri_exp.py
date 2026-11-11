"""ndtri_exp(y) = ndtri(e^y) from Chebyshev panels against jax's ndtri(exp(y)),
the composition pytensor's JAX backend lowers NdtriExp to (TruncatedNormal's icdf):
accuracy against mpmath, the operations each compiles to, and speed of the value
and of value_and_grad on one CPU core and on the GPU, float64 and float32.

chebax.ndtri_exp reads the centre, e^-1.5 <= e^y <= 1 - e^-1.5, from panels in
d = y + log 2 (z = d h(d), relative next to the median) and both tails from panels
in s = sqrt(-2 u), u = y below and log(1 - e^y) above, on 16 panels an octave up
to s = 2^30. Centre and tail panels lie in one table, so each element runs one
series. The derivative has its own table. "other gather" is the runtime with the
gather form it does not use on that device (one gather of each element's row of
coefficients on the CPU, one gather per coefficient elsewhere).

Sections:
  accuracy  the runtime against mpmath (50 digits) in bands of y: past exp's
            underflow (y < -745 in float64, < -87 in float32), the lower tail to
            the seam at -1.5, the centre, the upper tail to y = -1e-3, and on to
            the smallest normal; y = -log 2 +- 10^-k near the median. Relative
            error of z and of dz/dy, and the count of nonfinite results where
            the true value is finite. scipy's ndtri_exp (float64 only) for
            comparison.
  ops       transcendental and divide instructions in the compiled CPU programs.
  speed     ns per element at N = 2^20 (CPU) or 2^22 (GPU), median over E45_REPS
            alternations of the best of 7 calls; value_and_grad of sum(f(y)).
            "temp" is the compiled program's temporary memory from XLA, "compile"
            the time to trace, lower and compile it.

Usage:
  [E45_DEVICES=cpu,gpu E45_REPS=5 E45_CORE=10] python experiments/45_ndtri_exp.py

One CPU core is taskset's, with eigen's thread pool off; float32 runs with x64 off.

Not established: other CPUs or GPUs (consumer GPU float64 runs at 1/64 rate, so a
datacenter part will rank differently); speed inside a model, where ndtri_exp
shares a fused loop with the rest of the icdf.
"""

import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "src"


def _ref(mp, y):
    """(z, dz/dy) at 50 digits: Newton in log Phi, on the side whose tail
    probability is the smaller one, from log(1 - e^y) above the median."""
    y = mp.mpf(y)
    if y == 0:
        return mp.inf, mp.inf
    upper = y > -mp.log(2)
    u = mp.log(-mp.expm1(y)) if upper else y

    def erfcx(x):
        if x < 50:
            return mp.exp(x * x) * mp.erfc(x)
        t, term, total, n = 1 / (2 * x * x), mp.mpf(1), mp.mpf(1), 1
        while abs(term) > mp.eps:
            term *= -(2 * n - 1) * t
            total += term
            n += 1
        return total / (x * mp.sqrt(mp.pi))

    # z <= 0 with log Phi(z) = u; Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt 2)
    if u > -3:
        z = mp.sqrt(2) * mp.erfinv(2 * mp.exp(u) - 1)
    else:
        s = mp.sqrt(-2 * u)
        z = -(s - (mp.log(s) + mp.log(2 * mp.pi) / 2) / s)
    for _ in range(100):
        x = -z / mp.sqrt(2)
        mills = mp.sqrt(mp.pi / 2) * erfcx(x)
        step = (mp.log(erfcx(x) / 2) - x * x - u) * mills
        z -= step
        if abs(step) <= (abs(z) + mp.mpf(10) ** -30) * mp.eps * 16:
            break
    mills = mp.sqrt(mp.pi / 2) * erfcx(-z / mp.sqrt(2))
    if not upper:
        return z, mills
    # z -> -z; dz/dy = e^y / phi(z) = mills * e^y / (1 - e^y)
    return -z, mills * mp.exp(y) / -mp.expm1(y)


def _grid(np, dt):
    f64 = dt == np.float64
    big = np.finfo(dt).max
    under = 745.0 if f64 else 87.0
    rng = np.random.default_rng(45)
    pts = [
        -np.geomspace(under, big / 2, 299),
        -np.array([big]),
        -np.geomspace(1.5, under, 400),
        -rng.uniform(1.5, 40, 400),
        -rng.uniform(0.25, 1.5, 600),
        -np.geomspace(1e-3, 0.25, 300),
        -np.geomspace(np.finfo(dt).tiny, 1e-3, 300),
    ]
    for k in range(1, 16 if f64 else 8):
        pts.append(np.array([-np.log(2) - 10.0**-k, -np.log(2) + 10.0**-k]))
    m = np.asarray(-np.log(2), dt)
    pts.append(np.array([m, np.nextafter(m, 0), np.nextafter(m, -1)]))
    y = np.unique(np.concatenate(pts).astype(dt))
    return y[(y < 0) & np.isfinite(y)]


def _bands(np, f64):
    under = -745.0 if f64 else -87.0
    return (
        (f"[-max, {under:g})", -np.inf, under),
        (f"[{under:g}, -1.5)", under, -1.5),
        ("[-1.5, -0.25)", -1.5, -0.25),
        ("[-0.25, -1e-3)", -0.25, -1e-3),
        ("[-1e-3, -tiny]", -1e-3, 0.0),
    )


def run(device, dtype):
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, str(SRC))
    import jax

    jax.config.update("jax_enable_x64", dtype == "f64")
    import jax.numpy as jnp
    import numpy as np

    import chebax

    assert pathlib.Path(chebax.__file__).resolve().is_relative_to(SRC)
    dt = np.float64 if dtype == "f64" else np.float32
    funs = {
        "chebax": chebax.ndtri_exp,
        "other gather": _other_gather(device),
        "jax ndtri(exp)": lambda y: jax.scipy.special.ndtri(jnp.exp(y)),
    }
    tag = f"[{device} {dtype}]"
    if device == "cpu":
        _ops(tag, funs, jax, jnp)
        _accuracy(tag, funs, dt, jax, jnp, np)
    _speed(tag, device, funs, dt, jax, jnp, np)


def _other_gather(device):
    import jax

    from chebax._src.recipes import _common
    from chebax._src.recipes import ndtri_exp as N

    form = _common.row_series if device == "cpu" else _common.per_coefficient_series

    def with_form(fn):
        def g(y):
            saved, N._series = N._series, form
            try:
                return fn(y)
            finally:
                N._series = saved

        return g

    value, slope = with_form(N._value), with_form(N._slope)

    @jax.custom_jvp
    def f(y):
        return value(y)

    @f.defjvp
    def _(primals, tangents):
        (y,), (dy,) = primals, tangents
        return value(y), slope(y) * dy

    return lambda y: f(_common.canon_float(y))


def _vg(f, jax, jnp):
    return jax.value_and_grad(lambda y: jnp.sum(f(y)))


def _ops(tag, funs, jax, jnp):
    import collections
    import re

    names = (
        "log",
        "log-plus-one",
        "exponential",
        "exponential-minus-one",
        "sqrt",
        "rsqrt",
        "divide",
    )
    y = -jnp.linspace(1e-3, 100.0, 1024)
    print(f"{tag} instructions in the compiled program")
    for name, f in funs.items():
        if name == "other gather":
            continue
        for mode, g in (("value", f), ("value_and_grad", _vg(f, jax, jnp))):
            txt = jax.jit(g).lower(y).compile().as_text()
            c = collections.Counter(
                m.group(1)
                for m in (re.search(r"= \S+ (\S+?)\(", ln) for ln in txt.splitlines())
                if m and m.group(1) in names
            )
            print(f"  {name:16s} {mode:15s} {dict(sorted(c.items()))}")


def _accuracy(tag, funs, dt, jax, jnp, np):
    import mpmath as mp

    mp.mp.dps = 50
    f64 = dt == np.float64
    y = _grid(np, dt)
    ref = np.array([[float(v) for v in _ref(mp, float(t))] for t in y])
    yj = jnp.asarray(y)
    got = {}
    for name in ("chebax", "jax ndtri(exp)"):
        f = funs[name]
        got[name] = [
            np.asarray(jax.jit(f)(yj), float),
            np.asarray(jax.jit(jax.vmap(jax.grad(f)))(yj), float),
        ]
    if f64:
        import scipy.special as sc

        got["scipy"] = [sc.ndtri_exp(y), None]
    print(f"{tag} max relative error against mpmath, n nonfinite where finite")
    print(f"  {'y':18s} {'n':>5s} " + " ".join(f"{k:>28s}" for k in got))
    near = np.abs(y + np.log(2)) < 0.1
    for label, lo, hi in _bands(np, f64) + (("-log 2 +- 0.1", None, None),):
        m = near if lo is None else (y >= lo) & (y < hi)
        cells = []
        for name, (z, dz) in got.items():
            parts = []
            for g, w in ((z, ref[:, 0]), (dz, ref[:, 1])):
                if g is None:
                    parts.append(f"{'':>13s}")
                    continue
                bad = ~np.isfinite(g[m]) & np.isfinite(w[m])
                with np.errstate(all="ignore"):
                    e = np.abs(g[m] - w[m]) / np.abs(w[m])
                e = e[np.isfinite(g[m])]
                worst = e.max() if e.size else np.nan
                parts.append(f"{worst:8.1e} {bad.sum():4d}")
            cells.append(" ".join(parts))
        print(f"  {label:18s} {m.sum():5d} " + " ".join(f"{c:>28s}" for c in cells))
    print("  (each cell: z error, nonfinite z; dz/dy error, nonfinite dz/dy)")


def _speed(tag, device, funs, dt, jax, jnp, np):
    n = 2**20 if device == "cpu" else 2**22
    reps = int(os.environ.get("E45_REPS", 5))
    rng = np.random.default_rng(45)
    dists = {
        "log U, U in (0, 1)": np.log(rng.uniform(np.finfo(dt).tiny, 1, n)),
        "-|y| log-uniform 1e-6..1e6": -np.exp(
            rng.uniform(np.log(1e-6), np.log(1e6), n)
        ),
    }
    print(f"{tag} ns per element, N = {n}")
    for dname, ys in dists.items():
        y = jax.device_put(jnp.asarray(ys, dt))
        for mode in ("value", "value_and_grad"):
            comps, compile_s = {}, {}
            for name, f in funs.items():
                # a new function each time, so jax's cache does not serve the compile
                g = (lambda y, f=f: f(y)) if mode == "value" else _vg(f, jax, jnp)
                t0 = time.perf_counter()
                comps[name] = jax.jit(g).lower(y).compile()
                compile_s[name] = time.perf_counter() - t0
                jax.block_until_ready(comps[name](y))
            times = {k: [] for k in comps}
            for _ in range(reps):
                for k, c in comps.items():
                    best = np.inf
                    for _ in range(7):
                        t0 = time.perf_counter()
                        jax.block_until_ready(c(y))
                        best = min(best, time.perf_counter() - t0)
                    times[k].append(best)
            jx = np.median(times["jax ndtri(exp)"])
            for k, c in comps.items():
                t = np.median(times[k])
                temp = c.memory_analysis().temp_size_in_bytes / 2**20
                print(
                    f"  {dname:27s} {mode:15s} {k:16s} {t / n * 1e9:7.3f} ns "
                    f"{jx / t:5.2f}x jax  temp {temp:5.0f} MiB  "
                    f"compile {compile_s[k]:4.2f} s"
                )


def main():
    devices = os.environ.get("E45_DEVICES", "cpu,gpu").split(",")
    core = os.environ.get("E45_CORE", "4")
    for device in devices:
        for dtype in ("f64", "f32"):
            cmd = [sys.executable, __file__, "run", device, dtype]
            if device == "cpu":
                cmd = ["taskset", "-c", core, *cmd]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode:
                sys.exit(res.stderr[-3000:])
            print(res.stdout, end="", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run(*sys.argv[2:4])
    else:
        main()
