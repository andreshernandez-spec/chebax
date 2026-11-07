#!/usr/bin/env python3
"""What two of 0.5.0's gradient fixes cost in speed: each fix's merge commit
against its first parent, on the experiments/33 cells that moved between its
run on f1b74de and its run on v0.5.0.

WHAT THIS MEASURES
------------------
#63 (betainc's x-derivative is the Beta density from its logs) and #69
(gammainc_array's x-derivative is the density from its logs below the
smallest normal). For both, merge commit and parent:

    betainc        a = 2.5, b = 7 and a = 0.5, b = 50, x ~ Beta(a, b);
                   value_and_grad in x, and in (a, b, x) ("all")
    gammainc_array one a per element as in experiments/33 (a = mu^2 / 1.5^2,
                   mu log-normal, x ~ Gamma(a)); value_and_grad in (a, x)

ns per element (median, experiments/33's timer) of jit(sum f) and of
jit(value_and_grad(sum f)), one pinned CPU core at N = 2^16 and the GPU at
N = 2^20, float64 and float32. Every commit is timed in its own process
from a git archive of its src/, in two rounds, the second in reverse order;
the table shows the mean of the two rounds. Every cell is timed at all four
commits, so each fix also shows the other's cells as a control.

FINDINGS (2026-11-07, RTX 3080 Laptop, jax 0.11.0)
--------------------------------------------------
results/35_fix_costs_050.txt. Values do not move at either fix (ratios
0.95-1.01), and each fix leaves the other's cells within 6%.
- #63, betainc's value_and_grad in x at a = 2.5, b = 7: 1.70x on one CPU
  core in float64 (27.6 -> 46.7 ns), 1.22x in float32, 1.28x on the GPU in
  float64, and 0.87x on the GPU in float32. In all of (a, b, x): 1.44x,
  1.14x, 1.18x and 0.94x. At a = 0.5, b = 50 it is faster except on the
  CPU in float64 (1.04x, all 1.02x): 0.79x on the GPU in float64 and on
  the CPU in float32, and on the GPU in float32 the gradient in all of
  (a, b, x) takes half the time (1.3 -> 0.63 ns).
- #69, gammainc_array's value_and_grad: float32 1.86x on the CPU (191 ->
  350 ns) and 1.67x on the GPU (1.8 -> 3.0 ns), float64 1.08x and 1.32x.

WHAT IT DOES NOT ESTABLISH
--------------------------
Other inputs: betainc's cost depends on which of its paths the points take,
and only two (a, b) are timed. Whether the costs matter in a model: these
are the function alone. Which operations the cost is in.

Run (~55 min):

  JAX_PLATFORMS=cuda,cpu taskset -c 3 python experiments/35_fix_costs_050.py \\
      > results/35_fix_costs_050.txt
"""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np

PAIRS = (("#63", "f1b74de", "2adbbed"), ("#69", "5437cd6", "7e366e2"))
SIZES = {"cpu": 2**16, "gpu": 2**20}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timed(call, n, reps=5, budget=1.0):
    """Median ns per element, as in experiments/33."""
    import jax

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


def time_commit(commit, src, dts):
    """One commit, one dtype: a JSON line per (device, cell)."""
    import jax

    jax.config.update("jax_enable_x64", dts == "f64")
    import jax.numpy as jnp

    import chebax

    assert chebax.__file__.startswith(src), (chebax.__file__, src)
    dt = np.float64 if dts == "f64" else np.float32
    for dname, n in SIZES.items():
        dev = jax.devices(dname)[0]
        rng = np.random.default_rng(35)

        def put(v):
            return jax.device_put(jnp.asarray(v, dt), dev)

        cells = []
        for a, b in ((2.5, 7.0), (0.5, 50.0)):
            args = [put(a), put(b), put(rng.beta(a, b, n))]
            name = f"betainc a={a:g} b={b:g}"
            cells.append((name, chebax.betainc, args, (2,)))
            cells.append((name + ", all", chebax.betainc, args, (0, 1, 2)))
        mu = np.clip(np.exp(rng.normal(1.0, 0.5, n)), 0.6, 20.0)
        a = mu**2 / 1.5**2
        args = [put(a), put(rng.gamma(a))]
        cells.append(("gammainc_array", chebax.gammainc_array, args, (0, 1)))
        for name, f, args, argnums in cells:

            def total(*z, f=f):
                return jnp.sum(f(*z))

            cv = jax.jit(total).lower(*args).compile()
            cg = jax.jit(jax.value_and_grad(total, argnums)).lower(*args).compile()
            rec = {"commit": commit, "dtype": dts, "device": dname, "cell": name}
            if not name.endswith(", all"):
                rec["t_v"] = timed(lambda: cv(*args), n)
            rec["t_g"] = timed(lambda: cg(*args), n)
            if dname == "gpu":
                rec["gpu"] = dev.device_kind
            print(json.dumps(rec), flush=True)


def report(rows, meta):
    print("chebax gradient fixes in 0.5.0: merge commit against its first parent")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(
        "ns per element (median, mean of two rounds): value / value_and_grad;"
        " ratio = after / before"
    )
    mean = {}
    for r in rows:
        for q in ("t_v", "t_g"):
            if q in r:
                key = (r["dtype"], r["device"], r["cell"], r["commit"], q)
                mean.setdefault(key, []).append(r[q])
    mean = {k: float(np.mean(v)) for k, v in mean.items()}
    cells = list(dict.fromkeys(r["cell"] for r in rows))

    def pair(dts, dname, cell, c):
        g = mean[dts, dname, cell, c, "t_g"]
        v = mean.get((dts, dname, cell, c, "t_v"))
        return g if v is None else (v, g)

    def fmt(x):
        return f"- / {x:.3g}" if np.isscalar(x) else f"{x[0]:.3g} / {x[1]:.3g}"

    def ratio(b, a):
        if np.isscalar(a):
            return f"- / {a / b:.2f}"
        return f"{a[0] / b[0]:.2f} / {a[1] / b[1]:.2f}"

    head = f"{'cell':<24}"
    for pr, before, after in PAIRS:
        head += f"{before:>16}{after + ' (' + pr + ')':>22}{'ratio':>14}"
    for dts in ("f64", "f32"):
        for dname, n in SIZES.items():
            print(f"\n--- {dts} {dname} (N = {n})")
            print(head)
            for cell in cells:
                line = f"{cell:<24}"
                for _, before, after in PAIRS:
                    b = pair(dts, dname, cell, before)
                    a = pair(dts, dname, cell, after)
                    line += f"{fmt(b):>16}{fmt(a):>22}{ratio(b, a):>14}"
                print(line)


def main():
    if sys.argv[1:2] == ["time"]:
        return time_commit(*sys.argv[2:5])
    import jax

    commits = [c for _, before, after in PAIRS for c in (before, after)]
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "jax": jax.__version__,
        "cpu affinity": sorted(os.sched_getaffinity(0)),
        "load start": os.getloadavg()[0],
    }
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in commits:
            tar = subprocess.run(
                ["git", "-C", ROOT, "archive", c, "src"],
                capture_output=True,
                check=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(tar)) as t:
                t.extractall(os.path.join(tmp, c), filter="data")
        for order in (commits, commits[::-1]):
            for c in order:
                src = os.path.join(tmp, c, "src")
                for dts in ("f64", "f32"):
                    out = subprocess.run(
                        [sys.executable, __file__, "time", c, src, dts],
                        env={**os.environ, "PYTHONPATH": src},
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    rows += [
                        json.loads(s) for s in out.stdout.splitlines() if s[:1] == "{"
                    ]
                    print(f"{c} {dts} done {time.strftime('%H:%M')}", file=sys.stderr)
    meta["load end"] = os.getloadavg()[0]
    meta["gpu"] = next(r["gpu"] for r in rows if "gpu" in r)
    report(rows, meta)


if __name__ == "__main__":
    sys.exit(main())
