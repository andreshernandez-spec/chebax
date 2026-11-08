#!/usr/bin/env python3
"""The unscaled Bessel I tail's exponential: main against two fixes for the
float32 accuracy loss between x = 80 and 88.7.

e^x overflows before I_v does (x = 88.7 in float32, 709.8 in float64). On
main (44523c9) the unscaled tail, x > 8, is A / sqrt(2 pi x) times e^x up
to x = 80 (700 in float64) and one exponential of x + s, s = ln(A / sqrt(2
pi x)), past it; rounding x + s costs up to half an ulp of x, relative.
Sources compared:

    main        44523c9
    two-branch  0e1c2a0: past the switch exp(hi) (1 + lo), hi + lo = x + s
                exactly (Fast2Sum); the split branch below it unchanged
    single      the current tree: exp(hi) (1 + lo) (TwoSum) on the whole
                unscaled tail, no split branch

WHAT THIS MEASURES
------------------
1. Accuracy against mpmath at 40 dps, inputs rounded to the dtype: I_v and
   its first two x-derivatives, orders 0, 1, 2.5, 5, 9.97 and 10, x from 8
   to where I_v overflows (a log grid of 400 points, the points 8.0001,
   8.01, 8.1, and 400 points across the switch and the overflow), the
   frozen and the traced-order paths, CPU and GPU. Worst relative error at
   or below the switch and above it, over paths and devices; points where
   the value, a derivative or the reference is outside the dtype's range
   are skipped; non-finite results where the reference is finite counted.
2. Speed under jit: ns per element (median of five) of the sum of values
   and of value_and_grad of the sum, I0, I1 and I_2.5 (frozen), x ~ U(0, 50)
   and x ~ U(0, 100), one pinned CPU core at N = 2^16 and the GPU at
   N = 2^20. Two rounds alternate the source order; the faster round is
   kept. Each source and dtype runs in its own process.

FINDINGS
--------
(filled in from results/42_besseli_unscaled_exp.txt)

WHAT IT DOES NOT ESTABLISH
--------------------------
Orders between the six above, x below 8 (the tail starts there; the inner
series is unchanged), order derivatives, the scaled tail (unchanged), or
other hardware.

Run, with jax 0.12.0.1 first on PYTHONPATH and an idle machine (~25 min):

  JAX_PLATFORMS=cuda,cpu taskset -c 3 \\
      python experiments/42_besseli_unscaled_exp.py > results/42_besseli_unscaled_exp.txt
"""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (("main", "44523c9"), ("two-branch", "0e1c2a0"), ("single", None))
ORDERS = (0.0, 1.0, 2.5, 5.0, 9.97, 10.0)
SWITCH = {"f32": 80.0, "f64": 700.0}
TOP = {"f32": 91.9, "f64": 713.9}
SIZES = {"cpu": 2**16, "gpu": 2**20}
SPEED = (("I0", 0.0), ("I1", 1.0), ("I_2.5", 2.5))
DRAWS = (50.0, 100.0)


def grid(d):
    dt = np.float32 if d == "f32" else np.float64
    x = np.concatenate(
        [
            np.logspace(np.log10(8.0), np.log10(TOP[d]), 400),
            [8.0001, 8.01, 8.1],
            np.linspace(SWITCH[d] - 2.0, SWITCH[d] + 2.0, 200),
            np.linspace(TOP[d] - 12.0, TOP[d], 200),
        ]
    )
    return np.unique(x.astype(dt)).astype(np.float64)


def make_refs(path):
    import mpmath as mp

    mp.mp.dps = 40
    out = {}
    for d in ("f32", "f64"):
        x = grid(d)
        out[d + "|x"] = x
        for v in ORDERS:
            rows = []
            for t in x:
                X, V = mp.mpf(float(t)), mp.mpf(v)
                y = mp.besseli(V, X)
                d1 = (mp.besseli(V - 1, X) + mp.besseli(V + 1, X)) / 2
                d2 = (1 + V * V / (X * X)) * y - d1 / X
                rows.append([float(y), float(d1), float(d2)])
            out[f"{d}|{v}"] = np.array(rows).T
    np.savez(path, **out)


def timed(call, n, reps=5, budget=1.0):
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
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = call()
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts)) / n * 1e9


def child(label, d, refpath, accuracy):
    import jax

    jax.config.update("jax_enable_x64", d == "f64")
    import jax.numpy as jnp

    import chebax

    dt = np.float32 if d == "f32" else np.float64
    devs = {"cpu": jax.devices("cpu")[0], "gpu": jax.devices("gpu")[0]}
    print(json.dumps({"label": label, "dtype": d, "chebax": chebax.__file__}))
    if accuracy:
        refs = np.load(refpath)
        x_np = refs[d + "|x"]
        big = np.finfo(dt).max
        for dname, dev in devs.items():
            x = jax.device_put(jnp.asarray(x_np, dt), dev)
            for v in ORDERS:
                ref = refs[f"{d}|{v}"]
                paths = (
                    ("frozen", chebax.besseli_frozen(v)),
                    (
                        "traced",
                        lambda t, v=v: chebax.besseli_scalar(jnp.asarray(v, dt), t),
                    ),
                )
                for path, f in paths:
                    g1 = jax.grad(lambda t, f=f: f(t).sum())
                    g2 = jax.grad(lambda t, g1=g1: g1(t).sum())
                    for k, fun in enumerate((f, g1, g2)):
                        got = np.asarray(jax.jit(fun)(x), np.float64)
                        ok = np.all(np.abs(ref) <= big, axis=0)
                        for side, m in (
                            ("low", x_np <= SWITCH[d]),
                            ("high", x_np > SWITCH[d]),
                        ):
                            sel = ok & m
                            good = sel & np.isfinite(got)
                            e = np.abs(got - ref[k]) / np.abs(ref[k])
                            e = np.where(good, e, 0.0)
                            i = int(np.argmax(e))
                            print(
                                json.dumps(
                                    {
                                        "kind": "acc",
                                        "label": label,
                                        "dtype": d,
                                        "device": dname,
                                        "order": v,
                                        "path": path,
                                        "q": k,
                                        "side": side,
                                        "err": float(e[i]),
                                        "x": float(x_np[i]),
                                        "nonfinite": int(
                                            (sel & ~np.isfinite(got)).sum()
                                        ),
                                    }
                                )
                            )
    for dname, dev in devs.items():
        n = SIZES[dname]
        for top in DRAWS:
            xs = np.random.default_rng(42).uniform(0.0, top, n)
            x = jax.device_put(jnp.asarray(xs, dt), dev)
            for name, v in SPEED:
                f = chebax.besseli_frozen(v)
                total = lambda t, f=f: jnp.sum(f(t))  # noqa: E731
                cv = jax.jit(total).lower(x).compile()
                cg = jax.jit(jax.value_and_grad(total)).lower(x).compile()
                print(
                    json.dumps(
                        {
                            "kind": "speed",
                            "label": label,
                            "dtype": d,
                            "device": dname,
                            "fn": name,
                            "draw": top,
                            "t_v": timed(lambda: cv(x), n),
                            "t_g": timed(lambda: cg(x), n),
                        }
                    ),
                    flush=True,
                )


def report(rows, meta):
    print("Unscaled Bessel I tail: main against two fixes")
    print(json.dumps(meta, indent=2))
    labels = [s[0] for s in SOURCES]
    q_names = ("value", "d/dx", "d2/dx2")
    for d in ("f32", "f64"):
        print(
            f"\n=== ACCURACY {d}: worst relative error over frozen/traced and "
            f"CPU/GPU, x <= {SWITCH[d]:g} | x > {SWITCH[d]:g}; (k) = k non-finite"
        )
        hdr = ["order", "quantity"] + [f"{lb} low" for lb in labels]
        hdr += [f"{lb} high" for lb in labels]
        lines = [hdr]
        for v in ORDERS:
            for k, qn in enumerate(q_names):
                row = [f"{v:g}", qn]
                for side in ("low", "high"):
                    for lb in labels:
                        sel = [
                            r
                            for r in rows
                            if r.get("kind") == "acc"
                            and r["label"] == lb
                            and r["dtype"] == d
                            and r["order"] == v
                            and r["q"] == k
                            and r["side"] == side
                        ]
                        e = max(r["err"] for r in sel)
                        nf = sum(r["nonfinite"] for r in sel)
                        row.append(f"{e:.1e}" + (f" ({nf})" if nf else ""))
                lines.append(row)
        widths = [max(len(ln[i]) for ln in lines) for i in range(len(hdr))]
        for ln in lines:
            print("  ".join(c.ljust(w) for c, w in zip(ln, widths)))
    print(
        "\n=== SPEED, ns per element: value / value_and_grad, faster of two "
        "rounds; ratio = time / main's (below 1 is faster)"
    )
    for d in ("f32", "f64"):
        for dname in ("cpu", "gpu"):
            print(f"\n--- {d} {dname} (N = {SIZES[dname]})")
            hdr = ["setting"] + labels + [f"{lb} vs main" for lb in labels[1:]]
            lines = [hdr]
            for top in DRAWS:
                for name, _ in SPEED:
                    best = {}
                    for lb in labels:
                        sel = [
                            r
                            for r in rows
                            if r.get("kind") == "speed"
                            and r["label"] == lb
                            and r["dtype"] == d
                            and r["device"] == dname
                            and r["fn"] == name
                            and r["draw"] == top
                        ]
                        best[lb] = (
                            min(r["t_v"] for r in sel),
                            min(r["t_g"] for r in sel),
                        )
                    row = [f"{name}, x ~ U(0, {top:g})"]
                    row += [f"{best[lb][0]:.3g} / {best[lb][1]:.3g}" for lb in labels]
                    m = best["main"]
                    row += [
                        f"{best[lb][0] / m[0]:.2f}x / {best[lb][1] / m[1]:.2f}x"
                        for lb in labels[1:]
                    ]
                    lines.append(row)
            widths = [max(len(ln[i]) for ln in lines) for i in range(len(hdr))]
            for ln in lines:
                print("  ".join(c.ljust(w) for c, w in zip(ln, widths)))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1")
    import jax

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%h"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "jax": jax.__version__,
        "tree": commit,
        "affinity": sorted(os.sched_getaffinity(0)),
        "load_start": os.getloadavg()[0],
    }
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_refs(tmp / "refs.npz")
        srcs = []
        for label, rev in SOURCES:
            if rev is None:
                srcs.append((label, str(ROOT / "src")))
                continue
            archive = subprocess.run(
                ["git", "-C", str(ROOT), "archive", rev, "src"],
                capture_output=True,
                check=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(tmp / label, filter="data")
            srcs.append((label, str(tmp / label / "src")))
        for rnd in range(2):
            for label, src in srcs if rnd == 0 else srcs[::-1]:
                for d in ("f64", "f32"):
                    run = subprocess.run(
                        [sys.executable, __file__, "--child", label, d]
                        + [str(tmp / "refs.npz"), "1" if rnd == 0 else "0"],
                        env={
                            **os.environ,
                            "PYTHONPATH": src
                            + os.pathsep
                            + os.environ.get("PYTHONPATH", ""),
                        },
                        capture_output=True,
                        text=True,
                    )
                    if run.returncode:
                        raise RuntimeError(run.stdout + run.stderr)
                    got = [
                        json.loads(s)
                        for s in run.stdout.splitlines()
                        if s.startswith("{")
                    ]
                    assert got[0]["chebax"].startswith(src), got[0]
                    rows += got[1:]
                    print(
                        f"round {rnd + 1}: {label} {d} {time.strftime('%H:%M')}",
                        file=sys.stderr,
                    )
    meta["load_end"] = os.getloadavg()[0]
    report(rows, meta)


if __name__ == "__main__":
    main()
