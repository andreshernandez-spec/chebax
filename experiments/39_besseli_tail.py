#!/usr/bin/env python3
"""Direct Bessel I tails: speed and accuracy against main and standard JAX.

Compare main 10a82f8 with the current tree. Time I0/I1, scaled and unscaled,
and real order 2.5, x~U(0,50), on one CPU core (N=2**16) and GPU (N=2**20).
Report jit(sum f), grad(sum f), and value_and_grad(sum f), excluding compilation,
as the mean of four rounds' medians with alternating source order. Each source and
dtype runs in its own subprocess. JAX only provides orders 0 and 1.

Accuracy uses independent mpmath references at 60 dps and dtype-rounded inputs.
Include the origin, both sides of x=8, cancellation-prone points at large orders,
logarithmic tails, the overflow window, and very large scaled arguments. Cover
orders 0, 0.5, 1, 2.5, 5, 9.97 and 10. Compare values and the first two argument
derivatives. Use relative error for normal finite results, absolute error near
derivative zeros (a 1% window) and underflow, and count unexpected nonfinites.
Scaled derivatives exclude x=0, whose left/right conventions differ in JAX.
Above x=1e5 use 24 asymptotic terms to avoid cancellation in the reference.
This sweep measures errors; it is not a proof of the whole domain's accuracy.
Order and mixed derivatives have independent regression tests.

The degree check fits the anchored tail B=(sqrt(2 pi x) exp(-x) I_nu(x)-A8)/
(1-8/x) and the unscaled path's log tail at 80 nodes and 60 dps. Report omitted
coefficient mass at seven orders; this does not bound reconstruction, rounding,
or differentiation errors. Unscaled I retains its overflow-safe logarithmic
reconstruction. The full log tables still serve order derivatives.

Run accuracy separately from speed, with an idle CPU and a pinned caller:
  JAX_PLATFORMS=cuda,cpu python experiments/39_besseli_tail.py \
      --mode accuracy --devices cpu gpu > results/39_besseli_tail_accuracy_012.txt
  JAX_PLATFORMS=cuda,cpu taskset -c 3 python experiments/39_besseli_tail.py \
      --mode speed --devices cpu gpu --budget 0.1 > results/39_besseli_tail_speed_012.txt
Use the tested JAX on PYTHONPATH; versions and source hashes are captured below.
"""

import argparse
import hashlib
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
BASELINE = "10a82f8"
SIZES = {"cpu": 2**16, "gpu": 2**20}
CASES = [
    ("i0", 0.0, False),
    ("i1", 1.0, False),
    ("i0e", 0.0, True),
    ("i1e", 1.0, True),
    ("i2.5", 2.5, False),
    ("i2.5e", 2.5, True),
]
ACCURACY_CASES = CASES + [
    (f"i{nu}{'e' if scaled else ''}", nu, scaled)
    for nu in (0.5, 5.0, 9.97, 10.0)
    for scaled in (False, True)
]


def reference(nu, x, scaled, mp):
    v = mp.mpf(nu)
    x = x if isinstance(x, mp.mpf) else mp.mpf(float(x))
    if x > 1e5:
        c = mp.mpf(1)
        a = b = d = mp.mpf(0)
        for k in range(24):
            a += c
            b -= (k + mp.mpf(".5")) * c / x
            d += (k + mp.mpf(".5")) * (k + mp.mpf("1.5")) * c / (x * x)
            c *= ((2 * k + 1) ** 2 - 4 * v * v) / (8 * (k + 1) * x)
        scale = 1 / mp.sqrt(2 * mp.pi * x)
        a, b, d = a * scale, b * scale, d * scale
        if scaled:
            return a, b, d
        return mp.inf, mp.inf, mp.inf
    y = mp.besseli(v, x)
    dy = mp.besseli(v + 1, x) + (v * y / x if x else 0)
    if not x:
        dy = mp.inf if 0 < v < 1 else mp.mpf(".5") if v == 1 else mp.mpf(0)
        d2 = -mp.inf if 0 < v < 1 else mp.mpf(".5") if v == 0 else mp.mpf(0)
    else:
        d2 = (1 + v * v / (x * x)) * y - dy / x
    if scaled:
        e = mp.exp(-x)
        return y * e, (dy - y) * e, (d2 - 2 * dy + y) * e
    return y, dy, d2


def make_refs(path, dt):
    import mpmath as mp

    mp.mp.dps = 60
    seams = np.r_[np.sqrt(np.arange(5) * 16), 8 / (np.arange(1, 5) / 4)]
    pos = np.r_[
        0,
        np.geomspace(1e-12, 1e4, 220),
        np.linspace(0, 10, 100),
        [8.08, 8.32, 9.04, 9.68, 10.32, 13.2],
        seams,
        seams * (1 - 1e-7),
        seams * (1 + 1e-7),
        np.linspace(88, 93, 11) if dt == np.float32 else np.linspace(700, 715, 16),
    ]
    out = {}
    for name, nu, scaled in ACCURACY_CASES:
        nu = float(dt(nu))
        xs = np.unique(
            np.asarray(
                (
                    np.r_[
                        pos,
                        (
                            [1e10, 1e18]
                            if dt == np.float32
                            else [1e10, 1e100, 1e200, 1e308]
                        ),
                    ]
                    if scaled
                    else pos
                ),
                dt,
            )
        )
        out[name + "_x"] = xs
        out[name + "_ref"] = np.array(
            [[float(t) for t in reference(nu, x, scaled, mp)] for x in xs]
        ).T
        for k in (1, 2):
            r = out[name + "_ref"][k]
            cross = np.flatnonzero(np.signbit(r[:-1]) != np.signbit(r[1:]))
            roots = []
            for i in cross:
                if xs[i] > 0 and xs[i + 1] < 1000 and r[i] and r[i + 1]:
                    root = mp.findroot(
                        lambda x: reference(nu, x, scaled, mp)[k],
                        (mp.mpf(float(xs[i])), mp.mpf(float(xs[i + 1]))),
                        solver="anderson",
                    )
                    roots.append(float(root))
            out[name + f"_roots{k}"] = np.array(roots)
        extra = [
            root * (1 + delta)
            for k in (1, 2)
            for root in out[name + f"_roots{k}"]
            for delta in (-1e-7, 0.0, 1e-7)
        ]
        if extra:
            xs = np.unique(np.asarray(np.r_[xs, extra], dt))
            out[name + "_x"] = xs
            out[name + "_ref"] = np.array(
                [[float(t) for t in reference(nu, x, scaled, mp)] for x in xs]
            ).T
    np.savez(path, **out)


def degree_check():
    import mpmath as mp
    from chebax._src.recipes._gen_common import dct, nodes

    mp.mp.dps = 60
    out = {}
    for v in (0, 0.5, 1, 2.5, 5, 9.97, 10):
        v = mp.mpf(v)
        edge = mp.sqrt(16 * mp.pi) * mp.exp(-8) * mp.besseli(v, 8)
        values, logs = [], []
        for u in nodes(mp, 80):
            t = (u + 1) / 2
            x = 8 / t
            a = mp.sqrt(2 * mp.pi * x) * mp.exp(-x) * mp.besseli(v, x)
            values.append((a - edge) / (1 - t))
            logs.append(mp.log(a))
        out[str(float(v))] = {
            label: {
                str(n): float(sum(abs(a) for a in c[n:]))
                for n in (8, 10, 12, 24, 28, 32)
            }
            for label, c in (("B", dct(mp, values)), ("log A", dct(mp, logs)))
        }
    return out


def timed(call, n, budget):
    import jax

    jax.block_until_ready(call())
    start = time.perf_counter()
    while time.perf_counter() - start < 0.1:
        jax.block_until_ready(call())
    start = time.perf_counter()
    jax.block_until_ready(call())
    one = time.perf_counter() - start
    count = max(1, int(budget / 5 / max(one, 1e-6)))
    ts = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(count):
            out = call()
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - start) / count)
    return float(np.median(ts)) / n * 1e9


def accuracy(f, name, refs, dt, dev):
    import jax
    import jax.numpy as jnp

    x, ref = refs[name + "_x"], refs[name + "_ref"]
    d1 = jax.grad(lambda x: f(x).sum())
    d2 = jax.grad(lambda x: d1(x).sum())
    arg = jax.device_put(jnp.asarray(x), dev)
    out = {}
    for k, fun in enumerate((f, d1, d2)):
        got = np.asarray(jax.jit(fun)(arg), float)
        finite = np.isfinite(ref[k]) & (abs(ref[k]) <= np.finfo(dt).max)
        if name.endswith("e") and k:
            finite &= x > 0
        bad = finite & ~np.isfinite(got)
        valid = finite & (abs(ref[k]) >= np.finfo(dt).tiny) & ~bad
        near = np.zeros(x.size, bool)
        if k:
            roots = refs[name + f"_roots{k}"]
            for root in roots:
                near |= abs(x - root) < 0.01 * abs(root)
            if roots.size:
                out[f"d{k} zeros"] = [float(t) for t in roots]
        valid &= ~near
        out[f"d{k} nonfinite"] = int(bad.sum())
        if np.any(bad):
            out[f"d{k} nonfinite x"] = [float(t) for t in x[bad]]
        rel = np.full(x.size, -1.0, float)
        rel[valid] = abs(got[valid] / ref[k, valid] - 1)
        index = int(np.argmax(rel))
        out[f"d{k} rel"] = float(rel[index])
        out[f"d{k} worst x"] = float(x[index])
        tail = valid & (x > 8)
        index = int(np.argmax(np.where(tail, rel, -1)))
        out[f"d{k} tail rel"] = float(rel[index])
        out[f"d{k} tail worst x"] = float(x[index])
        bulk = finite & (x <= 8) & np.isfinite(got)
        out[f"d{k} bulk abs"] = float(np.max(abs(got[bulk] - ref[k, bulk])))
        for label, mask in (
            ("near zero abs", near),
            ("underflow abs", abs(ref[k]) < np.finfo(dt).tiny),
        ):
            keep = finite & mask & np.isfinite(got)
            out[f"d{k} {label}"] = (
                float(np.max(abs(got[keep] - ref[k, keep]))) if np.any(keep) else 0.0
            )
    return out


def child(label, src, dtype, refpath, devices, budget, check, mode):
    import jax

    jax.config.update("jax_enable_x64", dtype == "f64")
    import jax.numpy as jnp
    import jax.scipy.special as jsp
    import chebax

    assert chebax.__file__.startswith(src)
    dt = np.float64 if dtype == "f64" else np.float32
    measure = mode != "accuracy"
    check_accuracy = check and mode != "speed"
    refs = np.load(refpath) if check_accuracy else None
    for device in devices:
        dev, n = jax.devices(device)[0], SIZES[device]
        cases = ACCURACY_CASES if check_accuracy else CASES
        for name, nu, scaled in cases:
            host = np.random.default_rng(38).uniform(0, 50, n)
            x = jax.device_put(jnp.asarray(host, dt), dev)
            funcs = [
                (
                    label,
                    lambda x, nu=nu, scaled=scaled: chebax.besseli(
                        nu, x, scaled=scaled
                    ),
                )
            ]
            if label == "candidate" and nu in (0, 1):
                funcs.append(("jax", getattr(jsp, name)))
            for source, f in funcs:
                rec = dict(
                    source=source,
                    dtype=dtype,
                    device=device,
                    case=name,
                    hardware=dev.device_kind,
                    first_round=check,
                )
                total = lambda x: f(x).sum()
                if measure and (name, nu, scaled) in CASES:
                    for kind, fun in [
                        ("value", total),
                        ("grad", jax.grad(total)),
                        ("value_grad", jax.value_and_grad(total)),
                    ]:
                        start = time.perf_counter()
                        compiled = jax.jit(fun).lower(x).compile()
                        rec[kind + " compile"] = time.perf_counter() - start
                        rec[kind] = timed(lambda: compiled(x), n, budget)
                if check_accuracy:
                    rec["accuracy"] = accuracy(f, name, refs, dt, dev)
                print(json.dumps(rec), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", choices=SIZES, default=["cpu"])
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--budget", type=float, default=0.25)
    p.add_argument("--child", nargs=4)
    p.add_argument("--check", action="store_true")
    p.add_argument("--mode", choices=("all", "speed", "accuracy"), default="all")
    opts = p.parse_args()
    if opts.child:
        return child(*opts.child, opts.devices, opts.budget, opts.check, opts.mode)
    import jax
    import jaxlib

    rounds = 1 if opts.mode == "accuracy" else opts.rounds
    meta = dict(
        mode=opts.mode,
        date=time.strftime("%Y-%m-%d %H:%M %Z"),
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        baseline=BASELINE,
        affinity=sorted(os.sched_getaffinity(0)),
        rounds=rounds,
        budget=opts.budget,
        load_start=os.getloadavg()[0],
    )
    meta["candidate sha256"] = {
        f: hashlib.sha256(
            (ROOT / "src/chebax/_src/recipes" / f).read_bytes()
        ).hexdigest()
        for f in ("besseli.py", "besseli_gen.py", "besseli_table.py")
    }
    bounds = degree_check() if opts.mode != "speed" else None
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if opts.mode != "speed":
            for name, dt in [("f64", np.float64), ("f32", np.float32)]:
                make_refs(tmp / (name + ".npz"), dt)
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", BASELINE, "src"],
            capture_output=True,
            check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp / "baseline", filter="data")
        sources = [
            ("baseline", str(tmp / "baseline/src")),
            ("candidate", str(ROOT / "src")),
        ]
        for rnd in range(rounds):
            for label, src in (sources if rnd % 2 == 0 else sources[::-1]):
                for dt in ("f64", "f32"):
                    cmd = [
                        sys.executable,
                        __file__,
                        "--child",
                        label,
                        src,
                        dt,
                        str(tmp / (dt + ".npz")),
                        "--devices",
                        *opts.devices,
                        "--budget",
                        str(opts.budget),
                        "--mode",
                        opts.mode,
                    ]
                    if rnd == 0:
                        cmd.append("--check")
                    run = subprocess.run(
                        cmd,
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
                    rows.extend(
                        dict(json.loads(s), round=rnd + 1)
                        for s in run.stdout.splitlines()
                        if s.startswith("{")
                    )
                    print(
                        f'round {rnd+1}: {label} {dt} done {time.strftime("%H:%M")}',
                        file=sys.stderr,
                    )
    meta["load_end"] = os.getloadavg()[0]
    meta["hardware"] = {
        d: next(r["hardware"] for r in rows if r["device"] == d) for d in opts.devices
    }
    meta["cpu"] = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    print(
        "Bessel I tail accuracy"
        if opts.mode == "accuracy"
        else "Bessel I tail speed: ns per element, mean of round medians"
    )
    print(json.dumps(meta, indent=2))
    if bounds is not None:
        print("Tail B and log A: omitted coefficient mass at seven orders:")
        print(json.dumps(bounds, indent=2))
    if opts.mode != "accuracy":
        for device in opts.devices:
            for dt in ("f64", "f32"):
                print(f"\n{device} {dt}, N={SIZES[device]}")
                print(
                    f'{"function/source":<28}{"value":>12}{"grad":>12}{"value_grad":>14}'
                )
                for name in [c[0] for c in CASES]:
                    for source in ("baseline", "candidate", "jax"):
                        rr = [
                            r
                            for r in rows
                            if r["device"] == device
                            and r["dtype"] == dt
                            and r["case"] == name
                            and r["source"] == source
                        ]
                        if not rr:
                            continue
                        vals = [
                            float(np.mean([r[k] for r in rr])) if k in rr[0] else None
                            for k in ("value", "grad", "value_grad")
                        ]
                        print(
                            f'{name+" "+source:<28}'
                            + "".join(
                                f"{a:>12.4g}" if a is not None else f'{"-":>12}'
                                for a in vals
                            )
                        )
    if opts.mode != "speed":
        print("\nAccuracy at 60 dps:")
    for r in rows:
        if "accuracy" in r:
            print(
                r["device"],
                r["dtype"],
                r["case"],
                r["source"],
                json.dumps(r["accuracy"], sort_keys=True),
            )
    if opts.mode != "accuracy":
        print("\nCompile seconds (first round, value / grad / value_grad):")
    for r in rows:
        if r.get("first_round") and "value compile" in r:
            print(
                r["device"],
                r["dtype"],
                r["case"],
                r["source"],
                *[round(r[k + " compile"], 3) for k in ("value", "grad", "value_grad")],
            )
    if opts.mode != "accuracy":
        print(
            "\nRound medians: round device dtype function source value grad value_grad"
        )
        for r in rows:
            if "value" not in r:
                continue
            print(
                r["round"],
                r["device"],
                r["dtype"],
                r["case"],
                r["source"],
                *[f"{r[k]:.6g}" for k in ("value", "grad", "value_grad")],
            )


if __name__ == "__main__":
    main()
