#!/usr/bin/env python3
"""Bessel I speed, accuracy and the different standard-JAX I/J baselines.

Compare main f7efeb9, this working tree, and standard JAX. I cases use x~U(0,50),
J1 uses x~U(8,20), where JAX's 50-step recurrence is accurate in both dtypes.
CPU N=2**16, GPU N=2**20; caller pinned to one core. Time jit(sum f), grad(sum f),
and value_and_grad(sum f), excluding compilation, mean of two rounds' medians
with reversed source order. The standard counterparts for I are only orders 0,1.
Real order 2.5 also checks that the improvement extends beyond those kernels.

Accuracy is against mpmath at 60 dps at dtype-rounded inputs: the x=8 seam,
both sides, the origin, logarithmic bulk and tails, and the overflow window.
Values and x derivatives are relative where normal and finite in the compute
dtype, excluding a 1% window around each interior derivative zero located
at 60 dps. Report absolute errors there and below the normal range too.
Count every unexpected nonfinite explicitly.
Scaled derivatives exclude x=0: JAX extends with exp(-abs(x)), while Chebax
uses the right derivative on x>=0. Their endpoint conventions differ.
For x>1e5, form scaled I and its first two derivatives with 24 asymptotic terms,
so the reference avoids cancellation at very large x. The sweep is not a bound.

The degree check fits R(v,z) = (4(v+1)(0F1-1)/z - 1)/z on z in [0,64]
at 60 dps and reports omitted coefficient mass relative to min |R|, at six
orders. It does not bound floating-point reconstruction or AD. Values use
0F1 = 1 + z/(4(v+1)) (1+z R), avoiding small-x cancellation. The original
log tables retain stable order derivatives near nu=0 and the scaled tail.
Order derivatives have independent references in tests/test_besseli.py.

Run accuracy independently of timing (GPU access required):
  JAX_PLATFORMS=cuda,cpu python experiments/38_besseli_speed.py \
      --mode accuracy --devices cpu gpu > results/38_besseli_accuracy_012.txt
Then, with an idle CPU and the caller pinned to one core:
  JAX_PLATFORMS=cuda,cpu taskset -c 3 python experiments/38_besseli_speed.py \
      --mode speed --devices cpu gpu > results/38_besseli_speed_012.txt
These captures use JAX 0.12.0.1 on PYTHONPATH. The default mode runs both.
Hardware, source hashes, versions and timing details are captured below.
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
BASELINE = "f7efeb9"
SIZES = {"cpu": 2**16, "gpu": 2**20}
CASES = [
    ("i0", 0.0, False),
    ("i1", 1.0, False),
    ("i0e", 0.0, True),
    ("i1e", 1.0, True),
    ("i2.5", 2.5, False),
    ("i2.5e", 2.5, True),
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
        dy = mp.mpf(".5") if v == 1 else mp.mpf(0)
        d2 = mp.mpf(".5") if v == 0 else mp.mpf(0)
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
        seams,
        seams * (1 - 1e-7),
        seams * (1 + 1e-7),
        np.linspace(88, 93, 11) if dt == np.float32 else np.linspace(700, 715, 16),
    ]
    out = {}
    for name, nu, scaled in CASES:
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
                if xs[i] > 0 and xs[i + 1] < 100 and r[i] and r[i + 1]:
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
    out["j1_x"] = np.asarray(np.linspace(8, 20, 41), dt)
    out["j1_ref"] = np.array(
        [float(mp.besselj(1, mp.mpf(float(x)))) for x in out["j1_x"]]
    )
    np.savez(path, **out)


def degree_check():
    import mpmath as mp
    from chebax._src.recipes._gen_common import dct, nodes

    mp.mp.dps = 60
    worst = {k: mp.mpf(0) for k in (8, 10, 12, 14, 16)}
    for v in (0, 0.5, 1, 2.5, 5, 10):
        v = mp.mpf(v)
        values = []
        for t in nodes(mp, 40):
            z = 32 * (t + 1)
            values.append(((mp.hyp0f1(v + 1, z / 4) - 1) * 4 * (v + 1) / z - 1) / z)
        c = dct(mp, values)
        for k in worst:
            worst[k] = max(worst[k], sum(abs(a) for a in c[k:]) / min(values))
    return {str(k): float(a) for k, a in worst.items()}


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
        for name, nu, scaled in CASES:
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
                if measure:
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
        if label == "candidate":
            x = jax.device_put(
                jnp.asarray(
                    (
                        np.random.default_rng(38).uniform(8, 20, n)
                        if measure
                        else refs["j1_x"]
                    ),
                    dt,
                ),
                dev,
            )
            for source, f in [
                ("candidate", chebax.besselj_frozen(1.0)),
                ("jax", lambda x: jsp.bessel_jn(x, v=1)[1]),
            ]:
                rec = dict(
                    source=source,
                    dtype=dtype,
                    device=device,
                    case="j1 x~U(8,20)",
                    hardware=dev.device_kind,
                )
                if measure:
                    compiled = jax.jit(lambda x: f(x).sum()).lower(x).compile()
                    rec["value"] = timed(lambda: compiled(x), n, budget)
                rec["nonfinite"] = int(
                    np.count_nonzero(~np.isfinite(np.asarray(jax.jit(f)(x))))
                )
                assert rec["nonfinite"] == 0, rec
                if check_accuracy:
                    y = np.asarray(
                        jax.jit(f)(jax.device_put(jnp.asarray(refs["j1_x"]), dev))
                    )
                    rec["value sup error"] = float(
                        np.max(abs(y - refs["j1_ref"])) / np.max(abs(refs["j1_ref"]))
                    )
                print(json.dumps(rec), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", choices=SIZES, default=["cpu"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--budget", type=float, default=0.25)
    p.add_argument("--child", nargs=4)
    p.add_argument("--check", action="store_true")
    p.add_argument("--mode", choices=("all", "speed", "accuracy"), default="all")
    opts = p.parse_args()
    if opts.child:
        return child(*opts.child, opts.devices, opts.budget, opts.check, opts.mode)
    import jax

    rounds = 1 if opts.mode == "accuracy" else opts.rounds
    meta = dict(
        mode=opts.mode,
        date=time.strftime("%Y-%m-%d %H:%M"),
        jax=jax.__version__,
        baseline=BASELINE,
        affinity=sorted(os.sched_getaffinity(0)),
        rounds=rounds,
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
                        json.loads(s)
                        for s in run.stdout.splitlines()
                        if s.startswith("{")
                    )
                    print(
                        f'round {rnd+1}: {label} {dt} done {time.strftime("%H:%M")}',
                        file=sys.stderr,
                    )
    meta["load_end"] = os.getloadavg()[0]
    print(
        "Bessel I/J accuracy"
        if opts.mode == "accuracy"
        else "Bessel I/J speed: ns per element, mean of round medians"
    )
    print(json.dumps(meta, indent=2))
    if bounds is not None:
        print("Inner R: omitted coefficient mass / min R at six orders:")
        print(json.dumps(bounds, indent=2))
    if opts.mode != "accuracy":
        for device in opts.devices:
            for dt in ("f64", "f32"):
                print(f"\n{device} {dt}, N={SIZES[device]}")
                print(
                    f'{"function/source":<28}{"value":>12}{"grad":>12}{"value_grad":>14}'
                )
                for name in [c[0] for c in CASES] + ["j1 x~U(8,20)"]:
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
    if opts.mode != "speed":
        print("\nJ1 value accuracy on [8,20], sup-normalized:")
    for r in rows:
        if "value sup error" in r:
            print(
                r["device"],
                r["dtype"],
                r["source"],
                r["value sup error"],
                "nonfinite",
                r["nonfinite"],
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


if __name__ == "__main__":
    main()
