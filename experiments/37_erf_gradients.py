#!/usr/bin/env python3
"""Erf-family throughput and accuracy after short panels and analytic JVPs.

Compare main at 353ea37, the working src/, and this interpreter's standard JAX.
Time jit(sum f), grad(sum f), and value_and_grad(sum f), x ~ N(0,3) for erfcx
and N(0,5) for dawsn. CPU N=2**16, GPU N=2**20, median ns per element, averaged
over two rounds in reverse source order. Each source/dtype has its own process.
Compilation is excluded. The CPU caller is pinned to one core.

References: mpmath at 60 dps, dtype-rounded inputs, both signs, all panel seams,
the tail seam and far tails. Above |x|=40 use 20 asymptotic terms to avoid
cancellation in the reference derivatives, including at x=1e150. Report relative
errors where the reference is normal and finite in the compute dtype, excluding
windows of width .02 around Dawson's first-derivative zeros, and .05 around its
second-derivative zeros. Also report absolute errors on |x|<=6 for Dawson.
Unexpected nonfinite results are counted explicitly; the candidate must have none.
If standard JAX's vector second derivative raises, record the error and check
its vmapped scalar second derivative instead. The grid is not a worst-case
bound. Smaller batches and other
input distributions can have different speed ratios.

The degree check fits D(x)/x at 24 nodes per panel, at 60 dps, and prints the
largest omitted coefficient mass relative to the minimum function on each panel.
It explains the selected 64 panels, 9 terms in float64 and 5 in float32. It is
not a bound on floating-point evaluation; the reference sweep checks that too.

Run (~5 min per JAX version; GPU access required):
  JAX_PLATFORMS=cuda,cpu taskset -c 3 python experiments/37_erf_gradients.py \
      --devices cpu gpu > results/37_erf_gradients_011.txt
Use the same command with JAX 0.12.0.1 on PYTHONPATH for the latest comparison.
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
BASELINE = "353ea37"
SIZES = {"cpu": 2**16, "gpu": 2**20}


def reference(name, x, mp):
    x = mp.mpf(float(x))
    a = abs(x)
    if a > 40 and (name == "dawsn" or x > 0):
        u, c = 1 / (2 * a * a), mp.mpf(1)
        y = d1 = d2 = mp.mpf(0)
        for k in range(20):
            term = c * u**k * ((-1) ** k if name == "erfcx" else 1)
            y += term
            d1 += (2 * k + 1) * term
            d2 += (k + 1) * (2 * k + 1) * term
            c *= 2 * k + 1
        scale = 1 / mp.sqrt(mp.pi) if name == "erfcx" else mp.mpf(".5")
        y *= scale / a
        d1 *= -scale / (a * a)
        d2 *= 2 * scale / (a * a * a)
        if name == "dawsn" and x < 0:
            y, d2 = -y, -d2
        return y, d1, d2
    if name == "dawsn":
        y = mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)
        d1 = 1 - 2 * x * y
        return y, d1, -2 * y - 2 * x * d1
    y = mp.exp(x * x) * mp.erfc(x)
    d1 = 2 * x * y - 2 / mp.sqrt(mp.pi)
    return y, d1, 2 * y + 2 * x * d1


def make_refs(path, dt):
    import mpmath as mp

    mp.mp.dps = 60
    seams = np.r_[np.arange(65) * 6 / 64, 3 * (32 / np.arange(1, 33) - 1)]
    pos = np.r_[
        np.finfo(dt).tiny,
        2 * np.finfo(dt).tiny,
        np.linspace(0, 7, 351),
        np.geomspace(1e-10, 1e4, 100),
        seams,
        seams * (1 - 1e-7),
        seams * (1 + 1e-7),
        [1e10, 1e18] if dt == np.float32 else [1e10, 1e100, 1e150],
    ]
    arrays = {}
    for name in ("erfcx", "dawsn"):
        neg = (
            -np.linspace(0, 26.5 if dt == np.float64 else 9, 150)
            if name == "erfcx"
            else -pos
        )
        x = np.unique(np.asarray(np.r_[pos, neg], dt))
        refs = np.array([[float(v) for v in reference(name, t, mp)] for t in x]).T
        arrays[name + "_x"], arrays[name + "_ref"] = x, refs
    np.savez(path, **arrays)


def degree_check():
    import mpmath as mp
    from chebax._src.recipes._gen_common import dct, nodes

    mp.mp.dps = 60
    rows = {}
    for npan, terms in ((32, (6, 10, 11)), (64, (5, 8, 9))):
        worst = {m: mp.mpf(0) for m in terms}
        for j in range(npan):
            lo, hi = mp.mpf(6) * j / npan, mp.mpf(6) * (j + 1) / npan

            def e(x):
                return mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x) / x

            c = dct(mp, [e(lo + (v + 1) * (hi - lo) / 2) for v in nodes(mp, 24)])
            for m in terms:
                worst[m] = max(worst[m], sum(abs(v) for v in c[m:]) / e(hi))
        rows[str(npan) + " panels"] = {str(m): float(v) for m, v in worst.items()}
    return rows


def timed(call, n, budget):
    import jax

    jax.block_until_ready(call())
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.2:
        jax.block_until_ready(call())
    t0 = time.perf_counter()
    jax.block_until_ready(call())
    one = time.perf_counter() - t0
    inner = max(1, int(budget / 5 / max(one, 1e-6)))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = call()
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts)) / n * 1e9


def accuracy(f, name, refs, dt, dev, source):
    import jax
    import jax.numpy as jnp

    x, ref = refs[name + "_x"], refs[name + "_ref"]
    total = lambda x: f(x).sum()
    d1 = jax.grad(total)
    d2 = jax.grad(lambda x: d1(x).sum())
    arg = jax.device_put(jnp.asarray(x), dev)
    out = {}
    got = []
    for k, g in enumerate((f, d1, d2)):
        try:
            value = jax.jit(g)(arg)
        except TypeError as error:
            if source != "jax" or k != 2:
                raise
            out["d2 vector error"] = str(error).splitlines()[0]
            value = jax.jit(jax.vmap(jax.grad(jax.grad(f))))(arg)
        got.append(np.asarray(value))
    got = np.array(got, float)
    for k in range(3):
        valid = (abs(ref[k]) >= np.finfo(dt).tiny) & (abs(ref[k]) <= np.finfo(dt).max)
        if name == "dawsn":
            if k == 1:
                valid &= abs(abs(x) - 0.924138873) > 0.02
            elif k == 2:
                valid &= (abs(x) > 0.05) & (abs(abs(x) - 1.5) > 0.05)
        finite_ref = abs(ref[k]) <= np.finfo(dt).max
        bad = finite_ref & ~np.isfinite(got[k])
        keep = valid & ~bad
        err = abs(got[k, keep] / ref[k, keep] - 1)
        out[f"d{k} rel"] = float(err.max())
        out[f"d{k} nonfinite"] = int(bad.sum())
        small = (abs(ref[k]) < np.finfo(dt).tiny) & np.isfinite(got[k])
        out[f"d{k} underflow abs"] = (
            float(np.max(abs(got[k, small] - ref[k, small]))) if np.any(small) else 0.0
        )
        if name == "dawsn":
            bulk = abs(x) <= 6
            assert np.all(np.isfinite(got[k, bulk]))
            out[f"d{k} bulk abs"] = float(np.max(abs(got[k, bulk] - ref[k, bulk])))
    return out


def child(label, src, dtype, refpath, devices, budget, check):
    import jax

    jax.config.update("jax_enable_x64", dtype == "f64")
    import jax.numpy as jnp
    import jax.scipy.special as jsp

    import chebax

    assert chebax.__file__.startswith(src)
    dt = np.float64 if dtype == "f64" else np.float32
    refs = np.load(refpath)
    functions = [(label, chebax)] + ([("jax", jsp)] if label == "candidate" else [])
    for device in devices:
        dev, n = jax.devices(device)[0], SIZES[device]
        rng = np.random.default_rng(37)
        for name, scale in (("erfcx", 3), ("dawsn", 5)):
            x = jax.device_put(jnp.asarray(rng.normal(0, scale, n), dt), dev)
            for source, module in functions:
                f = getattr(module, name)
                total = lambda x: f(x).sum()
                rec = dict(
                    source=source,
                    dtype=dtype,
                    device=device,
                    family=name,
                    hardware=dev.device_kind,
                )
                for kind, fun in (
                    ("value", total),
                    ("grad", jax.grad(total)),
                    ("value_grad", jax.value_and_grad(total)),
                ):
                    compiled = jax.jit(fun).lower(x).compile()
                    rec[kind] = timed(lambda: compiled(x), n, budget)
                if check:
                    rec["accuracy"] = accuracy(f, name, refs, dt, dev, source)
                    if source == "candidate":
                        assert not any(
                            v
                            for k, v in rec["accuracy"].items()
                            if k.endswith("nonfinite")
                        ), rec
                print(json.dumps(rec), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", choices=SIZES, default=["cpu"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--budget", type=float, default=0.6)
    p.add_argument("--child", nargs=4)
    p.add_argument("--check", action="store_true")
    opts = p.parse_args()
    if opts.child:
        return child(*opts.child, opts.devices, opts.budget, opts.check)
    import jax

    meta = dict(
        date=time.strftime("%Y-%m-%d %H:%M"),
        jax=jax.__version__,
        baseline=BASELINE,
        affinity=sorted(os.sched_getaffinity(0)),
        rounds=opts.rounds,
        load_start=os.getloadavg()[0],
    )
    meta["candidate sha256"] = {
        n: hashlib.sha256(
            (ROOT / "src/chebax/_src/recipes" / n).read_bytes()
        ).hexdigest()
        for n in ("erf_family.py", "erf_gen.py", "erf_table.py")
    }
    bounds = degree_check()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for dts, dt in [("f64", np.float64), ("f32", np.float32)]:
            make_refs(tmp / (dts + ".npz"), dt)
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
        for rnd in range(opts.rounds):
            for label, src in (sources if rnd % 2 == 0 else sources[::-1]):
                for dts in ("f64", "f32"):
                    cmd = [
                        sys.executable,
                        __file__,
                        "--child",
                        label,
                        src,
                        dts,
                        str(tmp / (dts + ".npz")),
                        "--devices",
                        *opts.devices,
                        "--budget",
                        str(opts.budget),
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
                        f'round {rnd+1}: {label} {dts} done {time.strftime("%H:%M")}',
                        file=sys.stderr,
                    )
    meta["load_end"] = os.getloadavg()[0]
    print("erf-family speed: ns per element, mean of round medians")
    print(json.dumps(meta, indent=2))
    print("Dawson omitted coefficient mass / min E per panel:")
    print(json.dumps(bounds, indent=2))
    for device in opts.devices:
        for dts in ("f64", "f32"):
            print(f"\n{device} {dts}, N={SIZES[device]}")
            print(f'{"function/source":<28}{"value":>12}{"grad":>12}{"value_grad":>14}')
            for name in ("erfcx", "dawsn"):
                for source in ("baseline", "candidate", "jax"):
                    rr = [
                        r
                        for r in rows
                        if r["device"] == device
                        and r["dtype"] == dts
                        and r["family"] == name
                        and r["source"] == source
                    ]
                    if not rr:
                        continue
                    vals = {
                        k: float(np.mean([r[k] for r in rr]))
                        for k in ("value", "grad", "value_grad")
                        if k in rr[0]
                    }
                    print(
                        f'{name+" "+source:<28}'
                        + "".join(
                            f"{vals[k]:>12.4g}" if k in vals else f'{"-":>12}'
                            for k in ("value", "grad", "value_grad")
                        )
                    )
    print("\nAccuracy at 60 dps:")
    for r in rows:
        if "accuracy" in r:
            print(
                r["device"],
                r["dtype"],
                r["family"],
                r["source"],
                json.dumps(r["accuracy"], sort_keys=True),
            )


if __name__ == "__main__":
    main()
