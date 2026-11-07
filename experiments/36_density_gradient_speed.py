#!/usr/bin/env python3
"""Recover the cost of 0.5.0's density-gradient fixes without dropping them.

Compare 0.5.0 plus experiments/35 (353ea37) with the working src/. Beta reuses
its CDF's oriented logs for the density. Gamma computes the density fallback
only if a CDF in the array underflows. The formulas, tables and precision are
unchanged. This measures value and value_and_grad, not compilation time.

Same inputs and timer as experiments/35: scalar beta parameters, x drawn from
the beta; gamma's a = mu**2/1.5**2, mu clipped log-normal, x drawn from Gamma(a).
Also time gamma with one deep-tail element and with all elements deep in the
lower tail, so the fallback's cost is visible. Each source runs in a separate
process, two rounds in reverse order, pinned by the caller. Report median ns
per element, averaged over rounds; CPU N=2**16, GPU N=2**20 if requested.

Accuracy: mpmath at 60 dps, at dtype-rounded inputs. CDF absolute error,
x-gradient pointwise relative error where the reference density is normal,
a/b-gradient error divided by max(1, abs(reference)). Beta includes the two
benchmark pairs, reflected tails, shape seams and extreme shapes. Gamma
covers every zone, shape/argument seams, both tails and CDF underflow with a
representable density. Explicitly reject unexpected nonfinite results.
These grids supplement the acceptance suite; they are not a worst-case bound.

Run (~10 min, CPU):
  JAX_PLATFORMS=cpu taskset -c 3 python experiments/36_density_gradient_speed.py \
      > results/36_density_gradient_speed.txt
Pass --devices cpu gpu to include an available GPU. Runtime source hashes
identify the optimized files even when this is run before committing them.
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
RUNTIME = ("betainc_kernel.py", "gammainc_array.py")


def timed(call, n, budget):
    import jax

    jax.block_until_ready(call())
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
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


def time_source(label, src, dts, devices, budget, accuracy):
    import jax

    jax.config.update("jax_enable_x64", dts == "f64")
    import jax.numpy as jnp
    import chebax

    assert chebax.__file__.startswith(src), (chebax.__file__, src)
    dt = np.float64 if dts == "f64" else np.float32
    for dname in devices:
        dev, n = jax.devices(dname)[0], SIZES[dname]
        rng = np.random.default_rng(35)
        put = lambda v: jax.device_put(jnp.asarray(v, dt), dev)
        for a, b in ((2.5, 7.0), (0.5, 50.0)):
            args = [put(a), put(b), put(rng.beta(a, b, n))]
            total = lambda a, b, x: chebax.betainc(a, b, x).sum()
            cv = jax.jit(total).lower(*args).compile()
            for argnums, suffix in (((2,), "x"), ((0, 1, 2), "all")):
                cg = jax.jit(jax.value_and_grad(total, argnums)).lower(*args).compile()
                rec = dict(
                    source=label,
                    dtype=dts,
                    device=dname,
                    cell=f"beta({a:g},{b:g}) {suffix}",
                    hardware=dev.device_kind,
                )
                if suffix == "x":
                    rec["value"] = timed(lambda: cv(*args), n, budget)
                rec["grad"] = timed(lambda: cg(*args), n, budget)
                print(json.dumps(rec), flush=True)
        mu = np.clip(np.exp(rng.normal(1.0, 0.5, n)), 0.6, 20.0)
        a = mu**2 / 1.5**2
        x = rng.gamma(a)
        args = [put(a), put(x)]
        total = lambda a, x: chebax.gammainc_array(a, x).sum()
        cv = jax.jit(total).lower(*args).compile()
        cg = jax.jit(jax.value_and_grad(total, (0, 1))).lower(*args).compile()
        for kind in ("bulk", "one deep", "all deep"):
            if kind == "one deep":
                a, x = a.copy(), x.copy()
                a[0], x[0] = 2.0, (1e-160 if dts == "f64" else 1e-25)
                args = [put(a), put(x)]
            elif kind == "all deep":
                args = [put(np.full(n, 2.0)), put(np.full(n, x[0]))]
            rec = dict(
                source=label,
                dtype=dts,
                device=dname,
                cell=f"gamma {kind}",
                hardware=dev.device_kind,
            )
            if kind == "bulk":
                rec["value"] = timed(lambda: cv(*args), n, budget)
            rec["grad"] = timed(lambda: cg(*args), n, budget)
            print(json.dumps(rec), flush=True)
    if accuracy:
        print(
            json.dumps(dict(source=label, dtype=dts, accuracy=check_accuracy(dt))),
            flush=True,
        )


def check_accuracy(dt):
    import jax
    import jax.numpy as jnp
    import mpmath as mp
    import chebax
    from chebax._src.recipes import gammainc_array_table as t

    mp.mp.dps = 60
    tiny = np.finfo(dt).tiny
    result = {}

    def record(name, got, ref, metric):
        got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
        assert np.all(np.isfinite(got)), (name, got)
        assert np.all(np.isfinite(ref)), (name, ref)
        if metric == "relative":
            keep = np.abs(ref) >= tiny
            error = np.abs(got[keep] / ref[keep] - 1)
        else:
            error = np.abs(got - ref)
            if metric == "scaled":
                error /= np.maximum(1.0, np.abs(ref))
        assert error.size and np.all(np.isfinite(error)), name
        result[name] = max(result.get(name, 0.0), float(np.max(error)))

    rng = np.random.default_rng(36)
    pairs = (
        (2.5, 7.0),
        (0.5, 50.0),
        (100.0, 0.1),
        (0.1, 100.0),
        (100.0, 3.0),
        (3.0, 100.0),
        (7.0, 2.5),
        (50.0, 0.5),
        (2.0, 3.0),
        (1.0, 1.0),
        (10.0, 10.0),
        (100.0, 100.0),
    )
    bx = jax.jit(jax.value_and_grad(lambda a, b, x: chebax.betainc(a, b, x).sum(), 2))
    ba = jax.jit(jax.grad(chebax.betainc, (0, 1)))
    beta_points = 0
    for a, b in pairs:
        a, b = dt(a), dt(b)
        xs = np.asarray(
            [
                1e-160 if dt == np.float64 else 1e-25,
                1e-8,
                0.01,
                0.2,
                0.4999999,
                0.5,
                0.5000001,
                0.6,
                0.9,
                0.99,
                1 - 1e-7,
                *rng.beta(a, b, 12),
            ],
            dt,
        )
        xs = xs[(xs > 0) & (xs < 1)]
        beta_points += len(xs)
        _, dx = bx(jnp.asarray(a), jnp.asarray(b), jnp.asarray(xs))
        values = jax.jit(chebax.betainc)(
            jnp.asarray(a), jnp.asarray(b), jnp.asarray(xs)
        )
        am, bm = mp.mpf(float(a)), mp.mpf(float(b))
        refs, pdfs = [], []
        for x in xs:
            xm = mp.mpf(float(x))
            refs.append(float(mp.betainc(am, bm, 0, xm, regularized=True)))
            pdfs.append(
                float(
                    mp.exp(
                        (am - 1) * mp.log(xm)
                        + (bm - 1) * mp.log1p(-xm)
                        - mp.log(mp.beta(am, bm))
                    )
                )
            )
        record("beta value abs", values, refs, "absolute")
        record("beta dx rel", dx, pdfs, "relative")
        xm = mp.mpf(float(dt(0.3)))
        da, db = ba(jnp.asarray(a), jnp.asarray(b), jnp.asarray(dt(0.3)))
        refs = [
            float(mp.diff(lambda aa: mp.betainc(aa, bm, 0, xm, regularized=True), am)),
            float(mp.diff(lambda bb: mp.betainc(am, bb, 0, xm, regularized=True), bm)),
        ]
        record("beta da/db scaled", [da, db], refs, "scaled")

    # Every zone and its seams, plus both density-underflow regressions.
    aa, xx = [], []
    for a in (
        0.1,
        0.3,
        0.999999,
        1.0,
        1.000001,
        2.0,
        9.99999,
        10.0,
        10.00001,
        50.0,
        100.0,
        1000.0,
    ):
        for x in (a + 1, a * t.LAM_LO, a * t.LAM_HI, 0.01 * a, a, 5 * a, 20 * a):
            aa.append(a)
            xx.append(x)
    a = np.exp(rng.uniform(np.log(0.1), np.log(1e4), 60))
    aa.extend(a)
    xx.extend(rng.gamma(a))
    for a in (1.0, 2.0, 3.0, 100.0):
        aa.append(a)
        xx.append(1e-160 if dt == np.float64 else 1e-25)
    a, x = np.asarray(aa, dt), np.asarray(xx, dt)
    f = jax.jit(
        jax.value_and_grad(lambda a, x: chebax.gammainc_array(a, x).sum(), (0, 1))
    )
    _, (da, dx) = f(jnp.asarray(a), jnp.asarray(x))
    values = jax.jit(chebax.gammainc_array)(jnp.asarray(a), jnp.asarray(x))
    refs, shapes, pdfs = [], [], []
    for ak, xk in zip(a, x):
        am, xm = mp.mpf(float(ak)), mp.mpf(float(xk))
        refs.append(float(mp.gammainc(am, 0, xm, regularized=True)))
        shapes.append(
            float(mp.diff(lambda aa: mp.gammainc(aa, 0, xm, regularized=True), am))
        )
        pdfs.append(float(mp.exp((am - 1) * mp.log(xm) - xm - mp.loggamma(am))))
    record("gamma value abs", values, refs, "absolute")
    record("gamma dx rel", dx, pdfs, "relative")
    record("gamma da scaled", da, shapes, "scaled")
    result["beta points"] = beta_points
    result["gamma points"] = len(a)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", default=["cpu"], choices=SIZES)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--child", nargs=4, metavar=("LABEL", "SRC", "DTYPE", "ACCURACY"))
    opts = p.parse_args()
    if opts.child:
        label, src, dts, accuracy = opts.child
        return time_source(
            label, src, dts, opts.devices, opts.budget, accuracy == "yes"
        )
    import jax

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "jax": jax.__version__,
        "baseline": BASELINE,
        "cpu affinity": sorted(os.sched_getaffinity(0)),
        "load start": os.getloadavg()[0],
    }
    meta["optimized sha256"] = {
        name: hashlib.sha256(
            (ROOT / "src/chebax/_src/recipes" / name).read_bytes()
        ).hexdigest()
        for name in RUNTIME
    }
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(
            ["git", "-C", str(ROOT), "archive", BASELINE, "src"],
            capture_output=True,
            check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(tmp, filter="data")
        sources = [("0.5.0", str(Path(tmp) / "src")), ("optimized", str(ROOT / "src"))]
        for round_ in range(opts.rounds):
            for label, src in (sources if round_ % 2 == 0 else sources[::-1]):
                for dts in ("f64", "f32"):
                    cmd = [
                        sys.executable,
                        __file__,
                        "--child",
                        label,
                        src,
                        dts,
                        "yes" if round_ == 0 else "no",
                        "--devices",
                        *opts.devices,
                        "--budget",
                        str(opts.budget),
                    ]
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
                        f"round {round_+1}: {label} {dts} done {time.strftime('%H:%M')}",
                        file=sys.stderr,
                    )
    meta["load end"] = os.getloadavg()[0]
    print("density-gradient speed, 0.5.0 and optimized")
    print(json.dumps(meta, indent=2))
    for dts in ("f64", "f32"):
        for device in opts.devices:
            subset = [
                r for r in rows if r.get("device") == device and r["dtype"] == dts
            ]
            print(f"\n{dts} {device}, N={SIZES[device]}, {subset[0]['hardware']}")
            print(
                f"{'cell':<25}{'0.5.0 value/grad':>23}{'optimized value/grad':>25}{'grad speedup':>16}"
            )
            for cell in dict.fromkeys(r["cell"] for r in subset):
                means = {}
                for source in ("0.5.0", "optimized"):
                    rr = [
                        r for r in subset if r["cell"] == cell and r["source"] == source
                    ]
                    means[source] = {
                        k: float(np.mean([r[k] for r in rr]))
                        for k in ("value", "grad")
                        if k in rr[0]
                    }

                def fmt(r):
                    return (
                        f"{r['value']:.3g} / {r['grad']:.3g}"
                        if "value" in r
                        else f"- / {r['grad']:.3g}"
                    )

                print(
                    f"{cell:<25}{fmt(means['0.5.0']):>23}{fmt(means['optimized']):>25}"
                    f"{means['0.5.0']['grad']/means['optimized']['grad']:>15.2f}x"
                )
    print("\nAccuracy against mpmath 60 dps (worst errors on the grid):")
    for row in rows:
        if "accuracy" in row:
            print(
                row["source"], row["dtype"], json.dumps(row["accuracy"], sort_keys=True)
            )


if __name__ == "__main__":
    main()
