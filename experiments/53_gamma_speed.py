"""Speed and accuracy of gammainc_array, log_gammaincc_array and gammaincinv_array,
for comparing source trees (main and a branch) in one run.

Ops, per element with a different a on every element:
  P         gammainc_array(a, x)
  P vg      value_and_grad of sum(P) in (a, x)
  lnQ       log_gammaincc_array(a, x) (a censored model's log survival)
  lnQ vg    value_and_grad of sum(ln Q) in (a, x)
  inv       gammaincinv_array(a, p)
  inv vg_p  value_and_grad of sum(x) in p
  inv vg    value_and_grad of sum(x) in (a, p)

Regions: a log-uniform in [0.1, 1), [1, 20), [20, 1e4) crossed with the probability
band (lower tail p in [1e-20, 0.05), bulk, upper tail 1 - p in [1e-20, 0.05); for
gammaincinv the upper tail from 1e-15, and from 1e-6 in float32), as pymc-special
experiments 47 and 51 draw them; gammainc's x = scipy's quantile at that p. "mixed":
a log-uniform in [0.1, 1e3], x = a e^u with u uniform in [ln 0.05, ln 20], p uniform
in log-odds on [1e-12, 1 - 1e-12] (1e-6 in float32), as experiment 43. Inputs are
drawn in the dtype timed; float32 references are taken at the float32 inputs.

Speed: ns per element, N = 2^18 on one CPU core (taskset, single-threaded Eigen),
2^20 (float64) and 2^22 (float32) on the GPU, each region's 2^14 points tiled. Median
of at least 5 calls after a warm call. float32 runs with x64 off. Each tree runs in
its own process; "all" alternates the trees in every pass, and the report lists cells
whose passes differ by more than 1.25x. Compile time and XLA's temporary memory of
each compiled program are recorded too (compile time only where the persistent cache
missed).

Accuracy, against mpmath at 40 digits on the first 300 (a, x) and 150 (a, p) points of
every region, d/da of gammainc on the first 100 (refs):
  logs and P      |F - ref| / (|ref| max(1, |ln ref|)) for P, |ln F - ref| / max(1,
                  |ref|) for ln Q (the test suite's metric)
  gradients       |g - ref| / |ref|, refs below tiny/eps or past the largest float
                  left out (digits lost to the range);
                  d/da of ln Q as |g - ref| / max(1, |ref|)
  inv             two-sided: |x - x_ref| / x_ref times min(1, |d ln F / d ln x| /
                  max(1, |ln F|)), F the target's side: the error in x, or where x is
                  ill conditioned the error it makes in ln F, on the logs' metric
  inv slopes      dx/dp = 1/pdf and dx/da = -dP/da / pdf at the mpmath root, relative

Usage:
  python experiments/53_gamma_speed.py refs     # once, ~10 s on 6 processes
  [E53_DEVICES=cpu,gpu E53_DTYPES=float64,float32 E53_PASSES=2 E53_CORE=5
   E53_REGIONS=all|mixed E53_OPS=P,inv E53_RUN=<name> E53_NOTE=<header line>] \\
      python experiments/53_gamma_speed.py all main=<src> branch=<src>[@<jax dir>]
  python experiments/53_gamma_speed.py report <run> [<baseline label>] > results/...

Not established: other layouts (a uniform per call is gammainc_scalar's path, not
timed here), the seams and far tails (the test suite checks those), speed inside a
model.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE.parent / "results" / "53_cache"
A_BANDS = [("a<1", 0.1, 1.0), ("a1-20", 1.0, 20.0), ("a>20", 20.0, 1e4)]
REGIONS = [f"{a[0]} {b}" for a in A_BANDS for b in ("lower", "bulk", "upper")]
REGIONS.append("mixed")
OPS = ["P", "P vg", "lnQ", "lnQ vg", "inv", "inv vg_p", "inv vg"]
N_DISTINCT = 2**14
N_REF = {"ax": 300, "ap": 150}  # points per region with references
N_DA = 100  # of those, with d/da references
DTYPES = ("float64", "float32")


def n_of(device, dtype):
    if device == "cpu":
        return 2**18
    return 2**20 if dtype == "float64" else 2**22


# ------------------------------------------------------------------ inputs


def draw(kind, region, dtype):
    """(a, x) for kind "ax" or (a, p) for "ap", N_DISTINCT points, in dtype."""
    from scipy import special as sc

    seed = 53 + 101 * REGIONS.index(region) + (7 if kind == "ap" else 0)
    seed += 1000 * DTYPES.index(dtype)
    rng = np.random.default_rng(seed)
    n = N_DISTINCT
    tiny = np.finfo(dtype).tiny
    if region == "mixed":
        a = np.exp(rng.uniform(np.log(0.1), np.log(1e3), n)).astype(dtype)
        if kind == "ax":
            x = a * np.exp(rng.uniform(np.log(0.05), np.log(20.0), n))
            return a, x.astype(dtype)
        lo = 1e-12 if dtype == "float64" else 1e-6
        p = 1.0 / (1.0 + np.exp(-rng.uniform(np.log(lo), -np.log(lo), n)))
        return a, p.astype(dtype)
    aname, band = region.split(" ")
    _, alo, ahi = next(b for b in A_BANDS if b[0] == aname)
    a = np.exp(rng.uniform(np.log(alo), np.log(ahi), n)).astype(dtype)
    a64 = a.astype(np.float64)
    p = q = None
    if band == "lower":
        pmin = np.maximum(1e-20, sc.gammainc(a64, 1e3 * tiny))
        p = np.exp(rng.uniform(np.log(pmin), np.log(0.05)))
    elif band == "bulk":
        p = rng.uniform(0.05, 0.95, n)
    else:
        qlo = 1e-20 if kind == "ax" else (1e-15 if dtype == "float64" else 1e-6)
        q = np.exp(rng.uniform(np.log(qlo), np.log(0.05), n))
    if kind == "ap":
        p = p if q is None else 1.0 - q
        p = p.astype(dtype)
        # 1 - q can round to 1 in float32
        p[p >= 1.0] = np.nextafter(np.asarray(1.0, dtype), np.asarray(0.0, dtype))
        return a, p
    x = sc.gammaincinv(a64, p) if q is None else sc.gammainccinv(a64, q)
    x = x.astype(dtype)
    x[x < 1e3 * tiny] = 1e3 * tiny
    return a, x


# ------------------------------------------------------------------ references


def _ref_ax(job):
    """ln P, ln Q, ln pdf, d ln P/da, d ln Q/da (nan unless asked) at (a, x)."""
    import mpmath as mp

    mp.mp.dps = 40
    a, x, with_da = job
    am, xm = mp.mpf(a), mp.mpf(x)
    lower = xm < am

    def small(t):
        if lower:
            return mp.gammainc(t, 0, xm, regularized=True)
        return mp.gammainc(t, xm, mp.inf, regularized=True)

    s = small(am)
    o = 1 - s
    ls, lo_ = mp.log(s), mp.log(o)
    lp, lq = (ls, lo_) if lower else (lo_, ls)
    lpdf = (am - 1) * mp.log(xm) - xm - mp.loggamma(am)
    dp = dq = float("nan")
    if with_da:
        dls = mp.diff(lambda t: mp.log(small(t)), am)
        dlo = -dls * s / o
        dp, dq = (dls, dlo) if lower else (dlo, dls)
    return [float(v) for v in (lp, lq, lpdf, dp, dq)]


def _ref_ap(job):
    """x, ln F (target side), d ln F/d ln x, dx/dp, dx/da at the root of P(a, x) = p."""
    import mpmath as mp
    from scipy import special as sc

    mp.mp.dps = 40
    a, p = job
    am, pm = mp.mpf(a), mp.mpf(p)
    upper = p > 0.5
    target = 1 - pm if upper else pm

    def side(t, x):
        # the smaller tail directly; the other is >= ~1/2
        if x < t:
            pl = mp.gammainc(t, 0, x, regularized=True)
            return 1 - pl if upper else pl
        ql = mp.gammainc(t, x, mp.inf, regularized=True)
        return ql if upper else 1 - ql

    x0 = float(sc.gammainccinv(a, 1.0 - p) if upper else sc.gammaincinv(a, p))
    x0 = max(x0, 1e-300)
    v = mp.findroot(lambda v: mp.log(side(am, mp.exp(v)) / target), mp.log(x0),
                    tol=mp.mpf(10) ** -35)
    x = mp.exp(v)
    pdf = mp.exp((am - 1) * mp.log(x) - x - mp.loggamma(am))
    dlnf = (-1 if upper else 1) * x * pdf / target
    # dF/da at fixed x; dP/da = -dQ/da
    dfa = mp.diff(lambda t: side(t, x), am)
    dpa = -dfa if upper else dfa
    return [float(x), float(mp.log(target)), float(dlnf), float(1 / pdf),
            float(-dpa / pdf)]


def refs():
    from multiprocessing import Pool

    CACHE.mkdir(parents=True, exist_ok=True)
    procs = int(os.environ.get("E53_PROCS", 6))
    for dtype in DTYPES:
        out = {}
        for kind in ("ax", "ap"):
            pts, jobs = [], []
            for r in REGIONS:
                a, b = draw(kind, r, dtype)
                k = N_REF[kind]
                pts.append((a[:k], b[:k]))
                for i in range(k):
                    if kind == "ax":
                        jobs.append((float(a[i]), float(b[i]), i < N_DA))
                    else:
                        jobs.append((float(a[i]), float(b[i])))
            t0 = time.time()
            with Pool(procs) as pool:
                vals = pool.map(_ref_ax if kind == "ax" else _ref_ap, jobs, chunksize=8)
            print(f"{dtype} {kind}: {len(jobs)} points, {time.time() - t0:.0f} s",
                  flush=True)
            out[f"{kind}_a"] = np.concatenate([p[0] for p in pts])
            out[f"{kind}_b"] = np.concatenate([p[1] for p in pts])
            out[f"{kind}_ref"] = np.asarray(vals)
            out[f"{kind}_region"] = np.repeat(np.arange(len(REGIONS)), N_REF[kind])
        np.savez(CACHE / f"refs_{dtype}.npz", **out)


# ------------------------------------------------------------------ one tree, one device


def _ops(chebax, jax, jnp):
    def summed(f):
        return lambda a, b: jnp.sum(f(a, b))

    vg = jax.value_and_grad
    return {
        "P": (chebax.gammainc_array, "ax"),
        "P vg": (vg(summed(chebax.gammainc_array), (0, 1)), "ax"),
        "lnQ": (chebax.log_gammaincc_array, "ax"),
        "lnQ vg": (vg(summed(chebax.log_gammaincc_array), (0, 1)), "ax"),
        "inv": (chebax.gammaincinv_array, "ap"),
        "inv vg_p": (vg(summed(chebax.gammaincinv_array), 1), "ap"),
        "inv vg": (vg(summed(chebax.gammaincinv_array), (0, 1)), "ap"),
    }


def _rel(g, r, floor):
    g, r = np.asarray(g, np.float64), np.asarray(r, np.float64)
    keep = np.isfinite(r) & (np.abs(r) >= floor)
    e = np.full(r.shape, np.nan)
    e[keep] = np.abs(g[keep] - r[keep]) / np.abs(r[keep])
    e[keep & ~np.isfinite(g)] = np.inf
    return e


def _errors(op, out, ref, dtype):
    """name -> per-point error array for one op's outputs on the reference points."""
    # within 1/eps of the smallest normal a result loses digits to the range
    tiny = float(np.finfo(dtype).tiny / np.finfo(dtype).eps)
    big = float(np.finfo(dtype).max)

    def normal(v):
        return np.where((np.abs(v) >= tiny) & (np.abs(v) <= big), v, np.nan)

    if op in ("P", "P vg"):
        lp, lq, lpdf, dlp, _ = ref.T
        val = out if op == "P" else None
        errs = {}
        if val is not None:
            pref = np.exp(lp)
            e = _rel(val, normal(pref), tiny)
            errs["P"] = e / np.maximum(1.0, np.abs(lp))
        else:
            ga, gx = out
            errs["dP/da"] = _rel(ga, normal(np.exp(lp) * dlp), tiny)
            errs["dP/dx"] = _rel(gx, normal(np.exp(lpdf)), tiny)
        return errs
    if op in ("lnQ", "lnQ vg"):
        lp, lq, lpdf, _, dlq = ref.T
        if op == "lnQ":
            g = np.asarray(out, np.float64)
            e = np.abs(g - lq) / np.maximum(1.0, np.abs(lq))
            e[~np.isfinite(g)] = np.inf
            return {"lnQ": e}
        ga, gx = out
        ga = np.asarray(ga, np.float64)
        ea = np.abs(ga - dlq) / np.maximum(1.0, np.abs(dlq))
        ea[~np.isfinite(ga) & np.isfinite(dlq)] = np.inf
        return {"dlnQ/da": ea, "dlnQ/dx": _rel(gx, normal(-np.exp(lpdf - lq)), tiny)}
    xr, lnf, dlnf, dxdp, dxda = ref.T
    if op == "inv":
        cond = np.abs(dlnf) / np.maximum(1.0, np.abs(lnf))
        return {"x": _rel(out, normal(xr), tiny) * np.minimum(1.0, cond)}
    if op == "inv vg_p":
        return {"dx/dp": _rel(out, normal(dxdp), tiny)}
    ga, gp = out
    return {"dx/da": _rel(ga, normal(dxda), tiny), "dx/dp": _rel(gp, normal(dxdp), tiny)}


def child(label, src, device, dtype, out_path):
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, src)
    import jax

    jax.config.update("jax_enable_x64", dtype == "float64")
    if os.environ.get("E53_XLA_CACHE", "1") == "1":
        jax.config.update("jax_compilation_cache_dir", str(CACHE / f"xla_{device}"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    import jax.numpy as jnp

    import chebax

    here = pathlib.Path(chebax.__file__).resolve()
    assert here.is_relative_to(pathlib.Path(src).resolve()), here
    n = n_of(device, dtype)
    regions = REGIONS if os.environ.get("E53_REGIONS", "all") == "all" else ["mixed"]
    want = os.environ.get("E53_OPS")
    ops = _ops(chebax, jax, jnp)
    if want:
        ops = {k: v for k, v in ops.items() if k in want.split(",")}
    refd = np.load(CACHE / f"refs_{dtype}.npz")
    dev = jax.devices()[0]

    def spin():
        if device != "cpu":
            m = jnp.ones((2048, 2048), jnp.float32)
            t_end = time.perf_counter() + 0.5
            while time.perf_counter() < t_end:
                jax.block_until_ready(m @ m)

    result = {"label": label, "src": src, "device": device, "dtype": dtype, "n": n,
              "jax": jax.__version__, "cells": [], "acc": {}, "compile": {}}
    for op, (fn, kind) in ops.items():
        tiled = {}
        for r in regions:
            a, b = draw(kind, r, dtype)
            reps = n // a.size
            tiled[r] = (jax.device_put(np.tile(a, reps), dev),
                        jax.device_put(np.tile(b, reps), dev))
        args0 = tiled[regions[0]]
        t0 = time.perf_counter()
        comp = jax.jit(fn).lower(*args0).compile()
        cs = time.perf_counter() - t0
        mem = comp.memory_analysis()
        result["compile"][op] = {"s": cs, "temp_mib": mem.temp_size_in_bytes / 2**20}
        # accuracy on the reference points, padded to the compiled shape
        ra, rb, ref = refd[f"{kind}_a"], refd[f"{kind}_b"], refd[f"{kind}_ref"]
        reg = refd[f"{kind}_region"]
        reps = -(-n // ra.size)
        pa = jax.device_put(np.tile(ra.astype(dtype), reps)[:n], dev)
        pb = jax.device_put(np.tile(rb.astype(dtype), reps)[:n], dev)
        out = comp(pa, pb)
        if op.endswith("vg") or op.endswith("vg_p"):
            out = out[1]
        out = jax.tree.map(lambda v: np.asarray(v)[: ra.size], out)
        for name, e in _errors(op, out, ref, dtype).items():
            per = {}
            for i, r in enumerate(REGIONS):
                ei = e[reg == i]
                ei = ei[~np.isnan(ei)]
                per[r] = float(ei.max()) if ei.size else None
            result["acc"][f"{op}: {name}"] = per
        for r in regions:
            args = tiled[r]
            jax.block_until_ready(comp(*args))
            spin()
            ts = []
            t_end = time.perf_counter() + float(os.environ.get("E53_BUDGET", 1.0))
            while len(ts) < 5 or (time.perf_counter() < t_end and len(ts) < 31):
                t0 = time.perf_counter()
                jax.block_until_ready(comp(*args))
                ts.append(time.perf_counter() - t0)
            result["cells"].append({"op": op, "region": r,
                                    "ns": float(np.median(ts)) / n * 1e9,
                                    "calls": len(ts)})
        print(f"  {label} {device} {dtype} {op}: compile {cs:.1f} s", flush=True)
    pathlib.Path(out_path).write_text(json.dumps(result))


# ------------------------------------------------------------------ driver and report


def load_avg():
    return os.getloadavg()[0]


def run_all(specs):
    trees = [s.split("=", 1) for s in specs]
    devices = os.environ.get("E53_DEVICES", "cpu").split(",")
    dtypes = os.environ.get("E53_DTYPES", "float64,float32").split(",")
    passes = int(os.environ.get("E53_PASSES", 2))
    core = os.environ.get("E53_CORE", "5")
    run = os.environ.get("E53_RUN", time.strftime("%Y%m%d-%H%M"))
    out_dir = CACHE / run
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"trees": dict(trees), "devices": devices, "dtypes": dtypes, "passes": passes,
            "note": os.environ.get("E53_NOTE", "")}
    for label, spec in trees:
        src = spec.partition("@")[0]
        head = subprocess.run(["git", "-C", src, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", src, "status", "--porcelain", "src"],
                               capture_output=True, text=True).stdout.strip()
        meta[f"head {label}"] = head + (" (dirty)" if dirty else "")
    (out_dir / "meta.json").write_text(json.dumps(meta))
    for k in range(passes):
        order = trees if k % 2 == 0 else trees[::-1]
        for device in devices:
            for dtype in dtypes:
                for label, spec in order:
                    src, _, jaxdir = spec.partition("@")
                    out = out_dir / f"{label}_{device}_{dtype}_pass{k}.json"
                    cmd = [sys.executable, __file__, "child", label, src, device, dtype,
                           str(out)]
                    if device == "cpu":
                        cmd = ["taskset", "-c", core] + cmd
                    env = dict(os.environ)
                    if jaxdir:
                        env["PYTHONPATH"] = jaxdir
                    print(f"pass {k} {label} {device} {dtype} (load {load_avg():.1f})",
                          flush=True)
                    subprocess.run(cmd, check=True, env=env)
    print(f"run {run} done")


def report(run, base=None):
    out_dir = CACHE / run
    meta = json.loads((out_dir / "meta.json").read_text())
    res = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_pass*.json"))]
    labels = list(meta["trees"])
    base = base or labels[0]
    lines = [f"run {run}: " + ", ".join(f"{k} = {meta.get('head ' + k, '?')}"
                                         for k in labels)]
    if meta.get("note"):
        lines.append(meta["note"])
    jaxes = sorted({f"{r['label']} {r['jax']}" for r in res})
    lines.append("jax: " + ", ".join(jaxes))
    lines.append(f"ratio = {base} ns / variant ns (above 1: faster)")
    flagged = []
    for device in meta["devices"]:
        for dtype in meta["dtypes"]:
            rs = [r for r in res if r["device"] == device and r["dtype"] == dtype]
            if not rs:
                continue
            lines.append("")
            lines.append(f"## {device} {dtype}, ns per element (N = {rs[0]['n']})")
            cells = {}
            for r in rs:
                for c in r["cells"]:
                    cells.setdefault((c["op"], c["region"], r["label"]), []).append(c["ns"])
            ops = list(dict.fromkeys(c[0] for c in cells))
            regions = list(dict.fromkeys(c[1] for c in cells))
            head = "| op | region | " + " | ".join(labels)
            head += " | " + " | ".join(f"{l} ratio" for l in labels if l != base) + " |"
            lines.append(head)
            lines.append("|" + "---|" * (2 + 2 * len(labels) - 1))
            for op in ops:
                for reg in regions:
                    med = {}
                    for l in labels:
                        v = cells.get((op, reg, l))
                        if v:
                            med[l] = float(np.median(v))
                            if max(v) / min(v) > 1.25:
                                flagged.append(f"{device} {dtype} {op} {reg} {l}: "
                                               + ", ".join(f"{u:.1f}" for u in v))
                    row = f"| {op} | {reg} | " + " | ".join(
                        f"{med[l]:.1f}" if l in med else "-" for l in labels)
                    row += " | " + " | ".join(
                        f"{med[base] / med[l]:.2f}x" if l in med and base in med else "-"
                        for l in labels if l != base) + " |"
                    lines.append(row)
            lines.append("")
            lines.append(f"compile s / temp MiB ({device} {dtype}, largest over passes)")
            for op in ops:
                parts = []
                for l in labels:
                    cs = [r["compile"][op] for r in rs if r["label"] == l and op in r["compile"]]
                    if cs:
                        parts.append(f"{l} {max(c['s'] for c in cs):.1f} / "
                                     f"{max(c['temp_mib'] for c in cs):.1f}")
                lines.append(f"  {op}: " + "; ".join(parts))
            lines.append("")
            lines.append(f"accuracy, worst per region ({device} {dtype})")
            accs = {r["label"]: r["acc"] for r in rs}
            names = list(dict.fromkeys(k for a in accs.values() for k in a))
            for name in names:
                lines.append(f"  {name}")
                for l in labels:
                    per = accs.get(l, {}).get(name)
                    if per is None:
                        continue
                    vals = [per[r] for r in REGIONS]
                    worst = max((v for v in vals if v is not None), default=None)
                    lines.append(f"    {l:>8}: worst {worst:.2g} | " + " ".join(
                        "-" if v is None else f"{v:.1g}" for v in vals))
    lines.append("")
    lines.append("regions: " + ", ".join(REGIONS))
    lines.append("passes differing by more than 1.25x: " + ("none" if not flagged else ""))
    lines += [f"  {f}" for f in flagged]
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refs":
        refs()
    elif cmd == "child":
        child(*sys.argv[2:7])
    elif cmd == "all":
        run_all(sys.argv[2:])
    elif cmd == "report":
        report(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)
