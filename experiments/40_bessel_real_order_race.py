#!/usr/bin/env python3
"""chebax's Bessel functions of real order against TFP and scipy: accuracy
and speed of values and gradients, float64 and float32.

jax.scipy has no Bessel function of real order (only i0, i1, i0e, i1e and
integer-order bessel_jn, experiments/33), so the counterparts are
tensorflow_probability.substrates.jax.math (tfp-nightly 0.26.0-dev20261107,
the release that imports under jax 0.12; it has I and K only) and scipy.special
on the CPU (values only, no jit, no gradient).

WHAT THIS MEASURES
------------------
    jv      chebax.besselj_frozen(v)            scipy jv
    yv      chebax.bessely_frozen(v)            scipy yv
    iv      chebax.besseli(nu, x)               tfp bessel_ive(nu, x) e^x, scipy iv
    ive     besseli(nu, x, scaled=True)         tfp bessel_ive, scipy ive
    kv      chebax.besselk(nu, x)               tfp bessel_kve(nu, x) e^-x, scipy kv
    kve     besselk(nu, x, scaled=True)         tfp bessel_kve, scipy kve
    logkv   chebax.log_besselk(nu, x)           tfp log_bessel_kve - x, scipy log(kve) - x
    iratio  chebax.besseli_ratio(nu, x),        tfp bessel_iv_ratio(nu + 1, x),
            I_{nu+1} / I_nu                     scipy ive(nu + 1) / ive(nu)

J and Y take a concrete order (_frozen); I and K a traced one, and their
_frozen forms (iv, ive, kv) are timed beside it.

1. Accuracy against mpmath at 40 dps, orders 0.3, 1.7, 4.5, 6.3, 8.9, x from
   1e-6 to 1e4 (iv and kv to 700, past which I and K leave float64). Grids are
   rounded to float32 first, so one reference serves both dtypes. Values,
   d/dx and, for traced orders, d/dnu, per point (vmap of grad), CPU and GPU,
   the worse of the two shown; scipy on the CPU, values only. Metrics follow
   the test files: J and J' sup-normalized per order, Y and Y' modulus-relative
   (err / sqrt(J^2 + Y^2), and the same in the derivatives), both split at
   x = 50; log K error / max(1, |ln K|); everything else relative (a window of
   1% around an interior zero of the reference skipped). Points where the
   reference is outside the dtype's normal range are skipped (for a
   derivative, also where the value is); a nan or inf where the reference is
   finite is counted.
2. Speed under jit, as experiments/33: the sum of values and value_and_grad of
   the sum in x (chebax also in nu and x), ns per element (median), one pinned
   CPU core at N = 2^16 and the GPU at N = 2^20, orders 1.7 and 6.3, x ~
   U(0, 50); traced orders are 0-d arrays. scipy on the same core and N, on
   the same draws. Every cell is timed in two passes over all cells, minutes
   apart, and the faster pass is kept: other jobs on the laptop slowed single
   cells by up to 20x in a one-pass run. Cells whose passes differ by more
   than 1.25x are listed in the results.
3. Compile check: the ratio's value_and_grad in nu and x together does not
   finish compiling on the CPU in 300 s, so the speed run skips that cell.
   Mode "compile" times the ratio's gradient compiles by argument and device,
   each in a child process stopped at 300 s.

FINDINGS
--------
(results/40_bessel_real_order_race.txt, jax 0.12.0.1, RTX 3080 Laptop GPU)

Speed, chebax against TFP, value / value_and_grad in x:

                 float64 CPU     float64 GPU     float32 CPU     float32 GPU
    I            15x / 15x       15x / 15x       5.6-5.8x /      140-160x /
                                                 5.8-6x          110x
    I scaled     17x / 18x       16x / 16-17x    6.3-6.4x /      150-170x /
                                                 6.2-6.4x        130x
    K, K scaled  11x / 12x       10-11x / 11x    2.7-2.8x /      50-54x /
                                                 3-3.3x          58-61x
    log K        13-14x /        13-14x /        4-4.6x /        72-78x /
                 26-29x          20-21x          7.7-9x          100-110x
    I ratio      1.1-1.2x /      0.89-0.96x /    0.64-0.68x /    23-24x /
                 0.76-0.86x      0.88-0.97x      0.52-0.56x      6.4-6.8x

TFP's ratio is faster than chebax's except on the float32 GPU and for
float64 CPU values; on the CPU chebax's ratio gradient costs 2.4-2.9x its
value. Against scipy's values on one core, chebax's values are 3.7-16x
faster in float64 (J 9.7-10x, Y 8.7-16x, I 5.6-8.2x, K 3.7-5.4x, log K
4.4-6.3x, ratio 10-11x) and 7.9-34x in float32, where scipy takes as long
as in float64. The _frozen forms cost the same as the traced ones within
10%, except on the float32 GPU, where the frozen I is 1.3-1.6x faster and
the frozen K 1.1-1.15x. The gradient in nu and x together costs,
against the x-gradient alone, 4.9-6.1x for I on the CPU, 3.3-3.8x on the
float64 GPU and 15x on the float32 GPU; for K and log K 1.5-1.9x, and 5-5.4x
on the float32 GPU.

Compiling the ratio's value_and_grad in nu and x does not finish in 300 s on
the CPU in either dtype, where x alone takes 2 s and nu alone 9 s; the GPU
compiles it in 41 s (float64) and 56 s (float32).

Accuracy, float64: chebax's values are within 1.4e-15 (J, sup), 1.2e-14 (Y,
modulus), 5.7e-15 (I; 1.1e-15 frozen), 1.2e-14 (K), 3.7e-15 (log K) and
6.1e-15 (ratio). x-derivatives are within 4.8e-14 (ratio 1.9e-12). The
nu-derivatives are within 1.8e-13 for I, 6.4e-12 for K scaled and log K and
1.2e-12 for the ratio, the last three at x near 1e4; K's is within 3.5e-13
below x = 600, and 2.1e-9 on the CPU at x = 680, where dK/dnu is 9e-301.
TFP's values are within 4e-15 and its x-derivatives within 3.4e-11 (ratio
2.1e-8); its nu-derivative is 0 everywhere, for every function. scipy's
values are within 4.5e-14.

Accuracy, float32: chebax's values are within 5.7e-7 (J), 3.2e-6 (Y, and I
at x = 84.9), 6.6e-7 (I scaled), 6.5e-6 (K, K scaled), 2.9e-6 (log K) and
1.2e-6 (ratio). K' is inf at 2 points where |K'| is 2e38 or more, near the
top of the range, and off by 1.1e-2 at x = 84.9, v = 8.9, where K is 3e-38
(chebax returns -K there); K scaled' is within 9.9e-6 apart from the same
2 points. nu-derivatives lose accuracy where they are a small part of the
value: 3.6e-4 to 4.3e-4 at x near 1e4 for K scaled, log K and the ratio.
TFP returns 0 for I = 1.0e-37 (x = 8.4e-6, v = 6.3), its K scaled' is off
by 1.2e-2 at x = 7069 and its ratio' has relative error 14 at x = 1e4.
scipy has no log K or ratio: log(kve) - x is inf at 20 points (x <= 2e-4)
and ive(v + 1) / ive(v) nan at 12 (x <= 5e-5), where kve overflows or ive
underflows.

WHAT IT DOES NOT ESTABLISH
--------------------------
Other hardware, orders above 10 (outside every table; the _frozen forms
raise there, traced orders are not checked), per-element orders, or whole
models. TFP's and scipy's loops cost what their slowest lane costs,
so their times move with the input distribution.

Run (each stage writes into results/40_cache/, not checked in), with jax
0.12.0.1 and tfp-nightly first on PYTHONPATH (pip --target installs):

  python experiments/40_bessel_real_order_race.py refs              (~10 s)
  JAX_PLATFORMS=cuda,cpu taskset -c 3 \\
      python experiments/40_bessel_real_order_race.py run f64       (~20 min)
  ... run f32                                                       (~21 min)
  taskset -c 3 python experiments/40_bessel_real_order_race.py scipy     (~1 min)
  JAX_PLATFORMS=cuda,cpu taskset -c 3 \\
      python experiments/40_bessel_real_order_race.py compile       (~12 min)
  python experiments/40_bessel_real_order_race.py report > results/40_bessel_real_order_race.txt
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "40_cache")
SIZES = {"cpu": 2**16, "gpu": 2**20}
FAMILIES = ("jv", "yv", "iv", "ive", "kv", "kve", "logkv", "iratio")
TRACED = ("iv", "ive", "kv", "kve", "logkv", "iratio")
# the ratio's value_and_grad in nu and x together does not finish compiling on
# the CPU (mode "compile"), so the speed run skips it there
SKIP = {("iratio", "chebax_all", "cpu")}
COMPILE_LIMIT = 300.0
# every speed cell is timed in this many passes over all cells, the fastest kept
PASSES = 2
DISAGREE = 1.25


def r32(v):
    return np.unique(np.asarray(v, np.float64).astype(np.float32).astype(np.float64))


ORDERS = tuple(float(r32(v)[0]) for v in (0.3, 1.7, 4.5, 6.3, 8.9))
TIMED_ORDERS = tuple(float(r32(v)[0]) for v in (1.7, 6.3))


# ---------------------------------------------------------------- grids


def base_x():
    return r32(
        np.concatenate(
            [
                # 1e-6 rounds down in float32, below the domain of K and Y
                [float(np.nextafter(np.float32(1e-6), np.float32(1)))],
                np.logspace(-6, 0, 40)[1:],
                np.linspace(0, 60, 241)[1:],
                np.logspace(np.log10(60), 4, 60)[1:],
            ]
        )
    )


def family_x(fam):
    x = base_x()
    return x[x <= 700] if fam in ("iv", "kv") else x


# ----------------------------------------------------------- references


def _ref_chunk(task):
    import mpmath as mp

    base, v, xs = task
    mp.mp.dps = 40
    V = mp.mpf(v)
    rows = []
    for xf in xs:
        X = mp.mpf(float(xf))
        if base == "J":
            jm, j0, jp = (mp.besselj(V + k, X) for k in (-1, 0, 1))
            ym, y0, yp = (mp.bessely(V + k, X) for k in (-1, 0, 1))
            jd, yd = (jm - jp) / 2, (ym - yp) / 2
            rows.append(
                {
                    "jv": {"v": j0, "d_x": jd},
                    "yv": {
                        "v": y0,
                        "d_x": yd,
                        "mod": mp.sqrt(j0**2 + y0**2),
                        "dmod": mp.sqrt(jd**2 + yd**2),
                    },
                }
            )
        elif base == "I":
            im, i0, ip, ipp = (mp.besseli(V + k, X) for k in (-1, 0, 1, 2))
            dn0 = mp.diff(lambda s: mp.besseli(s, X), V)
            dn1 = mp.diff(lambda s: mp.besseli(s, X), V + 1)
            d0, d1 = (im + ip) / 2, (i0 + ipp) / 2
            e = mp.exp(-X)
            ratio = ip / i0
            rows.append(
                {
                    "iv": {"v": i0, "d_x": d0, "d_nu": dn0},
                    "ive": {"v": e * i0, "d_x": e * (d0 - i0), "d_nu": e * dn0},
                    "iratio": {
                        "v": ratio,
                        "d_x": (d1 - ratio * d0) / i0,
                        "d_nu": (dn1 - ratio * dn0) / i0,
                    },
                }
            )
        else:
            km, k0, kp = (mp.besselk(V + k, X) for k in (-1, 0, 1))
            dn = mp.diff(lambda s: mp.besselk(s, X), V)
            d = -(km + kp) / 2
            e = mp.exp(X)
            rows.append(
                {
                    "kv": {"v": k0, "d_x": d, "d_nu": dn},
                    "kve": {"v": e * k0, "d_x": e * (d + k0), "d_nu": e * dn},
                    "logkv": {"v": mp.log(k0), "d_x": d / k0, "d_nu": dn / k0},
                }
            )
    rows = [
        {f: {q: float(val) for q, val in r.items()} for f, r in row.items()}
        for row in rows
    ]
    return base, v, xs, rows


def refs():
    from multiprocessing import Pool

    import mpmath

    x = base_x()
    tasks = [
        (base, v, x[i : i + 24])
        for base in ("J", "I", "K")
        for v in ORDERS
        for i in range(0, len(x), 24)
    ]
    t0 = time.time()
    with Pool(4) as pool:
        done = pool.map(_ref_chunk, tasks, chunksize=1)
    table = {}
    for base, v, xs, rows in done:
        for xf, row in zip(xs, rows):
            for fam, r in row.items():
                table[(fam, v, float(xf))] = r
    flat = {}
    for fam in FAMILIES:
        xs = family_x(fam)
        pts = [(v, float(xf)) for v in ORDERS for xf in xs]
        flat[f"{fam}|nu"] = np.array([p[0] for p in pts])
        flat[f"{fam}|x"] = np.array([p[1] for p in pts])
        for q in table[(fam, ORDERS[0], float(xs[0]))]:
            flat[f"{fam}|{q}"] = np.array([table[(fam,) + p][q] for p in pts])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "refs.npz"), **flat)
    print(
        f"refs: {len(tasks)} chunks, mpmath {mpmath.__version__}, {time.time() - t0:.0f} s"
    )


# --------------------------------------------------------------- run


def implementations(jnp, chebax, tfp):
    """family -> {impl: (f, argnums, static)}. static: f(v) builds g(x) for a
    concrete order, argnums index g's arguments; otherwise f(nu, x)."""
    m = tfp.math
    imp = {
        "jv": {"chebax": (chebax.besselj_frozen, (0,), True)},
        "yv": {"chebax": (chebax.bessely_frozen, (0,), True)},
        "iv": {
            "chebax": (chebax.besseli, (1,), False),
            "chebax_frozen": (chebax.besseli_frozen, (0,), True),
            "tfp": (lambda n, x: m.bessel_ive(n, x) * jnp.exp(x), (1,), False),
        },
        "ive": {
            "chebax": (lambda n, x: chebax.besseli(n, x, scaled=True), (1,), False),
            "chebax_frozen": (
                lambda v: chebax.besseli_frozen(v, scaled=True),
                (0,),
                True,
            ),
            "tfp": (m.bessel_ive, (1,), False),
        },
        "kv": {
            "chebax": (chebax.besselk, (1,), False),
            "chebax_frozen": (chebax.besselk_frozen, (0,), True),
            "tfp": (lambda n, x: m.bessel_kve(n, x) * jnp.exp(-x), (1,), False),
        },
        "kve": {
            "chebax": (lambda n, x: chebax.besselk(n, x, scaled=True), (1,), False),
            "tfp": (m.bessel_kve, (1,), False),
        },
        "logkv": {
            "chebax": (chebax.log_besselk, (1,), False),
            "tfp": (lambda n, x: m.log_bessel_kve(n, x) - x, (1,), False),
        },
        "iratio": {
            "chebax": (chebax.besseli_ratio, (1,), False),
            "tfp": (lambda n, x: m.bessel_iv_ratio(n + 1.0, x), (1,), False),
        },
    }
    for fam in TRACED:
        f, _, _ = imp[fam]["chebax"]
        imp[fam]["chebax_all"] = (f, (0, 1), False)
    return imp


def scipy_fns():
    import scipy.special as sp

    return {
        "jv": sp.jv,
        "yv": sp.yv,
        "iv": sp.iv,
        "ive": sp.ive,
        "kv": sp.kv,
        "kve": sp.kve,
        "logkv": lambda n, x: np.log(sp.kve(n, x)) - x,
        "iratio": lambda n, x: sp.ive(n + 1, x) / sp.ive(n, x),
    }


def timing_settings():
    """(label, family, order); x ~ U(0, 50) for all."""
    names = {
        "jv": "J_v",
        "yv": "Y_v",
        "iv": "I_v",
        "ive": "I_v scaled",
        "kv": "K_v",
        "kve": "K_v scaled",
        "logkv": "log K_v",
        "iratio": "I_{v+1}/I_v",
    }
    return [
        (f"{names[f]} v={v:.1f}, x ~ U(0, 50)", f, v)
        for f in FAMILIES
        for v in TIMED_ORDERS
    ]


def draws(label, n):
    seed = sum(map(ord, label)) * 7919 + n
    return np.random.default_rng(seed).uniform(0, 50, n)


def timed(call, n, reps=5, budget=1.0):
    """Median ns per element; a warm-up spin first (the GPU clock drops
    while compiling). Same as experiments/33."""
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


def _meta(dname_dtype, extra):
    commit = subprocess.run(
        ["git", "-C", HERE, "log", "-1", "--format=%h %cd", "--date=short"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "dtype": dname_dtype,
        "chebax": commit,
        "load_start": os.getloadavg(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        **extra,
    }


def run(dname_dtype):
    wide = dname_dtype == "f64"
    import jax

    jax.config.update("jax_enable_x64", wide)
    import jax.numpy as jnp
    from tensorflow_probability.substrates import jax as tfp

    import chebax

    dt = np.float64 if wide else np.float32
    imp = implementations(jnp, chebax, tfp)
    ref = np.load(os.path.join(CACHE, "refs.npz"))
    devs = {"cpu": jax.devices("cpu")[0], "gpu": jax.devices("gpu")[0]}
    meta = _meta(
        dname_dtype,
        {
            "jax": jax.__version__,
            "tfp": tfp.__version__,
            "chebax_file": chebax.__file__,
            "devices": {k: d.device_kind for k, d in devs.items()},
        },
    )
    print(json.dumps(meta), flush=True)
    outs, notes = {}, {}

    def put(v, dev):
        return jax.device_put(jnp.asarray(v, dt), dev)

    # 1. accuracy; the widest argnums of each implementation
    acc_impls = {
        fam: {k: v for k, v in d.items() if k != "chebax"} for fam, d in imp.items()
    }
    for fam in ("jv", "yv"):
        acc_impls[fam] = imp[fam]
    for fam in TRACED:
        f = imp[fam]["tfp"][0]
        # tfp's order gradient, to show what it returns
        acc_impls[fam]["tfp"] = (f, (0, 1), False)
    for dname, dev in devs.items():
        for fam, impls in acc_impls.items():
            nus, xs = ref[f"{fam}|nu"], ref[f"{fam}|x"]
            for iname, (f, argnums, static) in impls.items():
                key = f"{fam}|{iname.removesuffix('_all')}|{dname}"
                t0 = time.time()
                try:
                    qs = ["v"] + [
                        "d_nu" if (not static and i == 0) else "d_x" for i in argnums
                    ]
                    vals = {q: [] for q in qs}
                    if not static:
                        vf = jax.jit(f)
                        gf = jax.jit(jax.vmap(jax.grad(f, argnums), in_axes=(None, 0)))
                    for v in ORDERS:
                        m = nus == v
                        xx = put(xs[m], dev)
                        if static:
                            g = f(v)
                            out = [jax.jit(g)(xx)]
                            out += list(jax.jit(jax.vmap(jax.grad(g, argnums)))(xx))
                        else:
                            nu = put(v, dev)
                            out = [vf(nu, xx)] + list(gf(nu, xx))
                        for q, o in zip(qs, out):
                            vals[q].append(np.asarray(o, np.float64))
                    for q, v in vals.items():
                        outs[f"{key}|{q}"] = np.concatenate(v)
                except Exception as e:  # recorded, not fatal
                    notes[key] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                print(
                    f"acc {key} {time.time() - t0:.1f} s {notes.get(key, '')}",
                    flush=True,
                )

    # 2. speed: compile every cell, then time all of them PASSES times
    rows = []
    for dname, dev in devs.items():
        n = SIZES[dname]
        progs = []
        for label, fam, v in timing_settings():
            x = put(draws(label, n), dev)
            nu = put(v, dev)
            for iname, (f, argnums, static) in imp[fam].items():
                if static:
                    g = f(v)
                    total = lambda x, g=g: jnp.sum(g(x))  # noqa: E731
                    args = (x,)
                else:
                    total = lambda nu, x, f=f: jnp.sum(f(nu, x))  # noqa: E731
                    args = (nu, x)
                rec = {"setting": label, "impl": iname, "device": dname, "n": n}
                rows.append(rec)
                if (fam, iname, dname) in SKIP:
                    rec["skipped"] = "compile does not finish, see compile check"
                    continue
                try:
                    t0 = time.perf_counter()
                    cv = jax.jit(total).lower(*args).compile()
                    rec["compile_v"] = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    cg = (
                        jax.jit(jax.value_and_grad(total, argnums))
                        .lower(*args)
                        .compile()
                    )
                    rec["compile_g"] = time.perf_counter() - t0
                    calls = (
                        {"t_g": cg} if iname == "chebax_all" else {"t_v": cv, "t_g": cg}
                    )
                    progs.append((rec, calls, args))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        for _ in range(PASSES):
            for rec, calls, args in progs:
                if "error" in rec:
                    continue
                try:
                    for k, c in calls.items():
                        rec.setdefault(k + "_passes", []).append(
                            timed(lambda c=c: c(*args), n)
                        )
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        for rec in rows:
            for k in ("t_v", "t_g"):
                if k + "_passes" in rec:
                    rec[k] = min(rec[k + "_passes"])
            if rec["device"] == dname:
                print("time", json.dumps(rec), flush=True)
    meta["load_end"] = os.getloadavg()
    np.savez_compressed(os.path.join(CACHE, f"acc_{dname_dtype}.npz"), **outs)
    with open(os.path.join(CACHE, f"run_{dname_dtype}.json"), "w") as fh:
        json.dump({"meta": meta, "notes": notes, "timing": rows}, fh, indent=1)


def compile_one(dname_dtype, argnums, dname):
    """Child of compile_check: compile value_and_grad of sum(besseli_ratio)
    as the speed run would and print the seconds."""
    import jax

    jax.config.update("jax_enable_x64", dname_dtype == "f64")
    import jax.numpy as jnp

    import chebax

    dt = jnp.float64 if dname_dtype == "f64" else jnp.float32
    dev = jax.devices(dname)[0]
    label = timing_settings()[-2][0]  # the ratio at the first timed order
    x = jax.device_put(jnp.asarray(draws(label, SIZES[dname]), dt), dev)
    nu = jax.device_put(jnp.asarray(TIMED_ORDERS[0], dt), dev)
    total = lambda nu, x: jnp.sum(chebax.besseli_ratio(nu, x))  # noqa: E731
    argnums = tuple(int(a) for a in argnums.split(","))
    t0 = time.perf_counter()
    jax.jit(jax.value_and_grad(total, argnums)).lower(nu, x).compile()
    print(json.dumps({"compile": time.perf_counter() - t0}), flush=True)


def compile_check():
    """Compile times of the ratio's value_and_grad by argument and device,
    each in a child process stopped at COMPILE_LIMIT seconds."""
    rows = []
    for dname_dtype in ("f64", "f32"):
        for argnums, dname in (
            ("1", "cpu"),
            ("0", "cpu"),
            ("0,1", "cpu"),
            ("0,1", "gpu"),
        ):
            rec = {"dtype": dname_dtype, "argnums": argnums, "device": dname}
            cmd = [sys.executable, __file__, "compile1", dname_dtype, argnums, dname]
            try:
                p = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=COMPILE_LIMIT,
                    env=dict(os.environ),
                )
                out = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                if out:
                    rec["compile"] = json.loads(out[-1])["compile"]
                else:
                    rec["error"] = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
            except subprocess.TimeoutExpired:
                rec["timeout"] = COMPILE_LIMIT
            rows.append(rec)
            print("compile", json.dumps(rec), flush=True)
    meta = _meta("both", {"limit_s": COMPILE_LIMIT})
    with open(os.path.join(CACHE, "compile_check.json"), "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1)


def _scipy_time(f, nu, x):
    """Median seconds per call of f(nu, x), and the last output."""
    with np.errstate(all="ignore"):
        f(nu, x)
        one = time.perf_counter()
        f(nu, x)
        one = time.perf_counter() - one
        inner = max(1, int(0.2 / max(one, 1e-6)))
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(inner):
                out = f(nu, x)
            ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts)), out


def run_scipy():
    """scipy values on the CPU: accuracy on the grids, then speed on the
    timing draws, float64 and float32 inputs."""
    import scipy

    fns = scipy_fns()
    ref = np.load(os.path.join(CACHE, "refs.npz"))
    meta = _meta("both", {"scipy": scipy.__version__})
    outs, rows = {}, []
    for dname_dtype, dt in (("f64", np.float64), ("f32", np.float32)):
        with np.errstate(all="ignore"):
            for fam, f in fns.items():
                nus, xs = ref[f"{fam}|nu"].astype(dt), ref[f"{fam}|x"].astype(dt)
                outs[f"{dname_dtype}|{fam}|scipy|cpu|v"] = np.asarray(
                    f(nus, xs), np.float64
                )
        n = SIZES["cpu"]
        recs = []
        for p in range(PASSES):
            for i, (label, fam, v) in enumerate(timing_settings()):
                t, out = _scipy_time(fns[fam], dt(v), draws(label, n).astype(dt))
                if p == 0:
                    recs.append(
                        {
                            "setting": label,
                            "impl": "scipy",
                            "device": "cpu",
                            "n": n,
                            "dtype": dname_dtype,
                            "out_dtype": str(out.dtype),
                            "t_v_passes": [],
                        }
                    )
                recs[i]["t_v_passes"].append(t / n * 1e9)
        for rec in recs:
            rec["t_v"] = min(rec["t_v_passes"])
            print("time", json.dumps(rec), flush=True)
        rows += recs
    meta["load_end"] = os.getloadavg()
    np.savez_compressed(os.path.join(CACHE, "acc_scipy.npz"), **outs)
    with open(os.path.join(CACHE, "run_scipy.json"), "w") as fh:
        json.dump({"meta": meta, "timing": rows}, fh, indent=1)


# ------------------------------------------------------------ report

REL = ("rel",)
METRICS = {
    "jv": {"v": ("sup",), "d_x": ("sup",)},
    "yv": {"v": ("mod",), "d_x": ("dmod",)},
    "logkv": {"v": ("lrel",), "d_x": REL, "d_nu": REL},
    **{
        f: {"v": REL, "d_x": REL, "d_nu": REL}
        for f in ("iv", "ive", "kv", "kve", "iratio")
    },
}
SPLIT = (("x<=50", lambda x: x <= 50), ("x>50", lambda x: x > 50))
SLICES = {"jv": SPLIT, "yv": SPLIT}
IMPLS = ("chebax", "chebax_frozen", "tfp", "scipy")


def _zero_windows(x, r, case):
    """Points within 1% of an interior sign change of r, per case."""
    skip = np.zeros(len(x), bool)
    for c in np.unique(case):
        idx = np.where(case == c)[0]
        idx = idx[np.argsort(x[idx])]
        xs, rs = x[idx], r[idx]
        for i in np.where(np.sign(rs[:-1]) * np.sign(rs[1:]) < 0)[0]:
            x0 = xs[i] - rs[i] * (xs[i + 1] - xs[i]) / (rs[i + 1] - rs[i])
            skip |= (case == c) & (np.abs(x - x0) <= 0.01 * max(abs(x0), 1e-3))
    return skip


def score(metric, y, r, x, case, scale, dt, val=None):
    """(worst error, nan count, points scored, index of the worst); val is
    the reference value for a derivative, which must be in range too."""
    fi = np.finfo(dt)

    def normal(a):
        return (
            np.isfinite(a) & (np.abs(a) <= fi.max) & ((a == 0) | (np.abs(a) >= fi.tiny))
        )

    ok = normal(r) if val is None else normal(r) & normal(val)
    if metric == "rel":
        ok &= (r != 0) & ~_zero_windows(x, r, case)
    bad = ok & ~np.isfinite(y)
    good = ok & np.isfinite(y)
    with np.errstate(all="ignore"):
        err = np.abs(y - r)
        if metric == "rel":
            err = err / np.abs(r)
        elif metric == "lrel":
            err = err / np.maximum(1.0, np.abs(r))
        elif metric in ("mod", "dmod"):
            err = err / scale
        elif metric == "sup":
            for c in np.unique(case):
                m = case == c
                err[m] = err[m] / np.max(np.abs(r[m & ok]))
    assert not np.isnan(err[good]).any(), metric
    err = np.where(good, err, -1.0)
    i = int(np.argmax(err))
    return float(err[i]), int(bad.sum()), int(ok.sum()), i


def report():
    ref = np.load(os.path.join(CACHE, "refs.npz"))
    runs = {}
    for dtn in ("f64", "f32"):
        with open(os.path.join(CACHE, f"run_{dtn}.json")) as fh:
            runs[dtn] = json.load(fh)
        runs[dtn]["acc"] = np.load(os.path.join(CACHE, f"acc_{dtn}.npz"))
    with open(os.path.join(CACHE, "run_scipy.json")) as fh:
        sci = json.load(fh)
    sci_acc = np.load(os.path.join(CACHE, "acc_scipy.npz"))
    print("chebax's Bessel functions of real order against TFP and scipy")
    for dtn, r in runs.items():
        m = r["meta"]
        print(
            f"  run {dtn}: {m['date']}, jax {m['jax']}, tfp {m['tfp']}, {m['devices']}, "
            f"chebax {m['chebax']}, load {m['load_start'][0]:.1f} -> {m['load_end'][0]:.1f}"
        )
        for k, v in r["notes"].items():
            print(f"    error {k}: {v}")
    m = sci["meta"]
    print(
        f"  run scipy: {m['date']}, scipy {m['scipy']}, load "
        f"{m['load_start'][0]:.1f} -> {m['load_end'][0]:.1f}"
    )
    print(f"  accuracy orders {ORDERS}, timed orders {TIMED_ORDERS}")

    for dtn, dt in (("f64", np.float64), ("f32", np.float32)):
        print(
            f"\n=== ACCURACY {dtn} (worst over CPU and GPU; scipy CPU, values only; "
            "'(k nan)' = k points nan/inf where the reference is finite)"
        )
        hdr = ["family", "quantity", "metric", *IMPLS]
        lines, where = [hdr], []
        for fam in FAMILIES:
            x, nus = ref[f"{fam}|x"], ref[f"{fam}|nu"]
            case = np.searchsorted(ORDERS, nus)
            for q, metrics in METRICS[fam].items():
                r = ref[f"{fam}|{q}"]
                # log K and the ratio stay in range where K and I do not
                val = (
                    ref[f"{fam}|v"]
                    if q != "v" and fam not in ("logkv", "iratio")
                    else None
                )
                for metric in metrics:
                    scale = (
                        ref[f"{fam}|{metric}"] if metric in ("mod", "dmod") else None
                    )
                    for sname, sel in SLICES.get(
                        fam, (("", lambda x: np.ones(len(x), bool)),)
                    ):
                        sl = sel(x)
                        row = [fam, q + (f" [{sname}]" if sname else ""), metric]
                        for iname in IMPLS:
                            if iname == "scipy":
                                srcs = [sci_acc] if q == "v" else []
                                keys = [f"{dtn}|{fam}|scipy|cpu|{q}"]
                            else:
                                srcs = [runs[dtn]["acc"]] * 2
                                keys = [
                                    f"{fam}|{iname}|{d}|{q}" for d in ("cpu", "gpu")
                                ]
                            worst, nbad, at, found = -1.0, 0, None, False
                            for src, key in zip(srcs, keys):
                                if key not in src:
                                    continue
                                found = True
                                e, b, _, i = score(
                                    metric,
                                    src[key][sl],
                                    r[sl],
                                    x[sl],
                                    case[sl],
                                    None if scale is None else scale[sl],
                                    dt,
                                    None if val is None else val[sl],
                                )
                                nbad += b
                                if e > worst:
                                    dev = key.split("|")[-2]
                                    worst, at = e, (
                                        dev,
                                        float(nus[sl][i]),
                                        float(x[sl][i]),
                                    )
                            if not found:
                                row.append("-")
                                continue
                            cell = f"{worst:.1e}" if worst >= 0 else "all bad"
                            if nbad:
                                cell += f" ({nbad} nan)"
                            row.append(cell)
                            where.append(
                                f"  {fam} {row[1]} {metric} {iname}: {cell} at {at}"
                            )
                        lines.append(row)
        widths = [max(len(str(ln[i])) for ln in lines) for i in range(len(hdr))]
        for ln in lines:
            print("  ".join(str(c).ljust(w) for c, w in zip(ln, widths)))
        print(f"\n  worst points (device, nu, x), {dtn}:")
        print("\n".join(where))

    print(
        "\n=== SPEED, ns per element (median): value / value_and_grad in x; "
        "compile s value + grad; scipy values only, one core, same draws"
    )
    for dtn in ("f64", "f32"):
        for dname in ("cpu", "gpu"):
            print(f"\n--- {dtn} {dname} (N = {SIZES[dname]})")
            hdr = ["setting", "chebax", "chebax frozen", "tfp"]
            hdr += (
                ["scipy", "chebax vs tfp", "chebax vs scipy"]
                if dname == "cpu"
                else ["chebax vs tfp"]
            )
            lines = [hdr]
            for label, fam, _ in timing_settings():
                cells = {
                    rec["impl"]: rec
                    for rec in runs[dtn]["timing"]
                    if rec["setting"] == label and rec["device"] == dname
                }
                if dname == "cpu":
                    for rec in sci["timing"]:
                        if rec["setting"] == label and rec["dtype"] == dtn:
                            cells["scipy"] = rec

                def fmt(rec):
                    if rec is None:
                        return "-"
                    if "error" in rec:
                        return "ERR " + rec["error"][:40]
                    if "skipped" in rec:
                        return "skipped (compile check)"
                    tv = f"{rec['t_v']:.3g}" if "t_v" in rec else "-"
                    if rec["impl"] == "scipy":
                        return tv
                    return f"{tv} / {rec['t_g']:.3g} ({rec['compile_v']:.1f}+{rec['compile_g']:.1f}s)"

                c = cells.get("chebax")
                row = [
                    label,
                    fmt(c),
                    fmt(cells.get("chebax_frozen")),
                    fmt(cells.get("tfp")),
                ]
                if dname == "cpu":
                    row.append(fmt(cells.get("scipy")))
                t = cells.get("tfp")
                ok = c is not None and "t_v" in c
                row.append(
                    f"{t['t_v'] / c['t_v']:.2g}x / {t['t_g'] / c['t_g']:.2g}x"
                    if ok and t and "t_v" in t
                    else "-"
                )
                if dname == "cpu":
                    s = cells.get("scipy")
                    row.append(f"{s['t_v'] / c['t_v']:.2g}x" if ok and s else "-")
                lines.append(row)
                ca = cells.get("chebax_all")
                if ca:
                    lines.append(
                        ["  chebax, grad in nu and x", fmt(ca)] + ["-"] * (len(hdr) - 2)
                    )
            widths = [max(len(str(ln[i])) for ln in lines) for i in range(len(hdr))]
            for ln in lines:
                print("  ".join(str(c).ljust(w) for c, w in zip(ln, widths)))
    dts = {(r["dtype"], r["out_dtype"]) for r in sci["timing"]}
    print(f"\n  scipy input -> output dtypes: {sorted(dts)}")

    print(
        f"\n=== TIMING PASSES: cells whose {PASSES} passes differ by more than "
        f"{DISAGREE:.2f}x (the faster is shown above)"
    )
    recs = [(dtn, rec) for dtn in ("f64", "f32") for rec in runs[dtn]["timing"]]
    recs += [(rec["dtype"], rec) for rec in sci["timing"]]
    worst, listed = 1.0, 0
    for dtn, rec in recs:
        for k in ("t_v", "t_g"):
            ps = rec.get(k + "_passes")
            if not ps:
                continue
            r = max(ps) / min(ps)
            worst = max(worst, r)
            if r > DISAGREE:
                listed += 1
                print(
                    f"  {dtn} {rec['device']} {rec['setting']} {rec['impl']} {k}: "
                    + " / ".join(f"{t:.3g}" for t in ps)
                )
    print(f"  {listed} listed; largest ratio over all cells {worst:.2f}x")

    with open(os.path.join(CACHE, "compile_check.json")) as fh:
        cc = json.load(fh)
    m = cc["meta"]
    print(
        "\n=== COMPILE CHECK: chebax.besseli_ratio, value_and_grad of the sum, "
        f"N as the speed run, child stopped at {m['limit_s']:.0f} s "
        f"({m['date']}, load {m['load_start'][0]:.1f})"
    )
    names = {"1": "x", "0": "nu", "0,1": "nu and x"}
    for r in cc["rows"]:
        if "compile" in r:
            res = f"{r['compile']:.1f} s"
        elif "timeout" in r:
            res = f"not finished at {r['timeout']:.0f} s"
        else:
            res = "ERR " + r["error"]
        print(f"  {r['dtype']} {r['device']} grad in {names[r['argnums']]}: {res}")


if __name__ == "__main__":
    {
        "refs": refs,
        "run": lambda: run(sys.argv[2]),
        "scipy": run_scipy,
        "compile": compile_check,
        "compile1": lambda: compile_one(*sys.argv[2:5]),
        "report": report,
    }[sys.argv[1]]()
