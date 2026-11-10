"""gammaln and digamma from Chebyshev panels against jax's (XLA's Lanczos lgamma and
its digamma): the panel layout, the operations each compiles to, accuracy against
mpmath, and speed of the value and of value_and_grad on one CPU core and on the GPU,
float64 and float32.

chebax.gammaln reads lgamma from panels of 1/32 octave on [1, 64) (below 1 through
lgamma(x + 1) - log x) and uses Stirling's series from 64 up. XLA computes every
branch for every element, so the branches share one log and one division. The
derivative, digamma (also public), comes from its own panel table (PSI), whose first
octave holds digamma / (x - X0) to keep relative accuracy next to the zero X0. Read
from the derivative of
the value's series instead ("value-series derivative" below), the program returning
value and gradient reads the gathered coefficients twice, and XLA writes them to
memory instead of fusing them. The coefficients are gathered one at a time on the CPU
and as one row per element elsewhere; "other gather" is the runtime with the form it
does not use on that device.

Sections:
  layouts   series error of two panel layouts, evaluated in numpy from mpmath fits:
            equal panels (g on [1, 3) by 1/16, lgamma on [3, 13) by 1/4 and on [13, 64)
            by 1) and 1/32 octave panels (g on [1, 4), lgamma on [4, 64)), at 8 and 9
            terms; for lgamma, its series' derivative against digamma, digamma's own
            series, and that series' derivative against trigamma. Nine points a panel,
            both edges included. Digamma's error is relative, except absolute within
            0.05 of its zero at 1.4616 (relative there measures nothing).
  ops       transcendental and divide instructions in the compiled CPU programs, for
            gammaln and for chebax.digamma against jax's digamma.
  accuracy  the runtime against mpmath (50 digits) on ~4,400 points in float64:
            bands of (0, 64) and log-uniform above, 1 +- 10^-k, 2 +- 10^-k and
            X0 +- 10^-k (digamma's zero), the panel edges; all relative. chebax's
            and jax's gammaln, digamma and digamma's derivative (trigamma).
  speed     ns per element at N = 2^20 (CPU) or 2^22 (GPU), median over E44_REPS
            alternations of the best of 7 calls; value_and_grad of sum(f(x)), for
            gammaln and for digamma.
            "temp" is the compiled program's temporary memory from XLA, "compile"
            the time to trace, lower and compile it.

Usage:
  python experiments/44_gammaln.py layouts           # layouts only
  [E44_DEVICES=cpu,gpu E44_REPS=5 E44_CORE=4] python experiments/44_gammaln.py all

One CPU core is taskset's, with eigen's thread pool off; float32 runs with x64 off.

Not established: other CPUs or GPUs (consumer GPU float64 runs at 1/64 rate, so a
datacenter part will rank differently); speed inside a model, where gammaln shares a
fused loop with other work.
"""

import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "src"
X0 = 1.4616321449683622  # digamma's positive zero, rounded


def layouts():
    import mpmath as mp
    import numpy as np

    sys.path.insert(0, str(SRC))
    from chebax._src.recipes._gen_common import DPS, dct, nodes

    ch = np.polynomial.chebyshev
    mp.mp.dps = DPS

    def g(x):
        return mp.loggamma(x) / ((x - 1) * (x - 2))

    equal = [(1 + j / 16, 1 / 16, True) for j in range(32)]
    equal += [(3 + j / 4, 1 / 4, False) for j in range(40)]
    equal += [(13 + j, 1.0, False) for j in range(51)]
    octave = [
        (2**e * (1 + j / 32), 2**e / 32, e < 2) for e in range(6) for j in range(32)
    ]
    s_eval = np.cos(np.pi * np.arange(9) / 8)
    refs = {}

    def ref(x):
        if x not in refs:
            m = mp.mpf(x)
            refs[x] = tuple(
                float(v) for v in (mp.loggamma(m), mp.digamma(m), mp.psi(1, m))
            )
        return refs[x]

    print("series error, max over nine points a panel (edges included)")
    print(
        f"{'layout':10s} {'terms':>5s} {'lgamma':>9s} {'its deriv':>17s} "
        f"{'digamma':>17s} {'its deriv':>17s}"
    )
    for name, panels in (("equal", equal), ("octave", octave)):
        fits = []
        for a, h, gform in panels:
            xs = [mp.mpf(a) + mp.mpf(h) * (u + 1) / 2 for u in nodes(mp, 24)]
            f = g if gform else mp.loggamma
            fits.append(
                (
                    [float(v) for v in dct(mp, [f(x) for x in xs])[:9]],
                    [float(v) for v in dct(mp, [mp.digamma(x) for x in xs])[:9]],
                )
            )
        for n in (8, 9):
            worst = np.zeros(4)
            where = [0.0] * 4
            for (a, h, gform), (cl, cp) in zip(panels, fits):
                cl, cp = np.array(cl[:n]), np.array(cp[:n])
                for s in s_eval:
                    x = a + h * (s + 1) / 2
                    sl = 2 * (x - a) / h - 1
                    lv, dv, tv = ref(x)
                    f, df = ch.chebval(sl, cl), ch.chebval(sl, ch.chebder(cl)) * 2 / h
                    if gform:
                        w, dw = (x - 1) * (x - 2), 2 * x - 3
                        f, df = w * f, dw * f + w * df
                    p, dp = ch.chebval(sl, cp), ch.chebval(sl, ch.chebder(cp)) * 2 / h
                    near = abs(x - X0) < 0.05
                    errs = (
                        0.0 if lv == 0 else abs(f - lv) / abs(lv),
                        abs(df - dv) / (1.0 if near else abs(dv)),
                        abs(p - dv) / (1.0 if near else abs(dv)),
                        abs(dp - tv) / abs(tv),
                    )
                    for k, e in enumerate(errs):
                        if e > worst[k]:
                            worst[k], where[k] = e, x
            cells = [f"{worst[0]:9.1e}"] + [
                f"{worst[k]:7.1e} at {where[k]:6.3g}" for k in (1, 2, 3)
            ]
            print(f"{name:10s} {n:5d} " + " ".join(cells))


def _gather(form, panels, i):
    import jax.numpy as jnp

    if form == "row":
        rows = jnp.take(panels, i, axis=0, mode="clip")
        return [rows[..., k] for k in range(panels.shape[1])]
    return [jnp.take(panels[:, k], i, mode="clip") for k in range(panels.shape[1])]


def _other_gather(device):
    """The runtime with the gather form it does not use on this device: one gather
    of each element's row of coefficients on the CPU, one gather per coefficient
    elsewhere."""
    import jax

    from chebax._src.recipes import gammaln as G

    series = G._row_series if device == "cpu" else G._per_coefficient_series

    def with_form(fn):
        def g(x):
            saved, G._series = G._series, series
            try:
                return fn(x)
            finally:
                G._series = saved

        return g

    lgamma, digamma = with_form(G._lgamma), with_form(G._digamma)

    @jax.custom_jvp
    def f(x):
        return lgamma(x)

    @f.defjvp
    def _(primals, tangents):
        (x,), (dx,) = primals, tangents
        return lgamma(x), digamma(x) * dx

    return f


def _value_series_derivative():
    """The rejected design: digamma from the derivative of the value's own series,
    sharing the gathers, the log and the division."""
    import jax
    import jax.numpy as jnp

    from chebax._src.recipes import gammaln as G

    t = G._t

    def parts(x):
        tab = G._tables()
        big, small, xl, y, i, s = G._locate(x)
        it, mb, bias = (
            (jnp.int64, 52, 1023) if y.dtype == jnp.float64 else (jnp.int32, 23, 127)
        )
        e = (jax.lax.bitcast_convert_type(y, it) >> mb) - bias
        scale = jax.lax.bitcast_convert_type((bias - e) << mb, y.dtype)
        c = jax.lax.platform_dependent(
            tab.lgamma,
            i,
            cpu=lambda p, i: _gather("per-coefficient", p, i),
            default=lambda p, i: _gather("row", p, i),
        )
        b1 = b2 = d1 = d2 = jnp.zeros_like(s)
        for ck in c[:0:-1]:
            d1, d2 = 2.0 * b1 + 2.0 * s * d1 - d2, d1
            b1, b2 = ck - b2 + 2.0 * s * b1, b1
        f = c[0] - b2 + s * b1
        df = (b1 + s * d1 - d2) * (2.0 * t.K) * scale
        log_x, inv = jnp.log(xl), 1.0 / xl
        gform = y < G._G_END
        q = jnp.where(small, xl, xl - 1.0)
        w = jnp.where(gform, q * (q - 1.0), 1.0)
        dw = jnp.where(gform, 2.0 * q - 1.0, 0.0)
        xb = jnp.where(big, xl, t.XB)
        ib = jnp.where(big, inv, 1.0 / t.XB)
        lb = jnp.where(big, log_x, G._LOG_XB)
        p = ib * ib
        stirling = (
            (xb - 0.5) * (lb - 1.0) + t.STIRLING_C + ib * G._horner(p, tab.stirling)
        )
        dstirling = lb - 0.5 * ib + p * G._horner(p, tab.dstirling)
        ok = x >= 0.0
        value = jnp.where(big, stirling, w * f - jnp.where(small, log_x, 0.0))
        deriv = jnp.where(big, dstirling, dw * f + w * df - jnp.where(small, inv, 0.0))
        return jnp.where(ok, value, jnp.nan), jnp.where(ok, deriv, jnp.nan)

    @jax.custom_jvp
    def f(x):
        return parts(x)[0]

    @f.defjvp
    def _(primals, tangents):
        (x,), (dx,) = primals, tangents
        v, d = parts(x)
        return v, d * dx

    return f


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
    families = {
        "gammaln": {
            "chebax": chebax.gammaln,
            "other gather": _other_gather(device),
            "value-series derivative": _value_series_derivative(),
            "jax": jax.scipy.special.gammaln,
        },
        "digamma": {"chebax": chebax.digamma, "jax": jax.scipy.special.digamma},
    }
    tag = f"[{device} {dtype}]"
    if device == "cpu":
        for family, funs in families.items():
            _ops(f"{tag} {family}", funs, jax, jnp)
        _accuracy(tag, families, dt, jax, jnp, np)
    for family, funs in families.items():
        _speed(f"{tag} {family}", device, funs, dt, jax, jnp, np)


def _vg(f, jax, jnp):
    return jax.value_and_grad(lambda x: jnp.sum(f(x)))


def _ops(tag, funs, jax, jnp):
    import collections
    import re

    names = ("log", "log-plus-one", "divide", "sine", "cosine", "exponential")
    x = jnp.linspace(0.5, 100.0, 1024)
    print(f"{tag} instructions in the compiled program")
    for name, f in funs.items():
        if name == "other gather":
            continue
        for mode, g in (("value", f), ("value_and_grad", _vg(f, jax, jnp))):
            if mode == "value" and name == "value-series derivative":
                continue
            txt = jax.jit(g).lower(x).compile().as_text()
            c = collections.Counter(
                m.group(1)
                for m in (re.search(r"= \S+ (\S+?)\(", ln) for ln in txt.splitlines())
                if m and m.group(1) in names
            )
            print(f"  {name:24s} {mode:15s} {dict(sorted(c.items()))}")


def _accuracy(tag, families, dt, jax, jnp, np):
    import mpmath as mp

    mp.mp.dps = 50
    f64 = dt == np.float64
    rng = np.random.default_rng(44)
    pts = [
        np.geomspace(1e-30, 1, 400),
        rng.uniform(0, 1, 400),
        rng.uniform(1, 4, 800),
        rng.uniform(4, 16, 800),
        rng.uniform(16, 64, 800),
        np.geomspace(64, 1e30, 400),
        rng.uniform(60, 70, 200),
    ]
    for c in (1.0, 2.0, X0):
        for k in range(1, 16 if f64 else 8):
            pts.append(np.array([c - 10.0**-k, c + 10.0**-k]))
    x0 = np.asarray(X0, dt)
    pts.append(np.array([x0, np.nextafter(x0, 0), np.nextafter(x0, 2)]))
    edges = np.concatenate([2.0**e * (1 + np.arange(33) / 32) for e in range(6)])
    pts += [edges, np.nextafter(edges, 0), np.nextafter(edges, 100)]
    x = np.unique(np.concatenate(pts).astype(dt))
    x = x[x > 0]
    ref = [
        np.array([float(f(mp.mpf(float(v)))) for v in x])
        for f in (mp.loggamma, mp.digamma, lambda v: mp.psi(1, v))
    ]
    xj = jnp.asarray(x)
    got = {}
    for name in ("chebax", "jax"):
        lg, dg = families["gammaln"][name], families["digamma"][name]
        fs = (lg, dg, jax.vmap(jax.grad(dg)))
        got[name] = [np.asarray(jax.jit(g)(xj)) for g in fs]
    zero = ref[0] == 0
    assert np.all(got["chebax"][0][zero] == 0), "lgamma(1) and lgamma(2) must be 0"
    # x^2 below the smallest normal flushes to 0, so trigamma overflows early there
    ovf = np.abs(ref[2]) > 1 / np.finfo(dt).tiny
    for k in range(3):
        bad = ~np.isfinite(got["chebax"][k]) & ~(ovf & (k == 2))
        assert not bad.any(), (k, x[bad])
    print(f"{tag} max relative error against mpmath (trigamma: digamma's derivative)")
    print(
        f"  {'x':16s} {'n':>5s} {'lgamma cheb':>20s} {'jax':>20s} "
        f"{'digamma cheb':>20s} {'jax':>20s} {'trigamma cheb':>20s} {'jax':>20s}"
    )
    for lo, hi in ((0, 1), (1, 4), (4, 16), (16, 64), (64, np.inf)):
        m = (x >= lo) & (x < hi) & ~zero & ~ovf
        cells = []
        for k in range(3):
            den = np.abs(ref[k])
            for name in ("chebax", "jax"):
                with np.errstate(all="ignore"):
                    e = np.abs(got[name][k][m].astype(float) - ref[k][m]) / den[m]
                e = np.where(np.isnan(e), np.inf, e)
                i = np.argmax(e)
                cells.append(f"{e[i]:9.1e} at {x[m][i]:8.4g}")
        print(
            f"  [{lo:g}, {hi:g}){'':{14 - len(f'[{lo:g}, {hi:g})')}s} {m.sum():5d} "
            + " ".join(cells)
        )


def _speed(tag, device, funs, dt, jax, jnp, np):
    n = 2**20 if device == "cpu" else 2**22
    reps = int(os.environ.get("E44_REPS", 5))
    rng = np.random.default_rng(44)
    dists = {
        "uniform [1, 13)": rng.uniform(1, 13, n),
        "log-uniform [1e-3, 1e4)": np.exp(rng.uniform(np.log(1e-3), np.log(1e4), n)),
    }
    print(f"{tag} ns per element, N = {n}")
    for dname, xs in dists.items():
        x = jax.device_put(jnp.asarray(xs, dt))
        for mode in ("value", "value_and_grad"):
            comps, compile_s = {}, {}
            for name, f in funs.items():
                if mode == "value" and name == "value-series derivative":
                    continue
                # a new function each time, so jax's cache does not serve the compile
                g = (lambda x, f=f: f(x)) if mode == "value" else _vg(f, jax, jnp)
                t0 = time.perf_counter()
                comps[name] = jax.jit(g).lower(x).compile()
                compile_s[name] = time.perf_counter() - t0
                jax.block_until_ready(comps[name](x))
            times = {k: [] for k in comps}
            for _ in range(reps):
                for k, c in comps.items():
                    best = np.inf
                    for _ in range(7):
                        t0 = time.perf_counter()
                        jax.block_until_ready(c(x))
                        best = min(best, time.perf_counter() - t0)
                    times[k].append(best)
            jx = np.median(times["jax"])
            for k, c in comps.items():
                t = np.median(times[k])
                temp = c.memory_analysis().temp_size_in_bytes / 2**20
                print(
                    f"  {dname:24s} {mode:15s} {k:24s} {t / n * 1e9:7.3f} ns "
                    f"{jx / t:5.2f}x jax  temp {temp:5.0f} MiB  "
                    f"compile {compile_s[k]:4.2f} s"
                )


def main():
    layouts()
    devices = os.environ.get("E44_DEVICES", "cpu,gpu").split(",")
    core = os.environ.get("E44_CORE", "4")
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
    if sys.argv[1] == "layouts":
        layouts()
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        main()
