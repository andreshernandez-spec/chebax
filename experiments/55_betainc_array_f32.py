"""betainc_array in float32: accuracy against the orientation split and the
number of coefficients read (NKEEP32), and speed against NKEEP32.

Points: float32-rounded (a, b, x), a and b log-uniform on the box [ALO,
100]^2 (ALO = 10^-2.125), x from the bulk (Beta draws), within 1e-3 of x_s,
the lower tail (x_s 10^-U(1, 30)), the upper tail (1 - x = (1 - x_s)
10^-U(1, 7)), the band past x_s where small-b lanes stay in the lower
orientation, and b below 0.1 to float32's last x below 1; x in float32's
normal range. Metrics as tests/test_betainc_array.py: I absolute,
ln I relative to max(1, |ln I|) where I > 1e-30.

The reference for all points is float64 betainc_array on the same float32
values (7.2e-16 against mpmath, tests), ~1e-8 of the errors measured here;
the 40 worst points of each metric and setting are then re-checked against
mpmath at 40 dps, and the mpmath column is what docstrings quote.

Settings: split S_SPLIT (float64's) or S_SPLIT32 (float32's, the shipped
one); NKEEP32 from tolerances on the dropped coefficients' sum (the shipped
TOL32 is 4e-7). The table and the split arrays are those of
betainc_array_table; only how many coefficients float32 reads changes.

    python experiments/55_betainc_array_f32.py acc                 # ~6 min, CPU
    python experiments/55_betainc_array_f32.py gpu                 # speed
    taskset -c 5 python experiments/55_betainc_array_f32.py cpu    # speed, one core

results/55_betainc_array_f32_acc.txt is jax 0.11; _acc_jax053.txt the same
run on jax 0.5.3 (CI's floor, from a pip --target directory first on
PYTHONPATH).

Speed: ns per element of value and value_and_grad in a, b, x, x64 off, a, b
log-uniform on the box and on [5, 30], x ~ Beta; two passes, min kept,
spread above 1.25x marked *.

Does NOT establish: float64 accuracy (the tests), accuracy outside these
points' kinds, or speed inside a model.
"""

import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "acc"
if MODE in ("acc", "cpu"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", MODE == "acc")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes import betainc_array as m  # noqa: E402
from chebax._src.recipes import betainc_array_table as T  # noqa: E402

F32 = np.float32
TOLS = (2e-8, 2e-7, 4e-7, 8e-7, 1.6e-6, 3.2e-6)
SPLITS = {"S_SPLIT32": T.S_SPLIT32, "S_SPLIT": T.S_SPLIT}


def nkeep(tol):
    A = np.abs(np.asarray(T.TABLE, np.float64))
    tail = np.cumsum(A[:, ::-1], axis=1)[:, ::-1]
    tail = np.concatenate([tail, np.zeros((A.shape[0], 1))], axis=1)
    return int(np.nonzero(np.all(tail <= tol, axis=0))[0].min())


def setting(n, split):
    T.NKEEP32 = n
    T.S_SPLIT32 = split
    m._table.cache_clear()


def points(n=40000, seed=11):
    rng = np.random.default_rng(seed)
    lo = np.log(T.ALO)
    a = np.exp(rng.uniform(lo, np.log(100.0), n))
    b = np.exp(rng.uniform(lo, np.log(100.0), n))
    xs = (a + 1.0) / (a + b + 2.0)
    kind = rng.integers(0, 4, n)
    x = np.select(
        [kind == 0, kind == 1, kind == 2],
        [
            rng.beta(a, b),
            xs * (1.0 + rng.uniform(-1e-3, 1e-3, n)),
            xs * 10.0 ** -rng.uniform(1.0, 30.0, n),
        ],
        1.0 - (1.0 - xs) * 10.0 ** -rng.uniform(1.0, 7.0, n),
    )
    k = n // 4
    a2 = np.exp(rng.uniform(lo, np.log(100.0), k))
    b2 = np.exp(rng.uniform(lo, np.log(3.0), k))
    xs2 = (a2 + 1.0) / (a2 + b2 + 2.0)
    x2 = -np.expm1(rng.uniform(0.8, 7.0, k) * np.log1p(-xs2))
    k3 = n // 8
    a3 = np.exp(rng.uniform(lo, np.log(100.0), k3))
    b3 = np.exp(rng.uniform(lo, np.log(0.1), k3))
    ls3 = np.log1p(-(a3 + 1.0) / (a3 + b3 + 2.0))
    x3 = -np.expm1(rng.uniform(0.8, 24 * np.log(2.0) / -ls3) * ls3)
    a, b, x = (
        np.concatenate(v).astype(F32) for v in ((a, a2, a3), (b, b2, b3), (x, x2, x3))
    )
    keep = (x >= np.finfo(F32).tiny) & (x < 1.0)
    return a[keep], b[keep], x[keep]


def f32_path(a, b, x):
    """float32 I and ln I through the internals (the public functions promote
    to float64 under x64)."""
    a, b, x = (jnp.asarray(v, jnp.float32) for v in (a, b, x))
    i = jax.jit(lambda a, b, x: m._cdf_parts(a, b, x, (False, False))[0])(a, b, x)
    l = jax.jit(lambda a, b, x: m._log_parts(a, b, x, (False, False))[0])(a, b, x)
    assert i.dtype == jnp.float32
    return np.asarray(i, np.float64), np.asarray(l, np.float64)


def x64_off_check(a, b, x, gi, gl):
    """The internal path under x64 equals the public one with x64 off."""
    import subprocess
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as d:
        np.savez(f"{d}/in.npz", a=a, b=b, x=x)
        code = textwrap.dedent(f"""
            import numpy as np, jax, chebax
            d = np.load({d + "/in.npz"!r})
            i = np.asarray(jax.jit(chebax.betainc_array)(d["a"], d["b"], d["x"]))
            l = np.asarray(jax.jit(chebax.log_betainc_array)(d["a"], d["b"], d["x"]))
            np.savez({d + "/out.npz"!r}, i=i, l=l)
        """)
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        env.pop("JAX_ENABLE_X64", None)
        r = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-2000:]
        o = np.load(f"{d}/out.npz")
        assert np.array_equal(o["i"], gi.astype(F32)), "x64-off I differs"
        assert np.array_equal(
            o["l"], gl.astype(F32), equal_nan=True
        ), "x64-off ln I differs"


def mp_ref(a, b, x):
    import mpmath as mp

    mp.mp.dps = 40
    out = np.empty((2, a.size))
    for k in range(a.size):
        v = mp.betainc(
            mp.mpf(float(a[k])),
            mp.mpf(float(b[k])),
            0,
            mp.mpf(float(x[k])),
            regularized=True,
        )
        out[:, k] = float(v), float(mp.log(v))
    return out


def acc():
    a, b, x = points()
    A, B, X = (v.astype(np.float64) for v in (a, b, x))
    ri = np.asarray(jax.jit(chebax.betainc_array)(A, B, X))
    rl = np.asarray(jax.jit(chebax.log_betainc_array)(A, B, X))
    live = ri > 1e-30
    print(f"jax {jax.__version__}, {a.size} points, CPU; shipped NKEEP32 = {T.NKEEP32}")
    print(
        "| split | TOL32 | NKEEP32 | I max | I max, mpmath | I > 5e-7 | ln I max | ln I max, mpmath | ln I > 5e-7 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    shipped = T.NKEEP32
    for sname, split in SPLITS.items():
        for tol in TOLS if sname == "S_SPLIT32" else (2e-8, 4e-7):
            n = nkeep(tol)
            setting(n, split)
            gi, gl = f32_path(a, b, x)
            if sname == "S_SPLIT32" and n == shipped:
                x64_off_check(a[:3000], b[:3000], x[:3000], gi[:3000], gl[:3000])
            ei = np.abs(gi - ri)
            el = np.where(live, np.abs(gl - rl) / np.maximum(1.0, np.abs(rl)), 0.0)
            worst = np.unique(
                np.concatenate([np.argsort(ei)[-40:], np.argsort(el)[-40:]])
            )
            mi, ml = mp_ref(a[worst], b[worst], x[worst])
            emi = np.max(np.abs(gi[worst] - mi))
            okw = mi > 1e-30
            eml = np.max(
                np.abs(gl[worst][okw] - ml[okw]) / np.maximum(1.0, np.abs(ml[okw]))
            )
            print(
                f"| {sname} | {tol:.1e} | {n} | {ei.max():.2e} | {emi:.2e} | {np.sum(ei > 5e-7)} | "
                f"{el.max():.2e} | {eml:.2e} | {np.sum(el > 5e-7)} |",
                flush=True,
            )


def speed(dev):
    d = jax.devices(dev)[0]
    n = 1 << (20 if dev == "gpu" else 16)
    rng = np.random.default_rng(1)
    regs = {}
    for name, lo, hi in (("box", T.ALO, 100.0), ("[5, 30]", 5.0, 30.0)):
        a = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        b = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        x = np.clip(rng.beta(a, b), 1e-30, 1 - 1e-7)
        regs[name] = tuple(
            jax.device_put(jnp.asarray(v, jnp.float32), d) for v in (a, b, x)
        )

    def tm(f, args, reps=15):
        for _ in range(4):
            jax.block_until_ready(f(*args))
        best = np.inf
        for _ in range(reps):
            t = time.perf_counter()
            jax.block_until_ready(f(*args))
            best = min(best, time.perf_counter() - t)
        return best / n * 1e9

    split = T.S_SPLIT32
    res = {}
    for _ in range(2):
        for tol in TOLS[:4]:
            setting(nkeep(tol), split)
            val = jax.jit(chebax.betainc_array)
            vg = jax.jit(
                jax.value_and_grad(
                    lambda a, b, x: chebax.betainc_array(a, b, x).sum(), (0, 1, 2)
                )
            )
            for r, args in regs.items():
                res.setdefault((tol, r), []).append((tm(val, args), tm(vg, args)))
    print(
        f"jax {jax.__version__} on {d}, float32, N = {n}; ns per element, min of 2 passes"
    )
    print(
        "| TOL32 | NKEEP32 | " + " | ".join(f"{r} value | {r} v+g" for r in regs) + " |"
    )
    print("|---|---|" + "---|---|" * len(regs))
    for tol in TOLS[:4]:
        cells = []
        for r in regs:
            v = np.array(res[(tol, r)])
            best, worst = v.min(0), v.max(0)
            cells += [
                f"{best[k]:.2f}{'*' if worst[k] / best[k] > 1.25 else ''}"
                for k in range(2)
            ]
        print(f"| {tol:.1e} | {nkeep(tol)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    acc() if MODE == "acc" else speed(MODE)
