"""hh / log_hh: accuracy against mpmath by regime, and speed.

acc: ln Hh_nu(z) from log_hh against mpmath (pcfd, Hh = e^(-z^2/4)
D_{-nu-1}(z), up to nu = 150; hh_gen's quadrature in the saddle's scaled
variable above), as |ln Hh - ref| / max(1, |ref|), on N points per regime:
nu log-uniform on [0.1, 10] with z uniform on [-10, 10] and at sigma =
z/sqrt(z^2 + 4(nu + 1)) uniform on (-1, 1); nu log-uniform on [10, 1e4] at
uniform sigma; the band where ln Hh is near 0 (z within 20% of the root of
|z|^nu sqrt(2 pi) = Gamma(nu + 1), nu log-uniform on [1, 1e5]), where nu and
nu ln v cancel; the noncentral t's nu in [1, 100], z in [-40, 40]; and the 4
corners of every patch, where every T_k is +-1. Next to it the two-term form
a jax user can write today,

    Hh = sqrt(pi) 2^(-(nu+1)/2) e^(-z^2/2) [M((nu+1)/2, 1/2, z^2/2)/Gamma(nu/2+1)
         - sqrt(2) z M(nu/2+1, 3/2, z^2/2)/Gamma((nu+1)/2)]

with jax.scipy's hyp1f1 and gammaln (it cancels for z > 0), and scipy's
pbdv(-nu-1, z) e^(-z^2/4) (float64 only); for those two, the worst error
where finite and the share of nonfinite results.

speed: ns per element of hh and log_hh (value; value_and_grad of the sum in
nu and z) and of the two-term form, over three regimes (noncentral t: nu
log-uniform on [1, 100], z uniform on [-6, 6]; box: nu log-uniform on [0.1,
1e3], sigma uniform; small nu: nu log-uniform on [0.1, 1], z uniform on [-6,
6]). Two passes, minimum kept, * where they differ by more than 1.25x.

    python experiments/58_hh.py acc [N] [f32]           # CPU
    taskset -c 6 python experiments/58_hh.py speed cpu [f32]
    python experiments/58_hh.py speed gpu [f32]

References are cached in results/58_cache (not committed). Does NOT
establish the slopes' accuracy (tests/test_hh.py) or speed inside a model.
"""

import os
import pathlib
import sys
import time

mode = sys.argv[1]
f32 = "f32" in sys.argv
dev = "cpu" if mode == "acc" else sys.argv[2]
if dev == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
    ).strip()

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", not f32)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes import hh_table as _t  # noqa: E402

DT = np.float32 if f32 else np.float64
CACHE = pathlib.Path(__file__).resolve().parent.parent / "results" / "58_cache"


def two_term(nu, z):
    """The two-term 1F1 form with jax.scipy (DLMF 12.7.14 with a = nu + 1/2)."""
    gl = jax.scipy.special.gammaln
    x = 0.5 * z * z
    m1 = jax.scipy.special.hyp1f1(0.5 * (nu + 1.0), 0.5, x)
    m2 = jax.scipy.special.hyp1f1(0.5 * nu + 1.0, 1.5, x)
    pre = 0.5 * jnp.log(jnp.pi) - 0.5 * (nu + 1.0) * jnp.log(2.0) - x
    return jnp.exp(pre) * (
        m1 * jnp.exp(-gl(0.5 * nu + 1.0))
        - jnp.sqrt(2.0) * z * m2 * jnp.exp(-gl(0.5 * (nu + 1.0)))
    )


def corners():
    box = np.asarray(_t.BOX).reshape(-1, 2, 2)
    tree = np.asarray(_t.TREE)
    high = np.zeros(len(box), bool)
    for zone, rows in enumerate(_t.ROOT):
        for row in rows:
            for root in row:
                stack = [root]
                while stack:
                    d, _, left, right = tree[stack.pop()]
                    if d >= 0:
                        stack += [int(left), int(right)]
                    else:
                        high[int(left)] = zone == 1
    pts = []
    for n in range(len(box)):
        mid, half = box[n].mean(axis=1), (box[n][:, 1] - box[n][:, 0]) / 2
        for cp in (-1, 1):
            for cs in (-1, 1):
                p, sg = mid + half * np.array([cp, cs]) * (1 - 1e-9)
                if abs(sg) > 1 - 1e-7 or (high[n] and p < 1e-6):
                    continue
                nu = _t.NU1 / p if high[n] else np.exp(p)
                pts.append((nu, 2.0 * np.sqrt(nu + 1.0) * sg / np.sqrt(1.0 - sg * sg)))
    return tuple(np.array(pts).T)


def regimes(n, seed=9):
    from scipy.special import gammaln

    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    zs = (
        lambda nu: 2.0
        * np.sqrt(nu + 1.0)
        * (s := rng.uniform(-0.999, 0.999, n))
        / np.sqrt(1 - s * s)
    )
    out = {}
    nu = lu(0.1, 10.0)
    out["nu 0.1-10, z in [-10, 10]"] = (nu, rng.uniform(-10.0, 10.0, n))
    nu = lu(0.1, 10.0)
    out["nu 0.1-10, sigma uniform"] = (nu, zs(nu))
    nu = lu(10.0, 1e4)
    out["nu 10-1e4, sigma uniform"] = (nu, zs(nu))
    nu = lu(1.0, 1e5)
    z0 = -np.exp((gammaln(nu + 1.0) - 0.5 * np.log(2 * np.pi)) / nu)
    out["ln Hh near 0, nu 1-1e5"] = (nu, z0 * rng.uniform(0.8, 1.2, n))
    out["noncentral t"] = (lu(1.0, 100.0), rng.uniform(-40.0, 40.0, n))
    out["patch corners"] = corners()
    return out


def _ref_chunk(p):
    import mpmath as mp

    from chebax._src.recipes import hh_gen

    mp.mp.dps = 40
    return [float(hh_gen.log_hh(mp, mp.mpf(float(a)), mp.mpf(float(b)))) for a, b in p]


def ref(name, nu, z):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{name.replace(' ', '_').replace(',', '')}_{nu.size}_{nu.dtype}.npy"
    if key.exists():
        return np.load(key)
    # forked workers run mpmath only
    with ProcessPoolExecutor(6) as ex:
        parts = ex.map(_ref_chunk, np.array_split(np.stack([nu, z], 1), 48))
        r = np.concatenate([np.array(v) for v in parts])
    np.save(key, r)
    return r


def worst(got, r):
    """Worst error where finite, and the share of nonfinite results."""
    e = np.abs(np.asarray(got, np.float64) - r) / np.maximum(1.0, np.abs(r))
    fin = np.isfinite(e)
    return f"{np.max(e[fin]) if fin.any() else np.nan:.1e} ({100 * (1 - fin.mean()):.0f}% nonfinite)"


def accuracy(n):
    from scipy.special import pbdv

    print(
        f"jax {jax.__version__}, {'float32' if f32 else 'float64'}, N = {n} per regime (corners: all); worst |ln Hh - ref| / max(1, |ref|)"
    )
    print("| regime | log_hh | (nu, z) at worst | two-term 1F1 (jax) | scipy pbdv |")
    print("|---|---|---|---|---|")
    f = jax.jit(chebax.log_hh)
    g = jax.jit(lambda a, b: jnp.log(two_term(a, b)))
    for name, (nu, z) in regimes(n).items():
        nu, z = nu.astype(DT), z.astype(DT)
        r = ref(name, nu, z)
        e = np.abs(np.asarray(f(nu, z), np.float64) - r) / np.maximum(1.0, np.abs(r))
        k = int(np.argmax(e))
        with np.errstate(all="ignore"):
            sp = "" if f32 else worst(np.log(pbdv(-nu - 1.0, z)[0]) - z * z / 4, r)
        print(
            f"| {name} | {e[k]:.2e} | ({nu[k]:.4g}, {z[k]:.4g}) | {worst(g(nu, z), r)} | {sp} |",
            flush=True,
        )


def timeit(fn, *args, reps=15):
    jax.block_until_ready(fn(*args))
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best / args[0].shape[0] * 1e9


def speed():
    N = 1 << (20 if dev == "gpu" else 16)
    d = jax.devices(dev)[0]

    def inputs(regime, seed):
        rng = np.random.default_rng(seed)
        lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), N))
        if regime == "noncentral t":
            nu, z = lu(1.0, 100.0), rng.uniform(-6.0, 6.0, N)
        elif regime == "box":
            nu = lu(0.1, 1e3)
            s = rng.uniform(-0.999, 0.999, N)
            z = 2.0 * np.sqrt(nu + 1.0) * s / np.sqrt(1 - s * s)
        else:
            nu, z = lu(0.1, 1.0), rng.uniform(-6.0, 6.0, N)
        return tuple(jax.device_put(jnp.asarray(v, DT), d) for v in (nu, z))

    vg = lambda fn: jax.jit(jax.value_and_grad(lambda a, b: fn(a, b).sum(), (0, 1)))
    ops = {
        "hh": jax.jit(chebax.hh),
        "hh v+g": vg(chebax.hh),
        "log_hh": jax.jit(chebax.log_hh),
        "log_hh v+g": vg(chebax.log_hh),
        "two-term 1F1 (jax)": jax.jit(two_term),
        "two-term 1F1 (jax) v+g": vg(two_term),
    }
    regs = ("noncentral t", "box", "small nu")
    print(
        f"jax {jax.__version__} on {d}, {'float32' if f32 else 'float64'}, N = {N}; ns per element, min of 2 passes"
    )
    print("| op | " + " | ".join(regs) + " |")
    print("|---|" + "---|" * len(regs))
    res = {}
    for _ in range(2):
        for r in regs:
            args = inputs(r, 1)
            for name, fn in ops.items():
                res.setdefault((name, r), []).append(timeit(fn, *args))
    for name in ops:
        cells = []
        for r in regs:
            v = res[(name, r)]
            cells.append(f"{min(v):.1f}{'*' if max(v) / min(v) > 1.25 else ''}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if mode == "acc":
    args = [v for v in sys.argv[2:] if v != "f32"]
    accuracy(int(args[0]) if args else 4000)
else:
    speed()
