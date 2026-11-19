"""nct_logpdf: accuracy against mpmath by regime, and speed.

acc: ln f from nct_logpdf against mpmath at 40 digits, as |ln f - ref| /
max(1, |ref|), on N points per regime. The reference is

    ln f = ln K + lnGamma(nu + 1) - (nu + 1)/2 ln P - (mu^2 - y^2)/2 + ln Hh_nu(-y),

P = nu + x^2, y = mu x/sqrt(P), K = 2 (nu/2)^(nu/2)/(Gamma(nu/2) sqrt(2 pi)),
with hh_gen's ln Hh (pcfd up to nu = 150, a quadrature in the saddle's
scaled variable above). Regimes: the noncentral t's usual range (nu
log-uniform on [1, 100], mu uniform on [-10, 10], x within 5 of mu); heavy
tails (nu on [0.1, 1], mu on [-5, 5], x on [-100, 100]); near normal (nu on
[100, 1e6], x within 5 of mu); large mu near the mode (mu on [10, 50], nu on
[1, 1e3], x/mu on [0.8, 1.3]); x opposite to mu (nu on [1, 100], |mu| <= 10,
|x| <= 20), where the textbook form cancels; and a wide box (nu on [0.1, 1e4],
mu on [-50, 50], x on [-200, 200]). Next to it the textbook two-1F1 form a jax
user can write today,

    f = nu^(nu/2) Gamma(nu + 1) e^(-mu^2/2) / (2^nu P^(nu/2) Gamma(nu/2))
        [sqrt(2) mu x/P M(nu/2 + 1, 3/2, w)/Gamma((nu + 1)/2)
         + M((nu + 1)/2, 1/2, w)/(sqrt(P) Gamma(nu/2 + 1))],  w = mu^2 x^2/(2P),

with jax.scipy's hyp1f1 and gammaln, and scipy.stats.nct.logpdf (float64
only); for those two, the worst error where finite and the share of
nonfinite results.

speed: ns per element of nct_logpdf (value; value_and_grad of the sum in nu,
mu and x) and of the two-1F1 form, over three regimes (noncentral t: nu
log-uniform on [1, 100], mu uniform on [-5, 5], x within 4 of mu; wide: nu
on [0.1, 1e3], mu on [-20, 20], x on [-50, 50]; near normal: nu on [100,
1e5], mu on [-5, 5], x within 4 of mu); on the CPU in float64 also
scipy.stats.nct.logpdf (numpy, no jit). Two passes, minimum kept, * where
they differ by more than 1.25x.

    python experiments/59_nct.py acc [N] [f32]           # CPU
    taskset -c 6 python experiments/59_nct.py speed cpu [f32]
    python experiments/59_nct.py speed gpu [f32]

References are cached in results/59_cache (not committed). Does NOT
establish the slopes' accuracy (tests/test_nct.py) or speed inside a model.
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

DT = np.float32 if f32 else np.float64
CACHE = pathlib.Path(__file__).resolve().parent.parent / "results" / "59_cache"


def two_term(nu, mu, x):
    """ln f by the textbook two-1F1 form with jax.scipy."""
    gl = jax.scipy.special.gammaln
    p = nu + x * x
    w = mu * mu * x * x / (2.0 * p)
    m1 = jax.scipy.special.hyp1f1(0.5 * nu + 1.0, 1.5, w)
    m2 = jax.scipy.special.hyp1f1(0.5 * (nu + 1.0), 0.5, w)
    pre = (
        0.5 * nu * jnp.log(nu)
        + gl(nu + 1.0)
        - 0.5 * mu * mu
        - nu * jnp.log(2.0)
        - 0.5 * nu * jnp.log(p)
        - gl(0.5 * nu)
    )
    br = jnp.sqrt(2.0) * mu * x / p * m1 * jnp.exp(-gl(0.5 * (nu + 1.0))) + m2 * jnp.exp(
        -gl(0.5 * nu + 1.0)
    ) / jnp.sqrt(p)
    return pre + jnp.log(br)


def regimes(n, seed=11):
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    un = lambda lo, hi: rng.uniform(lo, hi, n)
    out = {}
    mu = un(-10, 10)
    out["nu 1-100, |mu| <= 10, x within 5 of mu"] = (lu(1, 100), mu, mu + un(-5, 5))
    out["nu 0.1-1, |mu| <= 5, |x| <= 100"] = (lu(0.1, 1), un(-5, 5), un(-100, 100))
    mu = un(-10, 10)
    out["nu 100-1e6, x within 5 of mu"] = (lu(100, 1e6), mu, mu + un(-5, 5))
    mu = un(10, 50) * rng.choice([-1, 1], n)
    out["|mu| 10-50, nu 1-1e3, x/mu 0.8-1.3"] = (lu(1, 1e3), mu, mu * un(0.8, 1.3))
    mu = un(-10, 10)
    out["x opposite to mu, nu 1-100"] = (lu(1, 100), mu, -np.sign(mu) * un(0, 20))
    out["wide: nu 0.1-1e4, |mu| <= 50, |x| <= 200"] = (
        lu(0.1, 1e4),
        un(-50, 50),
        un(-200, 200),
    )
    return out


def _ref_chunk(rows):
    import mpmath as mp

    from chebax._src.recipes import hh_gen

    mp.mp.dps = 40
    out = []
    for a, b, c in rows:
        nu, mu, x = (mp.mpf(float(v)) for v in (a, b, c))
        p = nu + x * x
        y = mu * x / mp.sqrt(p)
        lnk = (
            mp.log(2)
            + nu / 2 * mp.log(nu / 2)
            - mp.loggamma(nu / 2)
            - mp.log(2 * mp.pi) / 2
        )
        v = (
            lnk
            + mp.loggamma(nu + 1)
            - (nu + 1) / 2 * mp.log(p)
            - (mu * mu - y * y) / 2
            + hh_gen.log_hh(mp, nu, -y)
        )
        out.append(float(v))
    return out


def ref(name, nu, mu, x):
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch for ch in name if ch.isalnum())
    key = CACHE / f"{tag}_{nu.size}_{nu.dtype}.npy"
    if key.exists():
        return np.load(key)
    # forked workers run mpmath only
    with ProcessPoolExecutor(4) as ex:
        parts = ex.map(_ref_chunk, np.array_split(np.stack([nu, mu, x], 1), 48))
        r = np.concatenate([np.array(v) for v in parts])
    np.save(key, r)
    return r


def worst(got, r):
    """Worst error where finite, and the share of nonfinite results."""
    e = np.abs(np.asarray(got, np.float64) - r) / np.maximum(1.0, np.abs(r))
    fin = np.isfinite(e)
    return f"{np.max(e[fin]) if fin.any() else np.nan:.1e} ({100 * (1 - fin.mean()):.0f}% nonfinite)"


def scipy_logpdf(nu, mu, x):
    """scipy.stats.nct.logpdf, nan where boost raises (tgamma overflows at
    large nu)."""
    from scipy.stats import nct

    try:
        return nct.logpdf(x, nu, mu)
    except OverflowError:
        out = np.empty(nu.shape)
        for k in range(nu.size):
            try:
                out[k] = nct.logpdf(x[k], nu[k], mu[k])
            except OverflowError:
                out[k] = np.nan
        return out


def accuracy(n):
    print(
        f"jax {jax.__version__}, {'float32' if f32 else 'float64'}, N = {n} per regime; worst |ln f - ref| / max(1, |ref|)"
    )
    print("| regime | nct_logpdf | (nu, mu, x) at worst | two-1F1 form (jax) | scipy nct.logpdf |")
    print("|---|---|---|---|---|")
    f = jax.jit(chebax.nct_logpdf)
    g = jax.jit(two_term)
    for name, (nu, mu, x) in regimes(n).items():
        nu, mu, x = nu.astype(DT), mu.astype(DT), x.astype(DT)
        r = ref(name, nu, mu, x)
        e = np.abs(np.asarray(f(nu, mu, x), np.float64) - r) / np.maximum(1.0, np.abs(r))
        k = int(np.argmax(e))
        with np.errstate(all="ignore"):
            sp = "" if f32 else worst(scipy_logpdf(nu, mu, x), r)
        print(
            f"| {name} | {e[k]:.2e} | ({nu[k]:.4g}, {mu[k]:.4g}, {x[k]:.4g}) | {worst(g(nu, mu, x), r)} | {sp} |",
            flush=True,
        )


def timeit(fn, *args, reps=15, sync=True):
    out = fn(*args)
    if sync:
        jax.block_until_ready(out)
    for _ in range(3):
        out = fn(*args)
        if sync:
            jax.block_until_ready(out)
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync:
            jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best / args[0].shape[0] * 1e9


def speed():
    N = 1 << (20 if dev == "gpu" else 16)
    d = jax.devices(dev)[0]

    def inputs(regime, seed):
        rng = np.random.default_rng(seed)
        lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), N))
        un = lambda lo, hi: rng.uniform(lo, hi, N)
        if regime == "noncentral t":
            nu, mu = lu(1.0, 100.0), un(-5, 5)
            x = mu + un(-4, 4)
        elif regime == "wide":
            nu, mu, x = lu(0.1, 1e3), un(-20, 20), un(-50, 50)
        else:
            nu, mu = lu(100.0, 1e5), un(-5, 5)
            x = mu + un(-4, 4)
        return tuple(np.asarray(v, DT) for v in (nu, mu, x))

    vg = lambda fn: jax.jit(jax.value_and_grad(lambda a, b, c: fn(a, b, c).sum(), (0, 1, 2)))
    ops = {
        "nct_logpdf": jax.jit(chebax.nct_logpdf),
        "nct_logpdf v+g": vg(chebax.nct_logpdf),
        "two-1F1 form (jax)": jax.jit(two_term),
        "two-1F1 form (jax) v+g": vg(two_term),
    }
    scipy_too = dev == "cpu" and not f32
    regs = ("noncentral t", "wide", "near normal")
    print(
        f"jax {jax.__version__} on {d}, {'float32' if f32 else 'float64'}, N = {N}; ns per element, min of 2 passes"
    )
    print("| op | " + " | ".join(regs) + " |")
    print("|---|" + "---|" * len(regs))
    res = {}
    for _ in range(2):
        for r in regs:
            host = inputs(r, 1)
            args = tuple(jax.device_put(jnp.asarray(v), d) for v in host)
            for name, fn in ops.items():
                res.setdefault((name, r), []).append(timeit(fn, *args))
            if scipy_too:
                from scipy.stats import nct

                sp = lambda a, b, c: nct.logpdf(c, a, b)
                try:
                    took = timeit(sp, *host, reps=3, sync=False)
                except OverflowError:  # boost's tgamma, at large nu
                    took = np.nan
                res.setdefault(("scipy nct.logpdf", r), []).append(took)
    names = list(ops) + (["scipy nct.logpdf"] if scipy_too else [])
    for name in names:
        cells = []
        for r in regs:
            v = res[(name, r)]
            if np.isnan(v).any():
                cells.append("raises")
                continue
            cells.append(f"{min(v):.1f}{'*' if max(v) / min(v) > 1.25 else ''}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if mode == "acc":
    args = [v for v in sys.argv[2:] if v != "f32"]
    accuracy(int(args[0]) if args else 4000)
else:
    speed()
