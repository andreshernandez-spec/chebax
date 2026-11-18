"""hyp1f1_array: accuracy against mpmath on a large sample, by region.

ln M from log_hyp1f1_array against mpmath at 40 dps, as |ln M - ref| /
max(1, |ref|) (M's relative error over max(1, ln M), the hyp1f1 contract),
on N points per regime: a, b log-uniform on [0.1, 10] with x uniform on [0,
X_S], log-uniform on [1e-8, X_S], within 50% of b, within 1e-3 of X_S, and
log-uniform on [X_S, 1e5] (X_S the seam of (a, b)'s column); the noncentral
t's a = nu/2 + 1 and (nu + 1)/2, nu in [1, 18], with b = 3/2 and 1/2; and the
8 corners of every cell, where every T_k is +-1 and the dropped coefficients
add up (shorter prefixes that passed on random points failed there). Also
hyp1f1_scalar on the first 400 points of each regime (the per-call path's
floor) and jax.scipy.special.hyp1f1 on all (inf where it returns nan or inf).
The slopes are in tests/test_hyp1f1_array.py.

    python experiments/57_hyp1f1_array_accuracy.py [N]      # float64
    python experiments/57_hyp1f1_array_accuracy.py [N] f32  # float32, x64 off

References are cached in results/57_cache (not committed). Does NOT
establish speed (experiments/56).
"""

import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
f32 = "f32" in sys.argv
args = [v for v in sys.argv[1:] if v != "f32"]
N = int(args[0]) if args else 4000

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", not f32)
import jax.numpy as jnp  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes import hyp1f1_array_table as _t  # noqa: E402


def seam(a, b):
    """X_S of each (a, b)'s base column."""
    w0 = (np.log(_t.AHI) - np.log(_t.ALO)) / _t.NB0
    i = [
        np.clip(np.floor((np.log(v) - np.log(_t.ALO)) / w0), 0, _t.NB0 - 1).astype(int)
        for v in (a, b)
    ]
    return np.asarray(_t.XS_COL)[i[0], i[1]]

mp.mp.dps = 40
CACHE = pathlib.Path(__file__).resolve().parent.parent / "results" / "57_cache"


def regimes(n, seed=3):
    rng = np.random.default_rng(seed)
    lo, hi = np.log(0.1), np.log(10.0)
    ab = lambda: (np.exp(rng.uniform(lo, hi, n)), np.exp(rng.uniform(lo, hi, n)))
    out = {}
    a, b = ab()
    out["x ~ U(0, X_S)"] = (a, b, rng.uniform(0.0, seam(a, b)))
    a, b = ab()
    out["x log-uniform 1e-8..X_S"] = (a, b, np.exp(rng.uniform(np.log(1e-8), np.log(seam(a, b)))))
    a, b = ab()
    out["x within 50% of b"] = (a, b, b * rng.uniform(0.5, 1.5, n))
    a, b = ab()
    out["x within 1e-3 of X_S"] = (a, b, seam(a, b) * (1.0 + rng.uniform(-1e-3, 1e-3, n)))
    a, b = ab()
    out["x log-uniform X_S..1e5"] = (a, b, np.exp(rng.uniform(np.log(seam(a, b)), np.log(1e5))))
    nu = rng.uniform(1.0, 18.0, n)
    b = np.where(rng.integers(0, 2, n) == 0, 0.5, 1.5)
    a = np.where(b == 1.5, nu / 2.0 + 1.0, (nu + 1.0) / 2.0)
    out["noncentral t"] = (a, b, rng.uniform(0.0, 32.0, n))
    out["cell corners"] = corners()
    return out


def corners():
    """The 8 corners of every cell, just inside (every T_k is +-1 there, so
    the dropped coefficients add up); tail corners at t = 0 left out."""
    box = np.asarray(_t.BOX).reshape(-1, 3, 2)
    tree = np.asarray(_t.TREE)
    tail, xs = np.zeros(len(box), bool), np.zeros(len(box))
    for r in range(2):
        for ia in range(_t.NB0):
            for ib in range(_t.NB0):
                stack = [_t.ROOT[r][ia][ib]]
                while stack:
                    d, _, left, right = tree[stack.pop()]
                    if d >= 0:
                        stack += [int(left), int(right)]
                    else:
                        tail[int(left)], xs[int(left)] = r == 1, _t.XS_COL[ia][ib]
    pts = []
    for n in range(len(box)):
        mid, half = box[n].mean(axis=1), (box[n][:, 1] - box[n][:, 0]) / 2
        for c in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3):
            la, lb, s = mid + half * c * (1 - 1e-9)
            if not (tail[n] and s < 1e-6):
                pts.append((np.exp(la), np.exp(lb), xs[n] / s if tail[n] else s))
    return tuple(np.array(pts).T)


def ref(name, a, b, x):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{name.replace(' ', '_').replace('/', '_')}_{a.size}_{a.dtype}.npy"
    if key.exists():
        return np.load(key)
    # forked workers run mpmath only
    with ProcessPoolExecutor(6) as ex:
        parts = ex.map(_ref_chunk, np.array_split(np.stack([a, b, x], 1), 48))
        r = np.concatenate([np.array(v) for v in parts])
    np.save(key, r)
    return r


def _ref_chunk(p):
    mp.mp.dps = 40
    return [
        float(mp.log(mp.hyp1f1(mp.mpf(float(u)), mp.mpf(float(v)), mp.mpf(float(w)))))
        for u, v, w in p
    ]


def _worst(got, r):
    """Worst error, nonfinite results counted as inf."""
    e = np.abs(np.asarray(got, np.float64) - r) / np.maximum(1.0, np.abs(r))
    return float(np.max(np.where(np.isfinite(e), e, np.inf)))


dt = np.float32 if f32 else np.float64
print(
    f"jax {jax.__version__}, {'float32' if f32 else 'float64'}, N = {N} per regime; "
    f"worst |ln M - ref| / max(1, |ref|)"
)
print("| regime | hyp1f1_array | (a, b, x) at worst | hyp1f1_scalar | jax.scipy hyp1f1 |")
print("|---|---|---|---|---|")
f_arr = jax.jit(chebax.log_hyp1f1_array)
f_sc = jax.jit(jax.vmap(chebax.log_hyp1f1_scalar))
f_jax = jax.jit(lambda a, b, x: jnp.log(jax.scipy.special.hyp1f1(a, b, x)))
for name, (a, b, x) in regimes(N).items():
    a, b, x = (v.astype(dt) for v in (a, b, x))
    r = ref(name, a, b, x)
    e = np.abs(np.asarray(f_arr(a, b, x), np.float64) - r) / np.maximum(1.0, np.abs(r))
    k = int(np.argmax(e))
    n = 400
    sc = "" if f32 else f"{_worst(f_sc(a[:n], b[:n], x[:n]), r[:n]):.1e} ({n} pts)"
    jx = _worst(f_jax(a, b, x), r)
    print(
        f"| {name} | {e[k]:.2e} | ({a[k]:.4g}, {b[k]:.4g}, {x[k]:.4g}) | {sc} | {jx:.1e} |",
        flush=True,
    )
