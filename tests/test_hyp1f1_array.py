"""Per-element hyp1f1: hyp1f1_array and log_hyp1f1_array, a and b per element
in [0.1, 10], x in [0, inf).

References are mpmath at 40 dps on 2,500 points: a, b log-uniform over the
box with x uniform on [0, X_S], log-uniform from 1e-8 to X_S, near x = b
(where the b-direction is hardest), next to the seam X_S of (a, b)'s column
and log-uniform in the tail to 1e5. M is positive, so errors are |ln M - ref|
/ max(1, |ref|) for both forms (M's relative error over max(1, ln M), the
hyp1f1 contract).

Every cell's 8 corners (every 8th cell here) are checked too: there all
T_k are +-1 and the dropped coefficients add up, so prefixes that looked fine
on random points failed there (experiments/57).

Measured worst, identical on jax 0.11 and 0.5.3 (CI's floor): float64 ln M
2.2e-16 on the random points and 4.4e-16 at the corners, M 3.2e-16 (bars 1e-15,
2e-15 and 1.5e-15). float32 3.1e-7 for ln M and 3.2e-7 for M on the
float32-rounded random points and corners (bar 1e-6). Slopes against mp.diff
and the contiguous-relation oracle, relative to max(1, |slope|): d/da
1.4e-15, d/db 2.0e-15, d/dx 7.1e-15 (bar 3e-14).
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import hyp1f1_array_gen
from chebax._src.recipes import hyp1f1_array_table as t

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40


def _seam(a, b):
    """X_S of each (a, b)'s base column."""
    w0 = (np.log(t.AHI) - np.log(t.ALO)) / t.NB0
    i = [
        np.clip(np.floor((np.log(v) - np.log(t.ALO)) / w0), 0, t.NB0 - 1).astype(int)
        for v in (a, b)
    ]
    return np.asarray(t.XS_COL)[i[0], i[1]]


def _points(n=2500, seed=5):
    rng = np.random.default_rng(seed)
    lo, hi = np.log(0.1), np.log(10.0)
    a = np.exp(rng.uniform(lo, hi, n))
    b = np.exp(rng.uniform(lo, hi, n))
    xs = _seam(a, b)
    kind = rng.integers(0, 5, n)
    x = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [
            rng.uniform(0.0, xs),
            np.exp(rng.uniform(np.log(1e-8), np.log(xs))),
            b * rng.uniform(0.5, 1.5, n),
            xs * (1.0 + rng.uniform(-1e-3, 1e-3, n)),
        ],
        np.exp(rng.uniform(np.log(xs), np.log(1e5))),
    )
    return a, b, x


def _corners(step=8):
    """(a, b, x) at the 8 corners of every step-th cell, just inside: where
    the dropped coefficients add up (every T_k is +-1). Tail corners at t = 0
    are left out (x = inf)."""
    box = np.asarray(t.BOX).reshape(-1, 3, 2)
    tree = np.asarray(t.TREE)
    tail, seam = np.zeros(len(box), bool), np.zeros(len(box))
    for r in range(2):
        for ia in range(t.NB0):
            for ib in range(t.NB0):
                stack = [t.ROOT[r][ia][ib]]
                while stack:
                    d, _, left, right = tree[stack.pop()]
                    if d >= 0:
                        stack += [int(left), int(right)]
                    else:
                        tail[int(left)], seam[int(left)] = r == 1, t.XS_COL[ia][ib]
    pts = []
    for n in range(0, len(box), step):
        mid, half = box[n].mean(axis=1), (box[n][:, 1] - box[n][:, 0]) / 2
        for corner in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3):
            la, lb, s = mid + half * corner * (1 - 1e-9)
            if tail[n] and s < 1e-6:
                continue
            pts.append((np.exp(la), np.exp(lb), seam[n] / s if tail[n] else s))
    return tuple(np.array(pts).T)


def _mp_ref(a, b, x):
    return np.array(
        [
            float(mp.log(mp.hyp1f1(mp.mpf(ak), mp.mpf(bk), mp.mpf(xk))))
            for ak, bk, xk in zip(a, b, x)
        ]
    )


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    assert np.all(np.isfinite(got)), "unexpected nonfinite values"
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


@pytest.fixture(scope="module")
def refs():
    a, b, x = _points()
    return a, b, x, _mp_ref(a, b, x)


def test_log_against_mpmath(refs):
    a, b, x, ln_m = refs
    got = jax.jit(chebax.log_hyp1f1_array)(a, b, x)
    assert np.max(_err(got, ln_m)) <= 1e-15


@pytest.fixture(scope="module")
def corners():
    a, b, x = _corners()
    return a, b, x, _mp_ref(a, b, x)


def test_log_at_cell_corners(corners):
    a, b, x, ln_m = corners
    got = jax.jit(chebax.log_hyp1f1_array)(a, b, x)
    assert np.max(_err(got, ln_m)) <= 2e-15


def test_values_relative(refs):
    a, b, x, ln_m = refs
    live = ln_m < 700.0
    got = np.asarray(jax.jit(chebax.hyp1f1_array)(a[live], b[live], x[live]))
    rel = np.abs(got / np.exp(ln_m[live]) - 1.0)
    assert np.max(rel / np.maximum(1.0, ln_m[live])) <= 1.5e-15


def _x64_off(tmp_path, body, **arrays):
    """Run body in a child process with x64 off (process-global): it reads
    the arrays from d and puts its results in the dict out."""
    np.savez(tmp_path / "in.npz", **arrays)
    code = (
        textwrap.dedent(f"""
        import numpy as np, jax, jax.numpy as jnp, chebax
        assert jnp.empty(()).dtype == jnp.float32, "x64 leaked in"
        d = np.load({str(tmp_path / "in.npz")!r})
        out = {{}}
    """)
        + textwrap.dedent(body)
        + textwrap.dedent(f"""
        np.savez({str(tmp_path / "out.npz")!r}, **out)
    """)
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, run.stderr[-3000:]
    return dict(np.load(tmp_path / "out.npz"))


def test_float32(refs, corners, tmp_path):
    """x64 off: the leading NKEEP32 coefficients of each cell, every closed
    form in float32, on the float32-rounded points and cell corners."""
    a, b, x = (
        np.concatenate([u[::4], v]).astype(np.float32)
        for u, v in zip(refs[:3], corners[:3])
    )
    got = _x64_off(
        tmp_path,
        """
        out["l"] = np.asarray(jax.jit(chebax.log_hyp1f1_array)(d["a"], d["b"], d["x"]))
        out["m"] = np.asarray(jax.jit(chebax.hyp1f1_array)(d["a"], d["b"], d["x"]))
        assert out["l"].dtype == np.float32
        """,
        a=a,
        b=b,
        x=x,
    )
    ln_m = _mp_ref(*(v.astype(np.float64) for v in (a, b, x)))
    assert np.max(_err(got["l"], ln_m)) <= 1e-6
    live = ln_m < 80.0
    rel = np.abs(got["m"][live] / np.exp(ln_m[live]) - 1.0)
    assert np.max(rel / np.maximum(1.0, ln_m[live])) <= 1e-6


def test_slopes_against_mpmath(refs):
    """d ln M/da, d/db against mp.diff and d/dx against (a/b) M(a+1, b+1, x)/M,
    relative to max(1, |slope|)."""
    a, b, x = (v[::40] for v in refs[:3])
    g = jax.jit(jax.vmap(jax.grad(chebax.log_hyp1f1_array, (0, 1, 2))))
    ga, gb, gx = (np.asarray(v) for v in g(a, b, x))
    worst = [0.0, 0.0, 0.0]
    for k in range(a.size):
        A, B, X = mp.mpf(a[k]), mp.mpf(b[k]), mp.mpf(x[k])
        lm = lambda u, v: mp.log(mp.hyp1f1(u, v, X))
        ref = (
            mp.diff(lambda u: lm(u, B), A),
            mp.diff(lambda v: lm(A, v), B),
            A / B * mp.hyp1f1(A + 1, B + 1, X) / mp.hyp1f1(A, B, X),
        )
        for m, (got, r) in enumerate(zip((ga[k], gb[k], gx[k]), ref)):
            worst[m] = max(worst[m], float(abs(got - r) / max(1, abs(r))))
    assert max(worst) <= 3e-14, worst


def test_both_modes_agree(refs):
    """Forward and reverse slopes, M finite (where M is inf, an explicit zero
    tangent gives inf * 0)."""
    live = refs[3] < 700.0
    a, b, x = (jnp.asarray(v[live][::50]) for v in refs[:3])
    for f in (chebax.hyp1f1_array, chebax.log_hyp1f1_array):
        rev = jax.grad(lambda a, b, x: f(a, b, x).sum(), (0, 1, 2))(a, b, x)
        for k, r in enumerate(rev):
            tan = [jnp.zeros_like(a)] * 3
            tan[k] = jnp.ones_like(a)
            fwd = jax.jvp(f, (a, b, x), tuple(tan))[1]
            np.testing.assert_allclose(fwd, r, rtol=1e-14, atol=0)


def test_endpoints_and_out_of_box():
    a = jnp.array([2.5, 2.5, 2.5, 0.05, 12.0, 2.0, jnp.nan, 1.0])
    b = jnp.array([0.7, 0.7, 0.7, 2.0, 2.0, jnp.nan, 2.0, 1.0])
    x = jnp.array([0.0, -1.0, jnp.inf, 0.3, 0.3, 0.3, 0.3, jnp.nan])
    nan = np.nan
    np.testing.assert_array_equal(
        chebax.hyp1f1_array(a, b, x), [1.0, nan, np.inf, nan, nan, nan, nan, nan]
    )
    np.testing.assert_array_equal(
        chebax.log_hyp1f1_array(a, b, x), [0.0, nan, np.inf, nan, nan, nan, nan, nan]
    )
    assert np.isnan(jax.grad(lambda a: chebax.hyp1f1_array(a, 2.0, 0.3))(12.0))


def test_origin_slope():
    for a, b in ((2.0, 3.0), (0.5, 1.5), (9.0, 0.2), (0.1, 10.0)):
        got = float(jax.grad(chebax.hyp1f1_array, 2)(a, b, 0.0))
        assert abs(got - a / b) <= 1e-15 * (a / b), (a, b, got)


def test_dispatch_and_broadcast():
    a = jnp.array([[1.5], [7.0]])
    xs = jnp.array([0.1, 5.0, 80.0])
    got = jax.jit(chebax.hyp1f1)(a, 2.5, xs)
    np.testing.assert_array_equal(got, chebax.hyp1f1_array(a, 2.5, xs))
    ref = [[float(mp.hyp1f1(v, 2.5, w)) for w in (0.1, 5.0, 80.0)] for v in (1.5, 7.0)]
    np.testing.assert_allclose(got, ref, rtol=2e-15 * 100, atol=0)
    np.testing.assert_array_equal(
        chebax.log_hyp1f1(a, 2.5, xs), chebax.log_hyp1f1_array(a, 2.5, xs)
    )
    with pytest.raises(ValueError, match="hyp1f1_array"):
        chebax.hyp1f1_scalar(jnp.array([1.0, 2.0]), 2.5, 1.0)


def test_tree_covers_each_column():
    """Every base column's leaves tile its box: volumes add up and each leaf
    lies inside the column."""
    w0 = (np.log(t.AHI) - np.log(t.ALO)) / t.NB0
    tree = np.asarray(t.TREE)
    box = np.asarray(t.BOX).reshape(-1, 3, 2)
    for r in range(2):
        for ia in range(t.NB0):
            for ib in range(t.NB0):
                top = t.XS_COL[ia][ib] if r == 0 else 1.0
                lo = np.log(t.ALO) + w0 * np.array([ia, ib])
                stack, vol = [t.ROOT[r][ia][ib]], 0.0
                while stack:
                    d, _, left, right = tree[stack.pop()]
                    if d >= 0:
                        stack += [int(left), int(right)]
                        continue
                    bx = box[int(left)]
                    assert np.all(bx[:2, 0] >= lo - 1e-12)
                    assert np.all(bx[:2, 1] <= lo + w0 + 1e-12)
                    assert bx[2, 0] >= 0.0 and bx[2, 1] <= top
                    vol += np.prod(bx[:, 1] - bx[:, 0])
                assert abs(vol - w0 * w0 * top) <= 1e-12 * w0 * w0 * top


@pytest.mark.slow
def test_column_regenerates_exact():
    """One base column of the tail, rebuilt (the whole module takes ~25 min on
    six cores and is compared only with CHEBAX_FULL_REGEN=1)."""
    cells, tree = hyp1f1_array_gen.column_cells("tail", 3, 0)
    simplex = hyp1f1_array_gen._simplex()
    pos = [simplex.index(ijk) for ijk in t.IDX]
    root = t.ROOT[1][3][0]
    leaves, stack = [], [root]
    while stack:
        d, _, left, right = t.TREE[stack.pop()]
        if d >= 0:
            stack += [int(right), int(left)]
        else:
            leaves.append(int(left))
    assert len(leaves) == len(cells)
    for n, (_, _, _, c, c0lo) in zip(leaves, cells):
        assert [c[m] for m in pos] == list(t.TABLE[n])
        assert c0lo == t.C0LO[n]


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CHEBAX_FULL_REGEN"),
    reason="~25 min on six cores; set CHEBAX_FULL_REGEN=1 (pre-release)",
)
def test_table_regenerates_bit_for_bit(tmp_path):
    hyp1f1_array_gen.main(out_dir=tmp_path)
    here = pathlib.Path(hyp1f1_array_gen.__file__).parent / "hyp1f1_array_table.py"
    assert (tmp_path / "hyp1f1_array_table.py").read_bytes() == here.read_bytes()
