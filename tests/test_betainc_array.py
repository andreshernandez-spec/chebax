"""Per-element betainc: betainc_array and log_betainc_array, a and b per
element in [0.1, 100], x in [0, 1].

References are mpmath at 40 dps on 2,500 points: a, b log-uniform over the
box, x from the bulk (Beta draws), within 1e-3 of x_s = (a+1)/(a+b+2), the
lower tail (x down to 1e-40 x_s) and the upper tail (1 - x down to 1e-15
(1 - x_s)); and 500 with b <= 3 in the band past x_s where those lanes stay
in the lower orientation (up to s = 7). ln I uses |ln I - ref| / max(1,
|ref|); I is absolute (the CDF contract).

Measured worst, the larger of jax 0.11 and 0.5.3 (CI's floor): float64
ln I 7.2e-16, I 6.7e-16, bars 3e-15. float32 on the float32-rounded points
with x in its normal range: I 1.5e-7, ln I 2.5e-7 (where I > 1e-30), bars
6e-7 and 1e-6. On 48,947 float32 points experiments/55 has I 2.3e-7 and
ln I 4.1e-7, ~3 float32 ulps, all float32 arithmetic (the target is 5).

dI/da, dI/db against mp.diff of the smaller side and dI/dx against the Beta
density, relative, on 76 points: 2.6e-14; bar 4e-13, 4x the 1.0e-13 that
3,000 probe points reached near the seam at large a, b.
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
from chebax._src.recipes import betainc_array_gen
from chebax._src.recipes import betainc_array_table as t

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40


def _points(n=2000, seed=11):
    """a, b log-uniform over the box with x from the bulk, next to x_s, both
    tails; and the band past x_s where small-b lanes stay in the lower
    orientation (x = 1 - (1 - x_s)^s, s up to 7)."""
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(0.1), np.log(100.0), n))
    b = np.exp(rng.uniform(np.log(0.1), np.log(100.0), n))
    xs = (a + 1.0) / (a + b + 2.0)
    kind = rng.integers(0, 4, n)
    x = np.select(
        [kind == 0, kind == 1, kind == 2],
        [
            rng.beta(a, b),
            xs * (1.0 + rng.uniform(-1e-3, 1e-3, n)),
            xs * 10.0 ** -rng.uniform(1.0, 40.0, n),
        ],
        1.0 - (1.0 - xs) * 10.0 ** -rng.uniform(1.0, 15.0, n),
    )
    m = n // 4
    a2 = np.exp(rng.uniform(np.log(0.1), np.log(100.0), m))
    b2 = np.exp(rng.uniform(np.log(0.1), np.log(3.0), m))
    xs2 = (a2 + 1.0) / (a2 + b2 + 2.0)
    x2 = -np.expm1(rng.uniform(0.8, 7.0, m) * np.log1p(-xs2))
    return (
        np.concatenate([a, a2]),
        np.concatenate([b, b2]),
        np.clip(np.concatenate([x, x2]), 1e-300, np.nextafter(1.0, 0.0)),
    )


def _mp_ref(a, b, x):
    """ln I and I, each I from its own side so the upper tail keeps its digits."""
    out = np.empty((2, a.size))
    for k, (ak, bk, xk) in enumerate(zip(a, b, x)):
        A, B, X = mp.mpf(ak), mp.mpf(bk), mp.mpf(xk)
        lo = mp.betainc(A, B, 0, X, regularized=True)
        out[:, k] = float(mp.log(lo)), float(lo)
    return out


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    assert np.all(np.isfinite(got)), "unexpected nonfinite values"
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


@pytest.fixture(scope="module")
def refs():
    a, b, x = _points()
    return (a, b, x, *_mp_ref(a, b, x))


def test_log_against_mpmath(refs):
    a, b, x, ln_i, _ = refs
    got = jax.jit(chebax.log_betainc_array)(a, b, x)
    assert np.max(_err(got, ln_i)) <= 3e-15


def test_values_absolute(refs):
    a, b, x, _, i = refs
    got = np.asarray(jax.jit(chebax.betainc_array)(a, b, x))
    assert np.max(np.abs(got - i)) <= 3e-15


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


def test_float32(refs, tmp_path):
    """x64 off: the leading NKEEP32 coefficients of each cell, every closed
    form in float32, on the float32-rounded points with x in float32's normal
    range (XLA flushes subnormal x to 0)."""
    a, b, x = (v[::4].astype(np.float32) for v in refs[:3])
    keep = x >= np.finfo(np.float32).tiny
    a, b, x = a[keep], b[keep], x[keep]
    got = _x64_off(
        tmp_path,
        """
        out["i"] = np.asarray(jax.jit(chebax.betainc_array)(d["a"], d["b"], d["x"]))
        out["l"] = np.asarray(jax.jit(chebax.log_betainc_array)(d["a"], d["b"], d["x"]))
        assert out["i"].dtype == np.float32
        """,
        a=a,
        b=b,
        x=x,
    )
    ln_i, i = _mp_ref(*(v.astype(np.float64) for v in (a, b, x)))
    assert np.max(np.abs(got["i"] - i)) <= 6e-7
    live = i > 1e-30  # float32's normal range
    assert np.max(_err(got["l"][live], ln_i[live])) <= 1e-6


def test_slopes_against_mpmath(refs):
    """dI/da, dI/db from mp.diff of the smaller side (I or 1 - I: of I itself,
    40 digits read 0 where I is within 1e-40 of 1) and dI/dx the density."""
    a, b, x = (v[::33] for v in refs[:3])
    g = jax.jit(jax.vmap(jax.grad(chebax.betainc_array, (0, 1, 2))))
    ga, gb, gx = (np.asarray(v) for v in g(a, b, x))
    worst = [0.0, 0.0, 0.0]
    for k in range(a.size):
        A, B, X = mp.mpf(a[k]), mp.mpf(b[k]), mp.mpf(x[k])
        if mp.betainc(A, B, 0, X, regularized=True) <= 0.5:
            f, sign = (lambda u, v: mp.betainc(u, v, 0, X, regularized=True)), 1
        else:
            f, sign = (lambda u, v: mp.betainc(v, u, 0, 1 - X, regularized=True)), -1
        ref = (
            sign * mp.diff(lambda u: f(u, B), A),
            sign * mp.diff(lambda v: f(A, v), B),
            mp.exp(
                (A - 1) * mp.log(X) + (B - 1) * mp.log1p(-X) - mp.log(mp.beta(A, B))
            ),
        )
        for m, (got, r) in enumerate(zip((ga[k], gb[k], gx[k]), ref)):
            if abs(r) > 1e-280:
                worst[m] = max(worst[m], float(abs(got - r) / abs(r)))
    assert max(worst) <= 4e-13, worst


def test_both_modes_agree(refs):
    a, b, x = (jnp.asarray(v[::50]) for v in refs[:3])
    for f in (chebax.betainc_array, chebax.log_betainc_array):
        rev = jax.grad(lambda a, b, x: f(a, b, x).sum(), (0, 1, 2))(a, b, x)
        for k, r in enumerate(rev):
            tan = [jnp.zeros_like(a)] * 3
            tan[k] = jnp.ones_like(a)
            fwd = jax.jvp(f, (a, b, x), tuple(tan))[1]
            np.testing.assert_allclose(fwd, r, rtol=1e-14, atol=0)


def test_endpoints_and_out_of_box():
    a = jnp.array([0.5, 2.0, 2.0, 0.05, 150.0, 2.0])
    b = jnp.array([3.0, 0.5, 2.0, 2.0, 2.0, jnp.nan])
    x = jnp.array([0.0, 1.0, jnp.nan, 0.3, 0.3, 0.3])
    np.testing.assert_array_equal(
        chebax.betainc_array(a, b, x), [0.0, 1.0, np.nan, np.nan, np.nan, np.nan]
    )
    np.testing.assert_array_equal(
        chebax.log_betainc_array(a, b, x),
        [-np.inf, 0.0, np.nan, np.nan, np.nan, np.nan],
    )
    g = jax.grad(lambda a: chebax.betainc_array(a, 2.0, 0.3))(150.0)
    assert np.isnan(g)


def test_endpoint_slopes_are_the_density():
    d = jax.grad(chebax.betainc_array, 2)
    assert d(1.0, 3.0, 0.0) == 3.0  # f(0) = b at a = 1
    assert d(3.0, 1.0, 1.0) == 3.0  # f(1) = a at b = 1
    assert d(2.0, 3.0, 0.0) == 0.0
    assert d(0.5, 3.0, 0.0) == np.inf


def test_broadcast_and_jit():
    a = jnp.array([[1.5], [70.0]])
    got = jax.jit(chebax.betainc_array)(a, 2.5, jnp.array([0.1, 0.5, 0.9]))
    assert got.shape == (2, 3)
    ref = [
        [float(mp.betainc(v, 2.5, 0, w, regularized=True)) for w in (0.1, 0.5, 0.9)]
        for v in (1.5, 70.0)
    ]
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-15)


@pytest.mark.slow
def test_column_regenerates_exact():
    """One column of the table, rebuilt (the whole module takes ~25 min on
    six cores and is compared only with CHEBAX_FULL_REGEN=1)."""
    ia, ib = 9, 3
    cuts, cells = betainc_array_gen.column_cells(ia, ib)
    col = ia * t.NAB + ib
    assert t.NS[col] == len(cuts) - 1
    assert list(t.SCUTS[col][: len(cuts)]) == cuts
    simplex = betainc_array_gen._simplex()
    pos = [simplex.index(ijk) for ijk in t.IDX]
    for n, (_, c) in enumerate(cells):
        assert [c[m] for m in pos] == list(t.TABLE[t.BASE[col] + n])
    assert t.S_SPLIT32[col] == betainc_array_gen.split_s32(ia, ib, cuts[-1])


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CHEBAX_FULL_REGEN"),
    reason="~25 min on six cores; set CHEBAX_FULL_REGEN=1 (pre-release)",
)
def test_table_regenerates_bit_for_bit(tmp_path):
    betainc_array_gen.main(out_dir=tmp_path)
    here = pathlib.Path(betainc_array_gen.__file__).parent / "betainc_array_table.py"
    assert (tmp_path / "betainc_array_table.py").read_bytes() == here.read_bytes()
