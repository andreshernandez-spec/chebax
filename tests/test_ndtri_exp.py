"""ndtri_exp acceptance: z = ndtri(e^y), dz/dy and d2z/dy2 against mpmath at
60 dps.

All three are pointwise-relative everywhere, z next to its zero at the
median y = -log 2 too (-log 2 +- 10^-k and the floats around it). The grid
holds every panel edge and its neighbours, centre panels in y and tail
panels in s mapped to both tails (y = -s^2/2 and y = log(1 - e^(-s^2/2))),
both seams and the s = 2^30 switch, and log-spaced points from the smallest
normal to the largest float. d2z/dy2 is checked for |y| <= 1e12, where its
reference keeps 30 digits, and where it is below the largest float (above
y = -3e-153 it overflows, and comes back nan, not inf).

Worst on this grid, jax 0.11: float64 z 4.7e-16 (5.7e-16 on jax 0.5.3),
dz/dy 6.3e-16, d2z/dy2 5.6e-15 (bars 2e-15, 2.5e-15, 2.2e-14); float32 2.7e-7, 3.3e-7, 1.6e-6 (bars
1.1e-6, 1.3e-6, 6e-6). d2z/dy2 differentiates the derivative's table, so it
is weakest on the centre's top panel. The grid stops at the smallest
normal (XLA flushes subnormal y to 0, where z is inf).
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import _common
from chebax._src.recipes import ndtri_exp as N
from chebax._src.recipes import ndtri_exp_gen
from chebax._src.recipes import ndtri_exp_table as nt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

LN2 = float(np.log(2.0))


def _grid(dtype):
    centre = nt.D_LO + np.arange(nt.NC + 1) / nt.WC - LN2
    octaves = range(int(np.log2(nt.S_BIG)) + 1)
    s = np.concatenate([2.0**e * (1 + np.arange(nt.K + 1) / nt.K) for e in octaves])
    s = s[s >= np.sqrt(-2 * nt.YC) * 0.97]
    s = np.concatenate([s, (s[1:] + s[:-1]) / 2])
    pts = [centre, (centre[1:] + centre[:-1]) / 2, -s * s / 2]
    with np.errstate(under="ignore"):
        pts.append(np.log1p(-np.exp(-s * s / 2)))
    pts.append(np.array([nt.YC, nt.Y_HI, -LN2]))
    eps_k = range(1, 16) if dtype == np.float64 else range(1, 8)
    pts += [np.array([-LN2 + d * 10.0**-k]) for d in (-1, 1) for k in eps_k]
    tiny, big = np.finfo(dtype).tiny, np.finfo(dtype).max
    pts += [-np.geomspace(tiny, 1, 60), -np.geomspace(1, big / 2, 120), [-big]]
    y = np.concatenate(pts).astype(dtype)
    y = np.concatenate([y, np.nextafter(y, dtype(0)), np.nextafter(y, dtype(-1))])
    return np.unique(y[(y < 0) & (y >= -big) & (np.abs(y) >= tiny)])


def _erfcx(x):
    if x < 50:
        return mp.exp(x * x) * mp.erfc(x)
    t, term, total, n = 1 / (2 * x * x), mp.mpf(1), mp.mpf(1), 1
    while abs(term) > mp.eps:
        term *= -(2 * n - 1) * t
        total += term
        n += 1
    return total / (x * mp.sqrt(mp.pi))


def _ref(y):
    """(z, dz/dy, d2z/dy2) for one float y, by Newton in log Phi on the side
    of the smaller tail probability u (log(1 - e^y) above the median)."""
    with mp.workdps(60):
        y = mp.mpf(float(y))
        upper = y > -mp.log(2)
        u = mp.log(-mp.expm1(y)) if upper else y
        if u > -3:
            z = mp.sqrt(2) * mp.erfinv(2 * mp.exp(u) - 1)
        else:
            s = mp.sqrt(-2 * u)
            z = -(s - (mp.log(s) + mp.log(2 * mp.pi) / 2) / s)
        for _ in range(100):
            x = -z / mp.sqrt(2)
            step = (mp.log(_erfcx(x) / 2) - x * x - u) * mp.sqrt(mp.pi / 2) * _erfcx(x)
            z -= step
            if abs(step) <= (abs(z) + mp.mpf(10) ** -40) * mp.eps * 16:
                break
        # dz/dy = e^y / phi(z) = Phi(z)/phi(z) below the median
        m = mp.sqrt(mp.pi / 2) * _erfcx(-z / mp.sqrt(2))
        if upper:
            z, m = -z, m * mp.exp(y) / -mp.expm1(y)
        return float(z), float(m), float(m * (1 + z * m))


def _refs(y):
    return [np.array(v) for v in zip(*(_ref(t) for t in y))]


Y = _grid(np.float64)


def _evaluate(y):
    f = chebax.ndtri_exp
    funs = (f, jax.vmap(jax.grad(f)), jax.vmap(jax.grad(jax.grad(f))))
    return [np.asarray(jax.jit(g)(jnp.asarray(y)), float) for g in funs]


def _errors(y, got, ref, big=np.finfo(np.float64).max):
    """Worst relative z, dz/dy and d2z/dy2 errors; d2 only for |y| <= 1e12
    and where it stays below big."""
    for g in got[:2]:
        assert np.all(np.isfinite(g)), y[~np.isfinite(g)]
    e = [np.abs(got[k] - ref[k]) / np.abs(ref[k]) for k in (0, 1)]
    d2 = (np.abs(y) <= 1e12) & (np.abs(ref[2]) < big)
    assert np.all(np.isfinite(got[2][d2])), y[d2 & ~np.isfinite(got[2])]
    e.append(np.abs(got[2][d2] - ref[2][d2]) / np.abs(ref[2][d2]))
    return [float(v.max()) for v in e]


def test_values_and_derivatives():
    errs = _errors(Y, _evaluate(Y), _refs(Y))
    assert np.all(np.array(errs) <= [2e-15, 2.5e-15, 2.2e-14]), errs


def test_relative_next_to_the_median():
    # z(-log 2 rounded) = 2.9e-17, not 0: d = y + log 2 is exact there
    y = np.array([-LN2, np.nextafter(-LN2, 0), np.nextafter(-LN2, -1)])
    got = np.asarray(chebax.ndtri_exp(jnp.asarray(y)))
    want = _refs(y)[0]
    assert np.all(np.abs(got - want) <= 4e-16 * np.abs(want)), (got, want)


def test_forward_and_reverse_agree():
    y = jnp.asarray(Y[Y > -1e6])
    rev = jax.vmap(jax.grad(chebax.ndtri_exp))(y)
    fwd = jax.jvp(chebax.ndtri_exp, (y,), (jnp.ones_like(y),))[1]
    np.testing.assert_allclose(np.asarray(fwd), np.asarray(rev), rtol=5e-16, atol=0)


def test_eager_matches_jit():
    # XLA fuses the jitted program differently: 1 ulp apart at a few points
    y = jnp.asarray(Y)
    np.testing.assert_allclose(
        np.asarray(chebax.ndtri_exp(y)),
        np.asarray(jax.jit(chebax.ndtri_exp)(y)),
        rtol=5e-16,
        atol=0,
    )


def test_row_gather_matches_per_coefficient():
    # the GPU's form (default=row_series), which CPU-only CI never lowers;
    # the two programs round differently at a few points, by 1 ulp
    y = jnp.asarray(Y)
    want = [np.asarray(jax.jit(f)(y)) for f in (N._value, N._slope)]
    saved, N._series = N._series, _common.row_series
    try:
        got = [
            np.asarray(jax.jit(lambda v, f=f: f(v))(y)) for f in (N._value, N._slope)
        ]
    finally:
        N._series = saved
    for w, g in zip(want, got):
        np.testing.assert_allclose(g, w, rtol=5e-16, atol=0)


def test_edges_of_the_domain():
    f, df = chebax.ndtri_exp, jax.grad(chebax.ndtri_exp)
    for y in (0.0, -0.0):
        assert float(f(y)) == np.inf and float(df(y)) == np.inf
    assert float(f(-np.inf)) == -np.inf and float(df(-np.inf)) == 0.0
    for y in (1e-300, 0.5, np.inf, np.nan):
        assert np.isnan(float(f(y))) and np.isnan(float(df(y)))
    # integers are floats: -3 is e^-3
    assert float(f(-3)) == float(f(-3.0))
    # the largest float: z = -sqrt(2 * 1.8e308), where -2y overflows
    big = np.finfo(np.float64).max
    want = -float(mp.sqrt(2 * mp.mpf(big)))
    assert abs(float(f(-big)) - want) <= 2.3e-16 * abs(want)


_F32 = """
import numpy as np, jax, jax.numpy as jnp, chebax, json, sys
assert jnp.empty(()).dtype == jnp.float32
y = jnp.asarray(np.array(json.loads(sys.stdin.read()), np.float32))
f = chebax.ndtri_exp
outs = [f(y), jax.vmap(jax.grad(f))(y), jax.vmap(jax.grad(jax.grad(f)))(y)]
assert all(o.dtype == jnp.float32 for o in outs)
print(json.dumps([np.asarray(o, float).tolist() for o in outs]))
"""


def test_float32():
    import json

    y = _grid(np.float32)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, "-c", _F32],
        input=json.dumps(y.astype(float).tolist()),
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    got = [np.array(v) for v in json.loads(out.stdout)]
    big = float(np.finfo(np.float32).max)
    errs = _errors(y.astype(float), got, _refs(y), big)
    assert np.all(np.array(errs) <= [1.1e-6, 1.3e-6, 6e-6]), errs


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    import pathlib

    ndtri_exp_gen.main(tmp_path)
    assert (tmp_path / "ndtri_exp_table.py").read_text() == pathlib.Path(
        nt.__file__
    ).read_text()
