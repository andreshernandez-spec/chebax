"""gammaln acceptance: values, digamma (its derivative) and trigamma (the
second), against mpmath at 40 dps.

Values are pointwise-relative everywhere, next to the zeros at 1 and 2
included (1 +- 10^-k), and exact zeros there. Digamma is relative except
within 0.05 of its zero at 1.4616, where it is absolute; trigamma is
relative. The grid holds every panel edge and its neighbours, the points
below 1 whose x + 1 is a first-octave edge, both sides of the Stirling
seam at 64, and log-spaced points from 1e-300 to 1e300.

Worst on this grid, the same on jax 0.11 and 0.5.3: float64 value 3.5e-16,
digamma 3.9e-16, trigamma 7.2e-16 (bars 1.4e-15, 1.6e-15, 3e-15); float32
1.7e-7, 1.2e-7, 1.8e-7 (bars 7e-7, 5e-7, 7e-7). The grid stops at the
smallest normal (XLA flushes subnormal x to 0, where lgamma is inf), and
trigamma is checked only where x^2 is normal.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import gammaln as G
from chebax._src.recipes import gammaln_gen
from chebax._src.recipes import gammaln_table as gt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

X0 = 1.4616321449683623  # digamma's positive zero


def _grid(dtype):
    edges = np.concatenate([2.0**e * (1 + np.arange(33) / gt.K) for e in range(6)])
    below = np.arange(gt.K + 1) / gt.K  # x + 1 on the first octave's edges
    pts = [edges, below, (edges[1:] + edges[:-1]) / 2]
    pts += [np.nextafter(pts[0], 0), np.nextafter(pts[0], 100)]
    pts += [np.nextafter(below, -1), np.nextafter(below, 2)]
    eps_k = range(1, 16) if dtype == np.float64 else range(1, 8)
    pts += [
        np.array([c + d * 10.0**-k]) for c in (1, 2) for d in (-1, 1) for k in eps_k
    ]
    pts += [np.geomspace(1e-300 if dtype == np.float64 else 1e-37, 1, 40)]
    pts += [np.geomspace(64, 1e300 if dtype == np.float64 else 1e36, 40)]
    x = np.unique(np.concatenate(pts).astype(dtype))
    # XLA flushes subnormals to 0, where lgamma is inf
    return x[x >= np.finfo(dtype).tiny]


def _refs(x):
    m = [mp.mpf(float(v)) for v in x]
    return [
        np.array([float(f(v)) for v in m])
        for f in (mp.loggamma, mp.digamma, lambda v: mp.psi(1, v))
    ]


X = _grid(np.float64)


def _errors(x, got, ref, tiny):
    """Worst value, digamma and trigamma errors in this file's metrics."""
    zero = ref[0] == 0
    assert np.all(got[0][zero] == 0), x[zero & (got[0] != 0)]
    e = [np.abs(got[0] - ref[0])[~zero] / np.abs(ref[0][~zero])]
    near = np.abs(x - X0) < 0.05
    e.append(np.abs(got[1] - ref[1]) / np.where(near, 1.0, np.abs(ref[1])))
    t = x >= tiny
    e.append(np.abs(got[2][t] - ref[2][t]) / ref[2][t])
    return [float(v.max()) for v in e]


def _evaluate(f, x):
    d1 = jax.vmap(jax.grad(f))
    d2 = jax.vmap(jax.grad(jax.grad(f)))
    return [np.asarray(jax.jit(g)(jnp.asarray(x))) for g in (f, d1, d2)]


def test_values_and_derivatives():
    got = _evaluate(chebax.gammaln, X)
    for k, g in enumerate(got):
        fine = X >= 1e-150 if k == 2 else np.ones(X.shape, bool)
        assert np.all(np.isfinite(g[fine])), (k, X[fine & ~np.isfinite(g)])
    errs = _errors(X, got, _refs(X), 1e-150)
    assert np.all(np.array(errs) <= [1.4e-15, 1.6e-15, 3e-15]), errs


def test_eager_matches_jit():
    # XLA fuses the jitted program differently: 1 ulp apart at a few points
    x = jnp.asarray(X)
    np.testing.assert_allclose(
        np.asarray(chebax.gammaln(x)),
        np.asarray(jax.jit(chebax.gammaln)(x)),
        rtol=5e-16,
        atol=0,
    )


def test_row_gather_matches_per_coefficient():
    # the GPU's form (default=_row_series), which CPU-only CI never lowers;
    # the two programs round differently at a few points, by 1 ulp
    x = jnp.asarray(X)
    want = [np.asarray(jax.jit(f)(x)) for f in (G._lgamma, G._digamma)]
    saved, G._series = G._series, G._row_series
    try:
        got = [
            np.asarray(jax.jit(lambda v, f=f: f(v))(x)) for f in (G._lgamma, G._digamma)
        ]
    finally:
        G._series = saved
    for w, g in zip(want, got):
        np.testing.assert_allclose(g, w, rtol=5e-16, atol=0)


def test_edges_of_the_domain():
    f = chebax.gammaln
    df = jax.grad(f)
    assert float(f(1.0)) == 0.0 and float(f(2.0)) == 0.0
    for x in (0.0, -0.0, np.inf):
        assert float(f(x)) == np.inf
    assert float(df(0.0)) == -np.inf
    assert float(df(np.inf)) == np.inf
    for x in (-0.5, -1.0, -np.inf, np.nan):
        assert np.isnan(float(f(x))) and np.isnan(float(df(x)))
    # past the largest float: lgamma(2.5e305) = 1.76e308, lgamma(2.6e305) overflows
    assert float(f(2.6e305)) == np.inf
    want = float(mp.loggamma(mp.mpf(2.5e305)))
    assert abs(float(f(2.5e305)) - want) <= 2e-16 * want


def test_forward_and_reverse_agree():
    x = jnp.asarray(X[(X > 1e-6) & (X < 1e6)])
    rev = jax.vmap(jax.grad(chebax.gammaln))(x)
    fwd = jax.jvp(chebax.gammaln, (x,), (jnp.ones_like(x),))[1]
    np.testing.assert_allclose(np.asarray(fwd), np.asarray(rev), rtol=5e-16, atol=0)


_F32 = """
import numpy as np, jax, jax.numpy as jnp, chebax, json, sys
assert jnp.empty(()).dtype == jnp.float32
x = jnp.asarray(np.array(json.loads(sys.stdin.read()), np.float32))
f = chebax.gammaln
outs = [f(x), jax.vmap(jax.grad(f))(x), jax.vmap(jax.grad(jax.grad(f)))(x)]
assert all(o.dtype == jnp.float32 for o in outs)
print(json.dumps([np.asarray(o, float).tolist() for o in outs]))
"""


def test_float32():
    import json

    x = _grid(np.float32)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, "-c", _F32],
        input=json.dumps(x.astype(float).tolist()),
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    got = [np.array(v) for v in json.loads(out.stdout)]
    tiny = 2 * np.sqrt(np.finfo(np.float32).tiny)  # x^2 stays normal
    errs = _errors(x.astype(float), got, _refs(x), tiny)
    assert np.all(np.array(errs) <= [7e-7, 5e-7, 7e-7]), errs


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    import pathlib

    gammaln_gen.main(tmp_path)
    assert (tmp_path / "gammaln_table.py").read_text() == pathlib.Path(
        gt.__file__
    ).read_text()
