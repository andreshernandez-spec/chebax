"""Lambert W, both real branches, against mpmath. Away from the branch
point the bar is 5e-15 * (1 + |W|); within 1e-4 of x = -1/e any double
routine is conditioning-limited by the cancellation in 1 + e x, so the bar
there is 4 sqrt(eps e / (x + 1/e)). Gradients are the implicit identity,
checked against mp.diff. Measured worst: 8.6e-16 (k=0), 6.4e-16 (k=-1)."""

import jax
import numpy as np
import pytest

import chebax

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40
BR = -float(mp.exp(-1))


def _check(xs, k):
    w = np.asarray(chebax.lambertw(xs, k))
    for x, v in zip(xs, w):
        ref = float(mp.lambertw(mp.mpf(x), k).real)
        e = abs(v - ref) / (1 + abs(ref))
        if x - BR < 1e-4:
            assert e <= 4 * np.sqrt(2.2e-16 * np.e / max(x - BR, 2.2e-16)), (k, x)
        else:
            assert e <= 5e-15, (k, x, e)


def test_branch0():
    xs = np.concatenate([BR + np.array([1e-12, 1e-9, 1e-6, 1e-3]),
                         [-0.3, -0.1, -0.01, 0.0, 0.3, 1.0, 2.0, np.e, 5.0],
                         np.logspace(1, 300, 20)])
    _check(xs, 0)


def test_branch_m1():
    xs = np.concatenate([BR + np.array([1e-12, 1e-9, 1e-6, 1e-3]),
                         [-0.3, -0.25, -0.1, -0.01], -np.logspace(-300, -3, 20)])
    _check(xs, -1)


def test_gradients():
    for x, k in [(0.5, 0), (5.0, 0), (-0.2, 0), (-0.2, -1), (-0.05, -1)]:
        g = float(jax.grad(lambda t: chebax.lambertw(t, k))(x))
        r = float(mp.diff(lambda t: mp.lambertw(t, k).real, mp.mpf(x)))
        assert abs(g - r) / abs(r) <= 1e-12, (x, k)


def test_invalid_inputs_nan():
    assert np.isnan(float(chebax.lambertw(-0.5, 0)))
    assert np.isnan(float(chebax.lambertw(0.5, -1)))
    with pytest.raises(ValueError, match="branches"):
        chebax.lambertw(1.0, 2)


def test_jit():
    import jax.numpy as jnp
    xs = jnp.asarray([0.5, 2.0, 10.0])
    np.testing.assert_allclose(np.asarray(jax.jit(chebax.lambertw)(xs)),
                               np.asarray(chebax.lambertw(xs)), rtol=0, atol=1e-15)


# ---- review 2026-07-30 regressions ------------------------------------------

def test_branch_point_both_roundings():
    # 1/math.e and math.exp(-1) can round to different doubles; both must
    # be accepted as the branch point (the old check rejected one with nan)
    import math
    for x in (-1.0 / math.e, -math.exp(-1.0)):
        assert float(chebax.lambertw(x)) == -1.0
        assert float(chebax.lambertw(x, k=-1)) == -1.0


def test_inf_and_boundary_derivatives():
    import math
    assert float(chebax.lambertw(np.inf)) == np.inf
    assert np.isnan(float(chebax.lambertw(np.nan)))
    br = -math.exp(-1.0)
    assert float(jax.grad(chebax.lambertw)(br)) == np.inf
    assert float(jax.grad(lambda x: chebax.lambertw(x, k=-1))(br)) == -np.inf
