"""chebax.pergroup: wiring tests for the per-group parameter wrapper.

The contract under test is exactness against the uniform path: the
wrapper must return what fn(params[g_i]..., x[i]) returns elementwise
(same tables, same arithmetic, so agreement is at rounding level, bars
at 5e-15 relative). Accuracy contracts live in the per-recipe test
files. Crossover cost is measured in experiments/07, not here.

The other contract is that an EMPTY group is inert: its parameters never
reach fn, so a nan or an out-of-table value parked there changes neither
the values nor any gradient, and its own gradient is exactly zero. The
mpmath anchor for the live groups next to a poisoned empty one is at 50
dps; measured worst there, besselk values 4.8e-15, dK/dnu 1.4e-15,
dK/dx 4.3e-15, bars at 2e-14.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax


def _loop_reference(fn, params, idx, x):
    out = np.empty(x.size)
    xf, gf = x.reshape(-1), idx.reshape(-1)
    for i in range(xf.size):
        out[i] = float(fn(*[p[gf[i]] for p in params], xf[i]))
    return out.reshape(x.shape)


def test_betainc_ragged_shuffled_with_empty_group():
    rng = np.random.default_rng(0)
    # ragged groups in shuffled order, group 4 empty, 2-D x
    idx = rng.choice([0, 1, 2, 3, 5], size=(9, 7), p=[.4, .3, .15, .1, .05])
    a = rng.uniform(0.5, 9.5, 6)
    b = rng.uniform(0.5, 9.5, 6)
    x = rng.uniform(0.01, 0.99, idx.shape)

    f = chebax.pergroup(chebax.betainc_fn, idx, num_groups=6)
    got = np.asarray(f(a, b, x))
    ref = _loop_reference(chebax.betainc_fn, (a, b), idx, x)
    np.testing.assert_allclose(got, ref, rtol=5e-15)
    assert got.shape == idx.shape


def test_gradients_match_loop_and_empty_group_is_zero():
    rng = np.random.default_rng(1)
    idx = np.array([2, 0, 0, 2, 0, 2, 2])  # group 1 empty
    a = jnp.asarray(rng.uniform(1.0, 5.0, 3))
    b = jnp.asarray(rng.uniform(1.0, 5.0, 3))
    x = jnp.asarray(rng.uniform(0.1, 0.9, idx.shape))

    f = chebax.pergroup(chebax.betainc_fn, idx, num_groups=3)
    ga, gb = jax.grad(lambda a, b: jnp.sum(f(a, b, x)), (0, 1))(a, b)

    def loop_sum(a, b):
        return sum(chebax.betainc_fn(a[g], b[g], x[i])
                   for i, g in enumerate(idx))

    ga_ref, gb_ref = jax.grad(loop_sum, (0, 1))(a, b)
    np.testing.assert_allclose(np.asarray(ga), np.asarray(ga_ref), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(gb), np.asarray(gb_ref), rtol=1e-13)
    assert float(ga[1]) == 0.0 and float(gb[1]) == 0.0


def test_besselk_single_param_and_order_gradient():
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 4, 40)
    nu = jnp.asarray(rng.uniform(0.3, 9.5, 4))
    x = jnp.asarray(np.exp(rng.uniform(np.log(0.2), np.log(50.0), 40)))

    f = chebax.pergroup(chebax.besselk_fn, idx)
    got = np.asarray(f(nu, x))
    ref = _loop_reference(chebax.besselk_fn, (np.asarray(nu),), idx,
                         np.asarray(x))
    np.testing.assert_allclose(got, ref, rtol=5e-15)
    g = jax.grad(lambda nu: jnp.sum(f(nu, x)))(nu)
    assert np.isfinite(np.asarray(g)).all()


def test_jit_and_traced_parameters():
    idx = np.array([0, 1, 1, 0, 1])
    x = jnp.asarray([0.2, 0.4, 0.6, 0.8, 0.5])
    f = jax.jit(chebax.pergroup(chebax.betainc_fn, idx))
    a, b = jnp.asarray([2.0, 6.0]), jnp.asarray([3.0, 1.5])
    got = np.asarray(f(a, b, x))
    ref = _loop_reference(chebax.betainc_fn,
                          (np.asarray(a), np.asarray(b)), idx, np.asarray(x))
    np.testing.assert_allclose(got, ref, rtol=5e-15)
    # values are free to change without retrace cost beyond the first
    got2 = np.asarray(f(a + 1.0, b, x))
    ref2 = _loop_reference(chebax.betainc_fn,
                           (np.asarray(a) + 1.0, np.asarray(b)), idx,
                           np.asarray(x))
    np.testing.assert_allclose(got2, ref2, rtol=5e-15)


EMPTY_LAYOUTS = [
    (np.array([0, 0, 2, 2, 0, 2]), 3, [1]),             # empty in the middle
    (np.array([0, 1, 0, 1, 1]), 4, [2, 3]),             # empty at the end
    (np.array([3, 3, 7, 1, 7, 3]), 8, [0, 2, 4, 5, 6]),  # several, group 0 too
]


@pytest.mark.parametrize("idx,ng,empty", EMPTY_LAYOUTS)
def test_empty_group_parameter_is_inert(idx, ng, empty):
    # 50.0 is outside besselk's nu table, nan and inf are worse: all three
    # used to reach fn and come back as 0 * nan in reverse mode
    rng = np.random.default_rng(3)
    x = jnp.asarray(np.exp(rng.uniform(np.log(0.2), np.log(30.0), idx.shape)))
    clean = rng.uniform(0.3, 9.5, ng)
    f = chebax.pergroup(chebax.besselk_fn, idx, num_groups=ng)

    def run(nu):
        nu = jnp.asarray(nu)
        v = f(nu, x)
        gnu, gx = jax.grad(lambda nu, x: jnp.sum(f(nu, x)), (0, 1))(nu, x)
        _, t = jax.jvp(lambda nu, x: f(nu, x), (nu, x),
                       (jnp.ones(ng), jnp.ones(idx.shape)))
        return [np.asarray(u) for u in (v, gnu, gx, t)]

    ref = run(clean)
    for poison in (np.nan, np.inf, 50.0):
        bad = clean.copy()
        bad[empty] = poison
        got = run(bad)
        for r, gt in zip(ref, got):
            np.testing.assert_array_equal(gt, r)
            assert np.isfinite(gt).all()
        assert (got[1][empty] == 0.0).all()


def test_empty_group_inert_with_two_parameters():
    idx = np.array([0, 0, 2, 2, 0])
    x = jnp.asarray([0.2, 0.45, 0.6, 0.85, 0.33])
    a0, b0 = np.array([2.0, 3.0, 4.5]), np.array([3.0, 2.5, 1.5])
    f = chebax.pergroup(chebax.betainc_fn, idx, num_groups=3)

    def run(a, b):
        a, b = jnp.asarray(a), jnp.asarray(b)
        v = f(a, b, x)
        g = jax.grad(lambda a, b, x: jnp.sum(f(a, b, x)), (0, 1, 2))(a, b, x)
        return [np.asarray(u) for u in (v,) + g]

    ref = run(a0, b0)
    for pa, pb in [(1e6, 2.5), (np.nan, np.nan), (-1.0, 0.0)]:
        a, b = a0.copy(), b0.copy()
        a[1], b[1] = pa, pb
        got = run(a, b)
        for r, gt in zip(ref, got):
            np.testing.assert_array_equal(gt, r)
            assert np.isfinite(gt).all()
        assert got[1][1] == 0.0 and got[2][1] == 0.0


def test_empty_group_neighbours_match_mpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    rng = np.random.default_rng(11)
    idx = np.array([0, 2, 0, 2, 2, 0, 2, 0])
    nu = np.array([rng.uniform(0.3, 9.0), np.nan, rng.uniform(0.3, 9.0), 1e9])
    x = np.exp(rng.uniform(np.log(0.2), np.log(40.0), 8))
    f = chebax.pergroup(chebax.besselk_fn, idx, num_groups=4)

    v = np.asarray(f(jnp.asarray(nu), jnp.asarray(x)))
    ref = np.array([float(mp.besselk(mp.mpf(nu[g]), mp.mpf(xi)))
                    for g, xi in zip(idx, x)])
    np.testing.assert_allclose(v, ref, rtol=2e-14)

    gnu, gx = jax.grad(lambda nu, x: jnp.sum(f(nu, x)), (0, 1))(
        jnp.asarray(nu), jnp.asarray(x))
    for g in (0, 2):
        d = sum((mp.diff(lambda t: mp.besselk(t, mp.mpf(xi)), mp.mpf(nu[g]))
                 for i, xi in enumerate(x) if idx[i] == g), mp.mpf(0))
        assert abs(float(gnu[g]) - float(d)) <= 2e-14 * abs(float(d))
    assert float(gnu[1]) == 0.0 and float(gnu[3]) == 0.0

    gxref = np.array([float(mp.diff(lambda t: mp.besselk(mp.mpf(nu[g]), t),
                                    mp.mpf(xi))) for g, xi in zip(idx, x)])
    np.testing.assert_allclose(np.asarray(gx), gxref, rtol=2e-14)


def test_errors():
    idx = np.array([0, 1, 2])
    f = chebax.pergroup(chebax.betainc_fn, idx)
    a3 = jnp.ones(3)
    with pytest.raises(ValueError, match="shape"):
        f(a3, a3, jnp.ones(4))            # x shape != group_idx shape
    with pytest.raises(ValueError, match="parameter"):
        f(jnp.ones(2), a3, jnp.ones(3))   # param length != num_groups
    with pytest.raises(ValueError, match="lie in"):
        chebax.pergroup(chebax.betainc_fn, idx, num_groups=2)
    with pytest.raises(ValueError, match="integers"):
        chebax.pergroup(chebax.betainc_fn, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="empty"):
        chebax.pergroup(chebax.betainc_fn, np.array([], dtype=int))
