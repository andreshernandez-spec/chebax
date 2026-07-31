"""chebax.pergroup: wiring tests for the per-group parameter wrapper.

The contract under test is exactness against the uniform path: the
wrapper must return what fn(params[g_i]..., x[i]) returns elementwise
(same tables, same arithmetic, so agreement is at rounding level, bars
at 5e-15 relative). Accuracy contracts live in the per-recipe test
files. Crossover cost is measured in experiments/07, not here.
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
