"""M2 acceptance tests for the besselj recipe (PROJECT.md section 4).

References are mpmath at 40 dps; errors sup-normalized per order. The f64
value and dJ/dx bars sit above the raw series floor because the (x/2)^v
prefactor evaluates through pow = exp(v*log), whose relative error grows
like v*|log(x/2)|*eps (~3e-15 at v=10). Measured worst cases over the 21
off-node orders below: 2.8e-15 value, 5.5e-15 dJ/dx, 9.2e-7 / 2.0e-6 f32.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import besselj_gen
from chebax._src.recipes import besselj_table as tab
from chebax._src.recipes.besselj import _digamma

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

_rng = np.random.default_rng(3)
NUS = np.concatenate([[0.5, 2.5], np.sort(_rng.uniform(0.05, 9.95, 19))])
XS = np.linspace(0.05, 8.0, 48)


@pytest.fixture(scope="module")
def refs():
    out = {}
    for v in NUS:
        J = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x))) for x in XS])
        Jp = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x), 1)) for x in XS])
        out[v] = (J, Jp)
    return out


def worst(err_fn):
    return max(err_fn(v) for v in NUS)


def test_values_f64(refs):
    def err(v):
        J, _ = refs[v]
        return np.max(np.abs(np.asarray(chebax.besselj(float(v))(XS)) - J)) / np.max(np.abs(J))
    assert worst(err) <= 5e-15


def test_dx_f64(refs):
    def err(v):
        _, Jp = refs[v]
        g = jax.vmap(jax.grad(chebax.besselj(float(v))))(jnp.asarray(XS))
        return np.max(np.abs(np.asarray(g) - Jp)) / np.max(np.abs(Jp))
    assert worst(err) <= 1e-14


def test_values_f32(refs):
    x32 = jnp.asarray(XS, jnp.float32)
    def err(v):
        J, _ = refs[v]
        y = chebax.besselj(float(v)).astype(jnp.float32)(x32)
        assert y.dtype == jnp.float32
        return np.max(np.abs(np.asarray(y) - J)) / np.max(np.abs(J))
    assert worst(err) <= 1e-6


def test_dx_f32(refs):
    x32 = jnp.asarray(XS, jnp.float32)
    def err(v):
        _, Jp = refs[v]
        g = jax.vmap(jax.grad(chebax.besselj(float(v)).astype(jnp.float32)))(x32)
        return np.max(np.abs(np.asarray(g) - Jp)) / np.max(np.abs(Jp))
    assert worst(err) <= 5e-6


def test_dnu():
    xd = np.linspace(0.3, 8.0, 16)
    for v in [0.3, 1.7, float(mp.pi), 5.5, 7.77, 9.6]:
        ref = np.array([float(mp.diff(lambda w: mp.besselj(w, mp.mpf(x)), mp.mpf(v)))
                        for x in xd])
        got = np.asarray(chebax.besselj_dnu(v)(xd))
        assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) <= 5e-15


def test_digamma_helper():
    for v in [0.0, 0.37, 0.5, 2.5, 7.77, 10.0]:
        assert abs(_digamma(v + 1.0) - float(mp.digamma(v + 1.0))) <= 1e-15


def test_x_zero():
    assert float(chebax.besselj(0.0)(0.0)) == 1.0
    assert float(chebax.besselj(2.5)(0.0)) == 0.0


def test_order_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.besselj(10.5)
    with pytest.raises(ValueError, match="table covers"):
        chebax.besselj_dnu(-0.1)


def test_factory_is_cached():
    assert chebax.besselj(2.5) is chebax.besselj(2.5)


def test_jit_with_static_order():
    xs = jnp.linspace(0.1, 8.0, 32)
    jv = chebax.besselj(2.5)
    y_arg = jax.jit(lambda q, x: q(x))(jv, xs)      # instance as pytree argument
    y_closed = jax.jit(lambda x: chebax.besselj(2.5)(x))(xs)  # built at trace time
    ref = np.asarray(jv(xs))
    np.testing.assert_allclose(np.asarray(y_arg), ref, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(y_closed), ref, rtol=0, atol=1e-15)


def test_table_regenerates_bit_for_bit():
    regen = besselj_gen.generate_table()
    assert regen.shape == tab.TABLE.shape
    assert np.array_equal(regen, tab.TABLE)
    assert tab.META["dps"] == besselj_gen.DPS
    assert tab.META["mpmath"] == mp.__version__
