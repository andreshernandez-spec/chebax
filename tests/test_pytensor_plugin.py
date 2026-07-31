"""chebax.pytensor: wiring tests for the opt-in JAX dispatch module.

These test the PLUGIN (registration, guards, fallbacks, gradient
plumbing on the numpyro-style path: jax_funcify the graph, then
jax.grad). Accuracy contracts live in the per-recipe test files; scipy
here is a wiring oracle at loose tolerances, not an accuracy reference.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytensor = pytest.importorskip("pytensor")
scipy = pytest.importorskip("scipy")

import pytensor.tensor as pt  # noqa: E402
from pytensor.graph.fg import FunctionGraph  # noqa: E402
from pytensor.link.jax.dispatch import jax_funcify  # noqa: E402
from scipy import special as sps  # noqa: E402

import chebax.pytensor  # noqa: E402,F401  (importing registers the dispatches)


def jaxify(inputs, output):
    fg = FunctionGraph([*inputs], [output], clone=True)
    fn = jax_funcify(fg)
    return lambda *args: fn(*args)[0]


def test_betainc_values_and_shape_gradients():
    a, b = pt.dscalar("a"), pt.dscalar("b")
    x = pt.dvector("x")
    f = jaxify([a, b, x], pt.betainc(a, b, x))
    xv = np.linspace(0.05, 0.95, 11)
    got = np.asarray(f(2.5, 3.5, xv))
    np.testing.assert_allclose(got, sps.betainc(2.5, 3.5, xv), rtol=1e-12)

    # the gap this plugin closes: grad w.r.t. a and b on the jax path
    def mean_out(av, bv):
        return jnp.mean(f(av, bv, jnp.asarray(xv)))

    ga, gb = jax.grad(mean_out, (0, 1))(2.5, 3.5)
    h = 1e-6
    ga_fd = (sps.betainc(2.5 + h, 3.5, xv).mean()
             - sps.betainc(2.5 - h, 3.5, xv).mean()) / (2 * h)
    gb_fd = (sps.betainc(2.5, 3.5 + h, xv).mean()
             - sps.betainc(2.5, 3.5 - h, xv).mean()) / (2 * h)
    assert abs(float(ga) - ga_fd) <= 1e-6
    assert abs(float(gb) - gb_fd) <= 1e-6

    # grad w.r.t. x is the Beta density
    gx = jax.grad(lambda xv1: f(2.5, 3.5, jnp.asarray([xv1]))[0])(0.4)
    pdf = 0.4 ** 1.5 * 0.6 ** 2.5 / sps.beta(2.5, 3.5)
    assert abs(float(gx) - pdf) <= 1e-10


def test_betainc_out_of_box_values_fine_grads_nan():
    a, b = pt.dscalar("a"), pt.dscalar("b")
    x = pt.dvector("x")
    f = jaxify([a, b, x], pt.betainc(a, b, x))
    xv = np.array([0.3, 0.7])
    got = np.asarray(f(20.0, 3.5, xv))  # a outside [0.1, 10]
    np.testing.assert_allclose(got, sps.betainc(20.0, 3.5, xv), rtol=1e-12)
    ga = jax.grad(lambda av: jnp.sum(f(av, 3.5, jnp.asarray(xv))))(20.0)
    assert np.isnan(float(ga))


def test_quantile_ops_values_and_gradients():
    a, b, p = pt.dscalar("a"), pt.dscalar("b"), pt.dvector("p")
    pv = np.array([0.05, 0.5, 0.95])

    f = jaxify([a, b, p], pt.betaincinv(a, b, p))
    np.testing.assert_allclose(np.asarray(f(2.0, 3.0, pv)),
                               sps.betaincinv(2.0, 3.0, pv), rtol=1e-10)
    g = jax.grad(lambda av: jnp.sum(f(av, 3.0, jnp.asarray(pv))))(2.0)
    assert np.isfinite(float(g))

    fg = jaxify([a, p], pt.gammaincinv(a, p))
    np.testing.assert_allclose(np.asarray(fg(3.5, pv)),
                               sps.gammaincinv(3.5, pv), rtol=1e-10)
    g = jax.grad(lambda av: jnp.sum(fg(av, jnp.asarray(pv))))(3.5)
    assert np.isfinite(float(g))

    fc = jaxify([a, p], pt.gammainccinv(a, p))
    np.testing.assert_allclose(np.asarray(fc(3.5, pv)),
                               sps.gammainccinv(3.5, pv), rtol=1e-10)


def test_bessel_ops():
    v, x = pt.dscalar("v"), pt.dvector("x")
    xv = np.array([0.5, 2.0, 8.0, 50.0])

    fi = jaxify([v, x], pt.ive(v, x))
    np.testing.assert_allclose(np.asarray(fi(1.7, xv)),
                               sps.ive(1.7, xv), rtol=1e-10)
    fk = jaxify([v, x], pt.kve(v, x))
    np.testing.assert_allclose(np.asarray(fk(2.5, xv)),
                               sps.kve(2.5, xv), rtol=1e-10)
    # kve stays finite far past where K underflows, and K_{-v} = K_v
    big = np.asarray(fk(-2.5, np.array([800.0])))
    np.testing.assert_allclose(big, sps.kve(2.5, 800.0), rtol=1e-10)
    # order gradient exists on the jax path (unique vs the tfp lowering)
    g = jax.grad(lambda vv: jnp.sum(fk(vv, jnp.asarray(xv))))(2.5)
    assert np.isfinite(float(g))
    # out-of-domain order is nan, not silently wrong
    assert np.isnan(np.asarray(fi(-1.7, np.array([2.0])))).all()


def test_erf_family_pure_jax():
    x = pt.dvector("x")
    fx = jaxify([x], pt.erfcx(x))
    xv = np.array([-3.0, 0.0, 1.3, 20.0])
    np.testing.assert_allclose(np.asarray(fx(xv)), sps.erfcx(xv), rtol=1e-12)
    fi = jaxify([x], pt.erfcinv(x))
    yv = np.array([0.1, 0.5, 1.0, 1.9])
    np.testing.assert_allclose(np.asarray(fi(yv)), sps.erfcinv(yv),
                               rtol=1e-10, atol=1e-12)


def test_batched_parameters_fall_back():
    try:
        import tensorflow_probability  # noqa: F401
        pytest.skip("tfp installed; fallback goes to tfp instead of raising")
    except ImportError:
        pass
    a, p = pt.dvector("a"), pt.dvector("p")
    f = jaxify([a, p], pt.gammaincinv(a, p))
    with pytest.raises(NotImplementedError, match="chebax.pytensor"):
        f(jnp.asarray([2.0, 3.0]), jnp.asarray([0.5, 0.5]))


def test_mode_jax_end_to_end():
    # the full linker path, not just jax_funcify
    a, b = pt.dscalar("a"), pt.dscalar("b")
    x = pt.dvector("x")
    fn = pytensor.function([a, b, x], pt.betaincinv(a, b, x), mode="JAX")
    pv = np.array([0.2, 0.8])
    np.testing.assert_allclose(np.asarray(fn(2.0, 3.0, pv)),
                               sps.betaincinv(2.0, 3.0, pv), rtol=1e-10)
