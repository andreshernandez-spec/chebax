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

import chebax  # noqa: E402
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


def test_betainc_values_drop_the_loop_in_box():
    # betainc VALUES used to stay jax's on every domain (only the a/b
    # gradients came from chebax), so the measured 79-133x reached no PyMC
    # user at all. Same lax.cond as the gamma pair: in box the loop is
    # gone, outside it jax's answers, and select would run both.
    from chebax.pytensor import _betainc_scalar_ab

    xs = jnp.zeros(8)
    in_box = jax.jit(lambda x: _betainc_scalar_ab(jnp.float64(2.5),
                                                  jnp.float64(3.5), x))
    assert in_box.lower(xs).compile().as_text().count("while(") == 0
    out_box = jax.jit(lambda x: _betainc_scalar_ab(jnp.float64(150.0),
                                                   jnp.float64(3.5), x))
    assert out_box.lower(xs).compile().as_text().count("while(") > 0
    plain = jax.jit(lambda x: jax.scipy.special.betainc(jnp.full(8, 2.5),
                                                        jnp.full(8, 3.5), x))
    assert plain.lower(xs).compile().as_text().count("while(") > 0


def test_forward_ops_keep_a_float32_graph_float32():
    # The tables are f64, so chebax answers f64 whatever it is handed. A
    # cond whose branches disagree does not trace AT ALL, so a
    # floatX="float32" model died where the op was lowered; widening to
    # f64 to make it trace would silently change the model's dtype
    # instead. Both cond'd registrations answer in jax's own dtype.
    a, b = pt.fscalar("a"), pt.fscalar("b")
    x = pt.fvector("x")
    xv = np.linspace(0.05, 0.95, 5, dtype=np.float32)
    # bar is absolute, the CDF convention; measured worst 5.0e-7 (betainc)
    # and 7.8e-7 (gammainc), which is the f32 round-trip and not the tables
    fb = jaxify([a, b, x], pt.betainc(a, b, x))
    got = fb(np.float32(2.5), np.float32(3.5), xv)
    assert got.dtype == np.float32
    np.testing.assert_allclose(np.asarray(got, np.float64),
                               sps.betainc(2.5, 3.5, xv.astype(np.float64)),
                               atol=5e-6)
    fg = jaxify([a, x], pt.gammainc(a, x))
    xg = np.linspace(0.5, 5.0, 5, dtype=np.float32)
    gotg = fg(np.float32(3.0), xg)
    assert gotg.dtype == np.float32
    np.testing.assert_allclose(np.asarray(gotg, np.float64),
                               sps.gammainc(3.0, xg.astype(np.float64)),
                               atol=5e-6)


def test_betainc_out_of_box_values_fine_grads_nan():
    a, b = pt.dscalar("a"), pt.dscalar("b")
    x = pt.dvector("x")
    f = jaxify([a, b, x], pt.betainc(a, b, x))
    xv = np.array([0.3, 0.7])
    got = np.asarray(f(150.0, 3.5, xv))  # a outside [0.1, 100]
    np.testing.assert_allclose(got, sps.betainc(150.0, 3.5, xv), rtol=1e-12)
    ga = jax.grad(lambda av: jnp.sum(f(av, 3.5, jnp.asarray(xv))))(150.0)
    assert np.isnan(float(ga))


def test_betainc_batched_shapes_keep_plain_jax():
    # a betainc with per-element (a, b) has no scalar predicate to branch
    # on, so it stays jax's op: values right everywhere, a/b gradients
    # missing exactly as in stock pytensor. Unlike the inverse CDFs this
    # never raises, because the plain jax op is a complete answer.
    a, b = pt.dvector("a"), pt.dvector("b")
    x = pt.dvector("x")
    f = jaxify([a, b, x], pt.betainc(a, b, x))
    av = np.array([2.5, 9.0])
    bv = np.array([3.5, 1.5])
    xv = np.array([0.3, 0.7])
    np.testing.assert_allclose(np.asarray(f(av, bv, xv)),
                               sps.betainc(av, bv, xv), rtol=1e-12)


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
    xv = np.array([-3.0, 0.0, 1.3, 20.0])
    if hasattr(jax.scipy.special, "erfcx"):
        fx = jaxify([x], pt.erfcx(x))
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


def test_forward_gamma_cdfs_values_and_gradients():
    # The forward pair was unregistered, so every censored or truncated
    # Gamma, Poisson, ChiSquared or NegativeBinomial on PyMC's JAX path kept
    # jax's looped igamma (experiments/19). Metric is absolute, the CDF
    # convention; measured worst 4.9e-14 at a = 60, bar 1e-12.
    a = pt.dscalar("a")
    x = pt.dvector("x")
    xs = jnp.asarray([0.05, 0.5, 2.0, 7.0, 30.0])
    for op, ref in ((pt.gammainc, sps.gammainc), (pt.gammaincc, sps.gammaincc)):
        f = jaxify([a, x], op(a, x))
        for av in (0.5, 3.0, 60.0):
            got = np.asarray(f(jnp.float64(av), xs))
            assert np.max(np.abs(got - ref(av, np.asarray(xs)))) <= 1e-12, (op, av)
    # dP/da through jax.grad of the LOWERED function, which is how PyMC's
    # JAX samplers differentiate a logp. pt.grad is a different path: it
    # builds pytensor's own ScalarLoop gradient graph, which has no JAX
    # dispatch (pytensor#299, open since 2023) and is not what this
    # registration touches. Reference is jax's own igamma_grad_a.
    f = jaxify([a, x], pt.gammainc(a, x))

    def total(av):
        return jnp.sum(f(av, xs))

    for av in (0.5, 3.0, 60.0):
        got = float(jax.grad(total)(jnp.float64(av)))
        ref = float(jnp.sum(jax.lax.igamma_grad_a(jnp.full(xs.shape, av), xs)))
        assert abs(got - ref) <= 1e-10 * max(1.0, abs(ref)), (av, got, ref)


def test_forward_gamma_falls_back_below_the_box():
    # a plain forward function must answer everywhere, so below chebax's
    # a >= 0.1 box this hands off to jax's igamma rather than returning nan
    # the way an inverse CDF does. lax.cond, never lax.select: select would
    # evaluate jax's while_loop on every lane and give back exactly what the
    # registration removes.
    a = pt.dscalar("a")
    x = pt.dvector("x")
    xs = jnp.asarray([0.01, 0.2, 1.0])
    f = jaxify([a, x], pt.gammainc(a, x))
    for av in (0.05, 0.02):
        got = np.asarray(f(jnp.float64(av), xs))
        assert np.all(np.isfinite(got)), (av, got)
        assert np.max(np.abs(got - sps.gammainc(av, np.asarray(xs)))) <= 1e-12


def test_forward_gamma_drops_the_loop_for_a_concrete_shape():
    # the point of the registration. A concrete shape lets XLA fold the
    # predicate and delete jax's branch; a TRACED shape (the sampled-alpha
    # case) keeps it in the module, where the conditional does not execute
    # it. Both are correct; only the first is visible in the HLO.
    from chebax.pytensor import _forward_gamma

    fwd = _forward_gamma(None, "igamma", chebax.gammainc_fn,
                         jax.scipy.special.gammainc)
    xs = jnp.zeros(8)
    concrete = jax.jit(lambda x: fwd(jnp.float64(3.0), x))
    assert concrete.lower(xs).compile().as_text().count("while(") == 0
    plain = jax.jit(lambda x: jax.scipy.special.gammainc(jnp.full(8, 3.0), x))
    assert plain.lower(xs).compile().as_text().count("while(") > 0
