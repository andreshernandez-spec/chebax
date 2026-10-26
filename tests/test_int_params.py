"""Integer and keyword arguments across the public API (reviews,
2026-08-02 and 2026-10-26).

nu=4 and kappa=5 are how anyone writes these calls, and a python or numpy
INTEGER is not a differentiable type: jax hands the custom_jvp rule a
float0 tangent for that primal and the rule's arithmetic dies on it. Four
public functions failed outright (besseli_fn, stdtr, stdtrit,
vonmises_cdf); the rules now sit on private names and the public entry
points promote every argument first (_common.float_params). Keyword calls
went around that promotion until the second review, and so did an integer
evaluation point of a ChebSeries whose coefficients are differentiated.

The metric is agreement with the same call written in floats, which must
be exact: promotion happens before anything numeric, so nothing about
the computation differs.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax
from chebax import ChebSeries, PiecewiseCheb

# (name, call taking (param, x), a float param, an x)
CASES = [
    ("besseli_fn", lambda p, x: chebax.besseli_fn(p, x), 2, 3.0),
    ("besseli_ratio", lambda p, x: chebax.besseli_ratio(p, x), 2, 3.0),
    ("besselk_fn", lambda p, x: chebax.besselk_fn(p, x), 2, 3.0),
    ("log_besselk_fn", lambda p, x: chebax.log_besselk_fn(p, x), 2, 3.0),
    ("betainc_fn", lambda p, x: chebax.betainc_fn(p, 2, x), 3, 0.4),
    ("log_betainc_fn", lambda p, x: chebax.log_betainc_fn(p, 2, x), 3, 0.4),
    ("gammainc_fn", lambda p, x: chebax.gammainc_fn(p, x), 3, 1.5),
    ("log_gammainc_fn", lambda p, x: chebax.log_gammainc_fn(p, x), 3, 1.5),
    ("hyp1f1_fn", lambda p, x: chebax.hyp1f1_fn(p, 3, x), 2, 1.5),
    ("matern", lambda p, x: chebax.matern(p, x), 2, 1.5),
    ("stdtr", lambda p, x: chebax.stdtr(p, x), 4, 1.0),
    ("log_stdtr", lambda p, x: chebax.log_stdtr(p, x), 4, 1.0),
    ("log_stdtr_sf", lambda p, x: chebax.log_stdtr_sf(p, x), 4, 1.0),
    ("stdtrit", lambda p, x: chebax.stdtrit(p, x), 4, 0.3),
    ("vonmises_cdf", lambda p, x: chebax.vonmises_cdf(p, x), 5, 0.3),
    ("vonmises_icdf", lambda p, x: chebax.vonmises_icdf(p, x), 5, 0.3),
    ("gammaincinv", lambda p, x: chebax.gammaincinv(p, x), 3, 0.3),
    ("gammainccinv", lambda p, x: chebax.gammainccinv(p, x), 3, 0.3),
    ("betaincinv", lambda p, x: chebax.betaincinv(p, 2, x), 3, 0.3),
    ("chi2inv", lambda p, x: chebax.chi2inv(p, x), 4, 0.3),
]


@pytest.mark.parametrize("name,fn,p,x", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("kind", ["python", "numpy"])
def test_integer_parameter_matches_float(name, fn, p, x, kind):
    ip = p if kind == "python" else np.int64(p)
    assert float(fn(ip, x)) == float(fn(float(p), x)), name
    gi = float(jax.grad(lambda z: fn(ip, z))(x))
    gf = float(jax.grad(lambda z: fn(float(p), z))(x))
    assert gi == gf, (name, gi, gf)


@pytest.mark.parametrize("name,fn,p,x", CASES, ids=[c[0] for c in CASES])
def test_integer_parameter_under_jit(name, fn, p, x):
    # against the FLOAT call under jit, not the eager one: jit fusion may
    # reassociate and shift the last ulp, which says nothing about the
    # parameter's dtype
    got = float(jax.jit(lambda z: fn(p, z))(x))
    ref = float(jax.jit(lambda z: fn(float(p), z))(x))
    assert got == ref, (name, got, ref)


# (name, uniform parameters by keyword, evaluation-point keyword, point)
KW_CASES = [
    ("besseli_fn", {"nu": 2}, "x", 3.0),
    ("besseli_ratio", {"nu": 2}, "x", 3.0),
    ("besselk_fn", {"nu": 2}, "x", 3.0),
    ("log_besselk_fn", {"nu": 2}, "x", 3.0),
    ("betainc_fn", {"a": 3, "b": 2}, "x", 0.4),
    ("log_betainc_fn", {"a": 3, "b": 2}, "x", 0.4),
    ("gammainc_fn", {"a": 3}, "x", 1.5),
    ("gammaincc_fn", {"a": 3}, "x", 1.5),
    ("log_gammainc_fn", {"a": 3}, "x", 1.5),
    ("log_gammaincc_fn", {"a": 3}, "x", 1.5),
    ("hyp1f1_fn", {"a": 2, "b": 3}, "x", 1.5),
    ("log_hyp1f1_fn", {"a": 2, "b": 3}, "x", 1.5),
    ("matern", {"nu": 2}, "r", 1.5),
    ("stdtr", {"nu": 4}, "t", 1.0),
    ("log_stdtr", {"nu": 4}, "t", 1.0),
    ("log_stdtr_sf", {"nu": 4}, "t", 1.0),
    ("stdtrit", {"nu": 4}, "p", 0.3),
    ("vonmises_cdf", {"kappa": 5}, "theta", 0.3),
    ("vonmises_icdf", {"kappa": 5}, "p", 0.3),
    ("gammaincinv", {"a": 3}, "p", 0.3),
    ("gammainccinv", {"a": 3}, "y", 0.3),
    ("betaincinv", {"a": 3, "b": 2}, "p", 0.3),
    ("chi2inv", {"k": 4}, "p", 0.3),
    ("log_wright_bessel0", {"rho": 2}, "z", 3.0),
]


@pytest.mark.parametrize("name,params,xname,x", KW_CASES,
                         ids=[c[0] for c in KW_CASES])
def test_keyword_call_matches_positional(name, params, xname, x):
    # stdtr(nu=4, t=t) skipped the promotion and raised under grad, and
    # stdtr(nu=jnp.array([4.]), t=.3) died on a coefficient shape
    fn = getattr(chebax, name)
    first, rest = next(iter(params)), list(params)[1:]

    def positional(q, z):
        return fn(q, *[float(params[k]) for k in rest], z)

    ref = jax.value_and_grad(positional, (0, 1))(float(params[first]), x)
    ref = [float(ref[0]), *map(float, ref[1])]
    # integer parameters, by keyword: value and x-gradient
    v, gx = jax.value_and_grad(lambda z: fn(**params, **{xname: z}))(x)
    assert [float(v), float(gx)] == [ref[0], ref[2]], name

    # a size-one parameter, by keyword, differentiated through
    def keyword(q, z):
        kw = {k: jnp.asarray([float(params[k])]) for k in rest}
        return fn(**{first: jnp.reshape(q, (1,))}, **kw, **{xname: z})

    v, (gq, gz) = jax.value_and_grad(keyword, (0, 1))(float(params[first]), x)
    assert jnp.shape(v) == ()
    assert [float(v), float(gq), float(gz)] == ref, name


def test_tweedie_keywords_match_positional():
    args = {"p": 1.5, "mu": 2.0, "phi": 1.0, "y": 1.5}
    g = jax.grad(lambda p, y: chebax.tweedie_logpdf(p, 2.0, 1.0, y), (0, 1))
    gk = jax.grad(lambda p, y: chebax.tweedie_logpdf(
        y=y, phi=1, mu=2, p=jnp.reshape(p, (1,))), (0, 1))
    assert float(chebax.tweedie_logpdf(**args)) == float(
        chebax.tweedie_logpdf(*args.values()))
    assert list(map(float, gk(1.5, 1.5))) == list(map(float, g(1.5, 1.5)))


def test_integer_evaluation_point_with_coefficient_gradient():
    # an integer x reached the series custom_jvp and its float0 tangent
    # broke the rule even though only the coefficients were differentiated
    c = jnp.asarray([1.0, 2.0, 0.5])
    cp = jnp.asarray([[1.0, 2.0, 0.5], [0.3, -1.0, 2.0]])
    for x in (1, np.int64(1), np.array([0, 1, 2])):
        xf = np.asarray(x, dtype=float)
        for make, coef in ((lambda c: ChebSeries(c, (-1.0, 2.0)), c),
                           (lambda c: PiecewiseCheb(c, (0.0, 1.0, 2.0)), cp)):
            gi = jax.grad(lambda c: jnp.sum(make(c)(x)))(coef)
            gf = jax.grad(lambda c: jnp.sum(make(c)(xf)))(coef)
            np.testing.assert_array_equal(np.asarray(gi), np.asarray(gf))
            assert float(jnp.sum(make(coef)(x))) == float(jnp.sum(make(coef)(xf)))
