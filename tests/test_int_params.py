"""Integer parameters across the public API (review, 2026-08-02).

nu=4 and kappa=5 are how anyone writes these calls, and a python or numpy
INTEGER is not a differentiable type: jax hands the custom_jvp rule a
float0 tangent for that primal and the rule's arithmetic dies on it. Four
public functions failed outright (besseli_fn, stdtr, stdtrit,
vonmises_cdf); the rules now sit on private names and the public entry
points promote every argument first (_common.float_params).

The metric is agreement with the same call written in floats, which must
be exact: promotion happens before anything numeric, so nothing about
the computation differs.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax

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
