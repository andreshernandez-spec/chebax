"""IEEE, validation and immutability contracts (review, 2026-08-02).

The medium-priority findings of that round, collected where the next
review can find them: infinities and negative arguments where a limit
exists, front-door validation that used to fail late or silently, and
the immutability of cached instances.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax
from chebax._src.series import ChebSeries


def test_limits_at_infinity():
    # each of these produced nan from masked tail arithmetic
    assert float(chebax.matern(1.5, jnp.inf)) == 0.0
    assert float(chebax.spherical_jn(2)(jnp.inf)) == 0.0
    assert float(chebax.spherical_yn(2)(jnp.inf)) == 0.0
    assert float(chebax.besseli_dnu(2.5, scaled=True)(jnp.inf)) == 0.0
    assert float(chebax.besseli(2.5, scaled=True)(jnp.inf)) == 0.0
    assert np.isposinf(float(chebax.hyp1f1_fn(1.0, 1.0, jnp.inf)))
    assert float(chebax.gammainc_fn(2.0, jnp.inf)) == 1.0
    assert float(chebax.gammaincc_fn(2.0, jnp.inf)) == 0.0


def test_negative_distance_is_a_domain_error():
    # a Matern kernel of a negative distance used to take the r = 0 branch
    # and answer 1, which reads as perfect correlation
    assert np.isnan(float(chebax.matern(1.5, -1.0)))
    assert np.isnan(float(chebax.matern(0.5, -1e-12)))
    assert float(chebax.matern(1.5, 0.0)) == 1.0


def test_cached_instances_cannot_be_mutated_or_emptied():
    s = chebax.besselj(2.5).g
    with pytest.raises(AttributeError, match="immutable"):
        s.coef = np.zeros(3)
    with pytest.raises(AttributeError, match="immutable"):
        del s.coef
    p = chebax.fit(np.exp, breaks=[-1.0, 0.0, 1.0])
    with pytest.raises(AttributeError, match="immutable"):
        del p.coef
    # and the shared instance is intact afterwards
    assert float(chebax.besselj(2.5)(1.0)) != 0.0


def test_truncate_validates_its_tolerance():
    inst = chebax.besselj(2.5)
    for bad in (-1.0, float("nan"), float("inf"), -0.0001):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            inst.truncate(bad)
    assert inst.truncate(1e-7).g.coef.size <= inst.g.coef.size
    assert inst.truncate(0.0).g.coef.size == inst.g.coef.size


def test_extreme_finite_domains_evaluate():
    # (2x - (b+a))/(b-a) overflows on a legal domain; midpoint/half-width
    # does not, and the endpoints must map to -1 and +1
    s = ChebSeries(np.array([1.0, 2.0, 0.5]), (-1e308, 1e308))
    for x, t in ((-1e308, -1.0), (0.0, 0.0), (1e308, 1.0)):
        ref = 1.0 + 2.0 * t + 0.5 * (2.0 * t * t - 1.0)
        assert abs(float(s(x)) - ref) <= 1e-14, (x, float(s(x)), ref)
    assert np.all(np.isfinite(np.asarray(s.deriv().coef)))


def test_front_door_validation():
    for bad in ((1,), (0.0, 1.0, 2.0), 5, None):
        with pytest.raises((ValueError, TypeError)):
            chebax.fit(np.exp, domain=bad)
    gi = np.array([0, 1, 0, 1])
    for bad in (2.9, True, 1.5):
        with pytest.raises(ValueError, match="num_groups"):
            chebax.pergroup(lambda a, x: a * x, gi, num_groups=bad)
    assert chebax.pergroup(lambda a, x: a * x, gi, num_groups=2) is not None


def test_emitted_names_are_valid_in_both_languages(tmp_path):
    from chebax import bake
    for bad in ("template", "class", "operator", "_Foo", "my__f", "lambda"):
        with pytest.raises(ValueError):
            bake.xsf_header(chebax.besselj(2.5), str(tmp_path / "h.h"),
                            name=bad)
    bake.xsf_header(chebax.besselj(2.5), str(tmp_path / "ok.h"),
                    name="besselj_2p5")


def test_truncated_icdf_rejects_invalid_probabilities():
    pytest.importorskip("numpyro")   # the 3.11 CI job installs gen,test only
    from chebax.numpyro import TruncatedGamma
    d = TruncatedGamma(3.0, 1.0, low=1.0, high=5.0)
    for bad in (-0.1, 1.1, np.nan):
        assert np.isnan(float(d.icdf(bad))), bad
    assert float(d.icdf(0.0)) == 1.0
    assert float(d.icdf(1.0)) == 5.0


# every traced entry point, with the parameter positions it takes and one
# valid argument set. The evaluation point is last, per the library's order.
_UNIFORM_PARAM_CALLS = [
    ("stdtr", lambda f, p: f(p(5.0), np.array([0.3, 0.7])), 1),
    ("log_stdtr", lambda f, p: f(p(5.0), np.array([0.3, 0.7])), 1),
    ("stdtrit", lambda f, p: f(p(5.0), np.array([0.3, 0.7])), 1),
    ("betainc_fn", lambda f, p: f(p(2.0), p(3.0), np.array([0.3, 0.7])), 2),
    ("log_betainc_fn", lambda f, p: f(p(2.0), p(3.0), np.array([0.3, 0.7])), 2),
    ("betaincinv", lambda f, p: f(p(2.0), p(3.0), np.array([0.3, 0.7])), 2),
    ("gammainc_fn", lambda f, p: f(p(2.0), np.array([0.3, 0.7])), 1),
    ("gammaincc_fn", lambda f, p: f(p(2.0), np.array([0.3, 0.7])), 1),
    ("gammaincinv", lambda f, p: f(p(2.0), np.array([0.3, 0.7])), 1),
    ("gammainccinv", lambda f, p: f(p(2.0), np.array([0.3, 0.7])), 1),
    ("chi2inv", lambda f, p: f(p(3.0), np.array([0.3, 0.7])), 1),
    ("besselk_fn", lambda f, p: f(p(1.5), np.array([1.3, 1.7])), 1),
    ("log_besselk_fn", lambda f, p: f(p(1.5), np.array([1.3, 1.7])), 1),
    ("besseli_fn", lambda f, p: f(p(1.5), np.array([1.3, 1.7])), 1),
    ("besseli_ratio", lambda f, p: f(p(1.5), np.array([1.3, 1.7])), 1),
    ("vonmises_cdf", lambda f, p: f(p(2.0), np.array([0.3, 0.7])), 1),
    ("hyp1f1_fn", lambda f, p: f(p(1.0), p(2.0), np.array([0.3, 0.7])), 2),
    ("log_hyp1f1_fn", lambda f, p: f(p(1.0), p(2.0), np.array([0.3, 0.7])), 2),
    ("matern", lambda f, p: f(p(1.5), np.array([0.3, 0.7])), 1),
]


@pytest.mark.parametrize("name,call,_n", _UNIFORM_PARAM_CALLS)
def test_size_one_parameter_is_the_scalar_it_holds(name, call, _n):
    # numpyro and pytensor hand back a broadcast (1,) array where the caller
    # wrote a scalar: numpyro's censoring wrapper broadcasts the base
    # distribution, so a censored StudentT's df reaches cdf with shape (1,).
    # Every entry point used to die on it inside the table reconstruction
    # ("coef must be a nonempty 1-D array, got shape (68, 1)").
    fn = getattr(chebax, name)
    scalar = np.asarray(call(fn, lambda v: v))
    for wrap in (lambda v: jnp.asarray([v]),            # (1,)
                 lambda v: jnp.asarray([[v]]),          # (1, 1)
                 lambda v: jnp.asarray(v)):             # 0-d
        got = np.asarray(call(fn, wrap))
        assert got.shape == scalar.shape, (name, got.shape)
        np.testing.assert_allclose(got, scalar, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("name,call,n", _UNIFORM_PARAM_CALLS)
def test_per_element_parameter_names_pergroup(name, call, n):
    # out of scope by design, so it has to say so rather than fail later
    fn = getattr(chebax, name)
    with pytest.raises(ValueError, match="pergroup"):
        call(fn, lambda v: jnp.asarray([v, v + 0.5]))
