"""A complete Generalized Inverse Gaussian for efax, using chebax for ln K_p.

The GIG density

    f(x; p, a, b) = (a/b)^(p/2) / (2 K_p(sqrt(ab))) * x^(p-1) e^{-(a x + b/x)/2}

is a three-parameter exponential family with sufficient statistics
(ln x, x, 1/x). Everything efax needs reduces to the log-normalizer

    A = ln 2 + (p/2) ln(b/a) + ln K_p(sqrt(ab))

and its derivatives, including d ln K_p / dp, a derivative in the ORDER of
the Bessel function. chebax.log_besselk_fn takes the order as a traced jax
scalar, so plain jax.grad supplies it, and optimistix Newton differentiates
through everything (second order included).

The classes below follow efax's own template (gamma.py / inverse_gaussian.py):
GigNP/GigEP with log_normalizer, closed-form to_exp built from ln K
differences and the order derivative, and the stock ExpToNat machinery for
the mean-to-natural conversion, with no custom solver code at all. The demo
verifies against scipy.stats.geninvgauss: log-pdf, entropy, the exact
NP -> EP -> NP roundtrip (efax's test_conversion), the same conversion from
200k sampled sufficient statistics, and finiteness of the entropy gradient.

Domains, stated up front: chebax's K tables cover order in [0, 10] and
argument sqrt(ab) >= 1e-6. to_exp needs orders |p|+1, so p is bounded to
[-8, 9] via the parameter support. The demo uses scalar-shaped
distributions; batched shapes need jax.vmap over these methods (each lane
then pays its own ~10k-flop coefficient reconstruction, which is fine for
parameter conversions).

This is the framework version of examples/gig_log_normalizer.py, which shows
the same math in plain jax with no efax dependency.

Run:  python examples/efax_gig.py   (needs pip install efax scipy)
"""

from __future__ import annotations

from typing import override

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from tjax import JaxArray, JaxRealArray, Shape  # noqa: E402
from tjax.dataclasses import dataclass  # noqa: E402

import optimistix as optx  # noqa: E402

from efax._src.expectation_parametrization import ExpectationParametrization  # noqa: E402
from efax._src.mixins.exp_to_nat.exp_to_nat import ExpToNat  # noqa: E402
from efax._src.mixins.exp_to_nat.optimistix import OptimistixRootFinder  # noqa: E402
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP  # noqa: E402
from efax._src.natural_parametrization import NaturalParametrization  # noqa: E402
from efax._src.parameter import (RealField, ScalarSupport,  # noqa: E402
                                 distribution_parameter, negative_support,
                                 positive_support)
from efax._src.parametrization import SimpleDistribution  # noqa: E402

import chebax  # noqa: E402


def _log_k(order, z):
    # K_{-v} = K_v; chebax tables cover v in [0, 10]
    return chebax.log_besselk_fn(jnp.abs(order), z)


@dataclass
class GigNP(HasEntropyNP["GigEP"],
            NaturalParametrization["GigEP", JaxRealArray],
            SimpleDistribution):
    """Natural parametrization of the GIG distribution.

    Args:
        p_minus_one: p - 1.
        negative_a_over_two: -a/2.
        negative_b_over_two: -b/2.
    """

    p_minus_one: JaxRealArray = distribution_parameter(
        ScalarSupport(ring=RealField(minimum=-9.0, maximum=8.0)))
    negative_a_over_two: JaxRealArray = distribution_parameter(
        ScalarSupport(ring=negative_support))
    negative_b_over_two: JaxRealArray = distribution_parameter(
        ScalarSupport(ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.p_minus_one.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=RealField(minimum=0.0))

    def _pab(self):
        return (self.p_minus_one + 1.0,
                -2.0 * self.negative_a_over_two,
                -2.0 * self.negative_b_over_two)

    @override
    def log_normalizer(self) -> JaxRealArray:
        p, a, b = self._pab()
        z = jnp.sqrt(a * b)
        return jnp.log(2.0) + 0.5 * p * (jnp.log(b) - jnp.log(a)) + _log_k(p, z)

    @override
    def to_exp(self) -> GigEP:
        # closed forms from ln K differences and the order derivative:
        #   E[x]    = sqrt(b/a) K_{p+1}/K_p
        #   E[1/x]  = sqrt(a/b) K_{p-1}/K_p
        #   E[ln x] = (1/2) ln(b/a) + d ln K_p / dp
        p, a, b = self._pab()
        z = jnp.sqrt(a * b)
        half_log_ba = 0.5 * (jnp.log(b) - jnp.log(a))
        log_kp = _log_k(p, z)
        mean = jnp.exp(half_log_ba + _log_k(p + 1.0, z) - log_kp)
        mean_reciprocal = jnp.exp(-half_log_ba + _log_k(p - 1.0, z) - log_kp)
        dlogk_dp = jax.grad(lambda q: _log_k(q, z))(p)
        mean_log = half_log_ba + dlogk_dp
        return GigEP(mean_log, mean, mean_reciprocal)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray,
                              **fixed_parameters: JaxArray) -> GigEP:
        return GigEP(jnp.log(x), x, 1.0 / x)


@dataclass
class GigEP(HasEntropyEP[GigNP], ExpToNat[GigNP], SimpleDistribution):
    """Expectation parametrization of the GIG distribution.

    Args:
        mean_log: E[ln x].
        mean: E[x].
        mean_reciprocal: E[1/x].
    """

    mean_log: JaxRealArray = distribution_parameter(ScalarSupport())
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))
    mean_reciprocal: JaxRealArray = distribution_parameter(
        ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=RealField(minimum=0.0))

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GigNP]:
        return GigNP

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    def __post_init__(self) -> None:
        # efax's stock minimizer is undamped Newton, which diverges for GIG's
        # curved gradient field (a full first step overflows a or b). Damped
        # Newton (Levenberg-Marquardt) converges from the default start; set
        # it before the mixin default kicks in, unless the caller chose one.
        if self.minimizer is None:
            object.__setattr__(self, "minimizer", OptimistixRootFinder(
                solver=optx.LevenbergMarquardt(rtol=1e-14, atol=1e-14),
                max_steps=200))
        super().__post_init__()


def _np_from(p, a, b):
    return GigNP(jnp.asarray(p - 1.0), jnp.asarray(-0.5 * a), jnp.asarray(-0.5 * b))


def main():
    p_t, a_t, b_t = 2.5, 1.5, 3.0
    nat = _np_from(p_t, a_t, b_t)
    # scipy's geninvgauss(p, b=z, scale=s) is GIG(p, a, b) with z = sqrt(ab),
    # s = sqrt(b/a)
    z, s = np.sqrt(a_t * b_t), np.sqrt(b_t / a_t)
    sp = stats.geninvgauss(p_t, z, scale=s)

    xs = np.linspace(0.05, 20.0, 9)
    lp = np.asarray(jax.vmap(nat.log_pdf)(jnp.asarray(xs)))
    err_pdf = np.abs(lp - sp.logpdf(xs)).max()
    print(f"log-pdf vs scipy.geninvgauss, max abs err:   {err_pdf:.2e}")

    exp = nat.to_exp()
    ent = float(exp.entropy())
    print(f"entropy: efax {ent:.12f}  scipy {sp.entropy():.12f}  "
          f"err {abs(ent - sp.entropy()):.2e}")

    # efax's test_conversion: exact NP -> EP -> NP roundtrip through the stock
    # ExpToNat machinery (optimistix Newton; zero custom solver code here)
    back = exp.to_nat()
    p_r = float(back.p_minus_one) + 1.0
    a_r = -2.0 * float(back.negative_a_over_two)
    b_r = -2.0 * float(back.negative_b_over_two)
    print("exact roundtrip NP -> EP -> NP:")
    print(f"  true:      p={p_t}, a={a_t}, b={b_t}")
    print(f"  recovered: p={p_r:.10f}, a={a_r:.10f}, b={b_r:.10f}")
    assert max(abs(p_r - p_t), abs(a_r - a_t), abs(b_r - b_t)) < 1e-6

    # the same conversion from sampled sufficient statistics
    rng = np.random.default_rng(0)
    draws = sp.rvs(size=200_000, random_state=rng)
    ep_hat = GigEP(jnp.asarray(np.mean(np.log(draws))),
                   jnp.asarray(np.mean(draws)),
                   jnp.asarray(np.mean(1.0 / draws)))
    nat_hat = ep_hat.to_nat()
    print("from 200k sampled sufficient statistics:")
    print(f"  recovered: p={float(nat_hat.p_minus_one) + 1.0:.4f}, "
          f"a={-2.0 * float(nat_hat.negative_a_over_two):.4f}, "
          f"b={-2.0 * float(nat_hat.negative_b_over_two):.4f}")

    # efax's test_entropy_gradient: finite, and matches finite differences
    def entropy_of(p, a, b):
        return _np_from(p, a, b).to_exp().entropy()

    g = jax.grad(entropy_of, (0, 1, 2))(p_t, a_t, b_t)
    eps = 1e-6
    fd = (float(entropy_of(p_t + eps, a_t, b_t))
          - float(entropy_of(p_t - eps, a_t, b_t))) / (2 * eps)
    print(f"entropy gradient: ({float(g[0]):+.8f}, {float(g[1]):+.8f}, "
          f"{float(g[2]):+.8f}); dH/dp vs finite diff err "
          f"{abs(float(g[0]) - fd):.2e}")
    assert all(bool(jnp.isfinite(gi)) for gi in g)
    return p_r, a_r, b_r


if __name__ == "__main__":
    main()
