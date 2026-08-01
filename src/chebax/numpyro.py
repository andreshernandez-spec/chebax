"""Truncated distributions for numpyro, backed by chebax quantiles.

    from chebax.numpyro import TruncatedGamma, TruncatedBeta, TruncatedStudentT

numpyro's truncated-distribution machinery covers location-scale families;
Gamma, Beta and Student-t need the inverse-CDF path (numpyro#969,
numpyro#1187, numpyro#1365), which historically stalled on jax having no
igammainv. chebax's differentiable quantiles fill that hole, so these
classes implement the full Distribution contract: reparameterized
sampling by inverse CDF (has_rsample = True, pathwise gradients in every
parameter including the shapes), truncation-normalized log_prob,
cdf/icdf, dependent interval support.

The chebax contract applies: shape parameters are uniform per call (a
scalar or one traced latent each; batched shape arrays are not served -
use chebax.pergroup for per-group shapes). Domains, from the underlying
tables: TruncatedBeta needs (concentration1, concentration0) in
[0.1, 10]^2; TruncatedStudentT needs df in [0.2, 200]; TruncatedGamma's
concentration solves through jax's own gammainc and needs only
concentration > 0. Rates, bounds and data are unrestricted.

numpyro is imported here and only here (the chebax runtime proper stays
jax + numpy). Install with:  pip install chebax[numpyro]
"""

import numpy as np

import jax
import jax.numpy as jnp

try:
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    from numpyro.distributions.util import promote_shapes, validate_sample
except ImportError as e:
    raise ImportError(
        "chebax.numpyro needs numpyro: pip install chebax[numpyro]") from e

from jax import lax, random
from jax.scipy.special import betaln, gammainc, gammaln

import chebax

__all__ = ["TruncatedGamma", "TruncatedBeta", "TruncatedStudentT"]


def _check_bounds_eagerly(low, high):
    try:  # traced bounds cannot be checked at construction time
        if not np.all(np.asarray(low) < np.asarray(high)):
            raise ValueError(f"need low < high, got {low}, {high}")
    except jax.errors.TracerArrayConversionError:
        pass


class TruncatedGamma(dist.Distribution):
    """Gamma(concentration, rate) truncated to [low, high].

    concentration must be uniform per call (a scalar or one traced
    latent); rate, low, high broadcast freely. high may be inf.
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "low": constraints.nonnegative,
        "high": constraints.positive,
    }
    reparametrized_params = ["concentration", "rate", "low", "high"]
    has_rsample = True

    def __init__(self, concentration, rate=1.0, *, low=0.0, high=jnp.inf,
                 validate_args=None):
        self.concentration = jnp.asarray(concentration)
        self.rate, self.low, self.high = promote_shapes(
            jnp.asarray(rate), jnp.asarray(low), jnp.asarray(high))
        _check_bounds_eagerly(self.low, self.high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration), jnp.shape(rate), jnp.shape(low),
            jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _bounds(self):
        flo = gammainc(self.concentration, self.rate * self.low)
        # high = inf means no upper truncation; keep the argument finite
        # so the unused branch cannot poison gradients
        finite = jnp.isfinite(self.high)
        xhi = jnp.where(finite, self.rate * self.high, 1.0)
        fhi = jnp.where(finite, gammainc(self.concentration, xhi), 1.0)
        return flo, fhi

    def rsample(self, key, sample_shape=()):
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)

    sample = rsample

    def icdf(self, q):
        flo, fhi = self._bounds()
        return chebax.gammaincinv(self.concentration,
                                  flo + q * (fhi - flo)) / self.rate

    def cdf(self, value):
        flo, fhi = self._bounds()
        f = gammainc(self.concentration, self.rate * value)
        return jnp.clip((f - flo) / (fhi - flo), 0.0, 1.0)

    @validate_sample
    def log_prob(self, value):
        flo, fhi = self._bounds()
        c = self.concentration
        base = (c * jnp.log(self.rate) + (c - 1.0) * jnp.log(value)
                - self.rate * value - gammaln(c))
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(inside, base - jnp.log(fhi - flo), -jnp.inf)


class TruncatedBeta(dist.Distribution):
    """Beta(concentration1, concentration0) truncated to [low, high].

    Both concentrations must be uniform per call and inside chebax's
    (a, b) box [0.1, 10]^2 (tables back the normalizer's gradient and
    the inverse CDF); low, high broadcast freely.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "low": constraints.unit_interval,
        "high": constraints.unit_interval,
    }
    reparametrized_params = ["concentration1", "concentration0", "low", "high"]
    has_rsample = True

    def __init__(self, concentration1, concentration0, *, low=0.0, high=1.0,
                 validate_args=None):
        self.concentration1 = jnp.asarray(concentration1)
        self.concentration0 = jnp.asarray(concentration0)
        self.low, self.high = promote_shapes(jnp.asarray(low),
                                             jnp.asarray(high))
        _check_bounds_eagerly(self.low, self.high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0),
            jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _bounds(self):
        a, b = self.concentration1, self.concentration0
        return (chebax.betainc_fn(a, b, self.low),
                chebax.betainc_fn(a, b, self.high))

    def rsample(self, key, sample_shape=()):
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)

    sample = rsample

    def icdf(self, q):
        flo, fhi = self._bounds()
        return chebax.betaincinv(self.concentration1, self.concentration0,
                                 flo + q * (fhi - flo))

    def cdf(self, value):
        flo, fhi = self._bounds()
        f = chebax.betainc_fn(self.concentration1, self.concentration0, value)
        return jnp.clip((f - flo) / (fhi - flo), 0.0, 1.0)

    @validate_sample
    def log_prob(self, value):
        flo, fhi = self._bounds()
        a, b = self.concentration1, self.concentration0
        base = ((a - 1.0) * jnp.log(value) + (b - 1.0) * jnp.log1p(-value)
                - betaln(a, b))
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(inside, base - jnp.log(fhi - flo), -jnp.inf)


class TruncatedStudentT(dist.Distribution):
    """Standard Student-t(df) truncated to [low, high].

    df must be uniform per call and inside chebax's table range
    [0.2, 200]; low and high broadcast freely (either may be infinite).
    For a located/scaled variant, truncate the standard one to
    ((low - loc)/scale, (high - loc)/scale) and map samples through
    loc + scale * x.
    """

    arg_constraints = {"df": constraints.positive,
                       "low": constraints.real,
                       "high": constraints.real}
    reparametrized_params = ["df", "low", "high"]
    has_rsample = True

    def __init__(self, df, *, low=-jnp.inf, high=jnp.inf, validate_args=None):
        self.df = jnp.asarray(df)
        self.low, self.high = promote_shapes(jnp.asarray(low),
                                             jnp.asarray(high))
        _check_bounds_eagerly(self.low, self.high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(df), jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _bounds(self):
        # infinite bounds are exact CDF limits; keep the arguments the
        # masked branch sees finite so gradients stay clean
        flo = jnp.where(jnp.isfinite(self.low),
                        chebax.stdtr(self.df, jnp.where(jnp.isfinite(self.low),
                                                        self.low, 0.0)), 0.0)
        fhi = jnp.where(jnp.isfinite(self.high),
                        chebax.stdtr(self.df, jnp.where(jnp.isfinite(self.high),
                                                        self.high, 0.0)), 1.0)
        return flo, fhi

    def rsample(self, key, sample_shape=()):
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)

    sample = rsample

    def icdf(self, q):
        flo, fhi = self._bounds()
        return chebax.stdtrit(self.df, flo + q * (fhi - flo))

    def cdf(self, value):
        flo, fhi = self._bounds()
        f = chebax.stdtr(self.df, value)
        return jnp.clip((f - flo) / (fhi - flo), 0.0, 1.0)

    @validate_sample
    def log_prob(self, value):
        flo, fhi = self._bounds()
        lp = jax.scipy.stats.t.logpdf(value, self.df) - jnp.log(fhi - flo)
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(inside, lp, -jnp.inf)
