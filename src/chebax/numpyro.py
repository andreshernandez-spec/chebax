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

The truncation algebra runs entirely on logs. The normalizer is
ln(F(hi) - F(lo)), built from a log-CDF and a log survival function
through logdiffexp and taken in whichever tail holds both endpoints;
icdf aims at a lower-tail or an upper-tail probability to match. A plain
F(hi) - F(lo) is exactly zero as soon as both endpoints land on the same
float, which happens early: under Gamma(3) both ends of [50, 51] give
F = 1.0, and that interval has perfectly ordinary conditional moments.
An interval in the BULK narrower than the CDF's own absolute error stays
unresolvable, in logs as much as out of them, and comes back with a
relative error of about eps over its mass.

The chebax contract applies: shape parameters are uniform per call (a
scalar or one traced latent each; batched shape arrays are not served,
use chebax.pergroup for per-group shapes), and a non-scalar shape is
rejected at construction. Domains, from the underlying tables and
checked eagerly when the value is not traced: TruncatedBeta needs
(concentration1, concentration0) in [0.1, 100]^2; TruncatedStudentT
needs df in [0.2, 200]; TruncatedGamma needs only concentration > 0,
its normalizer falling back to jax's own gammainc pair below chebax's
concentration = 0.1 (which costs the deep upper tail there, since jax's
logs underflow; from 0.1 up the recipe serves every concentration, the
Temme-zone tables taking over above 10). Rates, bounds and data are
unrestricted.

numpyro is imported here and only here (the chebax runtime proper stays
jax + numpy). Install with:  pip install chebax[numpyro]
"""

import math
from collections import namedtuple

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
from jax.scipy.special import betaln, gammaln

import chebax
from chebax import domains as _dom
from chebax._src.recipes._common import canon_float
# the log-CDF pair for the Student-t is not re-exported at the top level yet
from chebax._src.recipes.quantiles import (_t_logpdf, log_stdtr,
                                           log_stdtr_sf)

__all__ = ["TruncatedGamma", "TruncatedBeta", "TruncatedStudentT"]

_LN2 = math.log(2.0)
_DF_LO, _DF_HI = _dom.STUDENT_T_DF.lo, _dom.STUDENT_T_DF.hi
_REAL_OR_INF = constraints.interval(-jnp.inf, jnp.inf)


def _logdiffexp(a, b):
    """ln(e^a - e^b) for a >= b, without forming either exponential.

    b = -inf gives a back exactly, which is what an endpoint sitting at the
    edge of the support needs; a = -inf (both terms zero) gives -inf."""
    fin = a > -jnp.inf
    a_s = jnp.where(fin, a, 0.0)
    d = jnp.minimum(b - a_s, 0.0)          # rounding can push b just past a
    return jnp.where(fin, a_s + jnp.log(-jnp.expm1(d)), -jnp.inf)


def _pick(mask, when_true, when_false):
    """Take one tail branch or the other. Batched bounds need a select, but
    the usual scalar case takes a real branch, and a concrete one does not
    even build the quantile solve it is not going to run."""
    if jnp.ndim(mask) != 0:
        return jnp.where(mask, when_true(), when_false())
    try:
        taken = bool(mask)
    except jax.errors.TracerBoolConversionError:
        return lax.cond(mask, when_true, when_false)
    return when_true() if taken else when_false()


def _check_bounds_eagerly(low, high):
    try:  # traced bounds cannot be checked at construction time
        if not np.all(np.asarray(low) < np.asarray(high)):
            raise ValueError(f"need low < high, got {low}, {high}")
    except jax.errors.TracerArrayConversionError:
        pass


def _check_scalar(owner, name, v):
    if jnp.ndim(v) != 0:
        raise ValueError(
            f"{owner} needs a scalar {name}: chebax shape parameters are "
            "uniform per call, use chebax.pergroup for per-group shapes")


def _check_shape_range(owner, name, v, lo, hi):
    try:  # a traced shape cannot be checked at construction time
        x = float(np.asarray(v))
    except jax.errors.TracerArrayConversionError:
        return
    if not lo <= x <= hi:
        raise ValueError(f"{owner} needs {name} in [{lo}, {hi}], got {x}")


def _in_box(a):
    # the recipe's a-box: small tables to 10, Temme-zone tables above
    # (the public log forms dispatch between them internally)
    return (a >= _dom.GAMMAINC.lo) & (a <= _dom.GAMMAINC.hi)


def _log_pq(a, x, use_recipe):
    """(ln P(a, x), ln Q(a, x)). use_recipe is a python bool, so each
    lax.cond branch traces one: the chebax tables inside their a-box
    (fixed-degree polynomials, still accurate where P or Q underflows),
    jax's own gammainc pair outside it, whose logs bottom out where those
    functions do."""
    if use_recipe:
        return chebax.log_gammainc_fn(a, x), chebax.log_gammaincc_fn(a, x)
    return (jnp.log(jax.scipy.special.gammainc(a, x)),
            jnp.log(jax.scipy.special.gammaincc(a, x)))


def _gamma_log_pq(a, x):
    """(ln P, ln Q) for any a > 0. One scalar cond, only one branch runs."""
    a, x = canon_float(a), canon_float(x)
    return lax.cond(_in_box(a), lambda: _log_pq(a, x, True),
                    lambda: _log_pq(a, x, False))


_Norm = namedtuple("_Norm", "lnz lo_f lo_s hi_f hi_s upper")


class _Truncated(dist.Distribution):
    """Truncation algebra shared by the three classes, all of it in logs.

    A subclass supplies the base law's (ln F, ln S) pair, its log density,
    and the two one-sided inverses (from a lower-tail probability and from
    an upper-tail one). Every endpoint value the base law sees is masked
    to a safe interior point first: an infinite bound reaching an arithmetic
    op poisons reverse-mode tangents even when the result is selected away.
    """

    has_rsample = True

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _norm(self):
        lo_f, lo_s = self._log_cdf_sf(self.low)
        hi_f, hi_s = self._log_cdf_sf(self.high)
        # Work in the tail that holds both endpoints: below the median the
        # mass is a difference of log-CDFs, above it one of log survivals,
        # and each keeps its digits where the other has already rounded to
        # a constant. A straddling interval takes the lower form; it has
        # O(1) mass on at least one side.
        upper = lo_f >= -_LN2
        lnz = jnp.where(upper, _logdiffexp(lo_s, hi_s), _logdiffexp(hi_f, lo_f))
        return _Norm(lnz, lo_f, lo_s, hi_f, hi_s, upper)

    def rsample(self, key, sample_shape=()):
        u = random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)

    sample = rsample

    def icdf(self, q):
        n = self._norm()
        q = jnp.asarray(q)
        # p = F(lo) + q Z and s = S(hi) + (1-q) Z, by logaddexp so no
        # difference is formed; 1 - q is exact for q >= 1/2, where the upper
        # side needs it. An interval inside one tail keeps its own target
        # below 1/2 and so keeps full relative accuracy. A straddling one
        # takes the lower form and hits the untruncated quantile's own limit
        # near q = 1, no worse.
        # an invalid probability is masked before the logs, then answered
        # with nan: the core quantiles' policy, where this used to hand
        # back a plausible bound (icdf(-0.1) gave low; review, 2026-08-02)
        bad = jnp.isnan(q) | (q < 0.0) | (q > 1.0)
        q = jnp.where(bad, 0.5, q)
        lp = jnp.logaddexp(n.lo_f, jnp.log(q) + n.lnz)
        ls = jnp.logaddexp(n.hi_s, jnp.log1p(-q) + n.lnz)
        x = _pick(n.upper,
                  lambda: self._icdf_upper(jnp.exp(ls)),
                  lambda: self._icdf_lower(jnp.exp(lp)))
        x = jnp.clip(x, self.low, self.high)
        x = jnp.where(q <= 0.0, self.low, jnp.where(q >= 1.0, self.high, x))
        return jnp.where(bad, jnp.nan, x)

    def cdf(self, value):
        n = self._norm()
        v = jnp.clip(jnp.asarray(value), self.low, self.high)
        f, s = self._log_cdf_sf(v)
        num = jnp.where(n.upper, _logdiffexp(n.lo_s, s), _logdiffexp(f, n.lo_f))
        return jnp.clip(jnp.exp(num - n.lnz), 0.0, 1.0)

    @validate_sample
    def log_prob(self, value):
        value = jnp.asarray(value)
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(inside, self._log_density(value) - self._norm().lnz,
                         -jnp.inf)


class TruncatedGamma(_Truncated):
    """Gamma(concentration, rate) truncated to [low, high].

    concentration must be a positive scalar (or one traced latent); rate,
    low, high broadcast freely. high may be inf.
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "low": constraints.nonnegative,
        "high": constraints.positive,
    }
    reparametrized_params = ["concentration", "rate", "low", "high"]

    def __init__(self, concentration, rate=1.0, *, low=0.0, high=jnp.inf,
                 validate_args=None):
        _check_scalar("TruncatedGamma", "concentration", concentration)
        self.concentration = jnp.asarray(concentration)
        self.rate, self.low, self.high = promote_shapes(
            jnp.asarray(rate), jnp.asarray(low), jnp.asarray(high))
        _check_bounds_eagerly(self.low, self.high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration), jnp.shape(rate), jnp.shape(low),
            jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _log_cdf_sf(self, value):
        # mask before the multiply: high = inf otherwise reaches
        # rate * high, and reverse mode turns the masked-away 0 * inf into
        # a nan rate gradient (the default one-sided truncation)
        inside = (value > 0.0) & jnp.isfinite(value)
        v = jnp.where(inside, value, 1.0)
        f, s = _gamma_log_pq(self.concentration, self.rate * v)
        below = value <= 0.0
        return (jnp.where(inside, f, jnp.where(below, -jnp.inf, 0.0)),
                jnp.where(inside, s, jnp.where(below, 0.0, -jnp.inf)))

    def _log_density(self, value):
        c = self.concentration
        pos = (value > 0.0) & jnp.isfinite(value)
        v = jnp.where(pos, value, 1.0)
        core = (c * jnp.log(self.rate) + (c - 1.0) * jnp.log(v)
                - self.rate * v - gammaln(c))
        # x = 0 leaves (c-1) log 0, which is 0 * -inf at c = 1, where the
        # density is the rate itself
        at0 = jnp.where(c == 1.0, jnp.log(self.rate),
                        jnp.where(c < 1.0, jnp.inf, -jnp.inf))
        return jnp.where(pos, core, jnp.where(value <= 0.0, at0, -jnp.inf))

    def _icdf_lower(self, p):
        return chebax.gammaincinv(self.concentration, p) / self.rate

    def _icdf_upper(self, s):
        return chebax.gammainccinv(self.concentration, s) / self.rate


class TruncatedBeta(_Truncated):
    """Beta(concentration1, concentration0) truncated to [low, high].

    Both concentrations must be scalars inside chebax's (a, b) box
    [0.1, 100]^2 (tables back the normalizer's gradient and the inverse
    CDF); low, high broadcast freely.
    """

    arg_constraints = {
        "concentration1": constraints.interval(_dom.BETAINC.lo,
                                              _dom.BETAINC.hi),
        "concentration0": constraints.interval(_dom.BETAINC.lo,
                                               _dom.BETAINC.hi),
        "low": constraints.unit_interval,
        "high": constraints.unit_interval,
    }
    reparametrized_params = ["concentration1", "concentration0", "low", "high"]

    def __init__(self, concentration1, concentration0, *, low=0.0, high=1.0,
                 validate_args=None):
        for name, c in (("concentration1", concentration1),
                        ("concentration0", concentration0)):
            _check_scalar("TruncatedBeta", name, c)
            _check_shape_range("TruncatedBeta", name, c, _dom.BETAINC.lo,
                               _dom.BETAINC.hi)
        self.concentration1 = jnp.asarray(concentration1)
        self.concentration0 = jnp.asarray(concentration0)
        self.low, self.high = promote_shapes(jnp.asarray(low),
                                             jnp.asarray(high))
        _check_bounds_eagerly(self.low, self.high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0),
            jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _log_cdf_sf(self, value):
        a, b = self.concentration1, self.concentration0
        inside = (value > 0.0) & (value < 1.0)
        v = jnp.where(inside, value, 0.5)
        below = value <= 0.0
        # 1 - v is exact for v >= 1/2, which is the side the upper tail
        # needs; below the median the survival is O(1) and does not care
        return (jnp.where(inside, chebax.log_betainc_fn(a, b, v),
                          jnp.where(below, -jnp.inf, 0.0)),
                jnp.where(inside, chebax.log_betainc_fn(b, a, 1.0 - v),
                          jnp.where(below, 0.0, -jnp.inf)))

    def _log_density(self, value):
        a, b = self.concentration1, self.concentration0
        lnb = betaln(a, b)
        inside = (value > 0.0) & (value < 1.0)
        v = jnp.where(inside, value, 0.5)
        core = (a - 1.0) * jnp.log(v) + (b - 1.0) * jnp.log1p(-v) - lnb
        # at an endpoint one exponent multiplies -inf; the density is finite
        # there exactly when that concentration is 1
        at0 = jnp.where(a == 1.0, -lnb, jnp.where(a < 1.0, jnp.inf, -jnp.inf))
        at1 = jnp.where(b == 1.0, -lnb, jnp.where(b < 1.0, jnp.inf, -jnp.inf))
        return jnp.where(inside, core, jnp.where(value <= 0.0, at0, at1))

    def _icdf_lower(self, p):
        return chebax.betaincinv(self.concentration1, self.concentration0, p)

    def _icdf_upper(self, s):
        # 1 - X is Beta(b, a), so the upper tail inverts at full accuracy
        # in the distance from 1; only the final subtraction is absolute
        return 1.0 - chebax.betaincinv(self.concentration0,
                                       self.concentration1, s)


class TruncatedStudentT(_Truncated):
    """Standard Student-t(df) truncated to [low, high].

    df must be a scalar inside chebax's table range [0.2, 200]; low and
    high broadcast freely (either may be infinite). For a located/scaled
    variant, truncate the standard one to ((low - loc)/scale,
    (high - loc)/scale) and map samples through loc + scale * x.
    """

    # constraints.real rejects inf, so it would reject the untruncated
    # default bounds; interval's own test is inclusive
    arg_constraints = {"df": constraints.interval(_DF_LO, _DF_HI),
                       "low": _REAL_OR_INF,
                       "high": _REAL_OR_INF}
    reparametrized_params = ["df", "low", "high"]

    def __init__(self, df, *, low=-jnp.inf, high=jnp.inf, validate_args=None):
        _check_scalar("TruncatedStudentT", "df", df)
        _check_shape_range("TruncatedStudentT", "df", df, _DF_LO, _DF_HI)
        self.df = jnp.asarray(df)
        self.low, self.high = promote_shapes(jnp.asarray(low),
                                             jnp.asarray(high))
        _check_bounds_eagerly(self.low, self.high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(df), jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _log_cdf_sf(self, value):
        fin = jnp.isfinite(value)
        v = jnp.where(fin, value, 0.0)
        below = value < 0.0
        return (jnp.where(fin, log_stdtr(self.df, v),
                          jnp.where(below, -jnp.inf, 0.0)),
                jnp.where(fin, log_stdtr_sf(self.df, v),
                          jnp.where(below, 0.0, -jnp.inf)))

    def _log_density(self, value):
        return _t_logpdf(self.df, value)

    def _icdf_lower(self, p):
        return chebax.stdtrit(self.df, p)

    def _icdf_upper(self, s):
        return -chebax.stdtrit(self.df, s)
