"""Large-a path for the regularized incomplete gamma: a > 10,
x in [0, inf), from the Temme-zone tables (gammainc_large_gen).

Three zones of lambda = x/a, two hard selects:

- lambda <= 1/2:   P = exp(a ln x - x - lnGamma(a+1) + D(v, lambda)),
                   the inner kernel in scaled coordinates.
- x >= 6 (a - 1):  Q = exp((a-1) ln x - x - lnGamma(a) + U(v, s)),
                   s = (a-1)/x, the tail kernel in the uniform variable.
- between:         the Temme transition. With
                   w = lambda - 1 - ln lambda and
                   eta = sign(lambda - 1) sqrt(2 w),
                       Q = 1/2 erfc(eta sqrt(a/2)) + corr
                       P = 1/2 erfc(-eta sqrt(a/2)) - corr
                       corr = e^(-a w) / sqrt(2 pi a) * T(v, eta)
                   each side pairs its own erfc tail with the O(1/sqrt(a))
                   correction, so BOTH P and Q keep relative accuracy
                   through the transition (nothing is computed as
                   1-minus-the-other there).

w is evaluated as d - log1p(d) (d = lambda - 1) with a series switch
below |d| = 1e-3, and eta as d sqrt(2w/d^2) so AD at lambda = 1 gets
the smooth limit instead of sqrt(0)'s nan. v = 10/a is the parameter
axis of all three tables, reaching v = 0 (a = inf) where each kernel
tends to its asymptotic limit; jax.grad in a flows through it, eta, and
gammaln (dP/da for every a > 10, jax's igamma_grad_a's territory).

Shared machinery for gammainc.py, which owns the public dispatch at
a = 10; nothing here is exported.
"""

import functools

import jax
import jax.numpy as jnp

from chebax import domains
from chebax._src.pytree import Recipe
from chebax._src.recipes import gammainc_large_table as _glt
from chebax._src.recipes._common import param_coefs, traced_coefs
from chebax._src.recipes.erf_family import erfcx
from chebax._src.series import ChebSeries

_SQ2PI = 2.5066282746310002  # sqrt(2 pi)

# The tables run to v = 10/a = 0, which is a = inf, so this path has no
# upper end. Read off the table through chebax.domains, which owns it;
# gammainc and quantiles gate on the name.
AHI = domains.GAMMAINC.hi


def _w_eta(lam):
    """(w, eta): w = lambda - 1 - ln lambda, eta = sign * sqrt(2w),
    both smooth through lambda = 1."""
    return w_eta(lam - 1.0)


def w_eta(d):
    """_w_eta from d = lambda - 1 itself, for callers that can form d
    without rounding lambda first (the gamma density at large a)."""
    dm = jnp.where(d == 0.0, 1.0, d)
    ratio_direct = 2.0 * (d - jnp.log1p(d)) / (dm * dm)
    ratio_series = (1.0 - 2.0 * d / 3.0 + d * d / 2.0
                    - 2.0 * d ** 3 / 5.0 + d ** 4 / 3.0)
    ratio = jnp.where(jnp.abs(d) < 1e-3, ratio_series, ratio_direct)
    eta = d * jnp.sqrt(ratio)
    return 0.5 * eta * eta, eta


def temme_log_pq(y, aw, tval, rt):
    """(ln P, ln Q) in the Temme zone, from y = eta sqrt(a/2), a w = y^2,
    the table value T(v, eta) and rt = sqrt(2 pi a).

    erfc and corr underflow f64 once a w > ~745, so each side assembles in
    log space through erfcx: e.g. ln Q = -a w + ln(erfcx(y)/2 + T/rt). The
    erfcx argument is masked to its own sign so the wrong-side lanes cannot
    reach erfcx's e^(y^2) overflow (grad poison), and the far side is log1p
    of the other probability, whose untaken lanes see a dummy 1/2 (log1p(-1)
    is -inf, poisoning grads through the select). The quantile solver for
    huge a calls this with y as its unknown."""
    yq = jnp.where(y > 0.0, y, 0.0)
    yp = jnp.where(y < 0.0, -y, 0.0)
    lq_t = -aw + jnp.log(0.5 * erfcx(yq) + tval / rt)
    lp_t = -aw + jnp.log(0.5 * erfcx(yp) - tval / rt)
    corr = jnp.exp(-aw) / rt * tval
    p_in = jnp.where(y > 0.0, 0.5, 0.5 * jax.scipy.special.erfc(-y) - corr)
    q_in = jnp.where(y < 0.0, 0.5, 0.5 * jax.scipy.special.erfc(y) + corr)
    return (jnp.where(y < 0.0, lp_t, jnp.log1p(-q_in)),
            jnp.where(y > 0.0, lq_t, jnp.log1p(-p_in)))


def eval_large(a, x, temme, dlow, dup, kind):
    """kind: 'p' | 'q' | 'logp' | 'logq'. a scalar (traced ok), x array."""
    x = jnp.asarray(x)
    pos = x > 0.0
    lam = jnp.where(pos, x, 1.0) / a

    in_low = lam <= _glt.LHI
    in_up = x * _glt.SHI >= (a - 1.0)
    xl = jnp.where(pos & in_low, x, 0.25 * a)
    lp_low = (a * jnp.log(xl) - xl - jax.scipy.special.gammaln(a + 1.0)
              + dlow(xl / a))
    # at x = inf the bracket is inf - inf: mask it and set ln Q = -inf
    # by hand (the small-a path does the same)
    big = x == jnp.inf
    xu = jnp.where(in_up & ~big, x, 12.0 * a)
    lq_up = ((a - 1.0) * jnp.log(xu) - xu - jax.scipy.special.gammaln(a)
             + dup((a - 1.0) / xu))
    lq_up = jnp.where(big, -jnp.inf, lq_up)

    # masked lanes clipped to the Temme zone's own lambda range so eta
    # stays inside the table domain: eta(0.5) = -0.62, eta(6) = 2.53
    lam_t = jnp.clip(lam, _glt.LHI, 6.0)
    w, eta = _w_eta(lam_t)
    y = eta * jnp.sqrt(0.5 * a)
    rt = _SQ2PI * jnp.sqrt(a)
    tval = temme(eta)
    corr = jnp.exp(-a * w) / rt * tval
    p_t = 0.5 * jax.scipy.special.erfc(-y) - corr
    q_t = 0.5 * jax.scipy.special.erfc(y) + corr

    if kind == "p":
        core = jnp.where(in_low, jnp.exp(lp_low),
                         jnp.where(in_up, 1.0 - jnp.exp(lq_up), p_t))
        out = jnp.where(pos, core, 0.0)
    elif kind == "q":
        core = jnp.where(in_low, 1.0 - jnp.exp(lp_low),
                         jnp.where(in_up, jnp.exp(lq_up), q_t))
        out = jnp.where(pos, core, 1.0)
    else:
        # log forms: see temme_log_pq; y^2 = a w exactly
        lp_t, lq_t = temme_log_pq(y, a * w, tval, rt)
        if kind == "logp":
            core = jnp.where(in_low, lp_low,
                             jnp.where(in_up, jnp.log1p(-jnp.exp(lq_up)),
                                       lp_t))
            out = jnp.where(pos, core, -jnp.inf)
        else:
            core = jnp.where(in_low, jnp.log1p(-jnp.exp(lp_low)),
                             jnp.where(in_up, lq_up, lq_t))
            out = jnp.where(pos, core, 0.0)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)


def series_eager(a):
    v = _glt.ASPLIT / a
    return (ChebSeries(param_coefs(_glt.TABLE_TEMME, _glt.VLO, 1.0, v),
                       (_glt.ELO, _glt.EHI)),
            ChebSeries(param_coefs(_glt.TABLE_LOW, _glt.VLO, 1.0, v),
                       (0.0, _glt.LHI)),
            ChebSeries(param_coefs(_glt.TABLE_UP, _glt.VLO, 1.0, v),
                       (0.0, _glt.SHI)))


def series_traced(a):
    v = _glt.ASPLIT / a
    return (ChebSeries(traced_coefs(_glt.TABLE_TEMME, _glt.VLO, 1.0, v),
                       (_glt.ELO, _glt.EHI)),
            ChebSeries(traced_coefs(_glt.TABLE_LOW, _glt.VLO, 1.0, v),
                       (0.0, _glt.LHI)),
            ChebSeries(traced_coefs(_glt.TABLE_UP, _glt.VLO, 1.0, v),
                       (0.0, _glt.SHI)))


@jax.tree_util.register_pytree_node_class
class GammaIncLargeP(Recipe):
    """Callable P(a, x), a > 10. Built by gammainc(a)."""

    _static_fields = ("a",)
    _series_fields = ("temme", "dlow", "dup")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return eval_large(self.a, x, self.temme, self.dlow, self.dup, "p")


@jax.tree_util.register_pytree_node_class
class GammaIncLargeQ(Recipe):
    """Callable Q(a, x), a > 10. Built by gammaincc(a)."""

    _static_fields = ("a",)
    _series_fields = ("temme", "dlow", "dup")

    def _post_init(self):
        self.a = float(self.a)

    def __call__(self, x):
        return eval_large(self.a, x, self.temme, self.dlow, self.dup, "q")
