"""Large-a incomplete gamma, a > 10, from the Temme-zone tables
(gammainc_large_gen), in three zones of lambda = x/a:

- lambda <= 1/2:   P = exp(a ln x - x - lnGamma(a+1) + D(v, lambda))
- x >= 6 (a - 1):  Q = exp((a-1) ln x - x - lnGamma(a) + U(v, s)), s = (a-1)/x
- between:         Q = 1/2 erfc(eta sqrt(a/2)) + corr,
                   P = 1/2 erfc(-eta sqrt(a/2)) - corr,
                   corr = e^(-a w) / sqrt(2 pi a) * T(v, eta),
                   w = lambda - 1 - ln lambda, eta = sign(lambda - 1) sqrt(2 w),
                   so both P and Q keep relative accuracy.

v = 10/a reaches v = 0 (a = inf); grad in a flows through the tables, eta and
gammaln. Numerics only, shared by gammainc_scalar.py, gammainc_frozen.py and
gammaincinv.py.
"""

import jax
import jax.numpy as jnp

from chebax import domains
from chebax._src.recipes import gammainc_large_table as _glt
from chebax._src.recipes._common import param_coefs, traced_coefs
from chebax._src.recipes.erf_family import erfcx
from chebax._src.series import ChebSeries

_SQ2PI = 2.5066282746310002  # sqrt(2 pi)

# no upper end (v = 0 is a = inf); chebax.domains owns the bound
AHI = domains.GAMMAINC.hi


def _w_eta(lam):
    """(w, eta) of lambda, smooth through lambda = 1."""
    return w_eta(lam - 1.0)


def w_eta(d):
    """_w_eta from d = lambda - 1, unrounded (the gamma density at large a). A
    series below |d| = 1e-3, and eta = d sqrt(2w/d^2) so AD at d = 0 is finite."""
    dm = jnp.where(d == 0.0, 1.0, d)
    ratio_direct = 2.0 * (d - jnp.log1p(d)) / (dm * dm)
    ratio_series = 1.0 - 2.0 * d / 3.0 + d * d / 2.0 - 2.0 * d**3 / 5.0 + d**4 / 3.0
    ratio = jnp.where(jnp.abs(d) < 1e-3, ratio_series, ratio_direct)
    eta = d * jnp.sqrt(ratio)
    return 0.5 * eta * eta, eta


def temme_log_pq(y, aw, tval, rt):
    """(ln P, ln Q) in the Temme zone from y = eta sqrt(a/2), a w = y^2, T(v, eta)
    and rt = sqrt(2 pi a), through erfcx since erfc underflows past a w ~ 745.
    The side y points to is -a w + ln(erfcx(|y|)/2 +- T/rt), the other is
    log1p(-exp(that)): one erfcx per point, never at the sign whose e^(y^2)
    overflows. The own side is at most ~1/2, so log1p never reaches -1."""
    upper = y >= 0.0
    own = -aw + jnp.log(
        0.5 * erfcx(jnp.where(upper, y, -y)) + jnp.where(upper, tval, -tval) / rt
    )
    other = jnp.log1p(-jnp.exp(own))
    return jnp.where(upper, other, own), jnp.where(upper, own, other)


def eval_large(a, x, temme, dlow, dup, kind):
    """kind: 'p' | 'q' | 'logp' | 'logq'. a scalar (traced ok), x array."""
    x = jnp.asarray(x)
    pos = x > 0.0
    lam = jnp.where(pos, x, 1.0) / a

    in_low = lam <= _glt.LHI
    in_up = x * _glt.SHI >= (a - 1.0)
    xl = jnp.where(pos & in_low, x, 0.25 * a)
    lp_low = a * jnp.log(xl) - xl - jax.scipy.special.gammaln(a + 1.0) + dlow(xl / a)
    # inf - inf at x = inf: mask it and set ln Q = -inf
    big = x == jnp.inf
    xu = jnp.where(in_up & ~big, x, 12.0 * a)
    lq_up = (
        (a - 1.0) * jnp.log(xu)
        - xu
        - jax.scipy.special.gammaln(a)
        + dup((a - 1.0) / xu)
    )
    lq_up = jnp.where(big, -jnp.inf, lq_up)

    # clip to the Temme zone so eta stays in the table: eta(0.5) = -0.62, eta(6) = 2.53
    lam_t = jnp.clip(lam, _glt.LHI, 6.0)
    w, eta = _w_eta(lam_t)
    y = eta * jnp.sqrt(0.5 * a)
    rt = _SQ2PI * jnp.sqrt(a)
    tval = temme(eta)
    corr = jnp.exp(-a * w) / rt * tval
    p_t = 0.5 * jax.scipy.special.erfc(-y) - corr
    q_t = 0.5 * jax.scipy.special.erfc(y) + corr

    if kind == "p":
        core = jnp.where(
            in_low, jnp.exp(lp_low), jnp.where(in_up, 1.0 - jnp.exp(lq_up), p_t)
        )
        out = jnp.where(pos, core, 0.0)
    elif kind == "q":
        core = jnp.where(
            in_low, 1.0 - jnp.exp(lp_low), jnp.where(in_up, jnp.exp(lq_up), q_t)
        )
        out = jnp.where(pos, core, 1.0)
    else:
        # log forms: see temme_log_pq; y^2 = a w exactly
        lp_t, lq_t = temme_log_pq(y, a * w, tval, rt)
        if kind == "logp":
            core = jnp.where(
                in_low, lp_low, jnp.where(in_up, jnp.log1p(-jnp.exp(lq_up)), lp_t)
            )
            out = jnp.where(pos, core, -jnp.inf)
        else:
            core = jnp.where(
                in_low, jnp.log1p(-jnp.exp(lp_low)), jnp.where(in_up, lq_up, lq_t)
            )
            out = jnp.where(pos, core, 0.0)
    return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)


def series_eager(a):
    v = _glt.ASPLIT / a
    return (
        ChebSeries(
            param_coefs(_glt.TABLE_TEMME, _glt.VLO, 1.0, v), (_glt.ELO, _glt.EHI)
        ),
        ChebSeries(param_coefs(_glt.TABLE_LOW, _glt.VLO, 1.0, v), (0.0, _glt.LHI)),
        ChebSeries(param_coefs(_glt.TABLE_UP, _glt.VLO, 1.0, v), (0.0, _glt.SHI)),
    )


def series_traced(a):
    v = _glt.ASPLIT / a
    return (
        ChebSeries(
            traced_coefs(_glt.TABLE_TEMME, _glt.VLO, 1.0, v), (_glt.ELO, _glt.EHI)
        ),
        ChebSeries(traced_coefs(_glt.TABLE_LOW, _glt.VLO, 1.0, v), (0.0, _glt.LHI)),
        ChebSeries(traced_coefs(_glt.TABLE_UP, _glt.VLO, 1.0, v), (0.0, _glt.SHI)),
    )
