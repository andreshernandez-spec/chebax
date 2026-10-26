"""Student-t CDF, log-CDF, log survival function and quantile: stdtr,
log_stdtr, log_stdtr_sf, stdtrit (jax#20358).

None needs a solver of its own: both orientations of the incomplete-beta
reduction are used, t^2/(nu+t^2) near the median (exact first-order
behavior through t = 0, where the classic nu/(nu+t^2) form rounds to 1 and
loses everything) and nu/(nu+t^2) in the tails. The tail orientation is
evaluated from log|t| rather than from t^2, so it holds for any t,
including |t| past the point where t^2 overflows. The JVPs use the exact
Student-t density. log_stdtr/log_stdtr_sf are the log-CDF and log survival
function, accurate where the probability itself underflows.

All four run on the 2-D slice tables of the betainc kernel at a = 1/2
(b-panels to 100, stdtr_gen), so they cover nu in [0.2, 200] without the
3-D tensor; stdtrit inverts through the shared beta solve in
quantiles.py. nu is traceable and uniform per call, so jax.grad works in
both arguments.
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import stdtr_table as _st
from chebax._src.recipes._common import float_params, traced_coefs
from chebax._src.recipes.betainc import _log_direct as _log_betainc_direct
from chebax._src.recipes.quantiles import _betaincinv_core, _ln_beta
from chebax._src.series import ChebSeries

_LN2 = math.log(2.0)


def _t_logpdf(nu, t):
    # ln(1 + t^2/nu) from log|t| where t^2 would overflow (|t| ~ 1.3e154 in
    # float64, 1.8e19 in float32), where it is 2 ln|t| - ln nu to hundreds
    # of digits. The direct form is kept elsewhere: it is the one with the
    # exact zero t-derivative at t = 0, which second derivatives need.
    t2 = t * t
    ok = jnp.isfinite(t2)
    l1 = jnp.where(ok, jnp.log1p(jnp.where(ok, t2, 0.0) / nu),
                   2.0 * jnp.log(jnp.where(ok, 1.0, jnp.abs(t))) - jnp.log(nu))
    return (jax.scipy.special.gammaln(0.5 * (nu + 1.0))
            - jax.scipy.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(nu * jnp.pi) - 0.5 * (nu + 1.0) * l1)


def _half_series(hb):
    """x-series of the betainc kernel at (1/2, hb) and (hb, 1/2), from the
    stdtr slice tables: hb = nu/2 in [0.1, 100] (nu in [0.2, 200]),
    uniform per call, panel select at hb = 10. All four tables share the
    x-node count, so the selects act elementwise on coefficient vectors."""
    lo = hb <= _st.BSPLIT
    ca = jnp.where(lo,
                   traced_coefs(_st.TABLE_A_LO, _st.BLO, _st.BSPLIT, hb),
                   traced_coefs(_st.TABLE_A_HI, _st.BSPLIT, _st.BHI, hb))
    cb = jnp.where(lo,
                   traced_coefs(_st.TABLE_B_LO, _st.BLO, _st.BSPLIT, hb),
                   traced_coefs(_st.TABLE_B_HI, _st.BSPLIT, _st.BHI, hb))
    return (ChebSeries(ca, (0.0, _st.XSPLIT)),
            ChebSeries(cb, (0.0, _st.XSPLIT)))


def _stdtr_parts(nu, t):
    """(central mask, ln I_w(1/2, nu/2), ln I_x(nu/2, 1/2)).

    F is 1/2 (1 + sgn(t) I_w) near the median and 1/2 I_x in the tails,
    with w = t^2/(nu+t^2) and x = nu/(nu+t^2). w resolves F - 1/2 exactly
    through t = 0 (the classic x form rounds to 1 there and F degrades to
    exactly 0.5); x resolves the tail probability. Seam at t^2 = nu, where
    both are 1/2, which is also where each orientation leaves the tables'
    direct branch, so both logs come straight from the series.

    The tail is taken from log|t|: with z = 2 ln|t| - ln nu, x = 1/(1+e^z)
    is -softplus(z) and 1 - x is -softplus(-z), so nothing forms t^2
    (which overflows past |t| ~ 1.3e154 in float64, ~1.8e19 in float32)
    and the tail probability stays right down to where it underflows."""
    hb = 0.5 * nu
    half = jnp.asarray(0.5)
    ca, cb = _half_series(hb)
    t2 = t * t                        # may be inf; only its comparison is used
    central = t2 <= nu
    tc = jnp.where(central, t, 0.0)
    # <= 1/2 by construction, no overflow; masked lanes get a safe dummy so
    # that reverse mode does not multiply their zero cotangent by a d/dnu
    # of log(0)
    w = jnp.where(central, tc * tc / (nu + tc * tc), 0.25)
    # I <= 1, so clamping its log at 0 is exact and keeps 1 - I_w >= 0 when
    # the evaluation error swamps it (large nu just inside the seam)
    l_mid = jnp.minimum(_log_betainc_direct(half, hb, w, ca), 0.0)
    tt = jnp.where(central, jnp.sqrt(nu), jnp.abs(t))
    z = 2.0 * jnp.log(tt) - jnp.log(nu)
    lx, l1x = -jax.nn.softplus(z), -jax.nn.softplus(-z)
    l_tail = (hb * lx + 0.5 * l1x - jnp.log(hb) - _ln_beta(hb, half)
              + cb(jnp.exp(lx)))
    return central, l_mid, jnp.minimum(l_tail, 0.0)


def _stdtr_impl(nu, t):
    central, l_mid, l_tail = _stdtr_parts(nu, t)
    hp = 0.5 * jnp.exp(jnp.where(central, l_mid, l_tail))
    sgn = jnp.where(t >= 0.0, 1.0, -1.0)
    F = jnp.where(central, 0.5 + sgn * hp,
                  jnp.where(t >= 0.0, 1.0 - hp, hp))
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, F)


def _log_stdtr_impl(nu, t):
    central, l_mid, l_tail = _stdtr_parts(nu, t)
    up = t >= 0.0
    # every branch is a log of the series value, never of a difference of
    # two numbers near 1/2: below the median the lower tail is 1/2 I, above
    # it the survival half is, and log1p/expm1 keep the complement exact
    mid = jnp.where(up, jax.nn.softplus(l_mid),
                    jnp.log(-jnp.expm1(l_mid))) - _LN2
    tail = jnp.where(up, jnp.log1p(-0.5 * jnp.exp(l_tail)), l_tail - _LN2)
    L = jnp.where(central, mid, tail)
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, L)


@jax.custom_jvp
def _stdtr_cj(nu, t):
    """Student-t CDF with nu degrees of freedom (nu in [0.2, 200], traceable).

    jax.grad works with respect to nu as well: learnable degrees of freedom.
    First-order exact through t = 0 (dF/dt there is the density, not 0), and
    valid for any t, the tail being computed from log|t|."""
    return _stdtr_impl(nu, t)


@_stdtr_cj.defjvp
def _stdtr_jvp(primals, tangents):
    nu_c, t_c = primals
    dnu, dt = tangents
    F = _stdtr_impl(nu_c, t_c)
    # t-direction: the exact density (AD through the w = t^2/(nu+t^2)
    # branch hits a 0 * inf at t = 0; the density is finite there)
    dF_dt = jnp.exp(_t_logpdf(nu_c, jnp.where(jnp.isfinite(t_c), t_c, 0.0)))
    dF_dt = jnp.where(jnp.isfinite(t_c), dF_dt, 0.0)
    # nu-direction: AD of the implementation, at a safe t (t = 0 and
    # t = +-inf have exactly zero nu-sensitivity)
    t_safe = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), t_c, 1.0)
    _, dF_dnu = jax.jvp(lambda n: _stdtr_impl(n, t_safe), (nu_c,), (dnu,))
    dF_dnu = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), dF_dnu, 0.0)
    dF = dF_dt * dt + dF_dnu
    return F, jnp.where(jnp.isnan(F), jnp.nan, dF)


@jax.custom_jvp
def _log_stdtr_cj(nu, t):
    """ln F(t; nu), the Student-t log-CDF, with no underflow floor in the
    lower tail.

    Same contract as stdtr (nu in [0.2, 200], traceable, differentiable in
    both arguments). For t <= -sqrt(nu) the tail half-probability is a
    tabulated log, returned as such: relatively accurate to arbitrarily
    negative values (ln F ~ -nu ln|t|), which is far past where F itself
    underflows. Above the median it is log1p of a half-probability that is
    at most 1/2, and F is O(1) there, so ln F is accurate too.

    Between -sqrt(nu) and the median the tables only reach F through
    1 - I_w, which is absolutely accurate, so ln F there carries the CDF's
    absolute error over F. That band is harmless for small nu (F at its far
    edge is 0.058 at nu = 4, 8e-4 at nu = 15) and is the accuracy limit for
    large nu, whose far edge sits at F ~ 4e-11 (nu = 60) or 1e-34
    (nu = 200): the tables hold the (nu/2, 1/2) series only for x <= 1/2.
    Measured float64: 3e-6 at nu = 150, t = -6 (F = 7e-9), exact by
    t = -4. Where F underflows, t is always past -sqrt(nu) and the
    directly-tabulated branch has it.

    This is the log-CDF a truncated or censored Student-t likelihood wants:
    log(stdtr(...)) underflows to -inf first, and a normalizer built as
    log(F(hi) - F(lo)) collapses to log(0) whenever both tails are deep."""
    return _log_stdtr_impl(nu, t)


@_log_stdtr_cj.defjvp
def _log_stdtr_jvp(primals, tangents):
    nu_c, t_c = primals
    dnu, dt = tangents
    L = _log_stdtr_impl(nu_c, t_c)
    fin = jnp.isfinite(t_c)
    ts = jnp.where(fin, t_c, 0.0)
    # d/dt ln F = pdf / F, taken as a difference of logs so it survives
    # where F underflows (there it tends to nu/|t|, not to 0/0)
    dL_dt = jnp.where(fin, jnp.exp(_t_logpdf(nu_c, ts) - L), 0.0)
    t_safe = jnp.where(fin & (ts != 0.0), ts, 1.0)
    _, dL_dnu = jax.jvp(lambda n: _log_stdtr_impl(n, t_safe), (nu_c,),
                        (dnu,))
    dL_dnu = jnp.where(fin & (ts != 0.0), dL_dnu, 0.0)
    dL = dL_dt * dt + dL_dnu
    return L, jnp.where(jnp.isnan(L), jnp.nan, dL)


def log_stdtr_sf(nu, t):
    """ln (1 - F(t; nu)), the Student-t log survival function, with no
    underflow floor in the upper tail.

    The distribution is symmetric, so this is the log-CDF at -t and the
    negation is exact: no 1 - F cancellation anywhere, and the same
    accuracy in the upper tail that log_stdtr has in the lower."""
    return _log_stdtr_cj(nu, -t)


def _stdtrit_impl(nu, p):
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    upper = pc > 0.5
    # central: |t| from w = I^-1(|2p-1|; 1/2, nu/2), t = sqrt(nu w/(1-w));
    # tails: x = I^-1(2 min(p, 1-p); nu/2, 1/2), t = sqrt(nu (1-x)/x).
    # Both targets are exact in binary: 2p is a scaling, 1 - p is exact for
    # p >= 1/2, and 2p - 1 is exact for p in [1/4, 3/4]. The old tail
    # target 1 - |2p - 1| was not: at p = 1e-20 it rounded to 0 and the
    # quantile came back -inf.
    ptail = 2.0 * jnp.where(upper, 1.0 - pc, pc)
    aq = jnp.abs(2.0 * pc - 1.0)
    central = ptail >= 0.5
    ca, cb = _half_series(0.5 * nu)
    w = _betaincinv_core(jnp.asarray(0.5), 0.5 * nu,
                         jnp.where(central, aq, 0.25), ca, cb)
    x = _betaincinv_core(0.5 * nu, jnp.asarray(0.5),
                         jnp.where(central, 0.5, ptail), cb, ca)
    # sqrt(nu (1-x)) / sqrt(x), not sqrt(nu (1-x)/x): the ratio overflows
    # for x below ~1e-308 while the quantile itself is still representable
    mag = jnp.where(central, jnp.sqrt(nu * w / (1.0 - w)),
                    jnp.sqrt(nu * (1.0 - x)) / jnp.sqrt(x))
    t = jnp.where(pc < 0.5, -mag, mag)
    t = jnp.where(p <= 0.0, -jnp.inf, jnp.where(p >= 1.0, jnp.inf, t))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(nu)
    return jnp.where(oob, jnp.nan, t)


@jax.custom_jvp
def _stdtrit_cj(nu, p):
    """Student-t quantile with nu degrees of freedom (nu in [0.2, 200]).

    Closed form via betaincinv in both orientations: exact 0 at p = 1/2,
    +-inf at the endpoints, gradients from the implicit function theorem
    (dt/dp = 1/pdf is 8/3 at the median for nu = 4, not 0). The tail
    target is 2 min(p, 1-p), exact in binary, so deep-tail p resolves
    (nu = 4, p = 1e-300 gives -1.3e75). It bottoms out where the beta
    quantile behind it underflows and the +-inf endpoint takes over, which
    at small nu can be short of the representable t (nu = 1/2,
    p = 1e-100)."""
    return _stdtrit_impl(nu, p)


@_stdtrit_cj.defjvp
def _stdtrit_jvp(primals, tangents):
    nu_c, p = primals
    dnu, dp = tangents
    t = _stdtrit_impl(nu_c, p)
    finite = jnp.isfinite(t)
    ts = jnp.where(finite, t, 0.0)
    pdf = jnp.exp(_t_logpdf(nu_c, ts))
    # IFT: dt/dp = 1/pdf, which may legitimately overflow where the
    # density underflows; dt/dnu = -(dF/dnu)/pdf, which may NOT, since the
    # ratio stays finite where both parts vanish. At nu = 4, p = 1e-300
    # both are ~1e-320 and the quotient is 5.7e76, so forming them
    # separately gave inf (review, 2026-08-02). Take it through the logs:
    # dF/dnu = F dlnF/dnu, so the ratio is exp(lnF - ln pdf) dlnF/dnu and
    # nothing underflows on the way.
    t_safe = jnp.where(finite & (ts != 0.0), ts, 1.0)
    live = finite & (ts != 0.0)
    lnF, dlnF_dnu = jax.jvp(lambda n: _log_stdtr_impl(n, t_safe), (nu_c,),
                            (jnp.asarray(1.0),))
    dt_dnu = -jnp.exp(lnF - _t_logpdf(nu_c, t_safe)) * dlnF_dnu
    dt_dnu = jnp.where(live, dt_dnu, 0.0)
    dt = dp / pdf + dt_dnu * dnu
    dt = jnp.where(finite, dt, 0.0)
    return t, jnp.where(jnp.isnan(t), jnp.nan, dt)


# Public entry points. The custom_jvp rules live on the private names
# above and float_params normalizes every argument first, so an integer,
# keyword or size-one nu is a float scalar before any tangent exists.
stdtr = float_params(_stdtr_cj)
log_stdtr = float_params(_log_stdtr_cj)
stdtrit = float_params(_stdtrit_cj)
log_stdtr_sf = float_params(log_stdtr_sf)
