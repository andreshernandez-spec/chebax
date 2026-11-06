"""Student-t CDF, log-CDF, log survival function and quantile: stdtr, log_stdtr,
log_stdtr_sf, stdtrit (jax#20358).

Both incomplete-beta orientations: w = t^2/(nu+t^2) near the median, exact
through t = 0 where nu/(nu+t^2) rounds to 1, and x = nu/(nu+t^2) in the tails,
taken from log|t| so no t^2 overflows. All four use the betainc kernel's 2-D
slice tables at a = 1/2 (stdtr_gen, nu in [0.2, 200], betainc_kernel.py), not
the 3-D tensor. nu is traceable and uniform per call.
"""

import math

import jax
import jax.numpy as jnp

from chebax._src.recipes import stdtr_table as _st
from chebax._src.recipes._common import float_params, panel_coefs
from chebax._src.recipes.betainc_kernel import (
    betaincinv_core,
    ln_beta,
    log_direct,
    t_logpdf,
)
from chebax._src.series import ChebSeries

_LN2 = math.log(2.0)


def _half_series(hb):
    """x-series of the betainc kernel at (1/2, hb) and (hb, 1/2), hb = nu/2 in
    [0.1, 100], panels selected at hb = 10 on the coefficient vectors."""
    ca = panel_coefs(_st.TABLE_A_LO, _st.TABLE_A_HI, _st.BLO, _st.BSPLIT, _st.BHI, hb)
    cb = panel_coefs(_st.TABLE_B_LO, _st.TABLE_B_HI, _st.BLO, _st.BSPLIT, _st.BHI, hb)
    return (ChebSeries(ca, (0.0, _st.XSPLIT)), ChebSeries(cb, (0.0, _st.XSPLIT)))


def _stdtr_parts(nu, t):
    """(central mask, ln I_w(1/2, nu/2), ln I_x(nu/2, 1/2)), w = t^2/(nu+t^2),
    x = nu/(nu+t^2), seam at t^2 = nu. The tail takes ln x = -softplus(z),
    z = 2 ln|t| - ln nu, so nothing forms t^2 (it overflows past |t| ~ 1.3e154
    in float64, ~1.8e19 in float32)."""
    hb = 0.5 * nu
    half = jnp.asarray(0.5)
    ca, cb = _half_series(hb)
    t2 = t * t  # may be inf; only its comparison is used
    central = t2 <= nu
    tc = jnp.where(central, t, 0.0)
    # a dummy in masked lanes, so reverse mode never meets d/dnu of log(0)
    w = jnp.where(central, tc * tc / (nu + tc * tc), 0.25)
    # I <= 1, so the clamp is exact and keeps 1 - I_w >= 0 just inside the seam
    l_mid = jnp.minimum(log_direct(half, hb, w, ca), 0.0)
    tt = jnp.where(central, jnp.sqrt(nu), jnp.abs(t))
    z = 2.0 * jnp.log(tt) - jnp.log(nu)
    lx, l1x = -jax.nn.softplus(z), -jax.nn.softplus(-z)
    l_tail = hb * lx + 0.5 * l1x - jnp.log(hb) - ln_beta(hb, half) + cb(jnp.exp(lx))
    return central, l_mid, jnp.minimum(l_tail, 0.0)


def _stdtr_impl(nu, t):
    central, l_mid, l_tail = _stdtr_parts(nu, t)
    hp = 0.5 * jnp.exp(jnp.where(central, l_mid, l_tail))
    sgn = jnp.where(t >= 0.0, 1.0, -1.0)
    F = jnp.where(central, 0.5 + sgn * hp, jnp.where(t >= 0.0, 1.0 - hp, hp))
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, F)


def _log_stdtr_impl(nu, t):
    central, l_mid, l_tail = _stdtr_parts(nu, t)
    up = t >= 0.0
    # logs of series values, never of a difference near 1/2; log1p and expm1
    # keep the complements exact. ln(1 - I) sees l_mid only in its own lanes:
    # in the others I can round to 1 at large nu (the tail's dummy w = 1/4
    # too), and reverse mode turns log(0)'s inf times a zero cotangent into nan
    lm = jnp.where(central & ~up, l_mid, -1.0)
    mid = jnp.where(up, jax.nn.softplus(l_mid), jnp.log(-jnp.expm1(lm))) - _LN2
    tail = jnp.where(up, jnp.log1p(-0.5 * jnp.exp(l_tail)), l_tail - _LN2)
    L = jnp.where(central, mid, tail)
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, L)


@jax.custom_jvp
def _stdtr_cj(nu, t):
    """Student-t CDF, nu in [0.2, 200], differentiable in nu and t. Exact to
    first order through t = 0 and valid for any t."""
    return _stdtr_impl(nu, t)


@_stdtr_cj.defjvp
def _stdtr_jvp(primals, tangents):
    nu_c, t_c = primals
    dnu, dt = tangents
    F = _stdtr_impl(nu_c, t_c)
    # d/dt is the density: AD through w hits 0 * inf at t = 0
    dF_dt = jnp.exp(t_logpdf(nu_c, jnp.where(jnp.isfinite(t_c), t_c, 0.0)))
    dF_dt = jnp.where(jnp.isfinite(t_c), dF_dt, 0.0)
    # d/dnu by AD at a safe t; t = 0 and +-inf have zero nu-sensitivity
    t_safe = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), t_c, 1.0)
    _, dF_dnu = jax.jvp(lambda n: _stdtr_impl(n, t_safe), (nu_c,), (dnu,))
    dF_dnu = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), dF_dnu, 0.0)
    dF = dF_dt * dt + dF_dnu
    return F, jnp.where(jnp.isnan(F), jnp.nan, dF)


@jax.custom_jvp
def _log_stdtr_cj(nu, t):
    """ln F(t; nu), the Student-t log-CDF with no underflow floor in the lower
    tail; contract as stdtr. Below -sqrt(nu) it is a tabulated log, relatively
    accurate however small F is; above the median, log1p of at most 1/2.

    Between -sqrt(nu) and the median ln F carries F's absolute error (the tables
    hold the (nu/2, 1/2) series only for x <= 1/2): harmless at small nu (F at
    the band's edge 0.058 at nu = 4), the limit at large nu (edge F ~ 4e-11 at
    nu = 60): float64 3e-6 at nu = 150, t = -6 (F = 7e-9), exact by t = -4.
    For truncated or censored likelihoods, where log(F(hi) - F(lo)) hits log(0)."""
    return _log_stdtr_impl(nu, t)


@_log_stdtr_cj.defjvp
def _log_stdtr_jvp(primals, tangents):
    nu_c, t_c = primals
    dnu, dt = tangents
    L = _log_stdtr_impl(nu_c, t_c)
    fin = jnp.isfinite(t_c)
    ts = jnp.where(fin, t_c, 0.0)
    # pdf / F as a difference of logs, finite (-> nu/|t|) where F underflows
    dL_dt = jnp.where(fin, jnp.exp(t_logpdf(nu_c, ts) - L), 0.0)
    t_safe = jnp.where(fin & (ts != 0.0), ts, 1.0)
    _, dL_dnu = jax.jvp(lambda n: _log_stdtr_impl(n, t_safe), (nu_c,), (dnu,))
    dL_dnu = jnp.where(fin & (ts != 0.0), dL_dnu, 0.0)
    dL = dL_dt * dt + dL_dnu
    return L, jnp.where(jnp.isnan(L), jnp.nan, dL)


def log_stdtr_sf(nu, t):
    """ln (1 - F(t; nu)), the Student-t log survival function: log_stdtr at -t,
    exact by symmetry, as accurate in the upper tail as log_stdtr in the lower."""
    return _log_stdtr_cj(nu, -t)


def _stdtrit_impl(nu, p):
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    upper = pc > 0.5
    # central: w = I^-1(|2p-1|; 1/2, nu/2); tails: x = I^-1(2 min(p, 1-p);
    # nu/2, 1/2). Both targets are exact in binary; 1 - |2p - 1| is not, and
    # rounds to 0 at p = 1e-20
    ptail = 2.0 * jnp.where(upper, 1.0 - pc, pc)
    aq = jnp.abs(2.0 * pc - 1.0)
    central = ptail >= 0.5
    ca, cb = _half_series(0.5 * nu)
    w = betaincinv_core(
        jnp.asarray(0.5), 0.5 * nu, jnp.where(central, aq, 0.25), ca, cb
    )
    x = betaincinv_core(
        0.5 * nu, jnp.asarray(0.5), jnp.where(central, 0.5, ptail), cb, ca
    )
    # two square roots: the ratio overflows for x below ~1e-308, t does not
    mag = jnp.where(
        central, jnp.sqrt(nu * w / (1.0 - w)), jnp.sqrt(nu * (1.0 - x)) / jnp.sqrt(x)
    )
    t = jnp.where(pc < 0.5, -mag, mag)
    t = jnp.where(p <= 0.0, -jnp.inf, jnp.where(p >= 1.0, jnp.inf, t))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(nu)
    return jnp.where(oob, jnp.nan, t)


@jax.custom_jvp
def _stdtrit_cj(nu, p):
    """Student-t quantile, nu in [0.2, 200], by betaincinv in both orientations:
    exact 0 at p = 1/2, +-inf at the ends, gradients by the implicit function
    theorem. Deep tails resolve (nu = 4, p = 1e-300 gives -1.3e75) until the beta
    quantile underflows, short of the representable t at small nu (nu = 1/2,
    p = 1e-100)."""
    return _stdtrit_impl(nu, p)


@_stdtrit_cj.defjvp
def _stdtrit_jvp(primals, tangents):
    nu_c, p = primals
    dnu, dp = tangents
    # the custom function, not the solve: a second derivative then takes this
    # rule again instead of differentiating the Newton loop
    t = _stdtrit_cj(nu_c, p)
    finite = jnp.isfinite(t)
    ts = jnp.where(finite, t, 0.0)
    pdf = jnp.exp(t_logpdf(nu_c, ts))
    # dt/dp = 1/pdf may overflow; dt/dnu = -(dF/dnu)/pdf must not, though at
    # nu = 4, p = 1e-300 both parts are ~1e-320 (ratio 5.7e76). So through logs:
    # dt/dnu = -exp(lnF - ln pdf) dlnF/dnu
    t_safe = jnp.where(finite & (ts != 0.0), ts, 1.0)
    live = finite & (ts != 0.0)
    lnF, dlnF_dnu = jax.jvp(
        lambda n: _log_stdtr_impl(n, t_safe), (nu_c,), (jnp.asarray(1.0),)
    )
    dt_dnu = -jnp.exp(lnF - t_logpdf(nu_c, t_safe)) * dlnF_dnu
    dt_dnu = jnp.where(live, dt_dnu, 0.0)
    dt = dp / pdf + dt_dnu * dnu
    dt = jnp.where(finite, dt, 0.0)
    return t, jnp.where(jnp.isnan(t), jnp.nan, dt)


# float_params makes an integer, keyword or size-one nu a float scalar before
# the custom_jvp rules see a tangent
stdtr = float_params(_stdtr_cj)
log_stdtr = float_params(_log_stdtr_cj)
stdtrit = float_params(_stdtrit_cj)
log_stdtr_sf = float_params(log_stdtr_sf)
