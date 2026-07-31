"""Differentiable quantile functions: betaincinv, gammaincinv, stdtr, stdtrit.

These answer jax#2399, jax#5350 and jax#20358: inverse regularized
incomplete beta and gamma (the Beta and Gamma quantile functions), plus the
Student-t CDF and quantile built on them. All parameters are traceable
(uniform per call), so jax.grad works with respect to every argument —
which is what implicit-reparameterization gradients and copula models need.

The beta and gamma solvers are fixed-count safeguarded Newton (bisection
fallback keeps a bracket, so the iteration is branchless and jittable),
initialized in the deep lower tail from the exact leading asymptotics
(I_x ~ x^a / (a B(a,b)), P(a,x) ~ x^a / Gamma(a+1)); Newton from those
starts converges quadratically, where the generic start needs O(|log p|/a)
additive steps and used to saturate silently. Gradients do NOT
differentiate through the iteration: each inverse carries a custom_jvp
from the implicit function theorem,

    dx*/dp = 1 / pdf(x*),    dx*/dtheta = -(dCDF/dtheta)(x*) / pdf(x*),

with dCDF/da, dCDF/db coming from betainc_fn's traced gradients (or jax's
igamma_grad_a primitive for gamma), all evaluated at the solution.

Guarantees instead of silent saturation: every solve checks its final CDF
residual. A quantile whose true value underflows float64 returns 0.0 (the
scipy convention); anything else that failed to converge returns nan. NaN
inputs propagate to nan; p outside [0, 1] returns nan; the p = 0 and 1
endpoints return the exact distribution endpoints.

stdtr/stdtrit need no solver of their own: both orientations of the
incomplete-beta reduction are used, t^2/(nu+t^2) near the median (exact
first-order behavior through t = 0, where the classic nu/(nu+t^2) form
rounds to 1 and loses everything) and nu/(nu+t^2) in the tails. Their
JVPs use the exact Student-t density. betaincinv and stdtr/stdtrit need
(a, b) inside the betainc table domain [0.1, 10]; stdtr/stdtrit therefore
cover nu in [0.2, 20]. gammaincinv uses jax's own gammainc and has no
table domain. Inputs are computed in the canonical float dtype (float64
under x64), so explicit float32 arguments are promoted, not crashed on.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import betainc_table as _bt
from chebax._src.recipes._common import canon_float as _canon
from chebax._src.recipes._common import newton_bisect as _newton_bisect
from chebax._src.recipes.betainc import (betainc_fn, eval_betainc,
                                         tensor_coefs_traced)
from chebax._src.series import ChebSeries

_eval_betainc = eval_betainc
_tensor_coefs_traced = tensor_coefs_traced

# solved-in-log-space brackets: the edges of normal float64
_U_LO = -745.0
_U_HI = 745.0
# a solve is accepted when the CDF residual is this small relative to the
# target probability; healthy solves land at ~eps relative
_RESID_RTOL = 1e-6


def _ln_beta(a, b):
    return (jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)
            - jax.scipy.special.gammaln(a + b))


@jax.custom_jvp
def _dPda(a, x):
    """dP(a, x)/da (regularized lower gamma), differentiable in x.

    jax's igamma_grad_a primitive has no JVP of its own, which made
    gammaincinv only once differentiable even in p (the second derivative
    reaches this term through x*(p)). The x direction has the closed form
    d/dx dP/da = pdf(a, x) * (ln x - psi(a)); the a direction (d2P/da2)
    has no closed form here and raises."""
    return jax.lax.igamma_grad_a(a, x)


def _dPda_jvp(primals, tangents):
    a, x = primals
    da, dx = tangents
    g = _dPda(a, x)
    zero = jax.custom_derivatives.SymbolicZero
    if not isinstance(da, zero):
        raise NotImplementedError(
            "second derivative of gammaincinv with respect to a needs d2P/da2, "
            "which neither jax nor chebax provides")
    if isinstance(dx, zero):
        return g, jnp.zeros_like(g)
    pdf = jnp.exp((a - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a))
    d = pdf * (jnp.log(x) - jax.scipy.special.digamma(a)) * dx
    return g, d


_dPda.defjvp(_dPda_jvp, symbolic_zeros=True)


@jax.custom_jvp
def betaincinv(a, b, p):
    """Inverse of betainc in x: the Beta(a, b) quantile function.

    a, b are (traceable) scalars in [0.1, 10], uniform per call; p any shape.
    Quantiles below the smallest positive float64 return 0.0 (and, mirrored,
    quantiles within eps of 1 return 1.0, the scipy-shared representation
    limit; use betaincinv(b, a, 1-p) for the distance from 1). A solve that
    fails its CDF-residual check returns nan instead of a wrong value."""
    a = _canon(a)
    b = _canon(b)
    p = _canon(p)
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    cab = ChebSeries(_tensor_coefs_traced(a, b), (0.0, _bt.XSPLIT))
    cba = ChebSeries(_tensor_coefs_traced(b, a), (0.0, _bt.XSPLIT))
    ln_b = _ln_beta(a, b)

    # Solve every element as a LOWER-tail problem of the mirrored orientation
    # (1 - p is exact for p >= 1/2), in logit space so tail quantiles resolve
    # down to the float64 floor; df/du = pdf * x * (1-x) is a stable exp.
    swap = pc > 0.5
    ps2 = jnp.where(swap, 1.0 - pc, pc)
    aa = jnp.where(swap, b, a)

    def f_and_df(u):
        x = jax.nn.sigmoid(u)
        f = jnp.where(swap,
                      _eval_betainc(b, a, x, cba, cab),
                      _eval_betainc(a, b, x, cab, cba)) - ps2
        ls, l1s = -jax.nn.softplus(-u), -jax.nn.softplus(u)
        dfdu = jnp.exp(jnp.where(swap, b, a) * ls
                       + jnp.where(swap, a, b) * l1s - ln_b)
        return f, dfdu

    # Initialization: the generic start converges at ~1/a additive steps in
    # u, which cannot reach deep tails in a fixed count (measured: a = 1,
    # p = 1e-50 stalled at e^-64). In the tail the exact leading asymptotic
    # I_x(a,b) = x^a / (a B(a,b)) (1 + O(x)) inverts to a start within
    # O(x) relatively, and Newton converges quadratically from there.
    u_generic = jnp.where(swap, jnp.log(b / a), jnp.log(a / b)) + jnp.zeros_like(pc)
    u_asym = (jnp.log(ps2) + jnp.log(aa) + ln_b) / aa
    u0 = jnp.clip(jnp.where(ps2 < 0.01, u_asym, u_generic), _U_LO, _U_HI)
    u = _newton_bisect(f_and_df, u0, jnp.full_like(pc, _U_LO),
                       jnp.full_like(pc, _U_HI), 64)

    # Convergence is checked, never assumed. A solve pinned at the bracket
    # floor means the true quantile underflows float64 (or the CDF itself
    # is indistinguishable from 0 there): return the mirrored endpoint.
    # Any other unconverged element returns nan.
    f_fin, _ = f_and_df(u)
    at_floor = u <= _U_LO + 0.5
    x_low = jnp.where(at_floor, 0.0, jax.nn.sigmoid(u))
    bad = ~at_floor & (jnp.abs(f_fin) > _RESID_RTOL * ps2)
    x = jnp.where(swap, 1.0 - x_low, x_low)
    x = jnp.where(bad, jnp.nan, x)
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, 1.0, x))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(a) | jnp.isnan(b)
    return jnp.where(oob, jnp.nan, x)


@betaincinv.defjvp
def _betaincinv_jvp(primals, tangents):
    a, b, p = primals
    da, db, dp = tangents
    x = betaincinv(a, b, p)
    interior = (x > 0.0) & (x < 1.0)
    xs = jnp.where(interior, x, 0.5)
    pdf = jnp.exp((_canon(a) - 1) * jnp.log(xs)
                  + (_canon(b) - 1) * jnp.log1p(-xs) - _ln_beta(_canon(a), _canon(b)))
    _, d_cdf = jax.jvp(lambda aa, bb: betainc_fn(aa, bb, xs), (a, b), (da, db))
    dx = (_canon(dp) - d_cdf) / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


@jax.custom_jvp
def gammaincinv(a, p):
    """Inverse of jax.scipy.special.gammainc in x: the Gamma(a, 1) quantile.

    a is a (traceable) positive scalar, uniform per call; p any shape. Needs
    no chebax tables (jax's own gammainc supplies values and the a-gradient).
    Quantiles below the smallest positive float64 return 0.0; a solve that
    fails its CDF-residual check returns nan instead of a wrong value.

    Second derivatives in p and mixed p-a derivatives work (dP/da is
    wrapped with its exact x-derivative); a pure second derivative in a
    needs d2P/da2, which is unavailable and raises with a clear message."""
    a = _canon(a)
    p = _canon(p)
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    lga = jax.scipy.special.gammaln(a)

    # solve in log space: x = e^v, so small-a tail quantiles stay resolvable
    # and df/dv = pdf * x = exp(a v - e^v - lgamma(a))
    def f_and_df(v):
        x = jnp.exp(v)
        f = jax.scipy.special.gammainc(a, x) - pc
        dfdv = jnp.exp(a * v - x - lga)
        return f, dfdv

    # Initialization: Wilson-Hilferty for the bulk; the exact leading
    # asymptotic P(a, x) = x^a / Gamma(a+1) (1 + O(x)) in the lower tail,
    # where WH can start far off for large a (measured: a = 20, p = 1e-20
    # started at 0.56 for a root at 0.87 and the fixed count returned 1.53).
    z = jax.scipy.special.ndtri(pc)
    wh = a * (1.0 - 1.0 / (9.0 * a) + z / (3.0 * jnp.sqrt(a))) ** 3
    v_asym = (jnp.log(pc) + jax.scipy.special.gammaln(a + 1.0)) / a
    v_wh = jnp.where(wh > 1e-300, jnp.log(jnp.maximum(wh, 1e-300)), v_asym)
    v0 = jnp.clip(jnp.where(pc < 0.01, v_asym, v_wh), _U_LO, 709.0)
    v = _newton_bisect(f_and_df, v0, jnp.full_like(pc, _U_LO),
                       jnp.full_like(pc, 709.0), 40)

    f_fin, _ = f_and_df(v)
    at_floor = v <= _U_LO + 0.5
    x = jnp.where(at_floor, 0.0, jnp.exp(v))
    bad = ~at_floor & (jnp.abs(f_fin) > _RESID_RTOL * pc)
    x = jnp.where(bad, jnp.nan, x)
    x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, jnp.inf, x))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(a)
    return jnp.where(oob, jnp.nan, x)


def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    zero = jax.custom_derivatives.SymbolicZero
    x = gammaincinv(a, p)
    interior = jnp.isfinite(x) & (x > 0.0)
    xs = jnp.where(interior, x, 1.0)
    pdf = jnp.exp((_canon(a) - 1) * jnp.log(xs) - xs
                  - jax.scipy.special.gammaln(_canon(a)))
    # dP/da through the differentiable wrapper: second derivatives in p and
    # mixed p-a work; only d2/da2 remains unsupported (see _dPda). Symbolic
    # zeros matter here: a materialized 0 tangent would still route the
    # outer derivative through _dPda's a direction.
    num = _canon(dp) if not isinstance(dp, zero) else 0.0
    if not isinstance(da, zero):
        num = num - _dPda(_canon(a), xs) * _canon(da)
    dx = num / pdf
    return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))


gammaincinv.defjvp(_gammaincinv_jvp, symbolic_zeros=True)


def _t_logpdf(nu, t):
    return (jax.scipy.special.gammaln(0.5 * (nu + 1.0))
            - jax.scipy.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(nu * jnp.pi)
            - 0.5 * (nu + 1.0) * jnp.log1p(t * t / nu))


def _stdtr_impl(nu, t):
    t2 = t * t
    central = t2 <= nu
    # near the median: w = t^2/(nu+t^2) resolves F - 1/2 exactly through
    # t = 0 (the classic x = nu/(nu+t^2) rounds to 1 there and F degrades
    # to exactly 0.5); in the tails the classic orientation resolves the
    # tail probability. Seam at t^2 = nu, where both are mid-range.
    w = jnp.where(central, t2 / (nu + t2), 0.25)
    x = jnp.where(central, 0.25, nu / (nu + t2))
    half_central = 0.5 * betainc_fn(jnp.asarray(0.5), 0.5 * nu, w)
    half_tail = 0.5 * betainc_fn(0.5 * nu, jnp.asarray(0.5), x)
    sgn = jnp.where(t >= 0.0, 1.0, -1.0)
    F = jnp.where(central, 0.5 + sgn * half_central,
                  jnp.where(t >= 0.0, 1.0 - half_tail, half_tail))
    return jnp.where(jnp.isnan(t) | jnp.isnan(nu), jnp.nan, F)


@jax.custom_jvp
def stdtr(nu, t):
    """Student-t CDF with nu degrees of freedom (nu in [0.2, 20], traceable).

    jax.grad works with respect to nu as well: learnable degrees of freedom.
    First-order exact through t = 0 (dF/dt there is the density, not 0)."""
    return _stdtr_impl(_canon(nu), _canon(t))


@stdtr.defjvp
def _stdtr_jvp(primals, tangents):
    nu, t = primals
    dnu, dt = tangents
    nu_c, t_c = _canon(nu), _canon(t)
    F = _stdtr_impl(nu_c, t_c)
    # t-direction: the exact density (AD through the w = t^2/(nu+t^2)
    # branch hits a 0 * inf at t = 0; the density is finite there)
    dF_dt = jnp.exp(_t_logpdf(nu_c, jnp.where(jnp.isfinite(t_c), t_c, 0.0)))
    dF_dt = jnp.where(jnp.isfinite(t_c), dF_dt, 0.0)
    # nu-direction: AD of the implementation, at a safe t (t = 0 and
    # t = +-inf have exactly zero nu-sensitivity)
    t_safe = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), t_c, 1.0)
    _, dF_dnu = jax.jvp(lambda n: _stdtr_impl(n, t_safe), (nu_c,), (_canon(dnu),))
    dF_dnu = jnp.where(jnp.isfinite(t_c) & (t_c != 0.0), dF_dnu, 0.0)
    dF = dF_dt * _canon(dt) + dF_dnu
    return F, jnp.where(jnp.isnan(F), jnp.nan, dF)


def _stdtrit_impl(nu, p):
    interior = (p > 0.0) & (p < 1.0)
    pc = jnp.where(interior, p, 0.5)
    q = 2.0 * pc - 1.0
    aq = jnp.abs(q)
    central = aq <= 0.5
    # central: |t| from w = I^-1(|q|; 1/2, nu/2), t = sqrt(nu w/(1-w));
    # tails: x = I^-1(2 min(p,1-p); nu/2, 1/2), t = sqrt(nu (1-x)/x).
    # 1 - aq is 2p or 2(1-p) exactly, so the tail argument is exact.
    w = betaincinv(jnp.asarray(0.5), 0.5 * nu, jnp.where(central, aq, 0.25))
    x = betaincinv(0.5 * nu, jnp.asarray(0.5), jnp.where(central, 0.5, 1.0 - aq))
    mag = jnp.where(central, jnp.sqrt(nu * w / (1.0 - w)),
                    jnp.sqrt(nu * (1.0 - x) / x))
    t = jnp.where(q < 0.0, -mag, mag)
    t = jnp.where(p <= 0.0, -jnp.inf, jnp.where(p >= 1.0, jnp.inf, t))
    oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(nu)
    return jnp.where(oob, jnp.nan, t)


@jax.custom_jvp
def stdtrit(nu, p):
    """Student-t quantile with nu degrees of freedom (nu in [0.2, 20]).

    Closed form via betaincinv in both orientations: exact 0 at p = 1/2,
    +-inf at the endpoints, gradients from the implicit function theorem
    (dt/dp = 1/pdf is 8/3 at the median for nu = 4, not 0)."""
    return _stdtrit_impl(_canon(nu), _canon(p))


@stdtrit.defjvp
def _stdtrit_jvp(primals, tangents):
    nu, p = primals
    dnu, dp = tangents
    nu_c = _canon(nu)
    t = _stdtrit_impl(nu_c, _canon(p))
    finite = jnp.isfinite(t)
    ts = jnp.where(finite, t, 0.0)
    pdf = jnp.exp(_t_logpdf(nu_c, ts))
    # IFT: dt/dp = 1/pdf; dt/dnu = -(dF/dnu at t)/pdf
    t_safe = jnp.where(finite & (ts != 0.0), ts, 1.0)
    _, dF_dnu = jax.jvp(lambda n: _stdtr_impl(n, t_safe), (nu_c,), (jnp.asarray(1.0),))
    dF_dnu = jnp.where(finite & (ts != 0.0), dF_dnu, 0.0)
    dt = (_canon(dp) - dF_dnu * _canon(dnu)) / pdf
    dt = jnp.where(finite, dt, 0.0)
    return t, jnp.where(jnp.isnan(t), jnp.nan, dt)
