"""Opt-in pytensor JAX dispatches backed by chebax: `import chebax.pytensor`
registers them and is the whole API. Needs pytensor >= 3.2 (python >= 3.12);
pytensor is imported here only.

Each registration takes a scalar (0-d or size-one) shape parameter, which is
what a sampled scalar RV lowers to; batched shapes fall back to tfp when it
is installed and raise NotImplementedError otherwise, except where noted:

- Erfcx, Erfcinv: pure jax, no restrictions.
- GammaIncInv, GammaIncCInv, BetaIncInv: chebax's inverses; BetaIncInv is nan
  outside the (a, b) box [0.1, 100]^2.
- Ive, Kve: besseli_scalar and besselk_scalar(|v|) for |v| <= 10, nan outside.
- GammaInc, GammaIncC, BetaInc: chebax's tables in box, jax's own loop outside,
  through a lax.cond (a select would run the loop on every lane). BetaInc
  gets a/b derivatives from forward mode off the tables, nan out of box;
  batched shapes keep the plain jax op. Speed: experiments/18 and 19.

The cond'd registrations answer in the dtype plain jax would give, so a
float32 graph stays float32 (the f64 tables cast down; experiments/19 has the
cost).
"""

import jax
import jax.numpy as jnp

try:
    from pytensor.link.jax.dispatch import jax_funcify
    from pytensor.scalar.math import (
        BetaInc,
        BetaIncInv,
        Erfcinv,
        Erfcx,
        GammaInc,
        GammaIncC,
        GammaIncCInv,
        GammaIncInv,
        Ive,
        Kve,
    )
except ImportError as e:
    raise ImportError(
        "chebax.pytensor needs pytensor>=3.2: pip install chebax[pytensor]"
    ) from e

import chebax
from chebax import domains as _dom


def _is_uniform(p):
    return jnp.ndim(p) == 0 or getattr(p, "size", None) == 1


def _as_scalar(p):
    return jnp.reshape(jnp.asarray(p), ())


def _out_dtype(*args):
    """The dtype plain jax would give: both cond branches must agree, and
    registering chebax must not move a float32 graph to f64. All-integer
    arguments go to the default float, as in jax."""
    dt = jnp.result_type(*args)
    return dt if jnp.issubdtype(dt, jnp.inexact) else jnp.result_type(float)


def _tfp_fallback(op, name, shapes):
    try:
        import tensorflow_probability.substrates.jax.math as tfp_math
    except ImportError as e:
        raise NotImplementedError(
            f"chebax.pytensor lowers {type(op).__name__} only for scalar "
            f"(uniform-per-call) parameters; got shapes {shapes}. Install "
            f"tfp-nightly for batched parameters."
        ) from e
    return getattr(tfp_math, name)


@jax_funcify.register(Erfcx)
def _jax_funcify_Erfcx(op, **kwargs):
    if hasattr(jax.scipy.special, "erfcx"):
        return jax.scipy.special.erfcx

    # jax < 0.11 has no native erfcx; keep pytensor's tfp behavior there
    def erfcx_fallback(x):
        return _tfp_fallback(op, "erfcx", (jnp.shape(x),))(x)

    return erfcx_fallback


@jax_funcify.register(Erfcinv)
def _jax_funcify_Erfcinv(op, **kwargs):
    def erfcinv(y):
        # erfc(z) = 2 ndtr(-z sqrt(2))  =>  z = -ndtri(y/2)/sqrt(2)
        return -jax.scipy.special.ndtri(0.5 * y) / jnp.sqrt(2.0)

    return erfcinv


@jax_funcify.register(GammaIncInv)
def _jax_funcify_GammaIncInv(op, **kwargs):
    def gammaincinv(a, p):
        if not _is_uniform(a):
            return _tfp_fallback(op, "igammainv", (jnp.shape(a), jnp.shape(p)))(a, p)
        return chebax.gammaincinv(_as_scalar(a), p)

    return gammaincinv


@jax_funcify.register(GammaIncCInv)
def _jax_funcify_GammaIncCInv(op, **kwargs):
    def gammainccinv(a, p):
        if not _is_uniform(a):
            return _tfp_fallback(op, "igammacinv", (jnp.shape(a), jnp.shape(p)))(a, p)
        return chebax.gammainccinv(_as_scalar(a), p)

    return gammainccinv


@jax_funcify.register(GammaInc)
def _jax_funcify_GammaInc(op, **kwargs):
    return _forward_gamma(
        op, "igamma", chebax.gammainc_scalar, jax.scipy.special.gammainc
    )


@jax_funcify.register(GammaIncC)
def _jax_funcify_GammaIncC(op, **kwargs):
    return _forward_gamma(
        op, "igammac", chebax.gammaincc_scalar, jax.scipy.special.gammaincc
    )


def _forward_gamma(op, tfp_name, cheb, plain):
    """P(a, x) or Q(a, x): chebax's tables in box, jax's loop outside, through
    a lax.cond (a select would run the while_loop on every lane; the scalar
    shape makes the predicate scalar). A forward function answers everywhere,
    so no nan out of box."""

    def forward(a, x):
        if not _is_uniform(a):
            return _tfp_fallback(op, tfp_name, (jnp.shape(a), jnp.shape(x)))(a, x)
        dt = _out_dtype(a, x)
        a_s = _as_scalar(a).astype(dt)
        xs = jnp.asarray(x, dt)
        # clipped so the untaken branch stays finite: vmap turns a cond
        # with a batched predicate into a select, which evaluates both,
        # and a nan there would poison the gradient through the select
        lo = jnp.maximum(a_s, _dom.GAMMAINC.lo)
        return jax.lax.cond(
            a_s >= _dom.GAMMAINC.lo,
            lambda: cheb(lo, xs).astype(dt),
            lambda: plain(jnp.broadcast_to(a_s, jnp.shape(xs)), xs).astype(dt),
        )

    return forward


def _in_box(*params):
    ok = jnp.asarray(True)
    for p in params:
        ok = ok & (p >= _dom.BETAINC.lo) & (p <= _dom.BETAINC.hi)
    return ok


def _clip_box(p):
    return jnp.clip(p, _dom.BETAINC.lo, _dom.BETAINC.hi)


@jax_funcify.register(BetaIncInv)
def _jax_funcify_BetaIncInv(op, **kwargs):
    def betaincinv(a, b, p):
        if not (_is_uniform(a) and _is_uniform(b)):
            return _tfp_fallback(
                op, "betaincinv", (jnp.shape(a), jnp.shape(b), jnp.shape(p))
            )(a, b, p)
        a, b = _as_scalar(a), _as_scalar(b)
        out = chebax.betaincinv(_clip_box(a), _clip_box(b), p)
        return jnp.where(_in_box(a, b), out, jnp.nan)

    return betaincinv


@jax_funcify.register(Ive)
def _jax_funcify_Ive(op, **kwargs):
    def ive(v, x):
        if not _is_uniform(v):
            return _tfp_fallback(op, "bessel_ive", (jnp.shape(v), jnp.shape(x)))(v, x)
        v = _as_scalar(v)
        ok = (v >= 0.0) & (v <= 10.0) & (x >= 0.0)
        out = chebax.besseli_scalar(
            jnp.clip(v, 0.0, 10.0), jnp.where(x >= 0.0, x, 0.0), scaled=True
        )
        return jnp.where(ok, out, jnp.nan)

    return ive


@jax_funcify.register(Kve)
def _jax_funcify_Kve(op, **kwargs):
    def kve(v, x):
        if not _is_uniform(v):
            return _tfp_fallback(op, "bessel_kve", (jnp.shape(v), jnp.shape(x)))(v, x)
        v = jnp.abs(_as_scalar(v))  # K_{-v} = K_v
        ok = (v <= 10.0) & (x >= 1e-6)
        xs = jnp.where(x >= 1e-6, x, 1e-6)
        out = chebax.besselk_scalar(jnp.clip(v, 0.0, 10.0), xs, scaled=True)
        return jnp.where(ok, out, jnp.nan)

    return kve


def _jax_betainc(a, b, x, dt):
    return jax.scipy.special.betainc(a, b, x).astype(dt)


@jax.custom_jvp
def _betainc_scalar_ab(a, b, x):
    dt = _out_dtype(a, b, x)
    return jax.lax.cond(
        _in_box(a, b),
        lambda: chebax.betainc_scalar(_clip_box(a), _clip_box(b), x).astype(dt),
        lambda: _jax_betainc(a, b, x, dt),
    )


def _beta_density(a, b, x, dt):
    """d/dx I_x(a, b): the Beta density inside (0, 1), and at the
    endpoints the core recipe's one-sided limits: at x = 0, inf for a < 1,
    b for a = 1 (1/B(1, b) = b), 0 above, and mirrored at x = 1. Clipping
    x into the interior first gave a finite 5e153 at (a, b, x) = (.5, 2, 0)
    where the limit is inf (review, 2026-10-26)."""
    inside = (x > 0.0) & (x < 1.0)
    xs = jnp.where(inside, x, 0.5)
    core = jnp.exp(
        (a - 1.0) * jnp.log(xs)
        + (b - 1.0) * jnp.log1p(-xs)
        - jax.scipy.special.betaln(a, b)
    )
    at0 = jnp.where(a < 1.0, jnp.inf, jnp.where(a == 1.0, b, 0.0))
    at1 = jnp.where(b < 1.0, jnp.inf, jnp.where(b == 1.0, a, 0.0))
    edge = jnp.where(x == 0.0, at0, jnp.where(x == 1.0, at1, 0.0))
    return jnp.where(inside, core, edge).astype(dt)


def _betainc_scalar_ab_jvp(primals, tangents):
    a, b, x = primals
    da, db, dx = tangents
    zero = jax.custom_derivatives.SymbolicZero
    dt = _out_dtype(a, b, x)

    def in_box():
        # value and both shape derivatives off the same tables, forward mode
        ac, bc = _clip_box(a), _clip_box(b)
        one = jnp.ones_like(ac)
        p, ga = jax.jvp(lambda t: chebax.betainc_scalar(t, bc, x), (ac,), (one,))
        _, gb = jax.jvp(lambda t: chebax.betainc_scalar(ac, t, x), (bc,), (one,))
        return p.astype(dt), ga.astype(dt), gb.astype(dt)

    def out_of_box():
        # jax's value, and nan rather than a wrong shape derivative
        p = _jax_betainc(a, b, x, dt)
        nan = jnp.full(jnp.shape(p), jnp.nan, dt)
        return p, nan, nan

    # Only active directions enter the tangent. Out of the box the shape
    # partials are nan, and nan times the zero tangent of an undifferentiated
    # shape made the x derivative nan in forward mode (review, 2026-10-26).
    if isinstance(da, zero) and isinstance(db, zero):
        p = _betainc_scalar_ab(a, b, x)
        dp = jnp.zeros_like(p)
    else:
        p, ga, gb = jax.lax.cond(_in_box(a, b), in_box, out_of_box)
        dp = jnp.zeros_like(p)
        if not isinstance(da, zero):
            dp = dp + ga * da
        if not isinstance(db, zero):
            dp = dp + gb * db
    if not isinstance(dx, zero):
        dp = dp + _beta_density(a, b, x, dt) * dx
    return p, dp


_betainc_scalar_ab.defjvp(_betainc_scalar_ab_jvp, symbolic_zeros=True)


@jax_funcify.register(BetaInc)
def _jax_funcify_BetaInc(op, **kwargs):
    def betainc(a, b, x):
        if not (_is_uniform(a) and _is_uniform(b)):
            # values are fine everywhere; a/b gradients raise as in plain jax
            return jax.scipy.special.betainc(a, b, x)
        dt = _out_dtype(a, b, x)
        return _betainc_scalar_ab(
            _as_scalar(a).astype(dt), _as_scalar(b).astype(dt), jnp.asarray(x, dt)
        )

    return betainc
