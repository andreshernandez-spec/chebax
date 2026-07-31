"""Opt-in pytensor JAX dispatches backed by chebax.

    import chebax.pytensor  # noqa: F401

registers JAX lowerings for the pytensor scalar ops that otherwise need
tfp-nightly on the JAX backend (pm.sample with nuts_sampler="numpyro" or
"blackjax"), and gives betainc gradients with respect to its shape
parameters. Importing this module is the entire API; it does nothing
else. pytensor is imported here and only here (the chebax runtime proper
stays jax + numpy).

What gets registered, and the contracts:

- Erfcx, Erfcinv: pure jax (native erfcx; erfcinv via ndtri). No chebax
  tables, no tfp, no restrictions.
- GammaIncInv, GammaIncCInv: chebax.gammaincinv for a SCALAR shape
  (0-d or size-1; that is what a sampled scalar RV lowers to). The
  complemented inverse is solved at 1 - p, so its upper-tail depth is
  limited to ~eps. Batched shape arrays fall back to tfp when
  installed, else raise NotImplementedError at trace time.
- BetaIncInv: chebax.betaincinv for scalar (a, b) inside chebax's
  domain box (a, b) in [0.1, 10]^2; outside the box the result is nan
  (loud, never silently wrong). Batched shapes fall back as above.
- Ive: chebax.besseli_fn(scaled=True) for a scalar order v in [0, 10]
  and x >= 0; nan outside (I_{-v} differs from I_v, so negative orders
  are not folded). Batched orders fall back as above.
- Kve: exp(log_besselk_fn(|v|, x) + x) for scalar |v| <= 10, x >= 1e-6
  (K_{-v} = K_v). nan outside; batched orders fall back as above.
- BetaInc: values stay jax's own betainc on every domain; a custom_jvp
  adds the a/b derivatives (the "Betainc gradient with respect to a and
  b not supported" gap) from forward mode through chebax.betainc_fn,
  valid for scalar (a, b) in [0.1, 10]^2 and nan outside; the x
  derivative is the exact Beta density. Batched shapes keep the plain
  jax op (values fine, a/b gradients raise as before).
"""

import jax
import jax.numpy as jnp
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.scalar.math import (BetaInc, BetaIncInv, Erfcinv, Erfcx,
                                  GammaIncCInv, GammaIncInv, Ive, Kve)

import chebax

_BOX_LO, _BOX_HI = 0.1, 10.0


def _is_uniform(p):
    return jnp.ndim(p) == 0 or getattr(p, "size", None) == 1


def _as_scalar(p):
    return jnp.reshape(jnp.asarray(p), ())


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
    return jax.scipy.special.erfcx


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
        # solved at 1 - p: upper-tail resolution is limited to ~eps of 1
        return chebax.gammaincinv(_as_scalar(a), 1.0 - p)

    return gammainccinv


def _in_box(*params):
    ok = jnp.asarray(True)
    for p in params:
        ok = ok & (p >= _BOX_LO) & (p <= _BOX_HI)
    return ok


def _clip_box(p):
    return jnp.clip(p, _BOX_LO, _BOX_HI)


@jax_funcify.register(BetaIncInv)
def _jax_funcify_BetaIncInv(op, **kwargs):
    def betaincinv(a, b, p):
        if not (_is_uniform(a) and _is_uniform(b)):
            return _tfp_fallback(op, "betaincinv",
                                 (jnp.shape(a), jnp.shape(b), jnp.shape(p)))(a, b, p)
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
        out = chebax.besseli_fn(jnp.clip(v, 0.0, 10.0),
                                jnp.where(x >= 0.0, x, 0.0), scaled=True)
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
        out = jnp.exp(chebax.log_besselk_fn(jnp.clip(v, 0.0, 10.0), xs) + xs)
        return jnp.where(ok, out, jnp.nan)

    return kve


@jax.custom_jvp
def _betainc_scalar_ab(a, b, x):
    return jax.scipy.special.betainc(a, b, x)


@_betainc_scalar_ab.defjvp
def _betainc_scalar_ab_jvp(primals, tangents):
    a, b, x = primals
    da, db, dx = tangents
    p = jax.scipy.special.betainc(a, b, x)
    # d/dx is the Beta density, exact and valid for all a, b > 0
    xs = jnp.clip(x, 1e-300, 1.0 - 1e-16)
    log_pdf = ((a - 1.0) * jnp.log(xs) + (b - 1.0) * jnp.log1p(-xs)
               - jax.scipy.special.betaln(a, b))
    d_dx = jnp.exp(log_pdf)
    # d/da, d/db from forward mode through chebax's table representation,
    # valid inside the box, nan outside
    ac, bc = _clip_box(a), _clip_box(b)
    _, ga = jax.jvp(lambda t: chebax.betainc_fn(t, bc, x), (ac,), (jnp.ones_like(ac),))
    _, gb = jax.jvp(lambda t: chebax.betainc_fn(ac, t, x), (bc,), (jnp.ones_like(bc),))
    inbox = _in_box(a, b)
    ga = jnp.where(inbox, ga, jnp.nan)
    gb = jnp.where(inbox, gb, jnp.nan)
    return p, ga * da + gb * db + d_dx * dx


@jax_funcify.register(BetaInc)
def _jax_funcify_BetaInc(op, **kwargs):
    def betainc(a, b, x):
        if not (_is_uniform(a) and _is_uniform(b)):
            # values are fine everywhere; a/b gradients raise as in plain jax
            return jax.scipy.special.betainc(a, b, x)
        return _betainc_scalar_ab(_as_scalar(a), _as_scalar(b), x)

    return betainc
