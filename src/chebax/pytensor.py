"""Opt-in pytensor JAX dispatches backed by chebax.

    import chebax.pytensor  # noqa: F401

registers JAX lowerings for the pytensor scalar ops that otherwise need
tfp-nightly on the JAX backend (pm.sample with nuts_sampler="numpyro" or
"blackjax"), and gives betainc gradients with respect to its shape
parameters. Needs pytensor >= 3.2, which itself needs python >= 3.12. Importing this module is the entire API; it does nothing
else. pytensor is imported here and only here (the chebax runtime proper
stays jax + numpy).

What gets registered, and the contracts:

- Erfcx, Erfcinv: pure jax (native erfcx; erfcinv via ndtri). No chebax
  tables, no tfp, no restrictions.
- GammaIncInv, GammaIncCInv: chebax.gammaincinv / chebax.gammainccinv
  for a SCALAR shape (0-d or size-1; that is what a sampled scalar RV
  lowers to). The complemented inverse takes p directly, so the upper
  tail resolves as deep as the lower one. Batched shape arrays fall
  back to tfp when installed, else raise NotImplementedError at trace
  time.
- BetaIncInv: chebax.betaincinv for scalar (a, b) inside chebax's
  domain box (a, b) in [0.1, 100]^2; outside the box the result is nan
  (loud, never silently wrong). Batched shapes fall back as above.
- Ive: chebax.besseli_fn(scaled=True) for a scalar order v in [0, 10]
  and x >= 0; nan outside (I_{-v} differs from I_v, so negative orders
  are not folded). Batched orders fall back as above.
- Kve: besselk_fn(|v|, x, scaled=True) for scalar |v| <= 10, x >= 1e-6
  (K_{-v} = K_v). nan outside; batched orders fall back as above.
- GammaInc, GammaIncC: chebax.gammainc_fn / gammaincc_fn for a SCALAR
  shape from a = 0.1 up, jax's own igamma below that through a lax.cond
  (never a select, which would evaluate the loop on every lane). These
  are plain forward functions a user may call anywhere, so unlike the
  inverse CDFs they answer everywhere rather than nan out of box. The
  loop leaves the compiled program for a concrete shape; for a traced one
  it stays in the module and the conditional does not execute it.
  Measured (experiments/19, RTX 3080, f64, N = 4.2M): values 1.3-9.8x
  vs jax's igamma depending on the shape, dP/da 2.1-13.7x, the cond
  itself free (0.97-1.03x on values). Batched shapes fall back as above.
- BetaInc: chebax.betainc_fn for scalar (a, b) in [0.1, 100]^2, jax's
  own betainc outside the box, through the same lax.cond as the gamma
  pair and for the same reason. A custom_jvp adds the a/b derivatives
  (the "Betainc gradient with respect to a and b not supported" gap)
  from forward mode off the same tables, nan outside the box; the x
  derivative is the exact Beta density and is right everywhere.
  Measured (same run): values 108-131x vs jax's betainc, agreement
  <= 3.8e-14, the cond free again (0.95-1.02x). The a-derivative runs
  6-7x faster than jax.grad of chebax.betainc_fn itself, because the
  rule pushes one forward-mode tangent through the reconstruction
  where reverse mode transposes the whole contraction.
  Batched shapes keep the plain jax op (values fine, a/b gradients
  raise as before).

Both cond'd registrations answer in the dtype plain jax would have
given, so a float32 model stays float32: the tables are f64 and the
cond will not even trace with the two branches disagreeing. A float32
graph therefore still evaluates in f64 and casts down, which is a real
cost on a consumer part: betainc still wins 14.3x there, gammainc is a
wash (1.00x) at a = 3, which is jax's most favourable shape.
"""

import jax
import jax.numpy as jnp
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.scalar.math import (BetaInc, BetaIncInv, Erfcinv, Erfcx,
                                  GammaInc, GammaIncC, GammaIncCInv,
                                  GammaIncInv, Ive, Kve)

import chebax
from chebax import domains as _dom

_BOX_LO, _BOX_HI = _dom.BETAINC.lo, _dom.BETAINC.hi


def _is_uniform(p):
    return jnp.ndim(p) == 0 or getattr(p, "size", None) == 1


def _as_scalar(p):
    return jnp.reshape(jnp.asarray(p), ())


def _out_dtype(*args):
    """The dtype plain jax would have given this call.

    The tables are f64, so gammainc_fn and betainc_fn answer in f64
    whatever they are handed. Feeding that straight into a lax.cond
    against jax's own branch does not trace at all on a float32 graph
    (the two branches disagree), and widening to f64 to make it trace
    would quietly change the dtype of a pm.sample the user set up as
    float32. Registering chebax must not move a graph's dtypes, so both
    branches land here. jax sends all-integer arguments to the default
    float, so do the same.
    """
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
    return _forward_gamma(op, "igamma", chebax.gammainc_fn,
                          jax.scipy.special.gammainc)


@jax_funcify.register(GammaIncC)
def _jax_funcify_GammaIncC(op, **kwargs):
    return _forward_gamma(op, "igammac", chebax.gammaincc_fn,
                          jax.scipy.special.gammaincc)


def _forward_gamma(op, tfp_name, cheb_fn, jax_fn):
    """P(a, x) or Q(a, x): chebax's tables in box, jax's loop outside.

    Unlike the inverse CDFs, these are plain forward functions a user may
    call anywhere, so answering nan outside the table box is not
    defensible here. The fallback is a lax.cond and NOT a select: cond
    executes one branch, select would evaluate jax's while_loop on every
    lane and give back exactly what this registration removes. It can be
    a cond because the shape is already required to be scalar, so the
    predicate is scalar too and no lane can disagree with another.

    In box, the loop is gone: fixed-degree polynomial evaluation, which
    is the whole point (measured 2.4-2.9x end to end on a truncated Gamma
    with a sampled shape, six HLO whiles to zero; experiments/18).
    """
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
            lambda: cheb_fn(lo, xs).astype(dt),
            lambda: jax_fn(jnp.broadcast_to(a_s, jnp.shape(xs)), xs).astype(dt))

    return forward


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
        out = chebax.besselk_fn(jnp.clip(v, 0.0, 10.0), xs, scaled=True)
        return jnp.where(ok, out, jnp.nan)

    return kve


def _jax_betainc(a, b, x, dt):
    return jax.scipy.special.betainc(a, b, x).astype(dt)


@jax.custom_jvp
def _betainc_scalar_ab(a, b, x):
    dt = _out_dtype(a, b, x)
    return jax.lax.cond(
        _in_box(a, b),
        lambda: chebax.betainc_fn(_clip_box(a), _clip_box(b), x).astype(dt),
        lambda: _jax_betainc(a, b, x, dt))


@_betainc_scalar_ab.defjvp
def _betainc_scalar_ab_jvp(primals, tangents):
    a, b, x = primals
    da, db, dx = tangents
    dt = _out_dtype(a, b, x)

    def in_box():
        # value and both shape derivatives off the same tables, forward mode
        ac, bc = _clip_box(a), _clip_box(b)
        one = jnp.ones_like(ac)
        p, ga = jax.jvp(lambda t: chebax.betainc_fn(t, bc, x), (ac,), (one,))
        _, gb = jax.jvp(lambda t: chebax.betainc_fn(ac, t, x), (bc,), (one,))
        return p.astype(dt), ga.astype(dt), gb.astype(dt)

    def out_of_box():
        # jax's value, and nan rather than a wrong shape derivative
        p = _jax_betainc(a, b, x, dt)
        nan = jnp.full(jnp.shape(p), jnp.nan, dt)
        return p, nan, nan

    p, ga, gb = jax.lax.cond(_in_box(a, b), in_box, out_of_box)
    # d/dx is the Beta density, exact and valid for all a, b > 0
    fi = jnp.finfo(dt)
    xs = jnp.clip(x, fi.tiny, 1.0 - fi.eps)
    log_pdf = ((a - 1.0) * jnp.log(xs) + (b - 1.0) * jnp.log1p(-xs)
               - jax.scipy.special.betaln(a, b))
    d_dx = jnp.exp(log_pdf).astype(dt)
    return p, ga * da + gb * db + d_dx * dx


@jax_funcify.register(BetaInc)
def _jax_funcify_BetaInc(op, **kwargs):
    def betainc(a, b, x):
        if not (_is_uniform(a) and _is_uniform(b)):
            # values are fine everywhere; a/b gradients raise as in plain jax
            return jax.scipy.special.betainc(a, b, x)
        dt = _out_dtype(a, b, x)
        return _betainc_scalar_ab(_as_scalar(a).astype(dt),
                                  _as_scalar(b).astype(dt),
                                  jnp.asarray(x, dt))

    return betainc
