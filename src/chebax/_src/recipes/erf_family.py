"""dawsn and erfcx (recent jax ships both; kept for the C++ bake path).

Parameter-free, so these are plain functions, not factories.

- dawsn(x): odd, one hard select at |x| = 6. Inner: x * E(x^2) with E a
  Chebyshev series on [0, 36] — writing it that way (raw x, even series)
  makes the oddness and the gradient at 0 exact. Tail:
  sign(x) * G((6/x)^2) / |x|.
- erfcx(x): t h(t) with t = K / (K + |x|) in (0, 1], h from one of NP
  t-panels picked by floor(NP t), so no select on x; negative x by the
  reflection erfcx(-x) = 2 e^{x^2} - erfcx(x), which
  correctly overflows to inf past x ~ -26.6. x^2 is split exactly before
  the exp: rounded, it puts up to x^2 eps of relative error into e^{x^2}
  (5.7e-14 at x = -22.75). The |x| used for the positive-branch input is
  built from a select (not jnp.abs) so the derivative at exactly 0 is
  right; abs would zero it by its tie convention.

Test oracles: D' = 1 - 2 x D and erfcx' = 2 x erfcx - 2/sqrt(pi).
"""

import functools

import jax
import jax.numpy as jnp

from chebax._src.recipes import erf_table as _et
from chebax._src.recipes._common import canon_float, canon_tag
from chebax._src.series import ChebSeries


@functools.lru_cache(maxsize=4)
def _series_cached(_tag):
    # built lazily: module-level ChebSeries would initialize the jax backend
    # (and touch the device) as an import side effect. Built OUTSIDE any
    # active trace: the first call can come from inside a lax.cond branch
    # (gammainc's large-a path), and a cached series whose arrays were
    # created under that trace leaks its tracers into every later call.
    with jax.ensure_compile_time_eval():
        return (
            ChebSeries(_et.DAWSN_E, (0.0, _et.XS * _et.XS)),
            ChebSeries(_et.DAWSN_G, (0.0, 1.0)),
            canon_float(_et.ERFCX_P),
        )


def _series():
    """The two dawsn series and the erfcx panel table, keyed per x64 mode: a
    cache filled under x64 off would otherwise pin float32 coefficients
    after the flip."""
    return _series_cached(canon_tag())


@jax.custom_jvp
def _exp_sq(x):
    """e^{x^2} with x^2 = hi + lo split, hi exact. x_hi keeps the top half
    of the significand (a bit mask, so no fma or contraction can change
    the split); lo = (x - x_hi)(x + x_hi) is up to 2^-25 x^2 in f64, too
    big for 1 + lo to stand in for e^lo."""
    dt = x.dtype
    it, low = (jnp.int64, 27) if dt == jnp.float64 else (jnp.int32, 12)
    # an infinite x would make x - x_hi nan; e^{x^2} is inf long before
    inf = jnp.isinf(x)
    xs = jnp.where(inf, 0.0, x)
    bits = jax.lax.bitcast_convert_type(xs, it)
    xh = jax.lax.bitcast_convert_type(bits & ~((1 << low) - 1), dt)
    # e^hi (1 + expm1(lo)), not e^hi e^lo: under jit XLA folds a product of
    # exps into the exp of the sum, which rounds x^2 again. |lo| <= 2.1e-5
    # where e^hi is finite, so three terms of expm1 are exact to 1e-20.
    e = jnp.exp(xh * xh)
    lo = (xs - xh) * (xs + xh)
    y = e + e * (lo * (1.0 + lo * (0.5 + lo * (1.0 / 6.0))))
    return jnp.where(inf | jnp.isinf(e), jnp.inf, y)


@_exp_sq.defjvp
def _exp_sq_jvp(primals, tangents):
    # through the split, reverse mode gives inf * 0 = nan once e^{x^2}
    # overflows at an x whose lo is exactly 0 (x = -27); the derivative is
    # 2 x e^{x^2} whatever the split
    (x,), (dx,) = primals, tangents
    y = _exp_sq(x)
    return y, 2.0 * x * y * dx


def dawsn(x):
    """Dawson's integral D(x) = e^{-x^2} int_0^x e^{t^2} dt, any real x."""
    E, G, _ = _series()
    x = jnp.asarray(x)
    ax = jnp.where(x >= 0, x, -x)
    zi = jnp.where(ax <= _et.XS, x * x, _et.XS * _et.XS)
    inner = x * E(zi)
    xo = jnp.where(ax <= _et.XS, _et.XS, ax)
    tail = jnp.sign(x) * G((_et.XS / xo) ** 2) / xo
    return jnp.where(ax <= _et.XS, inner, tail)


def erfcx(x):
    """Scaled complementary error function e^{x^2} erfc(x), any real x.

    Overflows to inf (correctly) for x below ~ -26.6."""
    _, _, panels = _series()
    x = jnp.asarray(x)
    ax = jnp.where(x >= 0, x, -x)
    t = _et.K / (_et.K + ax)
    y = _et.NP * t
    fy = jnp.minimum(jnp.floor(y), _et.NP - 1)
    v = 2.0 * (y - fy) - 1.0
    # one gather per coefficient: on XLA's CPU half the time of gathering rows
    i = fy.astype(jnp.int32)
    c = [jnp.take(panels[:, k], i, mode="clip") for k in range(panels.shape[1])]
    b1 = b2 = jnp.zeros_like(v)
    for k in range(len(c) - 1, 0, -1):
        b1, b2 = c[k] - b2 + 2.0 * v * b1, b1
    pos = t * (c[0] - b2 + v * b1)
    # clamp the reflection's exp argument for the masked (x >= 0) lanes:
    # exp(x^2) = inf there would poison gradients via 0 * inf. The split
    # runs in the result's dtype (integer and f16 inputs promote).
    xn = jnp.where(x < 0, x, 0.0).astype(pos.dtype)
    return jnp.where(x >= 0, pos, 2.0 * _exp_sq(xn) - pos)
