"""Shared runtime for recipes: table reconstruction, domain checks, the
safeguarded root solver and its per-dtype constants, a float64 digamma.

A 2-D TABLE[k, j] is the j-th parameter coefficient of the k-th argument
coefficient, the parameter mapped from [lo, hi] onto [-1, 1] (_gen_common's axis
note). param_coefs is numpy for the frozen factories; traced_coefs is jax for the
_scalar paths, differentiable in the parameter.
"""

import functools
import inspect
import math
import warnings
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src import algorithms
from chebax._src.series import _chebval


def canon_tag():
    """Cache-key tag for the canonical float dtype, so x64 modes never share."""
    return str(jnp.empty(()).dtype)


def canon_float(x):
    """Promote to the canonical float dtype (float64 under x64): a float32 input
    under x64 would mismatch the solvers' fori_loop carry."""
    return jnp.asarray(x, dtype=jnp.empty(()).dtype)


def canon_param(p, who, name, array=None):
    """A uniform-per-call parameter: a scalar, or any array holding one value
    (numpyro and pytensor pass (1,) where the caller wrote a scalar, e.g. a
    censored StudentT's df). More values are refused here by name, pointing to
    the family's _array path (array) or chebax.pergroup, instead of failing deep
    in the table reconstruction.
    """
    a = canon_float(p)
    if a.ndim == 0:
        return a
    if a.size == 1:
        return jnp.reshape(a, ())
    per_elem = f"For {name} per element use chebax.{array}; for " if array else "For "
    raise ValueError(
        f"{who} takes {name} as a scalar (uniform per call), got shape "
        f"{a.shape}. {per_elem}one parameter set per group use "
        f"chebax.pergroup({who}, group_idx)."
    )


def per_element(*params):
    """True when any parameter holds more than one value; static, so free."""
    return any(jnp.size(p) > 1 for p in params)


def deprecated_factory(name, frozen, *params, **kw):
    """The 0.4 call name(params): the frozen callable, with a DeprecationWarning."""
    warnings.warn(
        f"{name}(...) without x is deprecated: call {frozen.__name__}(...) "
        f"for the cached callable, or {name}(..., x) for values.",
        DeprecationWarning,
        stacklevel=3,
    )
    return frozen(*params, **kw)


@jax.custom_jvp
def edge_slope(s, x):
    """Zero, carrying slope s in x: puts the density back as the slope at an
    endpoint a hard select makes constant. bake folds it by name, losing that
    slope (disclosed there)."""
    return jnp.zeros_like(x)


@edge_slope.defjvp
def _edge_slope_jvp(primals, tangents):
    s, x = primals
    return jnp.zeros_like(x), s * tangents[1]


def float_params(impl, array=None):
    """Public entry point for a custom_jvp'd impl: normalize every argument
    before the rule sees it. Inside, an integer parameter (stdtr(4, t)) already
    has a float0 tangent and a squeezed size-one one a mismatched tangent. Leading
    arguments go through canon_param, the last (the point) through canon_float;
    keyword calls are bound first. array is passed on to canon_param.
    """
    who = impl.__name__.lstrip("_").removesuffix("_cj")
    sig = inspect.signature(impl)
    *params, point = sig.parameters

    @functools.wraps(impl)
    def public(*args, **kw):
        arg = sig.bind(*args, **kw).arguments
        return impl(
            *(canon_param(arg[n], who, n, array) for n in params),
            canon_float(arg[point]),
        )

    public.__name__ = public.__qualname__ = who
    return public


def param_coefs(table, lo, hi, p):
    """Coefficient vector at parameter p: Clenshaw across table rows. numpy."""
    t = 2.0 * (p - lo) / (hi - lo) - 1.0
    return np.array([algorithms.chebval(t, row) for row in table])


def param_coefs_der(table, lo, hi, p):
    """d/dp of the coefficient vector, via chebder along the parameter axis."""
    t = 2.0 * (p - lo) / (hi - lo) - 1.0
    d = np.array([algorithms.chebval(t, algorithms.chebder(row)) for row in table])
    return d * (2.0 / (hi - lo))


def traced_coefs(table, lo, hi, p):
    """jax version of param_coefs: p may be a tracer (uniform per call)."""
    t = 2.0 * (p - lo) / (hi - lo) - 1.0
    return jax.vmap(lambda row: _chebval(t, row, (-1.0, 1.0)))(jnp.asarray(table))


def check_range(owner, name, v, lo, hi):
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{owner} table covers {name} in [{lo}, {hi}], got {v}")
    return v


def newton_bisect(f_and_df, x0, lo0, hi0, iters, snap=None):
    """Fixed-count Newton for increasing f, bracketed in [lo, hi], falling back to
    the midpoint (doubling while hi is inf). The bounds test is inclusive: else a
    converged step on a one-sided approach jumps to the bracket's midpoint. snap
    maps an iterate to the point f_and_df evaluates (the gamma solve rounds t)."""

    def step(_, carry):
        x, lo, hi = carry
        if snap is not None:
            x = snap(x)
        f, df = f_and_df(x)
        lo = jnp.where(f < 0, x, lo)
        hi = jnp.where(f < 0, hi, x)
        xn = x - f / df
        mid = jnp.where(jnp.isfinite(hi), 0.5 * (lo + hi), 2.0 * x + 1.0)
        ok = (xn >= lo) & (xn <= hi) & jnp.isfinite(xn)
        return jnp.where(ok, xn, mid), lo, hi

    x, _, _ = jax.lax.fori_loop(0, iters, step, (x0, lo0, hi0))
    return x


Limits = namedtuple("Limits", "lo hi v_hi n_beta n_gamma rtol")


@functools.lru_cache(maxsize=4)
def solver_limits(_tag):
    """Solver constants for the canonical float dtype, keyed per x64 mode. The
    log-space bracket is the dtype's normal range, ln(tiny) to ln(max) moved 4 ulps
    inward so XLA's exp stays normal: at the last reachable point the residual's
    sign can say "root below the floor". Whole e-folds (-708, 709) miss roots in
    [tiny, e^-708]. Iterations and the residual bar are measured per dtype
    (tests/test_quantiles.py), shared by the beta and gamma inverses."""
    dt = np.dtype(_tag).type
    f = np.finfo(dt)

    def inward(v):
        v = dt(v)
        for _ in range(4):
            v = np.nextafter(v, dt(0))
        return float(v)

    lo = inward(math.log(float(f.tiny)))
    wide = dt == np.float64
    return Limits(
        lo,
        -lo,
        inward(math.log(float(f.max))),
        64 if wide else 48,
        40 if wide else 24,
        1e-6 if wide else 1e-3,
    )


def digamma64(z):
    """float64 psi(z), z >= 1: recurrence to z >= 15, then A&S 6.3.18 to z^-12
    (truncation < 1e-16)."""
    acc = 0.0
    while z < 15.0:
        acc -= 1.0 / z
        z += 1.0
    zi = 1.0 / (z * z)
    tail = 1 / 12 - zi * (
        1 / 120 - zi * (1 / 252 - zi * (1 / 240 - zi * (1 / 132 - zi * (691 / 32760))))
    )
    return acc + math.log(z) - 0.5 / z - zi * tail
