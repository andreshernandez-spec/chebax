"""Shared runtime for recipes: table reconstruction, domain checks, the
safeguarded root solver and its per-dtype constants, a float64 digamma, the
per-element panel lookup.

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


def frozen_cache(build):
    """The bounded cache of a _frozen factory. The build runs outside any
    active trace: called first under jit, it would otherwise store tracers
    (jnp.asarray stages even numpy input), and every later call would fail
    with UnexpectedTracerError."""

    @functools.lru_cache(maxsize=128)
    @functools.wraps(build)
    def cached(*key):
        with jax.ensure_compile_time_eval():
            return build(*key)

    return cached


def canon_float(x):
    """Promote to the canonical float dtype (float64 under x64): a float32 input
    under x64 would mismatch the solvers' fori_loop carry."""
    return jnp.asarray(x, dtype=jnp.empty(()).dtype)


def _half_parts(x):
    # x/2 is subnormal below 2 tiny and XLA flushes it to 0; each branch sees a
    # masked x, so the other's nan gradient does not leak through the where
    small = x < 2 * jnp.finfo(x.dtype).tiny
    return small, jnp.where(small, x, 1.0), jnp.where(small, 2.0, x)


def pow_half(x, v):
    """(x/2)^v, as x^v 2^-v below 2 tiny."""
    small, lo, hi = _half_parts(x)
    scale = 2.0 ** -v if isinstance(v, (int, float)) else jnp.exp2(-v)
    return jnp.where(small, jnp.power(lo, v) * scale, jnp.power(hi / 2, v))


def log_half(x):
    """ln(x/2), as ln x - ln 2 below 2 tiny."""
    small, lo, hi = _half_parts(x)
    return jnp.where(small, jnp.log(lo) - math.log(2.0), jnp.log(hi / 2))


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
    """True unless every parameter holds exactly one value (an empty array goes
    per element, where it gives an empty result); static, so free."""
    return any(jnp.size(p) != 1 for p in params)


def renamed_order(who, nu, v):
    """The order passed as nu, or by its 0.4 keyword v= with a warning."""
    if v is None:
        if nu is None:
            raise TypeError(f"{who}() missing the order nu")
        return nu
    if nu is not None:
        raise TypeError(f"{who}() got the order twice, as nu and as v")
    warnings.warn(
        f"{who}(v=...) is deprecated: the order is {who}(nu=...) now",
        DeprecationWarning,
        stacklevel=3,
    )
    return v


def positional_domain(x, domain):
    """True for the 0.4 call besselj(v, (lo, hi)), whose second argument was
    the domain: a two-element tuple in x's place, with no domain= given."""
    return domain is None and isinstance(x, tuple) and len(x) == 2


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


def panel_coefs(table_lo, table_hi, lo, split, hi, p):
    """traced_coefs on two parameter panels [lo, split] and [split, hi]. Each
    panel sees p clamped into its own range: evaluated past it, the panel not
    taken overflows float32 (T_k grows like 7.9^k at twice the range), and
    reverse mode turns that inf times a zero cotangent into nan."""
    below = p <= split
    c_lo = traced_coefs(table_lo, lo, split, jnp.where(below, p, split))
    c_hi = traced_coefs(table_hi, split, hi, jnp.where(below, split, p))
    return jnp.where(below, c_lo, c_hi)


def check_range(owner, name, v, lo, hi):
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{owner} table covers {name} in [{lo}, {hi}], got {v}")
    return v


# XLA's float64 log1p is up to 128 ulp off on (-0.63, -0.25) before jax 0.12
# (increment 34); float32, and float64 from 0.12 on, are within 1.4 ulp there
_LOG1P_EXACT = jax.__version_info__ >= (0, 12)


def log1p(x):
    """log1p for x in (-1, inf), as log(1 + x) below -0.25 where XLA's own is
    off (5.5e-14 in gammainc_array's ln P, 3.7e-16 this way). XLA simplifies
    the textbook x log(u)/(u - 1) back to log1p."""
    if _LOG1P_EXACT or jnp.result_type(x) != jnp.float64:
        return jnp.log1p(x)
    return jnp.where(x < -0.25, jnp.log(1.0 + x), jnp.log1p(x))


def nan_outside(p, lo, hi):
    """check_range for a traced p: nan where p is outside [lo, hi], so the table
    reads nan instead of extrapolating. Added, not selected: a where would give
    p a zero cotangent in reverse mode."""
    return p + jnp.where((p < lo) | (p > hi), jnp.nan, 0.0)


def cheb_basis(s, n):
    """T_0(s), ..., T_{n-1}(s) along a new last axis."""
    b = [jnp.ones_like(s), s]
    for _ in range(2, n):
        b.append(2.0 * s * b[-1] - b[-2])
    return jnp.stack(b[:n], axis=-1)


def _panel_clenshaw(c, s):
    b1 = b2 = jnp.zeros_like(s)
    for ck in c[:0:-1]:
        b1, b2 = ck - b2 + 2.0 * s * b1, b1
    return c[0] - b2 + s * b1


def per_coefficient_series(panels, i, s):
    n = panels.shape[1]
    return _panel_clenshaw(
        [jnp.take(panels[:, k], i, mode="clip") for k in range(n)], s
    )


def row_series(panels, i, s):
    rows = jnp.take(panels, i, axis=0, mode="clip")
    return _panel_clenshaw([rows[..., k] for k in range(panels.shape[1])], s)


def panel_series(panels, i, s):
    """Row i of panels[i, k] as a Chebyshev series at s in [-1, 1], i and s
    per element. On the CPU, one gather per coefficient fuses into the
    recurrence, and one row gather instead runs 1.5-4.2x slower. On the GPU,
    the index those gathers share is written to memory first (gammaln's
    float32 values at 0.57x jax's), and one row gather reads it once
    (experiments/44)."""
    return jax.lax.platform_dependent(
        panels, i, s, cpu=per_coefficient_series, default=row_series
    )


def _patch_parts(tab, n, sy, sp, grad):
    """sum_kj tab[n, k, j] T_k(sy) T_j(sp), with grad also its sy and sp derivatives.
    One gather per coefficient of the flat table: elementwise, so XLA fuses the whole
    sum into one loop. Gathering the [k, j] block for an einsum stored it per element
    and ran a batched dot over it (experiment 43)."""
    ky, kp = tab.shape[1:]
    # Inside another custom_jvp rule (skewnorm's), jit hands the table over as a
    # numpy constant, which a traced index cannot gather from.
    flat = jnp.asarray(tab).reshape(-1)
    base = n * (ky * kp)
    ty, dty = _basis_and_slope(sy, ky)
    tp, dtp = _basis_and_slope(sp, kp)
    f = fy = fp = 0.0
    for k in range(ky):
        row = drow = 0.0
        for j in range(kp):
            c = flat[base + (k * kp + j)]
            row = row + c * tp[j]
            if grad:
                drow = drow + c * dtp[j]
        f = f + ty[k] * row
        if grad:
            fy, fp = fy + dty[k] * row, fp + ty[k] * drow
    return f, fy, fp


def _basis_and_slope(s, n):
    """T_0(s), ..., T_{n-1}(s) and their derivatives, as lists."""
    t, dt = [jnp.ones_like(s), s], [jnp.zeros_like(s), jnp.ones_like(s)]
    for _ in range(2, n):
        t.append(2.0 * s * t[-1] - t[-2])
        dt.append(2.0 * t[-2] + 2.0 * s * dt[-1] - dt[-2])
    return t[:n], dt[:n]


@jax.custom_jvp
def patch_sum(tab, n, sy, sp):
    """The 2-D Chebyshev series of patch n of tab ([patch, k, j] coefficients, k in
    sy, j in sp) at sy, sp in [-1, 1], n integer, all broadcast together."""
    return _patch_parts(tab, n, sy, sp, False)[0]


_PACK_BYTES = 512


def _patch_sum_jvp(primals, tangents):
    tab, n, sy, sp = primals
    dtab, _, dsy, dsp = tangents
    zero = jax.custom_derivatives.SymbolicZero
    f, fy, fp = _patch_parts(tab, n, sy, sp, True)
    if tab[0].size * tab.dtype.itemsize > _PACK_BYTES:
        # As three results XLA keeps the gathered coefficients for the reverse pass (402
        # MiB at 2^18 in float64, 128 MiB packed). Packing pays for a 12 x 12 float64
        # block, not a 7 x 7 float32 one, on the CPU (experiment 43, "other packing").
        parts = jax.lax.optimization_barrier(jnp.stack((f, fy, fp), axis=-1))
        f, fy, fp = parts[..., 0], parts[..., 1], parts[..., 2]
    out = jnp.zeros_like(f)
    if not isinstance(dsy, zero):
        out = out + fy * dsy
    if not isinstance(dsp, zero):
        out = out + fp * dsp
    if not isinstance(dtab, zero):
        out = out + patch_sum(dtab, n, sy, sp)
    return f, out


patch_sum.defjvp(_patch_sum_jvp, symbolic_zeros=True)


def locate_patch(val, cuts, n):
    """Interval index of val in each row of cuts (n intervals, last cut repeated
    as padding) and val mapped onto [-1, 1] inside it."""
    k = jnp.clip(jnp.sum(val[..., None] >= cuts[..., 1:], axis=-1), 0, n - 1)
    lo = jnp.take_along_axis(cuts, k[..., None], axis=-1)[..., 0]
    hi = jnp.take_along_axis(cuts, k[..., None] + 1, axis=-1)[..., 0]
    return k, 2.0 * (val - lo) / (hi - lo) - 1.0


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
