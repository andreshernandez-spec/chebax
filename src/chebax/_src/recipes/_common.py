"""Shared runtime machinery for recipes: parameter-table reconstruction,
domain checks, the safeguarded root solver, and a float64 digamma.

Reconstruction convention (matches _gen_common's emitted axis note): for a
2-D table, TABLE[k, j] is the j-th parameter-direction coefficient of the
k-th argument-direction coefficient, the parameter mapped affinely from
[lo, hi] onto [-1, 1]. `param_coefs` is the eager numpy path used by the
cached factories; `traced_coefs` is the jax path used by the *_fn traced
variants (differentiable in the parameter through the series custom_jvp).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms
from chebax._src.series import _chebval


def canon_tag():
    """Cache-key tag for the canonical float dtype: factory instances built
    under different x64 settings must not share a cache slot."""
    return str(jnp.empty(()).dtype)


def canon_float(x):
    """Promote to the canonical float dtype (float64 under x64).

    Solvers mix inputs with reconstructed float64 tables inside a fori_loop
    carry; an explicit float32 input under x64 raised a carry dtype
    mismatch instead of computing. One deliberate dtype for everything."""
    return jnp.asarray(x, dtype=jnp.empty(()).dtype)


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


def newton_bisect(f_and_df, x0, lo0, hi0, iters):
    """Fixed-count safeguarded Newton for increasing f: keep [lo, hi]
    bracketing the root, fall back to the midpoint (or doubling while hi is
    inf) when the Newton step leaves the bracket. The bounds test is
    INCLUSIVE: a Newton step that stalls exactly at the current iterate
    (converged) must be accepted, or a one-sided approach whose far bound
    never tightened catapults a converged iterate to the midpoint of the
    half-open bracket (measured, the quantile-toolkit bug)."""

    def step(_, carry):
        x, lo, hi = carry
        f, df = f_and_df(x)
        lo = jnp.where(f < 0, x, lo)
        hi = jnp.where(f < 0, hi, x)
        xn = x - f / df
        mid = jnp.where(jnp.isfinite(hi), 0.5 * (lo + hi), 2.0 * x + 1.0)
        ok = (xn >= lo) & (xn <= hi) & jnp.isfinite(xn)
        return jnp.where(ok, xn, mid), lo, hi

    x, _, _ = jax.lax.fori_loop(0, iters, step, (x0, lo0, hi0))
    return x


def digamma64(z):
    """float64 psi(z) for z >= 1: recurrence up to z >= 15, then the
    Bernoulli series through z^-12 (A&S 6.3.5 / 6.3.18). Truncation < 1e-16."""
    acc = 0.0
    while z < 15.0:
        acc -= 1.0 / z
        z += 1.0
    zi = 1.0 / (z * z)
    tail = 1 / 12 - zi * (1 / 120 - zi * (1 / 252 - zi * (1 / 240 - zi * (1 / 132 - zi * (691 / 32760)))))
    return acc + math.log(z) - 0.5 / z - zi * tail
