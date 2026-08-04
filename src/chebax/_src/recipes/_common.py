"""Shared runtime machinery for recipes: parameter-table reconstruction,
domain checks, the safeguarded root solver, and a float64 digamma.

Reconstruction convention (matches _gen_common's emitted axis note): for a
2-D table, TABLE[k, j] is the j-th parameter-direction coefficient of the
k-th argument-direction coefficient, the parameter mapped affinely from
[lo, hi] onto [-1, 1]. `param_coefs` is the eager numpy path used by the
cached factories; `traced_coefs` is the jax path used by the *_fn traced
variants (differentiable in the parameter through the series custom_jvp).
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

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


def canon_param(p, who, name):
    """A uniform-per-call parameter: scalar, or any array holding one value.

    numpyro and pytensor both hand back a broadcast (1,) array where the
    caller wrote a scalar. A censored StudentT is the case that found
    this: numpyro's censoring wrapper broadcasts the base distribution,
    so `self.df` reaches `cdf` with shape (1,), and every traced entry
    point here died deep inside the table reconstruction on it, with
    "coef must be a nonempty 1-D array, got shape (68, 1)" or "Branch
    index must be scalar". None of those name the parameter or say what
    to do.

    One value is uniform whatever its shape, so take it. Anything larger
    is a per-element parameter, which is out of scope by design and is
    what chebax.pergroup is for; say so instead of failing later.
    """
    a = canon_float(p)
    if a.ndim == 0:
        return a
    if a.size == 1:
        return jnp.reshape(a, ())
    raise ValueError(
        f"{who} takes {name} as a scalar (uniform per call), got shape "
        f"{a.shape}. For one parameter set per group use "
        f"chebax.pergroup({who}, group_idx).")


@jax.custom_jvp
def edge_slope(s, x):
    """Zero, carrying slope s in x.

    A CDF-like recipe picks its endpoint values with a hard select, and
    both branches there are constants, so AD sees no slope AT the endpoint
    even where the true one-sided derivative is exact and finite (the
    density). Adding this zero splices that slope back in without touching
    a value. bake knows this rule by name and folds it, which costs the
    artifact exactly that endpoint slope and is disclosed there."""
    return jnp.zeros_like(x)


@edge_slope.defjvp
def _edge_slope_jvp(primals, tangents):
    s, x = primals
    return jnp.zeros_like(x), s * tangents[1]


def float_params(impl):
    """Public entry point for a custom_jvp'd implementation: promote every
    argument to the canonical float dtype before the rule sees it.

    A python or numpy INTEGER parameter is not a differentiable type, so
    jax hands the rule a float0 tangent for it and the rule's own
    arithmetic dies on that: grad through stdtr(4, t), besseli_fn(2, x)
    and vonmises_cdf(5, t) all failed this way (review, 2026-08-02), and
    nu=4 or kappa=5 is how anyone would write those. Converting OUTSIDE
    the rule turns the parameter into an ordinary float constant, which
    is what the caller meant; converting inside cannot work, since by
    then the tangent already exists.

    Leading arguments are the parameters and the last is the evaluation
    point (the library's argument order throughout), so the parameters
    also go through canon_param: a broadcast size-1 array counts as the
    scalar it holds, and a genuinely per-element one is refused by name.
    """
    who = getattr(impl, "__name__", "this function").lstrip("_")
    who = who[:-3] if who.endswith("_cj") else who

    @functools.wraps(impl)
    def public(*args, **kw):
        if len(args) < 2:
            return impl(*[canon_float(a) for a in args], **kw)
        params = [canon_param(a, who, f"argument {i + 1}")
                  for i, a in enumerate(args[:-1])]
        return impl(*params, canon_float(args[-1]), **kw)
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
