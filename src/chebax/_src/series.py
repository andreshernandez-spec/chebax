"""Runtime evaluation in jax: single and segmented Chebyshev series.

Only two algorithms run per call: Clenshaw, and (for gradients) one more
Clenshaw on the derivative coefficients. Differentiation and integration act
on coefficients through dense (n x n) matrices built from the numpy
references in algorithms.py, so they work on traced values and agree with
the references exactly; for concrete coefficients jit constant-folds the
matrix product, so the per-call gradient cost is one extra Clenshaw, but
deriv() itself is O(n^2) work and is recomputed per call (caching a
possibly-traced result on the instance would leak tracers).

Series are pytrees: coefficients are leaves, domains and breakpoints are
static. Full float64 accuracy needs jax's x64 mode. Evaluation promotes x
and the coefficients to a common dtype first (a float32 x against float64
tables computes in float64, matching the segmented path).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from chebax._src import algorithms


@functools.lru_cache(maxsize=None)
def _der_matrix(n):
    """chebder as an (n-1, n) matrix (n=1: zero map), for traced coefficients."""
    if n == 1:
        return np.zeros((1, 1))
    return np.array([algorithms.chebder(e) for e in np.eye(n)]).T


@functools.lru_cache(maxsize=None)
def _int_matrix(n):
    """chebint as an (n+1, n) matrix, for traced coefficients."""
    return np.array([algorithms.chebint(e) for e in np.eye(n)]).T


def _clenshaw(t, coef, coef_last_axis=False):
    """Clenshaw with static degree. coef_last_axis: per-element coefficient
    vectors of shape t.shape + (n,), used by the segmented series."""
    n = coef.shape[-1] if coef_last_axis else coef.shape[0]
    ck = (lambda k: coef[..., k]) if coef_last_axis else (lambda k: coef[k])
    b1 = jnp.zeros(jnp.shape(t), jnp.result_type(t, coef))
    b2 = b1
    two_t = 2 * t
    for k in range(n - 1, 0, -1):
        b1, b2 = two_t * b1 - b2 + ck(k), b1
    return t * b1 - b2 + ck(0)


def _dcoef(coef, domain):
    a, b = domain
    d = jnp.asarray(_der_matrix(coef.shape[0]), dtype=coef.dtype)
    return (2.0 / (b - a)) * (d @ coef)


def _chebval_impl(x, coef, domain):
    a, b = domain
    # promote before mapping: a float32 x against float64 coefficients must
    # compute in float64, like the segmented path (same-dtype is a no-op,
    # so the traced graph of the normal path is unchanged); asarray, not
    # astype, since x may arrive as a scalar literal without methods
    x = jnp.asarray(x, dtype=jnp.result_type(x, coef))
    t = (2 * x - (b + a)) / (b - a)
    return _clenshaw(t, coef)


# The jvp rule calls the plain impl so the tangent computation is made of
# ordinary transposable primitives (reverse mode works), and so the value
# path and the deriv() path are literally the same ops.
_chebval = jax.custom_jvp(_chebval_impl, nondiff_argnums=(2,))


@_chebval.defjvp
def _chebval_jvp(domain, primals, tangents):
    x, coef = primals
    xdot, cdot = tangents
    y = _chebval_impl(x, coef, domain)
    dy = _chebval_impl(x, _dcoef(coef, domain), domain) * xdot + _chebval_impl(x, cdot, domain)
    return y, dy


@jax.tree_util.register_pytree_node_class
class ChebSeries:
    """sum(c[k] T_k(t)), t the affine image of x taking [a, b] to [-1, 1].

    Plain-c0, lowest-first coefficients (the numpy.polynomial convention).
    """

    def __init__(self, coef, domain=(-1.0, 1.0)):
        coef = jnp.asarray(coef)
        # shape/dtype checks only: coef may be a tracer (the *_fn paths
        # build series from traced reconstructions), so no value checks
        if coef.ndim != 1 or coef.shape[0] == 0:
            raise ValueError(f"coef must be a nonempty 1-D array, got shape {coef.shape}")
        if not jnp.issubdtype(coef.dtype, jnp.floating):
            # integer coefficients truncate under integ() and break
            # coefficient AD with float0 tangents
            coef = coef.astype(jnp.empty(()).dtype)
        self.coef = coef
        a, b = float(domain[0]), float(domain[1])
        if not (np.isfinite(a) and np.isfinite(b)) or not a < b:
            raise ValueError(f"domain must be finite with a < b, got ({a}, {b})")
        self.domain = (a, b)

    @property
    def degree(self):
        return self.coef.shape[0] - 1

    def __call__(self, x):
        return _chebval(jnp.asarray(x), self.coef, self.domain)

    def deriv(self):
        return ChebSeries(_dcoef(self.coef, self.domain), self.domain)

    def integ(self):
        """Antiderivative vanishing at the domain midpoint."""
        a, b = self.domain
        m = jnp.asarray(_int_matrix(self.coef.shape[0]), dtype=self.coef.dtype)
        return ChebSeries(((b - a) / 2.0) * (m @ self.coef), self.domain)

    def astype(self, dtype):
        """Round the coefficients; error floor grows to ~eps(dtype)*sum|c|."""
        return ChebSeries(self.coef.astype(dtype), self.domain)

    def truncate(self, tol):
        """Drop the converged tail: keep coefficients through the last one
        above tol * max|c|. Adds up to ~tol relative error - pair with
        astype(float32) at tol ~ 1e-7, where the f64 fit carries roughly
        twice the terms f32 accuracy needs. Concrete coefficients only
        (a traced tail has no static length)."""
        c = np.asarray(self.coef)
        keep = np.nonzero(np.abs(c) > tol * np.abs(c).max())[0]
        n = int(keep.max()) + 1 if keep.size else 1
        return ChebSeries(self.coef[:n], self.domain)

    def __repr__(self):
        return f"ChebSeries(degree={self.degree}, domain={self.domain}, dtype={self.coef.dtype})"

    def tree_flatten(self):
        return (self.coef,), self.domain

    @classmethod
    def tree_unflatten(cls, domain, children):
        obj = object.__new__(cls)
        obj.coef = children[0]
        obj.domain = domain
        return obj


def _pw_dcoef(coef, breaks):
    d = jnp.asarray(_der_matrix(coef.shape[1]), dtype=coef.dtype)
    widths = np.diff(np.asarray(breaks))
    return (coef @ d.T) * jnp.asarray(2.0 / widths, dtype=coef.dtype)[:, None]


def _pw_chebval_impl(x, coef, breaks):
    dtype = jnp.result_type(x, coef)
    br = np.asarray(breaks)
    idx = jnp.clip(jnp.searchsorted(jnp.asarray(br), x, side="right") - 1, 0, len(breaks) - 2)
    a = jnp.asarray(br[:-1], dtype)[idx]
    b = jnp.asarray(br[1:], dtype)[idx]
    t = (2 * x.astype(dtype) - (b + a)) / (b - a)
    return _clenshaw(t, coef[idx], coef_last_axis=True)


_pw_chebval = jax.custom_jvp(_pw_chebval_impl, nondiff_argnums=(2,))


@_pw_chebval.defjvp
def _pw_chebval_jvp(breaks, primals, tangents):
    x, coef = primals
    xdot, cdot = tangents
    y = _pw_chebval_impl(x, coef, breaks)
    dy = (_pw_chebval_impl(x, _pw_dcoef(coef, breaks), breaks) * xdot
          + _pw_chebval_impl(x, cdot, breaks))
    return y, dy


@jax.tree_util.register_pytree_node_class
class PiecewiseCheb:
    """Segmented series: row i of coef is a ChebSeries on [breaks[i], breaks[i+1]],
    all padded to one degree, evaluated by gather + one Clenshaw (no branches).
    x outside [breaks[0], breaks[-1]] uses the nearest end segment's polynomial.
    """

    def __init__(self, coef, breaks):
        coef = jnp.asarray(coef)
        self.breaks = tuple(float(t) for t in breaks)
        if len(self.breaks) < 2:
            raise ValueError("breaks needs at least two knots")
        if coef.ndim != 2 or coef.shape[0] != len(self.breaks) - 1 or coef.shape[1] == 0:
            raise ValueError("coef must have shape (len(breaks) - 1, degree + 1)")
        if not jnp.issubdtype(coef.dtype, jnp.floating):
            coef = coef.astype(jnp.empty(()).dtype)
        self.coef = coef
        if not all(np.isfinite(t) for t in self.breaks):
            raise ValueError("breaks must be finite")
        if not all(u < v for u, v in zip(self.breaks, self.breaks[1:])):
            raise ValueError("breaks must be strictly increasing")

    @property
    def degree(self):
        return self.coef.shape[1] - 1

    @property
    def domain(self):
        return (self.breaks[0], self.breaks[-1])

    def __call__(self, x):
        return _pw_chebval(jnp.asarray(x), self.coef, self.breaks)

    def deriv(self):
        return PiecewiseCheb(_pw_dcoef(self.coef, self.breaks), self.breaks)

    def astype(self, dtype):
        return PiecewiseCheb(self.coef.astype(dtype), self.breaks)

    def truncate(self, tol):
        """Drop trailing coefficient COLUMNS below tol * max|c| across all
        segments (rows share one degree by construction)."""
        c = np.asarray(self.coef)
        col = np.abs(c).max(axis=0)
        keep = np.nonzero(col > tol * col.max())[0]
        n = int(keep.max()) + 1 if keep.size else 1
        return PiecewiseCheb(self.coef[:, :n], self.breaks)

    def __repr__(self):
        return (f"PiecewiseCheb(segments={len(self.breaks) - 1}, degree={self.degree}, "
                f"domain={self.domain}, dtype={self.coef.dtype})")

    def tree_flatten(self):
        return (self.coef,), self.breaks

    @classmethod
    def tree_unflatten(cls, breaks, children):
        obj = object.__new__(cls)
        obj.coef = children[0]
        obj.breaks = breaks
        return obj
