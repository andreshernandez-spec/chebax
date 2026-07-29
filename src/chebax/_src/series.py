"""Runtime evaluation in jax: single and segmented Chebyshev series.

Only two algorithms run per call: Clenshaw, and (for gradients) one more
Clenshaw on the derivative coefficients. Differentiation and integration act
on coefficients through matrices built from the numpy references in
algorithms.py, so they work on traced values and agree with the references
exactly. Gradients come from a custom_jvp bound to the derivative series:
jax.grad(p)(x) and p.deriv()(x) are the same computation.

Series are pytrees: coefficients are leaves, domains and breakpoints are
static. Full float64 accuracy needs jax's x64 mode.
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
        self.coef = jnp.asarray(coef)
        self.domain = (float(domain[0]), float(domain[1]))

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
        self.coef = jnp.asarray(coef)
        self.breaks = tuple(float(t) for t in breaks)
        if self.coef.ndim != 2 or self.coef.shape[0] != len(self.breaks) - 1:
            raise ValueError("coef must have shape (len(breaks) - 1, degree + 1)")
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
