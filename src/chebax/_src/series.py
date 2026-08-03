"""Runtime evaluation in jax: single and segmented Chebyshev series.

Only two algorithms run per call: Clenshaw, and (for gradients) one more
Clenshaw on the derivative coefficients. Differentiation and integration act
on coefficients through the closed forms of the numpy references in
algorithms.py, written in jax so they work on traced values, in O(n) time
and with no cached state. For concrete coefficients jit constant-folds them,
so the per-call gradient cost is one extra Clenshaw; deriv() itself is O(n)
and is recomputed per call (caching a possibly-traced result on the instance
would leak tracers).

Series are pytrees: coefficients are leaves, domains and breakpoints are
static. Full float64 accuracy needs jax's x64 mode. Evaluation promotes x
and the coefficients to a common dtype first (a float32 x against float64
tables computes in float64, matching the segmented path).
"""

import jax
import jax.numpy as jnp
import numpy as np


def _der_coefs(coef):
    """chebder along the last axis, for traced coefficients.

    Closed form of algorithms.chebder's downward recurrence: d_k is twice
    the sum of j*c_j over j > k of the opposite parity, with d_0 halved.
    Reshaping to (m, 2) puts the two parities in separate columns, so one
    reverse cumsum down the rows does both stride-2 suffix sums. O(n).
    n = 1 is the zero map."""
    n = coef.shape[-1]
    if n == 1:
        return jnp.zeros_like(coef)
    # the index vectors are static, so build them in numpy (2j and the
    # halving factor are exact in either float dtype)
    a = coef * np.arange(0, 2 * n, 2, dtype=np.float64).astype(coef.dtype)
    if n % 2:
        a = jnp.concatenate([a, jnp.zeros(a.shape[:-1] + (1,), a.dtype)], axis=-1)
    pairs = a.reshape(a.shape[:-1] + (-1, 2))
    s = jax.lax.cumsum(pairs, axis=pairs.ndim - 2, reverse=True).reshape(a.shape)
    half = np.where(np.arange(n - 1) == 0, 0.5, 1.0).astype(coef.dtype)
    return s[..., 1:n] * half


def _int_coefs(coef):
    """chebint along the last axis, for traced coefficients.

    algorithms.chebint has no recurrence to unroll: out_k = (c_{k-1} -
    c_{k+1}) / 2k for k >= 2, out_1 = c_0 - c_2/2 (plain-c0 convention),
    and out_0 last so the antiderivative vanishes at t = 0."""
    n = coef.shape[-1]
    z = jnp.zeros(coef.shape[:-1] + (1,), coef.dtype)
    hi = jnp.concatenate([coef[..., 2:], z, z], axis=-1)[..., :n]   # c_{k+1}, k = 1..n
    k = np.arange(1, n + 1)
    # divided term by term, like the reference, so the rounding matches
    lo_div = np.where(k == 1, 1.0, 2.0 * k).astype(coef.dtype)
    hi_div = (2.0 * k).astype(coef.dtype)
    out = coef / lo_div - hi / hi_div
    tk = np.where(k % 2 == 1, 0.0, np.where(k % 4 == 0, 1.0, -1.0)).astype(coef.dtype)  # T_k(0)
    return jnp.concatenate([-jnp.sum(out * tk, axis=-1, keepdims=True), out], axis=-1)


def _as_float_coef(coef):
    """Coefficients as a float jax array, with no silent conversion loss.

    jnp.asarray narrows int64 to int32 first, so a big integer wraps under
    x64 off, and casting complex to float drops the imaginary part. Value
    checks run on concrete input only: coef may be a tracer (the *_fn paths
    build series from traced reconstructions)."""
    traced = isinstance(coef, jax.core.Tracer)
    if not traced and not isinstance(coef, jax.Array):
        coef = np.asarray(coef)  # numpy keeps the full integer width
    if jnp.issubdtype(coef.dtype, jnp.complexfloating):
        raise ValueError(f"complex coefficients are not supported, got dtype {coef.dtype}")
    if jnp.issubdtype(coef.dtype, jnp.floating):
        return jnp.asarray(coef)
    # integer coefficients truncate under integ() and break coefficient AD
    # with float0 tangents, so promote
    canon = jnp.empty(()).dtype
    if traced:
        return coef.astype(canon)
    src = np.asarray(coef)
    out = src.astype(canon)
    if not np.array_equal(out.astype(src.dtype), src):
        raise ValueError(f"integer coefficients are not exactly representable in {canon}; "
                         "convert them yourself if the rounding is acceptable")
    return jnp.asarray(out)


def _check_tol(tol):
    """A truncation tolerance must be a finite, nonnegative real.

    A negative one was a silent no-op and nan or inf silently collapsed a
    nonconstant series to degree zero, both of which bake forwards
    straight into an artifact (review, 2026-08-02)."""
    t = float(tol)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError(f"tol must be finite and nonnegative, got {tol!r}")
    return t


def _check_float_dtype(dtype):
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"astype needs a floating dtype, got {np.dtype(dtype)}")


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
    # 1/half-width, formed the same way as the map: 2/(b - a) is 0 on an
    # extreme finite domain, which silently zeroed every derivative
    return (1.0 / (0.5 * b - 0.5 * a)) * _der_coefs(coef)


def _chebval_impl(x, coef, domain):
    a, b = domain
    # promote before mapping: a float32 x against float64 coefficients must
    # compute in float64, like the segmented path (same-dtype is a no-op,
    # so the traced graph of the normal path is unchanged); asarray, not
    # astype, since x may arrive as a scalar literal without methods
    x = jnp.asarray(x, dtype=jnp.result_type(x, coef))
    # midpoint/half-width, not (2x - (b+a))/(b-a): both b + a and b - a
    # overflow on a legal domain like (-1e308, 1e308), where this form
    # gives t = 0 and +-1 at the endpoints as it should (review,
    # 2026-08-02, which measured nan at both ends)
    mid = 0.5 * a + 0.5 * b
    half = 0.5 * b - 0.5 * a
    t = (x - mid) / half
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
    Frozen after construction: the recipe factories cache instances and hand
    the same series to every caller, so mutation would corrupt the cache.
    """

    _frozen = False

    def __init__(self, coef, domain=(-1.0, 1.0)):
        coef = _as_float_coef(coef)
        if coef.ndim != 1 or coef.shape[0] == 0:
            raise ValueError(f"coef must be a nonempty 1-D array, got shape {coef.shape}")
        self.coef = coef
        a, b = float(domain[0]), float(domain[1])
        if not (np.isfinite(a) and np.isfinite(b)) or not a < b:
            raise ValueError(f"domain must be finite with a < b, got ({a}, {b})")
        self.domain = (a, b)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"ChebSeries is immutable (cached instances are shared); cannot set {name!r}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # deleting a coefficient array from a CACHED instance corrupts
        # every later caller sharing the slot; assignment was guarded and
        # deletion was not (review, 2026-08-02)
        if self._frozen:
            raise AttributeError(
                f"ChebSeries is immutable (cached instances are shared); "
                f"cannot delete {name!r}")
        object.__delattr__(self, name)

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
        return ChebSeries(((b - a) / 2.0) * _int_coefs(self.coef), self.domain)

    def astype(self, dtype):
        """Round the coefficients; error floor grows to ~eps(dtype)*sum|c|."""
        _check_float_dtype(dtype)
        return ChebSeries(self.coef.astype(dtype), self.domain)

    def truncate(self, tol):
        """Drop the converged tail: keep coefficients through the last one
        above tol * max|c|. Adds up to ~tol relative error - pair with
        astype(float32) at tol ~ 1e-7, where the f64 fit carries roughly
        twice the terms f32 accuracy needs. Concrete coefficients only
        (a traced tail has no static length)."""
        tol = _check_tol(tol)
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
        object.__setattr__(obj, "_frozen", True)
        return obj


def _pw_dcoef(coef, breaks):
    br = np.asarray(breaks)
    half = 0.5 * br[1:] - 0.5 * br[:-1]     # see _dcoef on the overflow
    return _der_coefs(coef) * jnp.asarray(1.0 / half, dtype=coef.dtype)[:, None]


def _pw_chebval_impl(x, coef, breaks):
    dtype = jnp.result_type(x, coef)
    br = np.asarray(breaks)
    idx = jnp.clip(jnp.searchsorted(jnp.asarray(br), x, side="right") - 1, 0, len(breaks) - 2)
    a = jnp.asarray(br[:-1], dtype)[idx]
    b = jnp.asarray(br[1:], dtype)[idx]
    mid = 0.5 * a + 0.5 * b                 # see _chebval_impl on the overflow
    t = (x.astype(dtype) - mid) / (0.5 * b - 0.5 * a)
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
    Frozen after construction, like ChebSeries.
    """

    _frozen = False

    def __init__(self, coef, breaks):
        coef = _as_float_coef(coef)
        self.breaks = tuple(float(t) for t in breaks)
        if len(self.breaks) < 2:
            raise ValueError("breaks needs at least two knots")
        if coef.ndim != 2 or coef.shape[0] != len(self.breaks) - 1 or coef.shape[1] == 0:
            raise ValueError("coef must have shape (len(breaks) - 1, degree + 1)")
        self.coef = coef
        if not all(np.isfinite(t) for t in self.breaks):
            raise ValueError("breaks must be finite")
        if not all(u < v for u, v in zip(self.breaks, self.breaks[1:])):
            raise ValueError("breaks must be strictly increasing")
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"PiecewiseCheb is immutable (cached instances are shared); "
                f"cannot set {name!r}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if self._frozen:
            raise AttributeError(
                f"PiecewiseCheb is immutable (cached instances are shared); "
                f"cannot delete {name!r}")
        object.__delattr__(self, name)

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
        _check_float_dtype(dtype)
        return PiecewiseCheb(self.coef.astype(dtype), self.breaks)

    def truncate(self, tol):
        """Drop trailing coefficient COLUMNS below tol * max|c| across all
        segments (rows share one degree by construction)."""
        tol = _check_tol(tol)
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
        object.__setattr__(obj, "_frozen", True)
        return obj
