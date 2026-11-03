"""Gamma quantiles gammaincinv (P(a, x) = p), gammainccinv (Q(a, x) = y) and
chi2inv, each as name, name_scalar (one a, any a > 0) and name_array (a per
element, a >= 0.1). Both paths run gammaincinv_kernel.py's solve on their own
gammainc, and each imports its implementation on first call.
"""

import jax.numpy as jnp

from chebax._src.recipes._common import canon_float, canon_param, per_element


def _broadcast(a, p):
    return jnp.broadcast_arrays(canon_float(a), canon_float(p))


# the functions: one a or an array of a


def gammaincinv(a, p):
    """x with P(a, x) = p: the _scalar or _array path, chosen from a's shape at
    trace time."""
    return gammaincinv_array(a, p) if per_element(a) else gammaincinv_scalar(a, p)


def gammainccinv(a, y):
    """x with Q(a, x) = y, dispatched like gammaincinv; resolves y below eps."""
    if per_element(a):
        return gammainccinv_array(a, y)
    return gammainccinv_scalar(a, y)


def chi2inv(k, p):
    """2 gammaincinv(k/2, p), the chi-squared quantile, dispatched likewise."""
    return chi2inv_array(k, p) if per_element(k) else chi2inv_scalar(k, p)


# one a per call


def gammaincinv_scalar(a, p):
    """x with P(a, x) = p for one traceable a > 0 (nan otherwise) and any p.

    a >= 0.1 uses chebax's gammainc tables (above 1e10 the solve runs in
    Temme's variable), below that jax's gammainc. The upper half solves
    against Q, so p near 1 resolves. p = 0 gives 0, p = 1 inf, quantiles below
    the smallest normal 0.0, and a solve neither within the residual bar nor a
    float step from the root nan.

    Derivatives by the implicit function theorem: dx/dp = 1/pdf(x),
    dx/da = -(dP/da)(x)/pdf(x). A second derivative in a alone raises (jax has
    no d2P/da2; gammaincinv_array has it). Above a = 1e10 the first derivatives
    use Temme's variable and higher ones differentiate its fixed iteration."""
    from chebax._src.recipes import gammaincinv_scalar as _scalar

    return _scalar.gammaincinv(a, p)


def gammainccinv_scalar(a, y):
    """x with Q(a, x) = y, gammaincinv_scalar with the tails swapped, so y below
    eps resolves. y = 0 gives inf, y = 1 gives 0, quantiles past the largest
    float inf."""
    from chebax._src.recipes import gammaincinv_scalar as _scalar

    return _scalar.gammainccinv(a, y)


def chi2inv_scalar(k, p):
    """2 gammaincinv_scalar(k/2, p) for one traceable k > 0 (nan otherwise), real k
    included: the tables from k = 0.2, jax's gammainc below."""
    k = canon_param(k, "chi2inv_scalar", "k", "chi2inv_array")
    return 2.0 * gammaincinv_scalar(0.5 * k, p)


# a per element


def gammaincinv_array(a, p):
    """x with P(a, x) = p for a >= 0.1 per element (nan below), p in [0, 1].
    Endpoints and failed solves as in gammaincinv_scalar; above a = 1e36 (1e18
    in float32) the quantile is a. Derivatives in a and p, both AD modes, to
    second order, by the implicit function theorem on gammainc_array."""
    from chebax._src.recipes import gammaincinv_array as _array

    return _array.inv(*_broadcast(a, p))


def gammainccinv_array(a, y):
    """x with Q(a, x) = y, gammaincinv_array with the tails swapped."""
    from chebax._src.recipes import gammaincinv_array as _array

    return _array.cinv(*_broadcast(a, y))


def chi2inv_array(k, p):
    """2 gammaincinv_array(k/2, p), k >= 0.2 per element (nan below)."""
    return 2.0 * gammaincinv_array(0.5 * canon_float(k), p)
