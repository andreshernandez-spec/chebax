"""Gamma quantiles: gammaincinv (the x with P(a, x) = p), gammainccinv (the
x with Q(a, x) = y) and chi2inv; every public entry point.

Each function has one name that takes any a, and one name per path for
callers who know which path they want:

    gammaincinv(a, p)          a holding one value runs gammaincinv_scalar,
                               an array that broadcasts with p runs
                               gammaincinv_array (decided from the shape)
    gammaincinv_scalar(a, p)   one a for the whole call, traceable, any
                               a > 0                (gammaincinv_scalar.py)
    gammaincinv_array(a, p)    a per element, broadcast with p, a >= 0.1
                                                    (gammaincinv_array.py)

and the same for gammainccinv and for chi2inv (k = 2a). Both paths run
the same solve (gammaincinv_kernel.py) on their own gammainc: the
scalar path on gammainc_scalar's tables, the array path on
gammainc_array's patches. Each entry point imports its implementation
when first called, so a program that uses one path loads only that
path's tables.
"""

import jax.numpy as jnp

from chebax._src.recipes._common import canon_float, canon_param, per_element


def _broadcast(a, p):
    return jnp.broadcast_arrays(canon_float(a), canon_float(p))


# the functions: one a or an array of a


def gammaincinv(a, p):
    """The Gamma(a, 1) quantile, x with P(a, x) = p, for any a:
    gammaincinv_scalar when a holds one value, gammaincinv_array when a is
    an array that broadcasts with p. The choice is made from the shape of
    a, so it is fixed at trace time and costs nothing at run time. See
    those two for the domain, endpoints and derivatives."""
    return gammaincinv_array(a, p) if per_element(a) else gammaincinv_scalar(a, p)


def gammainccinv(a, y):
    """The Gamma(a, 1) survival quantile, x with Q(a, x) = y, for any a,
    dispatched like gammaincinv. Resolves y below eps, where
    gammaincinv(a, 1 - y) cannot (1 - y rounds to 1)."""
    if per_element(a):
        return gammainccinv_array(a, y)
    return gammainccinv_scalar(a, y)


def chi2inv(k, p):
    """The chi-squared quantile at real degrees of freedom k, for any k,
    dispatched like gammaincinv: 2 gammaincinv(k/2, p)."""
    return chi2inv_array(k, p) if per_element(k) else chi2inv_scalar(k, p)


# one a per call


def gammaincinv_scalar(a, p):
    """x with P(a, x) = p, with a one (traceable) value for the whole call
    and p any array.

    a must be finite and positive: anything else (0, negative, inf, nan)
    gives nan for every p, endpoints included. From a = 0.1 up the
    residual runs on the chebax gammainc tables (fixed-degree
    polynomials; above 10 via the Temme-zone path, and above 1e10 the
    solve itself runs in Temme's variable, which keeps every a up to the
    largest float); below it falls back to jax's gammainc, so the domain
    stays all of a > 0. The upper half is solved against Q, so p within
    rounding of 1 still resolves (P(a, x) - p there has no digits left).
    p = 0 gives 0 and p = 1 gives inf. Quantiles below the smallest normal
    float of the canonical dtype return 0.0; a solve that is neither
    within the residual bar nor one float step from the root returns nan
    instead of a wrong value.

    Differentiable in a and p by the implicit function theorem, not
    through the iteration: dx/dp = 1/pdf(x), dx/da = -(dP/da)(x)/pdf(x).
    Second derivatives in p and mixed p-a derivatives work; a pure second
    derivative in a needs d2P/da2, which jax does not provide, and raises
    (gammaincinv_array has it). Above a = 1e10 first derivatives come from
    the implicit rule in Temme's variable and higher ones differentiate
    its fixed iteration."""
    from chebax._src.recipes import gammaincinv_scalar as _scalar

    return _scalar.gammaincinv(a, p)


def gammainccinv_scalar(a, y):
    """x with Q(a, x) = y, same contract as gammaincinv_scalar with the
    tails swapped: the target enters as y, exact for the upper tail this
    exists for, and the solve runs against ln Q there. gammaincinv(a, 1 - y)
    cannot serve: 1 - y is exactly 1 for every y below eps, so the whole
    deep upper tail collapses onto one point. y = 0 gives inf and y = 1
    gives 0; quantiles past the largest float return inf. Derivatives from
    the same rule, with dQ/dx = -pdf and dQ/da = -dP/da."""
    from chebax._src.recipes import gammaincinv_scalar as _scalar

    return _scalar.gammainccinv(a, y)


def chi2inv_scalar(k, p):
    """The chi-squared quantile, 2 gammaincinv_scalar(k/2, p), with k one
    (traceable) value for the whole call and p any array.

    chi2(k) is Gamma(k/2, rate 1/2), so the quantile is a rescaled Gamma
    quantile: differentiable in k and p, real non-integer dof included. k
    must be finite and positive, inheriting gammaincinv_scalar's check:
    k <= 0 (and inf, nan) gives nan for every p. The recipe residual
    serves k from 0.2 up (a = k/2 >= 0.1); below that the jax fallback
    takes over. p = 0 gives 0 and p = 1 gives inf."""
    k = canon_param(k, "chi2inv_scalar", "k", "chi2inv_array")
    return 2.0 * gammaincinv_scalar(0.5 * k, p)


# a per element


def gammaincinv_array(a, p):
    """x with P(a, x) = p, with a per element: a and p broadcast together.

    a >= 0.1 (nan below, and for a = inf or nan), p in [0, 1]: p = 0 gives
    0, p = 1 gives inf, and nan p or p outside [0, 1] gives nan. Quantiles
    below the smallest normal float return 0.0, and a solve that is
    neither within the residual bar nor one float step from the root
    returns nan. The upper half is solved against Q, so p within rounding
    of 1 still resolves; any finite a keeps its resolution (the solve
    works in ln(x/a)).

    Differentiable in a and p, in both AD modes and to any order (second
    derivatives in a included): the implicit function theorem on the same
    per-element ln P and ln Q the solve inverted (gammainc_array's).
    Active jax float dtype."""
    from chebax._src.recipes import gammaincinv_array as _array

    return _array.inv(*_broadcast(a, p))


def gammainccinv_array(a, y):
    """x with Q(a, x) = y, with a per element, same contract as
    gammaincinv_array with the tails swapped: y = 0 gives inf, y = 1 gives
    0, quantiles past the largest float return inf, and y below eps
    resolves."""
    from chebax._src.recipes import gammaincinv_array as _array

    return _array.cinv(*_broadcast(a, y))


def chi2inv_array(k, p):
    """The chi-squared quantile, 2 gammaincinv_array(k/2, p), with k per
    element: k >= 0.2 (nan below), broadcast with p."""
    return 2.0 * gammaincinv_array(0.5 * canon_float(k), p)
