"""The regularized incomplete gamma functions, P(a, x) = gamma(a, x)/Gamma(a)
and Q = 1 - P, for a >= 0.1 and x in [0, inf): every public entry point.

Each function has one name that takes any a, and one name per path for
callers who know which path they want:

    gammainc(a, x)          a holding one value runs gammainc_scalar, an
                            array that broadcasts with x runs
                            gammainc_array (decided from the shape)
    gammainc_scalar(a, x)   one a for the whole call, traceable
                                                    (gammainc_scalar.py)
    gammainc_array(a, x)    a per element, broadcast with x
                                                    (gammainc_array.py)
    gammainc_frozen(a)      a cached callable of x for one concrete a,
                            what bake takes         (gammainc_frozen.py)

and the same for gammaincc (Q) and for the logs log_gammainc and
log_gammaincc, which have no frozen form. The scalar and frozen paths
share the small-a tables (gammainc_small.py, a <= 10) and the Temme-zone
tables (gammainc_large.py, a > 10); the array path has its own patch
table. Each entry point imports its implementation when first called, so
a program that uses one path loads only that path's tables.

The one-argument gammainc(a) and gammaincc(a) are the 0.4 factories and
still return the frozen callable, with a DeprecationWarning.
"""

import jax.numpy as jnp

from chebax._src.recipes._common import (
    canon_float,
    canon_param,
    canon_tag,
    deprecated_factory,
    per_element,
)


def _broadcast(a, x):
    return jnp.broadcast_arrays(canon_float(a), canon_float(x))


def _scalar_a(a, who):
    return canon_param(a, who, "a", array=who.replace("_scalar", "_array"))


# the functions: one a or an array of a


def gammainc(a, x=None):
    """P(a, x) for any a >= 0.1: gammainc_scalar when a holds one value,
    gammainc_array when a is an array that broadcasts with x. The choice is
    made from the shape of a, so it is fixed at trace time and costs
    nothing at run time. See those two for accuracy, endpoints and
    derivatives (in a and x, both AD modes)."""
    if x is None:
        return deprecated_factory("gammainc", gammainc_frozen, a)
    return gammainc_array(a, x) if per_element(a) else gammainc_scalar(a, x)


def gammaincc(a, x=None):
    """Q(a, x) = 1 - P(a, x) for any a >= 0.1, dispatched like gammainc;
    computed directly, so relatively accurate in the upper tail."""
    if x is None:
        return deprecated_factory("gammaincc", gammaincc_frozen, a)
    return gammaincc_array(a, x) if per_element(a) else gammaincc_scalar(a, x)


def log_gammainc(a, x):
    """ln P(a, x) for any a >= 0.1, dispatched like gammainc, with no
    underflow floor in the lower tail."""
    return log_gammainc_array(a, x) if per_element(a) else log_gammainc_scalar(a, x)


def log_gammaincc(a, x):
    """ln Q(a, x) for any a >= 0.1, dispatched like gammainc, with no
    underflow ceiling in the upper tail: the log survival function a
    censored Gamma likelihood wants."""
    if per_element(a):
        return log_gammaincc_array(a, x)
    return log_gammaincc_scalar(a, x)


# one a per call


def gammainc_scalar(a, x):
    """P(a, x) with a one (traceable) value for the whole call, x any array.

    a must hold one value and be at least 0.1 (unchecked under trace;
    above 10 it runs on the Temme-zone tables, whose v = 10/a axis reaches
    a = inf). Coefficients are reconstructed once per call, not per point;
    jit constant-folds them when a is static. Differentiable in both
    arguments: d/da through the tables' parameter axis, d/dx through the
    formula, whose exact value is the Gamma density. At x = 0 that density
    (inf below a = 1, 1 at it, 0 above) is spliced in by hand; x < 0 and
    x = +inf are flat. x <= 0 gives 0, x = inf gives 1, nan propagates."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.p(_scalar_a(a, "gammainc_scalar"), jnp.asarray(x))


def gammaincc_scalar(a, x):
    """Q(a, x) = 1 - P(a, x), same contract as gammainc_scalar; relatively
    accurate in the upper tail (computed directly, not as 1 - P)."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.q(_scalar_a(a, "gammaincc_scalar"), jnp.asarray(x))


def log_gammainc_scalar(a, x):
    """ln P(a, x), same contract as gammainc_scalar but with no underflow
    floor in the lower tail.

    For x <= 8 (a <= 10) the tables already hold the log; ln P evaluates
    directly (ln P ~ a ln x as x -> 0, arbitrarily negative). For x > 8 it
    is log1p(-exp(ln Q)): absolutely accurate in P, error in ln P is
    ~eps Q/P, small everywhere there since Q <= 0.3 in the box. x <= 0
    returns -inf, x = inf returns 0. For a > 10 the large path assembles
    in log space through erfcx, valid past the linear form's underflow."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.log_p(_scalar_a(a, "log_gammainc_scalar"), jnp.asarray(x))


def log_gammaincc_scalar(a, x):
    """ln Q(a, x), same contract as gammaincc_scalar but with no underflow
    ceiling in the tail.

    For x > 8 (a <= 10) the tables already hold the log; ln Q evaluates
    directly (ln Q ~ -x, valid for arbitrarily large x where Q itself
    underflows past x ~ 700). For x <= 8 it is log1p(-exp(ln P)):
    absolutely accurate in Q, so the error in ln Q is ~eps/Q in the wedge
    x -> 8 at small a where Q reaches ~1e-6. x <= 0 returns 0, x = inf
    returns -inf."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.log_q(_scalar_a(a, "log_gammaincc_scalar"), jnp.asarray(x))


# a per element


def gammainc_array(a, x):
    """P(a, x) with a per element: a and x broadcast together.

    a >= 0.1 (nan below, and for a = inf or nan), x anywhere: x <= 0 gives
    0, x = inf gives 1. Absolutely accurate, and relatively accurate where
    P is small. Differentiable in a and x in both AD modes; dP/dx at x = 0
    is the density (inf below a = 1). Active jax float dtype."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.p(*_broadcast(a, x))


def gammaincc_array(a, x):
    """Q(a, x) = 1 - P(a, x) with a per element, same contract as
    gammainc_array; relatively accurate where Q is small, the upper tail
    included (computed directly, not as 1 - P)."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.q(*_broadcast(a, x))


def log_gammainc_array(a, x):
    """ln P(a, x) with a per element, same contract as gammainc_array and
    no underflow floor: ln P ~ a ln x - x - lnGamma(a+1) as x -> 0 and
    -a phi as x/a -> 0 for large a. x <= 0 gives -inf, x = inf gives 0."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.log_p(*_broadcast(a, x))


def log_gammaincc_array(a, x):
    """ln Q(a, x) with a per element, same contract as gammaincc_array and
    no underflow ceiling (ln Q ~ -x as x -> inf): the log survival
    function of a Gamma likelihood with a shape per observation. x <= 0
    gives 0, x = inf gives -inf."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.log_q(*_broadcast(a, x))


# one concrete a, cached


def gammainc_frozen(a):
    """P(a, .) on [0, inf) for one concrete a >= 0.1: a cached Recipe
    callable of x, jittable, differentiable in x and bakeable. No mpmath.

    a may be a python number or a concrete jax scalar; the bounded cache
    is keyed per x64 mode. a > 10 runs on the Temme-zone tables."""
    from chebax._src.recipes import gammainc_frozen as _frozen

    return _frozen.p_at(float(a), canon_tag())


def gammaincc_frozen(a):
    """Q(a, .) = 1 - P(a, .) for one concrete a >= 0.1, cached like
    gammainc_frozen."""
    from chebax._src.recipes import gammainc_frozen as _frozen

    return _frozen.q_at(float(a), canon_tag())
