"""Regularized incomplete gamma P(a, x) = gamma(a, x)/Gamma(a) and Q = 1 - P,
a >= 0.1, x in [0, inf): every public entry point, as name, name_scalar,
name_array and name_frozen (the logs have no frozen form). The scalar and frozen
paths share gammainc_small.py (a <= 10) and gammainc_large.py (a > 10); the array
path has its own table. Each path imports its tables on first call.
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
    """P(a, x) for any a >= 0.1: gammainc_scalar when a holds one value, else
    gammainc_array. x=None is the deprecated 0.4 factory."""
    if x is None:
        return deprecated_factory("gammainc", gammainc_frozen, a)
    return gammainc_array(a, x) if per_element(a) else gammainc_scalar(a, x)


def gammaincc(a, x=None):
    """Q(a, x), dispatched like gammainc. Relatively accurate where Q is small,
    except gammaincc_scalar below x = 8 (see there)."""
    if x is None:
        return deprecated_factory("gammaincc", gammaincc_frozen, a)
    return gammaincc_array(a, x) if per_element(a) else gammaincc_scalar(a, x)


def log_gammainc(a, x):
    """ln P(a, x), dispatched like gammainc, no underflow floor."""
    return log_gammainc_array(a, x) if per_element(a) else log_gammainc_scalar(a, x)


def log_gammaincc(a, x):
    """ln Q(a, x), dispatched like gammainc, no underflow in the upper tail."""
    if per_element(a):
        return log_gammaincc_array(a, x)
    return log_gammaincc_scalar(a, x)


# one a per call


def gammainc_scalar(a, x):
    """P(a, x) for one traceable a >= 0.1 per call (unchecked under trace; above
    10 the Temme tables reach a = inf), coefficients built once per call.
    Differentiable in a and x; at x = 0 the slope is the density (inf below
    a = 1). x <= 0 gives 0, x = inf gives 1, nan propagates."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.p(_scalar_a(a, "gammainc_scalar"), jnp.asarray(x))


def gammaincc_scalar(a, x):
    """Q(a, x) as gammainc_scalar. For a <= 10 the tables hold Q for x > 8;
    below, Q = 1 - P is only absolutely accurate, ~eps/Q relative (worst near
    x = 8 at small a, where Q ~ 1e-6). gammaincc_array is relative there."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.q(_scalar_a(a, "gammaincc_scalar"), jnp.asarray(x))


def log_gammainc_scalar(a, x):
    """ln P(a, x) as gammainc_scalar, no underflow floor. For a <= 10 the tables
    hold ln P for x <= 8; above, log1p(-exp(ln Q)) errs ~eps Q/P with Q <= 0.3.
    a > 10 works in log space through erfcx. x <= 0 gives -inf."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.log_p(_scalar_a(a, "log_gammainc_scalar"), jnp.asarray(x))


def log_gammaincc_scalar(a, x):
    """ln Q(a, x) as gammaincc_scalar, no underflow in the tail. For a <= 10 the
    tables hold ln Q for x > 8; below, log1p(-exp(ln P)) errs ~eps/Q, worst near
    x = 8 at small a where Q ~ 1e-6. x = inf gives -inf."""
    from chebax._src.recipes import gammainc_scalar as _scalar

    return _scalar.log_q(_scalar_a(a, "log_gammaincc_scalar"), jnp.asarray(x))


# a per element


def gammainc_array(a, x):
    """P(a, x) with a per element, broadcast with x. a >= 0.1 (else nan),
    relatively accurate where P is small, differentiable in a and x. Endpoints as
    in gammainc_scalar."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.p(*_broadcast(a, x))


def gammaincc_array(a, x):
    """Q(a, x) as gammainc_array, relatively accurate where Q is small."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.q(*_broadcast(a, x))


def log_gammainc_array(a, x):
    """ln P(a, x) as gammainc_array, no underflow floor."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.log_p(*_broadcast(a, x))


def log_gammaincc_array(a, x):
    """ln Q(a, x) as gammaincc_array, no underflow (ln Q ~ -x as x -> inf)."""
    from chebax._src.recipes import gammainc_array as _array

    return _array.log_q(*_broadcast(a, x))


# one concrete a, cached


def gammainc_frozen(a):
    """P(a, .) for one concrete a >= 0.1: a cached Recipe callable of x,
    jittable, differentiable in x and bakeable."""
    from chebax._src.recipes import gammainc_frozen as _frozen

    return _frozen.p_at(float(a), canon_tag())


def gammaincc_frozen(a):
    """Q(a, .) for one concrete a >= 0.1, cached like gammainc_frozen."""
    from chebax._src.recipes import gammainc_frozen as _frozen

    return _frozen.q_at(float(a), canon_tag())
