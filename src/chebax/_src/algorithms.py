"""The textbook algorithms, in plain numpy, on [-1, 1].

Coefficient convention is numpy.polynomial's: plain c[0], lowest order first,
p(t) = sum(c[k] * T_k(t)). These are the reference implementations; the jax
runtime in series.py must agree with them (tested against numpy.polynomial).
Domain handling lives in the series classes, not here.
"""

import numpy as np


def chebpts(n):
    """First-kind Chebyshev points cos(pi*(2i+1)/(2n)), decreasing from ~1 to ~-1."""
    i = np.arange(n)
    return np.cos(np.pi * (2 * i + 1) / (2 * n))


def chebfit_dct(samples):
    """Interpolation coefficients from samples taken at chebpts(len(samples)).

    Discrete orthogonality of cosines on the first-kind points; O(n^2), which
    is fine at generator degrees.
    """
    n = len(samples)
    i = np.arange(n)
    j = np.arange(n)[:, None]
    # reduce the angle with exact integer arithmetic; the raw product reaches
    # hundreds of radians at large n and costs ~angle*eps of cosine accuracy
    weights = np.cos(np.pi * ((j * (2 * i + 1)) % (4 * n)) / (2 * n))
    c = (2.0 / n) * (weights @ np.asarray(samples, dtype=float))
    c[0] /= 2
    return c


def chebval(t, c):
    """Clenshaw evaluation of a plain Chebyshev series."""
    t = np.asarray(t, dtype=float)
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for ck in np.asarray(c, dtype=float)[:0:-1]:
        b1, b2 = 2 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + c[0]


def chebder(c):
    """Coefficients of the derivative (numpy.polynomial's exact recurrence)."""
    c = np.array(c, dtype=float)
    n = c.size - 1
    if n == 0:
        return np.zeros(1)
    der = np.zeros(n)
    for j in range(n, 2, -1):
        der[j - 1] = 2 * j * c[j]
        c[j - 2] += j * c[j] / (j - 2)
    if n > 1:
        der[1] = 4 * c[2]
    der[0] = c[1]
    return der


def chebint(c):
    """Coefficients of the antiderivative that vanishes at t=0
    (numpy.polynomial's k=0, lbnd=0 convention)."""
    c = np.asarray(c, dtype=float)
    n = c.size
    out = np.zeros(n + 1)
    out[1] = c[0]
    if n > 1:
        out[2] = c[1] / 4
    for j in range(2, n):
        out[j + 1] = c[j] / (2 * (j + 1))
        out[j - 1] -= c[j] / (2 * (j - 1))
    out[0] -= chebval(0.0, out)
    return out
