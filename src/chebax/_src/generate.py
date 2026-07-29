"""Build-time fitting: sample at Chebyshev points, DCT to coefficients.

The float64 path uses numpy. Passing dps switches sampling and the DCT to
mpmath at that precision; mpmath is imported inside that branch only (the
runtime-imports rule in CLAUDE.md). Adaptive fitting doubles the node count
until the last two coefficients fall below tol relative to the largest, then
truncates the converged tail.
"""

import numpy as np

from chebax._src import algorithms
from chebax._src.series import ChebSeries, PiecewiseCheb

# ~2*eps: the practical floor of the float64 build path. Use dps= to go below.
_DEFAULT_TOL = 5e-16


def fit(f, domain=(-1.0, 1.0), *, deg=None, tol=None, breaks=None, dps=None, max_deg=256):
    """Chebyshev interpolant of f, as a ChebSeries (or PiecewiseCheb if breaks given).

    f:       callable on the domain. Vectorized over numpy arrays if possible;
             with dps set it must accept mpmath floats instead.
    domain:  (a, b). Ignored when breaks is given.
    deg:     fixed degree; otherwise the degree adapts until tol is met.
    tol:     target for the relative coefficient tail (default 5e-16).
    breaks:  strictly increasing knots, endpoints included; one series per
             segment, each fit independently, padded to a common degree.
    dps:     mpmath working precision for sampling and the DCT. Derivatives of
             a float64-built fit sit at a ~deg^2 * eps floor (differentiation
             amplifies sample noise); build with dps when that matters.
    max_deg: adaptive-path cap; exceeded means f is not smooth enough on the
             interval (split the domain) or genuinely needs a higher degree.
    """
    if tol is None:
        tol = _DEFAULT_TOL
    if breaks is not None:
        br = [float(t) for t in breaks]
        cs = [_fit_interval(f, br[i], br[i + 1], deg, tol, max_deg, dps)
              for i in range(len(br) - 1)]
        n = max(c.size for c in cs)
        coef = np.zeros((len(cs), n))
        for i, c in enumerate(cs):
            coef[i, : c.size] = c
        return PiecewiseCheb(coef, br)
    a, b = float(domain[0]), float(domain[1])
    return ChebSeries(_fit_interval(f, a, b, deg, tol, max_deg, dps), (a, b))


def _fit_interval(f, a, b, deg, tol, max_deg, dps):
    dct = _dct_mp(f, a, b, dps) if dps is not None else _dct_np(f, a, b)
    if deg is not None:
        return dct(deg + 1)
    return _adaptive(dct, tol, max_deg, (a, b))


def _adaptive(dct, tol, max_deg, interval):
    n = 17
    while True:
        c = dct(n)
        scale = np.max(np.abs(c))
        if scale == 0.0:
            return np.zeros(1)
        if np.max(np.abs(c[-2:])) <= tol * scale:
            keep = np.nonzero(np.abs(c) > tol * scale)[0]
            return c[: keep[-1] + 1] if keep.size else c[:1]
        if n > max_deg:
            raise ValueError(
                f"no convergence to tol={tol:g} by degree {max_deg} on {interval}; "
                "f may not be smooth there (split the domain) or needs a higher max_deg")
        n = 2 * n - 1


def _dct_np(f, a, b):
    def dct(n):
        xk = a + (b - a) * (algorithms.chebpts(n) + 1) / 2
        return algorithms.chebfit_dct(_sample(f, xk))
    return dct


def _sample(f, xk):
    # scalar-only callables (wrapped special functions) are common; fall back.
    try:
        y = np.asarray(f(xk), dtype=float)
        if y.shape == xk.shape:
            return y
    except (TypeError, ValueError):
        pass
    return np.array([float(f(x)) for x in xk])


def _dct_mp(f, a, b, dps):
    import mpmath as mp

    def dct(n):
        with mp.workdps(dps):
            mid = (mp.mpf(b) + mp.mpf(a)) / 2
            half = (mp.mpf(b) - mp.mpf(a)) / 2
            fk = [f(mid + half * mp.cos(mp.pi * (2 * i + 1) / (2 * n))) for i in range(n)]
            c = []
            for j in range(n):
                s = mp.fsum(fk[i] * mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n))
                c.append(2 * s / n)
            c[0] /= 2
            return np.array([float(x) for x in c])
    return dct
