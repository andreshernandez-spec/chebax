"""Build-time fitting: sample at Chebyshev points, DCT to coefficients.

The float64 path uses numpy. Passing dps switches sampling and the DCT to
mpmath at that precision; mpmath is imported inside that branch only (the
runtime-imports rule in CLAUDE.md).

Adaptive fitting doubles the node count until the coefficient tail sits at
the fit's own noise floor, then truncates there. A converged-looking tail
on one grid is never trusted by itself: on n first-kind points T_{2n-k}
samples exactly as -T_k, so an aliased fit can look converged (T_34 on 17
nodes reproduces -T_0 to the last bit). Every candidate is therefore
validated against fresh samples at off-grid points before it is returned.

The noise floor is estimated from the trailing quarter of the candidate's
own coefficients rather than assumed: float64 sampling noise scales with
the function's derivative (measured: ~1e-16 relative for exp, ~2.5e-15 for
sin(50x)), so a fixed tolerance either fails to converge on oscillatory
functions or keeps hundreds of noise coefficients.
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
    tol:     target for the relative coefficient tail (default 5e-16). The
             effective chop level is max(tol, the fit's measured noise floor);
             float64 sampling noise can sit above tol for functions with
             large derivatives. Build with dps to control the floor.
    breaks:  strictly increasing knots, endpoints included; one series per
             segment, each fit independently, padded to a common degree.
    dps:     mpmath working precision for sampling and the DCT. Derivatives of
             a float64-built fit sit at a ~deg^2 * eps floor (differentiation
             amplifies sample noise); build with dps when that matters.
    max_deg: hard cap on the returned degree. Reaching it without a validated
             converged fit raises: f is not smooth enough on the interval
             (split the domain) or genuinely needs a higher degree.
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
    dct, sample_unit = (_dct_mp(f, a, b, dps) if dps is not None
                        else _dct_np(f, a, b))
    if deg is not None:
        return dct(deg + 1)
    return _adaptive(dct, sample_unit, tol, max_deg, (a, b))


def _chop(c, tol):
    """Truncate at the noise level; returns (candidate, absolute level).

    The level is max(tol, noise floor), where the noise floor is read off
    the last eighth of the coefficients: for a still-decaying tail those
    sit below tol and the requested tol governs (measured: exp's last
    coefficients decay through 1e-16); for a converged fit they are the
    sampling-noise plateau (sin(50x): ~5e-16 relative at n=129, edge-
    singular sampling can push higher) and doubling the largest of them
    clears the plateau's own scatter."""
    scale = np.max(np.abs(c))
    noise = 2.0 * float(np.max(np.abs(c[-max(2, c.size // 16):])))
    level = max(tol * scale, noise)
    keep = np.nonzero(np.abs(c) > level)[0]
    return (c[: keep[-1] + 1] if keep.size else c[:1]), level


def _validated(cand, level, sample_unit):
    """Check the candidate against fresh samples at off-grid points.

    chebpts(m) for m != n shares no points with the fitting grid, so an
    aliased fit cannot pass: its off-grid error is O(scale), ~1e15x the
    chop level, while an honest fit's is ~1x. The 1000x margin makes this
    a gross-error detector, not the accuracy contract.
    """
    m = cand.size + 11
    t = algorithms.chebpts(m)
    y = sample_unit(t)
    err = np.max(np.abs(algorithms.chebval(t, cand) - y))
    allow = max(1000.0 * level,
                100.0 * np.finfo(float).eps * float(np.sum(np.abs(cand))))
    return err <= allow


def _adaptive(dct, sample_unit, tol, max_deg, interval):
    n = min(17, max_deg + 1)
    prev_rel = np.inf
    while True:
        c = dct(n)
        scale = float(np.max(np.abs(c)))
        if scale == 0.0:
            return np.zeros(1)
        cand, level = _chop(c, tol)
        rel = level / scale
        # The tail has settled if the chop level reached the requested tol,
        # or if it stopped improving under node doubling while being far
        # below the function scale: that is the sampling-noise plateau, and
        # more nodes cannot beat it. A small-but-still-shrinking tail (a
        # segment one doubling short of converged) must escalate, not chop.
        settled = level <= 2.0 * tol * scale or (rel <= 1e-10 and rel >= prev_rel / 4.0)
        if settled and cand.size <= c.size - max(2, c.size // 8):
            if _validated(cand, level, sample_unit):
                return cand
        prev_rel = rel
        if n >= max_deg + 1:
            raise ValueError(
                f"no validated convergence to tol={tol:g} by degree {max_deg} on "
                f"{interval}; f may not be smooth there (split the domain), may "
                "need a higher max_deg, or float64 sampling noise may sit above "
                "tol (build with dps)")
        n = min(2 * n - 1, max_deg + 1)


def _dct_np(f, a, b):
    def sample_unit(t):
        xk = a + (b - a) * (t + 1) / 2
        return _sample(f, xk)

    def dct(n):
        return algorithms.chebfit_dct(sample_unit(algorithms.chebpts(n)))
    return dct, sample_unit


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

    def sample_unit(t):
        with mp.workdps(dps):
            mid = (mp.mpf(b) + mp.mpf(a)) / 2
            half = (mp.mpf(b) - mp.mpf(a)) / 2
            return np.array([float(f(mid + half * mp.mpf(float(tk)))) for tk in t])

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
    return dct, sample_unit
