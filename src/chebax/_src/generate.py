"""Build-time fitting: sample at Chebyshev points, DCT to coefficients.

The float64 path uses numpy. Passing dps switches sampling and the DCT to
mpmath at that precision; mpmath is imported inside that branch only (the
runtime-imports rule in CLAUDE.md).

Adaptive fitting doubles the node count until the coefficient tail sits at
the fit's own noise floor, then truncates there. A converged-looking tail
on one grid is never trusted by itself: on n first-kind points T_{2n-k}
samples exactly as -T_k, so an aliased fit can look converged (T_34 on 17
nodes reproduces -T_0 to the last bit). Every candidate the adaptive path
returns, the all-zero fit included, is first validated against fresh
samples on two finer grids (see _validated). A fixed deg= is not: there an
unresolved f is a legitimate request, so under-resolution cannot be told
from aliasing.

The noise floor is estimated from the trailing sixteenth of the fit's own
coefficients rather than assumed: float64 sampling noise scales with
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
    domain:  (a, b), finite with a < b. Ignored when breaks is given.
    deg:     fixed degree, at most max_deg; otherwise the degree adapts until
             tol is met. A fixed degree is taken as an instruction: the
             interpolant is returned as asked, with no convergence check and
             no aliasing check (fit(T_29, deg=5) returns T_5, which is what
             interpolating T_29 at 6 points gives). Leave deg unset for a
             validated fit.
    tol:     target for the relative coefficient tail (default 5e-16), positive
             and finite. The effective chop level is max(tol, the fit's
             measured noise floor); float64 sampling noise can sit above tol
             for functions with large derivatives. Build with dps to control
             the floor.
    breaks:  strictly increasing finite knots, at least two, endpoints
             included; one series per segment, each fit independently, padded
             to a common degree.
    dps:     mpmath working precision for sampling and the DCT. Derivatives of
             a float64-built fit sit at a ~deg^2 * eps floor (differentiation
             amplifies sample noise); build with dps when that matters.
    max_deg: hard cap on the returned degree. Reaching it without a validated
             converged fit raises: f is not smooth enough on the interval
             (split the domain) or genuinely needs a higher degree.
    """
    max_deg = _check_int("max_deg", max_deg, 1)
    if deg is not None:
        deg = _check_int("deg", deg, 0)
        if deg > max_deg:
            raise ValueError(f"deg={deg} is above the max_deg={max_deg} cap")
    if dps is not None:
        dps = _check_int("dps", dps, 1)
    if tol is None:
        tol = _DEFAULT_TOL
    else:
        tol = float(tol)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError(f"tol must be positive and finite, got {tol}")
    if breaks is not None:
        br = [float(t) for t in breaks]
        if len(br) < 2:
            raise ValueError("breaks needs at least two knots")
        if not all(np.isfinite(t) for t in br):
            raise ValueError("breaks must be finite")
        if not all(u < v for u, v in zip(br, br[1:])):
            raise ValueError("breaks must be strictly increasing")
        cs = [_fit_interval(f, br[i], br[i + 1], deg, tol, max_deg, dps)
              for i in range(len(br) - 1)]
        n = max(c.size for c in cs)
        coef = np.zeros((len(cs), n))
        for i, c in enumerate(cs):
            coef[i, : c.size] = c
        return PiecewiseCheb(coef, br)
    a, b = float(domain[0]), float(domain[1])
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        raise ValueError(f"domain must be finite with a < b, got ({a}, {b})")
    return ChebSeries(_fit_interval(f, a, b, deg, tol, max_deg, dps), (a, b))


def _check_int(name, v, low):
    kind = "a nonnegative integer" if low == 0 else "a positive integer"
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < low:
        raise ValueError(f"{name} must be {kind}, got {v!r}")
    return int(v)


def _fit_interval(f, a, b, deg, tol, max_deg, dps):
    dct, sample_unit = (_dct_mp(f, a, b, dps) if dps is not None
                        else _dct_np(f, a, b))
    if deg is not None:
        return dct(deg + 1)
    return _adaptive(dct, sample_unit, tol, max_deg, (a, b))


def _chop(c, tol):
    """Truncate at the noise level; returns (candidate, absolute level).

    The level is max(tol, noise floor), where the noise floor is read off
    the last sixteenth of the coefficients: for a still-decaying tail those
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


def _validated(cand, level, sample_unit, n):
    """Check the candidate against fresh samples on the 2n and 4n grids.

    Two grids, both fixed by the fitting resolution n rather than by the
    candidate's size, which could land back on the fitting grid itself.
    chebpts(p) and chebpts(q) share a point only when p/g and q/g are both
    odd (g their gcd), so n, 2n and 4n are pairwise disjoint. Disjoint
    points are not the argument though, the aliasing is: on m first-kind
    points T_K samples as +-T_j with j and the sign fixed by K mod 4m, and
    the smallest K whose (j, sign) agrees across n, 2n and 4n is 15n + 1,
    folding onto j = n - 1, wider than any candidate the loop accepts below
    the cap. So an aliased fit is off by O(scale) on at least one of the two
    grids, ~1e15x the chop level, where an honest fit is at ~1x. The 1000x
    margin makes this a gross-error detector, not the accuracy contract.
    """
    allow = max(1000.0 * level,
                100.0 * np.finfo(float).eps * float(np.sum(np.abs(cand))))
    for m in (2 * n, 4 * n):
        t = algorithms.chebpts(m)
        err = np.max(np.abs(algorithms.chebval(t, cand) - sample_unit(t)))
        if err > allow:
            return False
    return True


def _adaptive(dct, sample_unit, tol, max_deg, interval):
    n = min(17, max_deg + 1)
    prev_rel = np.inf
    while True:
        c = dct(n)
        scale = float(np.max(np.abs(c)))
        if scale == 0.0:
            # all-zero samples prove nothing off the grid, T_n vanishes on
            # chebpts(n), so the zero series is validated like any other
            cand, level, settled = np.zeros(1), 0.0, True
        else:
            cand, level = _chop(c, tol)
            rel = level / scale
            # The tail has settled if the chop level reached the requested
            # tol, or if it stopped improving under node doubling while being
            # far below the function scale: that is the sampling-noise
            # plateau, and more nodes cannot beat it. A tail that is small but
            # still shrinking (a segment one doubling short of converged) must
            # escalate, not chop.
            settled = level <= 2.0 * tol * scale or (rel <= 1e-10 and rel >= prev_rel / 4.0)
            prev_rel = rel
        at_cap = n >= max_deg + 1
        # A candidate holding nearly every coefficient of its grid has not
        # shown a resolved tail, so double instead. At the cap there is no
        # doubling left and validation is the only evidence available.
        chopped = cand.size <= c.size - max(2, c.size // 8)
        if settled and (chopped or at_cap) and _validated(cand, level, sample_unit, n):
            return cand
        if at_cap:
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
