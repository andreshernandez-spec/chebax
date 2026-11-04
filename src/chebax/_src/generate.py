"""Build-time fitting: sample at Chebyshev points, DCT to coefficients. numpy
in float64, or mpmath at dps (imported in that branch only).

Adaptive fitting doubles the node count until the coefficient tail sits at
the fit's own noise floor, then truncates there. A converged-looking tail on
one grid is not trusted: on n first-kind points T_{2n-k} samples as -T_k, so
every adaptive candidate is validated against fresh samples on finer grids
(_validated). A fixed deg= is taken as an instruction and not validated. The
noise floor is read off the trailing sixteenth of the coefficients, since
float64 sampling noise scales with the function's derivative (~1e-16 relative
for exp, 2.5e-15 for sin(50x)).
"""

import numpy as np

from chebax._src import algorithms
from chebax._src.series import ChebSeries, PiecewiseCheb

# ~2*eps: the practical floor of the float64 build path. Use dps= to go below.
_DEFAULT_TOL = 5e-16


def fit(
    f, domain=(-1.0, 1.0), *, deg=None, tol=None, breaks=None, dps=None, max_deg=256
):
    """Chebyshev interpolant of f: a ChebSeries, or a PiecewiseCheb with breaks.

    f:       callable on the domain, vectorized over numpy arrays; with dps
             set it takes mpmath floats.
    domain:  (a, b), finite, a < b; ignored with breaks.
    deg:     fixed degree (at most max_deg), returned as asked with no
             convergence or aliasing check; unset, the degree adapts to tol.
    tol:     relative coefficient tail (default 5e-16); the chop level is
             max(tol, the fit's measured noise floor).
    breaks:  increasing finite knots, endpoints included; one series per
             segment, padded to a common degree.
    dps:     mpmath precision for sampling and the DCT. Derivatives of a
             float64 fit sit at a ~deg^2 eps floor; build with dps when that
             matters.
    max_deg: cap on the degree; reaching it without a validated fit raises.
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
        cs = [
            _fit_interval(f, br[i], br[i + 1], deg, tol, max_deg, dps)
            for i in range(len(br) - 1)
        ]
        n = max(c.size for c in cs)
        coef = np.zeros((len(cs), n))
        for i, c in enumerate(cs):
            coef[i, : c.size] = c
        return PiecewiseCheb(coef, br)
    try:
        n_dom = len(domain)
    except TypeError:
        n_dom = -1
    if n_dom != 2:
        # (1,) used to reach an IndexError and a 3-tuple silently dropped
        # its third element (review, 2026-08-02)
        raise ValueError(f"domain must be a pair (a, b), got {domain!r}")
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
    dct, sample_unit = _dct_mp(f, a, b, dps) if dps is not None else _dct_np(f, a, b)
    if deg is not None:
        return dct(deg + 1)
    return _adaptive(dct, sample_unit, tol, max_deg, (a, b))


_NOISE_CEIL = 1e-8  # a window this close to the function scale is signal
_PHI = 0.6180339887498949  # (sqrt(5) - 1)/2, irrational: see _offgrid


def _chop(c, tol):
    """Truncate at max(tol, noise floor); returns (candidate, absolute level,
    plateau relative to the scale, inf if none). The floor is twice the
    largest of the last sixteenth of the coefficients, which is noise only
    when it sits under _NOISE_CEIL of the scale: fitting T20 on 21 points puts
    c[20] = 1 there, and reading that as noise chopped the exact fit to one
    coefficient."""
    scale = np.max(np.abs(c))
    win = float(np.max(np.abs(c[-max(2, c.size // 16) :])))
    measured = win <= _NOISE_CEIL * scale
    noise = 2.0 * win if measured else 0.0
    level = max(tol * scale, noise)
    keep = np.nonzero(np.abs(c) > level)[0]
    return (
        (c[: keep[-1] + 1] if keep.size else c[:1]),
        level,
        noise / scale if measured else np.inf,
    )


def _offgrid(m):
    """m points cos(pi (i + PHI) / m), PHI irrational: not the nodes of any
    Chebyshev grid. On m first-kind points T_K samples as +-T_j with j fixed
    by K mod 4m, so the grids n, 2n and 4n all read T_{16n} as +1 (fit(T272)
    once came back as the constant 1). Here an alias needs (K -+ j)(i + PHI)/m
    even for every i, which forces q PHI integral for some integer q, so only
    the candidate's own mode can hide."""
    i = np.arange(m)
    return np.cos(np.pi * (i + _PHI) / m)


def _validated(cand, level, sample_unit, n):
    """Check the candidate against fresh samples off the fit grid: a
    gross-error detector (an aliased fit is off by the scale, ~1e15x the chop
    level; the margin is 1000x). The off-grid sets carry the argument and the
    2n grid catches ordinary mistakes; sizes follow the fitting resolution n,
    so they never land on the fitting grid. The second term is Clenshaw's own
    evaluation floor, which grows with the degree (a degree-544 candidate is
    6e-13 off its own samples)."""
    allow = max(
        1000.0 * level,
        100.0 * np.finfo(float).eps * cand.size * float(np.sum(np.abs(cand))),
    )
    for t in (algorithms.chebpts(2 * n), _offgrid(2 * n + 1), _offgrid(4 * n + 1)):
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
            cand, level, rel = _chop(c, tol)
            # The tail has settled if the chop level reached the requested
            # tol, or if the MEASURED plateau stopped improving under node
            # doubling while being far below the function scale: that is the
            # sampling-noise plateau, and more nodes cannot beat it. A tail
            # that is small but still shrinking (a segment one doubling short
            # of converged) must escalate, not chop. rel is inf when the
            # window held signal rather than noise, so a resolution with no
            # plateau to report cannot seed the comparison either way.
            settled = level <= 2.0 * tol * scale or (
                rel <= 1e-10 and rel >= prev_rel / 4.0
            )
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
                "tol (build with dps)"
            )
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
    try:
        import mpmath as mp
    except ImportError as e:
        raise ImportError(
            "chebax.fit(dps=...) needs mpmath: pip install chebax[gen]"
        ) from e

    def sample_unit(t):
        with mp.workdps(dps):
            mid = (mp.mpf(b) + mp.mpf(a)) / 2
            half = (mp.mpf(b) - mp.mpf(a)) / 2
            return np.array([float(f(mid + half * mp.mpf(float(tk)))) for tk in t])

    def dct(n):
        with mp.workdps(dps):
            mid = (mp.mpf(b) + mp.mpf(a)) / 2
            half = (mp.mpf(b) - mp.mpf(a)) / 2
            fk = [
                f(mid + half * mp.cos(mp.pi * (2 * i + 1) / (2 * n))) for i in range(n)
            ]
            c = []
            for j in range(n):
                s = mp.fsum(
                    fk[i] * mp.cos(mp.pi * j * (2 * i + 1) / (2 * n)) for i in range(n)
                )
                c.append(2 * s / n)
            c[0] /= 2
            return np.array([float(x) for x in c])

    return dct, sample_unit
