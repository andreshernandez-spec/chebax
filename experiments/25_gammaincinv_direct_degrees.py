#!/usr/bin/env python3
"""Experiment 25: can the Gamma quantile be tabulated directly?

WHAT IS MEASURED
----------------
gammaincinv runs a fixed-count Newton (40 steps on the scalar path, 8 Halley
steps per element on the array path), each step one evaluation of the
gammainc tables. A Chebyshev table of the quantile itself, built by root
finding in mpmath at the nodes, would replace the loop by one table
evaluation and at most one Newton step (none in float32). The table does not
care that its generator iterates; it cares about smoothness, and the quantile
is analytic wherever the density is positive. What this script measures is
the coefficient tail, along the argument and the parameter axis, of the
quantile written in three candidate coordinate pairs, one per zone, at the
accuracy a start needs for one Newton step (1e-8) and at the floor (1e-15).
"last > tol" is the highest coefficient index still above tol in absolute
value, at its worst over the other axis; a value of NODES - 1 means the fit
did not converge on the box.

The zones and what is tabulated (a is the shape, p the lower-tail target, q
the upper-tail one; every tabulated quantity is an O(1) correction whose
absolute error is a relative error in x):

- lower, all a:   y = (p Gamma(a+1))^(1/a) is the first term of the lower
                  tail, s = y / (a+1) in [0, S]; tabulated rho = x/y - 1,
                  rho(0) = 0. For large a, s -> lambda e^-lambda, singular
                  only at lambda = 1 (s = 1/e), so S stays at 0.3 there;
                  below a = 1 the zones only meet with S = 0.45 or tau up to
                  1.5 (the coverage table below).
- centre, a >= 1: Temme's inversion. eta0 = ndtri(p)/sqrt(a); the true eta
                  (eta^2/2 = lambda - 1 - ln lambda, lambda = x/a) is
                  eta0 + eps(eta0, w), w = 1/a, eps = O(w). Tabulated eps on
                  the Temme band and ln lambda(eta) as a 1-D series.
- upper, all a:   r = -ln q - lnGamma(a) and x_W solving x - (a-1) ln x = r
                  (three cheap Newton steps on an elementary function);
                  tabulated delta = x/x_W - 1 on tau = (a+1)/x_W in [0, T].
                  delta is C-infinity at tau = 0 (the asymptotic series of
                  Gamma(a, x) diverges), as gammainc_array's B0 and U zones
                  are, so the tail is measured on sub-boxes toward 0 too.

Parameter axes: u = ln(1 + a) on [0.1, 1] and [1, 10], v = 10/a on
[0.01, 1] (a in [10, 1000]; mpmath's gammainc stalls above). The centre zone
uses w = 1/a on [0.001, 0.1] and [0.1, 1].

WHAT IT MEANS
-------------
The degrees size the table: the product of the two "last > 1e-8" counts per
zone, summed over the zones an element can land in, against today's 144
coefficients x 8 steps on the array path and 68 x 40 on the scalar path.
Where a zone converges slowly, that coordinate pair is the wrong one or the
box needs patches, as gammainc_array_gen cuts them.

WHAT THIS DOES NOT ESTABLISH
----------------------------
The zone boundaries' coverage of every (a, p) (checked at a few a below, not
proven), the accuracy after one Newton step on the real tables, and speed.
Those are the prototype's job (stream 4, stage 2).

Run:  python experiments/25_gammaincinv_direct_degrees.py        # ~30 min
"""

import sys
import time

import mpmath as mp
import numpy as np
import scipy.special as sps

sys.path.insert(0, "src")
from chebax._src.recipes._gen_common import DPS, dct, nodes, param_fit  # noqa: E402
from chebax._src.recipes.gammainc_large_gen import (  # noqa: E402
    _hyp1f1_1,
    _lam_of_eta,
    _log_q,
)

# the generator's series and continued fraction stop at 10^-(DPS + 5)
mp.mp.dps = DPS
NODES = 28
TOLS = (1e-8, 1e-15)


def log_p(a, x):
    """ln P(a, x): the 1F1 series below x = a + 1, 1 - Q from the continued
    fraction above (mpmath's own gammainc is seconds per call at a = 1000)."""
    if x > a + 1:
        return mp.log1p(-mp.exp(_log_q(mp, a, x)))
    return a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))


def log_q(a, x):
    if x > a + 1:
        return _log_q(mp, a, x)
    lp = a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))
    return mp.log1p(-mp.exp(lp))


def solve(f, target, x0):
    """x with f(x) = target, f monotone in ln x, from the float64 start x0."""
    if target >= 0:
        raise ValueError(f"no quantile for a log probability of {target}")
    g = lambda t: f(mp.exp(t)) - target
    t0 = mp.log(x0)
    try:
        t = mp.findroot(
            g, (t0, t0 + mp.mpf("1e-6")), solver="secant", tol=1e-30, maxsteps=60
        )
        if abs(g(t)) < mp.mpf("1e-25") * max(1, abs(target)):
            return mp.exp(t)
    except (ValueError, ZeroDivisionError):
        pass
    lo, hi = t0 - 1, t0 + 1
    for _ in range(12):
        if g(lo) * g(hi) <= 0:
            break
        lo, hi = lo - 2 * (t0 - lo), hi + 2 * (hi - t0)
    else:
        raise ValueError(f"no bracket around ln x = {t0}")
    for _ in range(140):
        m = (lo + hi) / 2
        if g(m) * g(lo) > 0:
            lo = m
        else:
            hi = m
    return mp.exp((lo + hi) / 2)


def start_lower(a, ltp, y):
    """scipy's quantile where ln p is a float, else the asymptotic y."""
    if ltp > -700:
        x = sps.gammaincinv(float(a), float(mp.exp(ltp)))
        if np.isfinite(x) and x > 0:
            return x
    return y


def start_upper(a, ltq, xw):
    if ltq > -700:
        x = sps.gammainccinv(float(a), float(mp.exp(ltq)))
        if np.isfinite(x) and x > 0:
            return x
    return xw


# the three tabulated corrections


def rho_lower(a, s):
    y = s * (a + 1)
    ltp = a * mp.log(y) - mp.loggamma(a + 1)
    x = solve(lambda x: log_p(a, x), ltp, start_lower(a, ltp, y))
    return x / y - 1


def eps_centre(a, eta0):
    z = eta0 * mp.sqrt(a)
    # the small tail as the target, started from Wilson-Hilferty
    if z < 0:
        ltp = mp.log(mp.ncdf(z))
        x = solve(lambda x: log_p(a, x), ltp, start_lower(a, ltp, a))
    else:
        ltq = mp.log(mp.ncdf(-z))
        x = solve(lambda x: log_q(a, x), ltq, start_upper(a, ltq, a))
    lam = x / a
    w = lam - 1 - mp.log(lam)
    eta = mp.sign(lam - 1) * mp.sqrt(2 * w)
    return eta - eta0


def delta_upper(a, tau):
    xw = (a + 1) / tau
    r = xw - (a - 1) * mp.log(xw)
    ltq = -r - mp.loggamma(a)
    x = solve(lambda x: log_q(a, x), ltq, start_upper(a, ltq, xw))
    return x / xw - 1


def a_of(kind, p):
    if kind == "u":
        return mp.expm1(p)
    if kind == "v":
        return 10 / p
    return 1 / p


def fit2d(f, kind, pbox, ybox):
    """C[k][j]: j-th parameter coefficient of the k-th argument coefficient."""
    p0, p1 = (mp.mpf(v) for v in pbox)
    y0, y1 = (mp.mpf(v) for v in ybox)
    t = nodes(mp, NODES)
    rows = []
    for tp in t:
        a = a_of(kind, p0 + (tp + 1) / 2 * (p1 - p0))
        rows.append(dct(mp, [f(a, y0 + (ty + 1) / 2 * (y1 - y0)) for ty in t]))
    return np.array([[float(v) for v in row] for row in param_fit(mp, rows)])


def last_above(c, tol):
    """(argument index, parameter index) of the last coefficients above tol."""
    big = np.abs(c) > tol
    ks = np.nonzero(big.any(axis=1))[0]
    js = np.nonzero(big.any(axis=0))[0]
    return (int(ks.max()) if ks.size else -1, int(js.max()) if js.size else -1)


def report(label, c):
    scale = float(np.max(np.abs(c)))
    parts = [f"max {scale:.2e}"]
    for tol in TOLS:
        k, j = last_above(c, tol)
        parts.append(f"last>{tol:.0e}: arg {k:2d} par {j:2d}")
    print(f"  {label:34s} " + " | ".join(parts), flush=True)


PANELS = {
    "u [0.1, 1]": ("u", (float(mp.log1p(mp.mpf("0.1"))), float(mp.log(2)))),
    "u [1, 10]": ("u", (float(mp.log(2)), float(mp.log(11)))),
    "v a in [10, 1000]": ("v", (0.01, 1.0)),
}


def lower():
    print("lower zone: rho(s, a) = x/y - 1, y = (p Gamma(a+1))^(1/a), s = y/(a+1)")
    for S in (0.3, 0.4, 0.45):
        for name, (kind, pbox) in PANELS.items():
            if S > 0.3 and kind != "u":
                continue
            if S == 0.45 and pbox[1] > 1.0:
                continue
            report(f"S = {S}, {name}", fit2d(rho_lower, kind, pbox, (0.0, S)))


def centre():
    print("centre zone (a >= 1): eps(eta0, w) = eta - eta0, eta0 = ndtri(p)/sqrt(a)")
    for band in ((-0.7, 2.6), (-1.0, 3.0)):
        for wbox in ((0.001, 0.1), (0.1, 1.0)):
            report(f"eta0 {band}, w {wbox}", fit2d(eps_centre, "w", wbox, band))
    print("centre zone: ln lambda(eta), one series")
    for band in ((-1.0, 3.0), (-1.3, 3.3)):
        lo, hi = (mp.mpf(v) for v in band)
        c = dct(
            mp,
            [
                mp.log(_lam_of_eta(mp, lo + (t + 1) / 2 * (hi - lo)))
                for t in nodes(mp, 2 * NODES)
            ],
        )
        c = np.array([[float(v)] for v in c])
        report(f"eta {band}", c)


def upper():
    print(
        "upper zone: delta(tau, a) = x/x_W - 1, x_W - (a-1) ln x_W = -ln q - lnGamma(a)"
    )
    for tbox in (
        (0.0, 1.0),
        (0.0, 0.5),
        (0.5, 1.0),
        (0.0, 0.25),
        (1.0, 1.5),
        (0.0, 1.5),
    ):
        for name, (kind, pbox) in PANELS.items():
            if tbox[1] > 1.0 and pbox[1] > 1.0:
                continue
            report(f"tau {tbox}, {name}", fit2d(delta_upper, kind, pbox, tbox))


def coverage():
    """Where the zones meet, at a few a: the lower zone's top (s = S) and the
    upper zone's bottom (tau = T) against the centre band, as quantiles."""
    print("zone edges as lower-tail probabilities p (nan: not in the float range)")
    print(
        f"  {'a':>8s} {'s=0.3':>10s} {'s=0.4':>10s} {'eta0=-0.7':>10s} {'eta0=2.6':>10s} {'tau=1':>10s} {'tau=0.5':>10s}"
    )
    for a in (0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1000.0):
        am = mp.mpf(a)
        row = []
        for S in (0.3, 0.4):
            y = S * (am + 1)
            ltp = am * mp.log(y) - mp.loggamma(am + 1)
            if ltp >= 0:
                # s cannot reach S at this a (s -> lambda e^-lambda <= 1/e)
                row.append(float("nan"))
                continue
            x = solve(lambda x: log_p(am, x), ltp, start_lower(am, ltp, y))
            row.append(float(mp.exp(log_p(am, x))))
        for eta0 in (-0.7, 2.6):
            z = eta0 * mp.sqrt(am)
            row.append(float(mp.ncdf(z)) if a >= 1 else float("nan"))
        for T in (1.0, 0.5):
            xw = (am + 1) / T
            r = xw - (am - 1) * mp.log(xw)
            q = mp.exp(-r - mp.loggamma(am))
            row.append(float(1 - q))
        print(f"  {a:8.1f} " + " ".join(f"{v:10.4g}" for v in row), flush=True)


def main(which):
    t0 = time.time()
    print(f"mpmath {mp.__version__}, dps {mp.mp.dps}, {NODES} nodes per axis\n")
    for name, fn in (
        ("coverage", coverage),
        ("lower", lower),
        ("centre", centre),
        ("upper", upper),
    ):
        if which and name not in which:
            continue
        fn()
        print(f"  [{time.time() - t0:.0f} s]\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
