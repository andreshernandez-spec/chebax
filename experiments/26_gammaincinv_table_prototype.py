#!/usr/bin/env python3
"""Experiment 26: the Gamma quantile from a table plus one Newton step.

(The prototype. The shipped table is gammaincinv_array_gen, increment 44,
with the centre zone scaled to reach a = inf; experiments/31 measures it.)

WHAT IS MEASURED
----------------
A prototype of experiment 25's design for a per element in [0.1, 1000]:
tables of the quantile in three zones, fitted in mpmath at 1e-8 by root
finding at the nodes, with gammainc_array_gen's patch refinement (10 x 10
blocks), then one Newton step in t = ln(x/a) on gammainc_array's ln P and
ln Q, the same step the shipped solve takes. Zones and coordinates, with the
panel in a (u = ln(1 + a) below 10, v = 10/a above; w = 1/a for the centre):

    lower   s = (p Gamma(a+1))^(1/a) / (a+1) <= S (0.4 below a = 1, 0.3 above):
            x = (a+1) s (1 + rho(s, a))
    centre  a >= 1 and eta0 = ndtri(p)/sqrt(a) <= 2.6:
            eta = eta0 + eps(eta0, 1/a), x = a lambda(eta) from a 1-D series
    upper   the rest: x_W from x - (a-1) ln x = -ln q - lnGamma(a) (four
            Newton steps on that elementary equation), tau = (a+1)/x_W,
            x = x_W (1 + delta(tau, a))

Measured: the number of patches and coefficients per element; the round
trip |F(x)/target - 1| on the target's own side against the 40-dps series
and continued fraction, over a grid of a and both tails, for the table alone,
after one Newton step, and for chebax.gammaincinv_array; and the time per
element against gammaincinv_array at N = 2^14 on the CPU and 2^20 on the
GPU when there is one.

WHAT IT MEANS
-------------
If one step reaches gammaincinv_array's own floor and the time drops by the
step count, stage 3 (a generator and runtime, float32 with no step) is
justified. The table alone is the float32 answer.

WHAT THIS DOES NOT ESTABLISH
----------------------------
a < 0.1 or above 1000 (the tables here stop at 1000; the real ones take
v = 10/a to 0), float32, derivatives (the implicit function theorem on
_log_pq, unchanged), and the far ends of the float range.

Run:  python experiments/26_gammaincinv_table_prototype.py        # ~10 min
"""

import functools
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

sys.path.insert(0, "src")
jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402
from chebax._src.recipes import gammainc_array as _ga  # noqa: E402
from chebax._src.recipes import gammainc_array_gen as _gag  # noqa: E402
from chebax._src.recipes import gammaincinv_kernel as _k  # noqa: E402
from chebax._src.recipes._gen_common import DPS, dct, nodes  # noqa: E402
from chebax._src.recipes.gammainc_large_gen import (  # noqa: E402
    _hyp1f1_1,
    _lam_of_eta,
    _log_q,
)

mp.mp.dps = DPS
TOL = 1e-8
KY = KP = 10
NODES = 14
ETA_LO, ETA_HI = -0.7, 2.6
A_MAX = 1000.0
S_SMALL, S_BIG = 0.4, 0.3
U1, U10 = math.log(2.0), math.log(11.0)


# references, as in experiment 25


def log_p(a, x):
    if x > a + 1:
        return mp.log1p(-mp.exp(_log_q(mp, a, x)))
    return a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))


def log_q(a, x):
    if x > a + 1:
        return _log_q(mp, a, x)
    lp = a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(_hyp1f1_1(mp, a + 1, x))
    return mp.log1p(-mp.exp(lp))


def solve(f, target, x0):
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


# the tabulated corrections, in gammainc_array_gen's f(mp, a, y) form


def f_lower(mp, a, s):
    y = s * (a + 1)
    ltp = a * mp.log(y) - mp.loggamma(a + 1)
    return solve(lambda x: log_p(a, x), ltp, y) / y - 1


def f_centre(mp, a, eta0):
    z = eta0 * mp.sqrt(a)
    c = z / (3 * mp.sqrt(a)) - 1 / (9 * a)
    x0 = a * c**3 if c > 0 else a * mp.exp(3 * c)
    if z < 0:
        x = solve(lambda x: log_p(a, x), mp.log(mp.ncdf(z)), x0)
    else:
        x = solve(lambda x: log_q(a, x), mp.log(mp.ncdf(-z)), x0)
    lam = x / a
    w = lam - 1 - mp.log(lam)
    return mp.sign(lam - 1) * mp.sqrt(2 * w) - eta0


def f_upper(mp, a, tau):
    xw = (a + 1) / tau
    r = xw - (a - 1) * mp.log(xw)
    ltq = -r - mp.loggamma(a)
    return solve(lambda x: log_q(a, x), ltq, xw) / xw - 1


# zone id = 3 * kind + panel, kind 0 lower, 1 centre, 2 upper; panel 0 a < 1,
# 1 a in [1, 10), 2 a >= 10 (the centre has no panel 0)
ZONES = {
    0: (f_lower, "u", (math.log1p(0.1), U1), (0.0, S_SMALL)),
    1: (f_lower, "u", (U1, U10), (0.0, S_BIG)),
    2: (f_lower, "v", (10.0 / A_MAX, 1.0), (0.0, S_BIG)),
    4: (f_centre, "w", (0.1, 1.0), (ETA_LO, ETA_HI)),
    5: (f_centre, "w", (1.0 / A_MAX, 0.1), (ETA_LO, ETA_HI)),
    6: (f_upper, "u", (math.log1p(0.1), U1), (0.0, 1.5)),
    7: (f_upper, "u", (U1, U10), (0.0, 1.0)),
    8: (f_upper, "v", (10.0 / A_MAX, 1.0), (0.0, 0.5)),
}
LAM_BAND = (-1.1, 3.1)


def build():
    """Patch tables for every zone, and the ln lambda(eta) series."""
    _gag.NODES, _gag.KY, _gag.KP, _gag.TOL = NODES, KY, KP, TOL
    blocks, pcuts, ycuts, npar, narg, offset = [], {}, {}, {}, {}, {}
    t0 = time.time()
    for z, (f, kind, pbox, ybox) in ZONES.items():
        pc, yc, patches = _gag._refine(mp, f, kind, pbox, ybox)
        offset[z] = len(blocks)
        npar[z], narg[z] = len(pc) - 1, len(yc) - 1
        pcuts[z], ycuts[z] = pc, yc
        for c in patches:
            blocks.append([[float(c[k][j]) for j in range(KP)] for k in range(KY)])
        print(
            f"  zone {z}: {npar[z]} x {narg[z]} patches  [{time.time() - t0:.0f} s]",
            flush=True,
        )
    lo, hi = (mp.mpf(v) for v in LAM_BAND)
    c = dct(
        mp,
        [mp.log(_lam_of_eta(mp, lo + (t + 1) / 2 * (hi - lo))) for t in nodes(mp, 40)],
    )
    c = np.array([float(v) for v in c])
    keep = np.nonzero(np.abs(c) > 1e-10)[0].max() + 1
    npad = max(len(v) for v in pcuts.values()), max(len(v) for v in ycuts.values())
    tab = {
        "TABLE": np.array(blocks),
        "PCUTS": np.array(
            [
                pcuts.get(z, [0.0])
                + [pcuts.get(z, [0.0])[-1]] * (npad[0] - len(pcuts.get(z, [0.0])))
                for z in range(9)
            ]
        ),
        "YCUTS": np.array(
            [
                ycuts.get(z, [0.0])
                + [ycuts.get(z, [0.0])[-1]] * (npad[1] - len(ycuts.get(z, [0.0])))
                for z in range(9)
            ]
        ),
        "NPAR": np.array([npar.get(z, 1) for z in range(9)]),
        "NARG": np.array([narg.get(z, 1) for z in range(9)]),
        "OFFSET": np.array([offset.get(z, 0) for z in range(9)]),
        "LAM": c[:keep],
    }
    print(
        f"  {len(blocks)} patches of {KY} x {KP}, {len(blocks) * KY * KP} coefficients; "
        f"ln lambda(eta): {keep} terms  [{time.time() - t0:.0f} s]"
    )
    return tab


# runtime


def _clenshaw(c, t):
    b1 = b2 = jnp.zeros_like(t)
    for ck in c[:0:-1]:
        b1, b2 = 2.0 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + c[0]


def _xw(a, r):
    """x with x - (a-1) ln x = r, four Newton steps from max(r, 1.5 a + 1)."""
    x = jnp.maximum(r, 1.5 * a + 1.0)
    for _ in range(4):
        g = x - (a - 1.0) * jnp.log(x) - r
        x = jnp.maximum(x - g / (1.0 - (a - 1.0) / x), 0.5 * x)
    return x


def make_quantile(tab):
    table = {k: jnp.asarray(v) for k, v in tab.items()}
    lam_c = [float(v) for v in tab["LAM"]]

    def start(a, ltp, ltq, z, upper):
        """t0 = ln(x0/a) from the tables, and the zone taken."""
        panel = jnp.where(a < 1.0, 0, jnp.where(a < 10.0, 1, 2))
        ln_y = (ltp + jax.scipy.special.gammaln(a + 1.0)) / a
        s = jnp.exp(ln_y - jnp.log1p(a))
        s_max = jnp.where(a < 1.0, S_SMALL, S_BIG)
        eta0 = z / jnp.sqrt(a)
        r = -ltq - jax.scipy.special.gammaln(a)
        xw = _xw(a, r)
        tau = (a + 1.0) / xw
        kind = jnp.where(s <= s_max, 0, jnp.where((a >= 1.0) & (eta0 <= ETA_HI), 1, 2))
        zone = 3 * kind + panel
        par = jnp.select([kind == 1, panel == 2], [1.0 / a, 10.0 / a], jnp.log1p(a))
        arg = jnp.select([kind == 0, kind == 1], [s, eta0], tau)
        i, sp = _ga._locate(par, table["PCUTS"][zone], table["NPAR"][zone])
        j, sy = _ga._locate(arg, table["YCUTS"][zone], table["NARG"][zone])
        block = table["TABLE"][table["OFFSET"][zone] + i * table["NARG"][zone] + j]
        f = jnp.einsum(
            "...kj,...k,...j->...", block, _ga._basis(sy, KY), _ga._basis(sp, KP)
        )
        eta = jnp.clip(eta0 + f, LAM_BAND[0], LAM_BAND[1])
        te = 2.0 * (eta - LAM_BAND[0]) / (LAM_BAND[1] - LAM_BAND[0]) - 1.0
        ln_lam = _clenshaw(lam_c, te)
        t0 = jnp.select(
            [kind == 0, kind == 1],
            [ln_y - jnp.log(a) + jnp.log1p(f), ln_lam],
            jnp.log(xw) - jnp.log(a) + jnp.log1p(f),
        )
        return t0, zone

    def newton(a, t, ltp, ltq, upper):
        x = a * jnp.exp(t)
        lp, lq = _ga._log_pq(a, x)
        f = jnp.where(upper, ltq - lq, lp - ltp)
        df = jnp.exp(_k.log_xpdf(a, t) - jnp.where(upper, lq, lp))
        return t - f / df

    @functools.partial(jax.jit, static_argnums=2)
    def quantile(a, p, steps):
        """x with P(a, x) = p: table start, `steps` Newton steps (static)."""
        a = jnp.clip(a, 0.1, A_MAX)
        ltp, ltq, z, upper = _k.inv_targets(p)
        t, _ = start(a, ltp, ltq, z, upper)
        for _ in range(steps):
            t = newton(a, t, ltp, ltq, upper)
        return a * jnp.exp(t)

    @functools.partial(jax.jit, static_argnums=2)
    def cquantile(a, q, steps):
        a = jnp.clip(a, 0.1, A_MAX)
        ltp, ltq, z, upper = _k.cinv_targets(q)
        t, _ = start(a, ltp, ltq, z, upper)
        for _ in range(steps):
            t = newton(a, t, ltp, ltq, upper)
        return a * jnp.exp(t)

    return quantile, cquantile, start


# accuracy

A_GRID = [
    0.1,
    0.17,
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.5,
    5.0,
    9.0,
    10.0,
    17.0,
    50.0,
    180.0,
    640.0,
    1000.0,
]
P_GRID = [1e-300, 1e-100, 1e-20, 1e-6, 1e-3, 0.02, 0.1, 0.3, 0.5]
Q_GRID = [0.4, 0.2, 0.05, 1e-3, 1e-6, 1e-20, 1e-100, 1e-300]


def grid():
    """(a, p, q) arrays over the grid, lower targets with q = nan and upper
    with p = nan, dropping targets whose quantile leaves the float range."""
    rows = []
    for a in A_GRID:
        am = mp.mpf(a)
        for p in P_GRID:
            # ln x > -700 at the asymptotic start
            if (mp.log(mp.mpf(p)) + mp.loggamma(am + 1)) / am > -700:
                rows.append((a, p, np.nan))
        for q in Q_GRID:
            rows.append((a, np.nan, q))
    return (np.array(c) for c in zip(*rows))


def round_trip(a, p, q, x):
    """|F(x)/target - 1| on the target's side, 40 dps."""
    out = np.empty(len(a))
    for i in range(len(a)):
        am, xm = mp.mpf(float(a[i])), mp.mpf(float(x[i]))
        if not np.isfinite(x[i]) or x[i] <= 0:
            out[i] = np.inf
            continue
        if np.isnan(q[i]):
            out[i] = float(abs(mp.expm1(log_p(am, xm) - mp.log(mp.mpf(float(p[i]))))))
        else:
            out[i] = float(abs(mp.expm1(log_q(am, xm) - mp.log(mp.mpf(float(q[i]))))))
    return out


def accuracy(quantile, cquantile):
    a, p, q = grid()
    lower = np.isnan(q)
    print(
        f"\nround trip |F(x)/target - 1| on {len(a)} points, a in [0.1, 1000], both tails"
    )
    print(f"  {'method':28s} {'worst':>9s} {'median':>9s}  worst at")

    def run(name, fn_p, fn_q):
        x = np.empty(len(a))
        x[lower] = np.asarray(fn_p(jnp.asarray(a[lower]), jnp.asarray(p[lower])))
        x[~lower] = np.asarray(fn_q(jnp.asarray(a[~lower]), jnp.asarray(q[~lower])))
        err = round_trip(a, p, q, x)
        k = int(np.argmax(err))
        tgt = f"p={p[k]:.0e}" if lower[k] else f"q={q[k]:.0e}"
        print(f"  {name:28s} {err.max():9.2e} {np.median(err):9.2e}  a={a[k]:g}, {tgt}")
        return err

    for steps in (0, 1, 2):
        run(
            f"table + {steps} Newton step(s)",
            lambda aa, pp: quantile(aa, pp, steps),
            lambda aa, qq: cquantile(aa, qq, steps),
        )
    run("chebax.gammaincinv_array", chebax.gammaincinv_array, chebax.gammainccinv_array)


# timing


def timed(call, reps=7):
    jax.block_until_ready(call())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def speed(quantile):
    rng = np.random.default_rng(0)
    for dev in jax.devices("cpu") + [d for d in jax.devices() if d.platform != "cpu"]:
        n = 2**14 if dev.platform == "cpu" else 2**20
        a = jax.device_put(
            jnp.asarray(np.exp(rng.uniform(np.log(0.1), np.log(1000.0), n))), dev
        )
        p = jax.device_put(jnp.asarray(rng.uniform(1e-6, 1 - 1e-6, n)), dev)
        print(f"\n{dev.platform}, N = {n}, float64, ns per element (median of 7)")
        for steps in (0, 1):
            with jax.default_device(dev):
                t = timed(lambda: quantile(a, p, steps))
            print(f"  table + {steps} Newton step(s): {1e9 * t / n:8.1f}")
        with jax.default_device(dev):
            f = jax.jit(chebax.gammaincinv_array)
            t = timed(lambda: f(a, p))
        print(f"  chebax.gammaincinv_array:  {1e9 * t / n:8.1f}")


def main():
    print(
        f"tables at TOL = {TOL}, {KY} x {KP} blocks, {NODES} nodes per patch, mpmath {mp.__version__}"
    )
    cache = os.environ.get("EXP26_TABLE_CACHE")
    if cache and os.path.exists(cache):
        tab = dict(np.load(cache))
        print(f"  tables from {cache}: {tab['TABLE'].shape[0]} patches")
    else:
        tab = build()
        if cache:
            np.savez(cache, **tab)
    quantile, cquantile, _ = make_quantile(tab)
    accuracy(quantile, cquantile)
    speed(quantile)


if __name__ == "__main__":
    main()
