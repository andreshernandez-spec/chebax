"""Tweedie densities, 1.05 <= p <= 1.95, and ln Phi(rho, 0; z) under them.

References are mpmath at 40 dps: Phi summed over a window around its peak
term (the terms are log-concave in n), the density from its definition.
Both are logarithms, so the error is ABSOLUTE in the log (relative in W and
f) and, where the log is large, relative to it: |err| / max(1, |ref|).
Large logs carry eps |ln W| from forming Z itself, which no method avoids.

Measured worst, float64 (bar about 4x in each test):
- ln W: values 7.7e-16; d/dz 1.0e-14 relative (oracle: dPhi/dz is
  Phi(rho, rho; z)); d/drho 5.2e-15, 8.7e-15 on jax 0.5.3 (the CI floor), both
  just above the split; z to 1e300 against the leading saddle
  term, 3.2e-16. Both branches at the split agree to 4.4e-16 in value and
  3.5e-15 in d/du. The table's s -> 0 end matches
  c1(rho) = -(2 rho + 1)(rho + 2)/(24 rho) to 8.9e-16.
- density on CASES: values 3.3e-13; d/dmu 3.1e-16, d/dy 1.2e-13,
  d/dphi 7.5e-13, d/dp 3.6e-12.
- density on MORE, chebtorch's fixture points, which reach just below the
  split at small p where CASES does not: values 1.1e-12, d/dp 4.7e-11,
  d/dln mu 3.9e-16, d/dln phi 9.3e-12, d/dln y 7.8e-12. All worst near
  p = 1.05 in the direct sum, where lgamma(rho n) reaches ~4000 near the
  split and its rounding, eps |lgamma|, survives the cancellation down to
  an O(1) log-density; d/dp also carries drho/dp = -1/(p-1)^2 = -400 there.
- float32 (x64 off): ln W 7.8e-7, density 7.4e-6 on test_x64_off's grid.
  On MORE, against float64 at the same float32 inputs, by p:
                density  d/dp     d/dln mu  d/dln phi  d/dln y
    p < 1.1     4.5e-4   1.7e-2   1.7e-7    5.1e-3     4.9e-3
    p < 1.3     3.4e-5   8.9e-4   2.3e-7    1.1e-4     5.4e-5
    p >= 1.3    5.6e-6   2.7e-5   1.9e-7    5.3e-6     2.9e-6
  The same cancellation at float32's eps, plus XLA's float32 lgamma
  (~10 ulp near small arguments), which dominates at p >= 1.3.
"""

import math
import os
import pathlib
import random
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import chebax
from chebax._src.recipes import tweedie_gen
from chebax._src.recipes import tweedie_table as tt
from chebax._src.recipes.tweedie import _half_deviance, _log_series, _over, _split

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40
HERE = pathlib.Path(__file__).parent

RHOS = [1 / 19, 0.07, 0.111, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 14.0, 19.0]


def log_phi(rho, beta, z):
    """ln Phi(rho, beta; z) for z > 0, summed around its peak term."""
    rho, beta, z = mp.mpf(rho), mp.mpf(beta), mp.mpf(z)
    lz = mp.log(z)
    nmin = 1 if beta == 0 else 0

    def lt(n):
        return n * lz - mp.loggamma(n + 1) - mp.loggamma(rho * n + beta)

    n0 = max(nmin, int((rho * z) ** (1 / (1 + rho)) / rho))
    while n0 > nmin and lt(n0 - 1) > lt(n0):
        n0 -= 1
    while lt(n0 + 1) > lt(n0):
        n0 += 1
    peak, s = lt(n0), mp.mpf(0)
    for step, start in ((-1, n0 - 1), (1, n0)):
        n = start
        while n >= nmin:
            d = lt(n) - peak
            if d < -120:
                break
            s += mp.exp(d)
            n += step
    return peak + mp.log(s)


def z_of(rho, big_z):
    """z at saddle scale Z = (1+rho)/rho (rho z)^(1/(1+rho))."""
    rho, big_z = mp.mpf(rho), mp.mpf(big_z)
    return (rho * big_z / (1 + rho)) ** (1 + rho) / rho


def zgrid(rho):
    z1 = tt.ZA + tt.S1 * (1 + rho) ** 2
    return [
        1e-6,
        0.01,
        0.5,
        3.0,
        0.9 * z1,
        z1 * (1 - 1e-9),
        z1 * (1 + 1e-9),
        1.5 * z1,
        10 * z1,
        1e3,
        1e4,
        1e5,
    ]


def ref_logpdf(p, mu, phi, y):
    p, mu, phi, y = map(mp.mpf, (p, mu, phi, y))
    lam = mu ** (2 - p) / (phi * (2 - p))
    if y == 0:
        return -lam
    theta = phi * (p - 1) * mu ** (p - 1)
    rho = (2 - p) / (p - 1)
    u = mp.log(lam) + rho * (mp.log(y) - mp.log(theta))
    return -lam - y / theta - mp.log(y) + log_phi(rho, 0, mp.exp(u))


CASES = [
    (p, mu, phi, y)
    for p in (1.05, 1.08, 1.2, 1.5, 1.8, 1.95)
    for mu, phi in ((1.0, 1.0), (50.0, 0.05), (0.2, 3.0), (3.0, 0.3), (8.0, 1.7))
    for y in (0.0, 1e-6, 0.05, 0.9, 3.0, 20.0, 150.0)
    if y == 0 or y ** (2 - p) / (phi * (2 - p) * (p - 1)) <= 1e5
]


def _rel(got, ref):
    return abs(got - ref) / max(1.0, abs(ref))


def test_log_wright_bessel0_values():
    worst = 0.0
    for rho in RHOS:
        for big_z in zgrid(rho):
            z = float(z_of(rho, big_z))
            got = float(chebax.log_wright_bessel0(rho, z))
            assert np.isfinite(got), (rho, z, got)
            worst = max(worst, _rel(got, float(log_phi(rho, 0, z))))
    assert worst <= 3e-15, worst


def test_log_wright_bessel0_dz():
    # d ln W/dz = Phi(rho, rho; z) / Phi(rho, 0; z)
    worst = 0.0
    for rho in RHOS:
        for big_z in zgrid(rho):
            z = float(z_of(rho, big_z))
            ref = float(mp.exp(log_phi(rho, rho, z) - log_phi(rho, 0, z)))
            got = float(jax.grad(lambda zz: chebax.log_wright_bessel0(rho, zz))(z))
            assert np.isfinite(got), (rho, z, got)
            worst = max(worst, abs(got - ref) / abs(ref))
    assert worst <= 4e-14, worst


def test_log_wright_bessel0_drho():
    worst = 0.0
    for rho in (1 / 19, 0.25, 1.0, 4.0, 19.0 - 1e-9):
        for big_z in (0.5, 3.0, 1.5 * (tt.ZA + tt.S1 * (1 + rho) ** 2), 1e3, 1e4):
            z = float(z_of(rho, big_z))
            ref = float(mp.diff(lambda r: log_phi(r, 0, z), mp.mpf(rho)))
            got = float(jax.grad(lambda r: chebax.log_wright_bessel0(r, z))(rho))
            assert np.isfinite(got), (rho, z, got)
            worst = max(worst, _rel(got, ref))
    assert worst <= 3.5e-14, worst


def _branches(rho):
    rho = jnp.asarray(rho)
    z1, tail = _split(rho)
    lr = jnp.log(rho)
    u1 = (1 + rho) * jnp.log(rho * z1 / (1 + rho)) - lr

    def inner(u):
        return _log_series(rho, u)

    def outer(u):
        big_z = (1 + rho) / rho * jnp.exp((lr + u) / (1 + rho))
        return (
            big_z
            + 0.5 * jnp.log(rho * big_z / (1 + rho))
            - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log1p(rho)
            + tail(z1 / big_z) / big_z
        )

    return u1, inner, outer


def test_branches_agree_at_the_split():
    # the series and the tabulated saddle form, both at the split point
    for rho in np.exp(np.linspace(-np.log(19), np.log(19), 23)):
        u1, inner, outer = _branches(rho)
        vi, vo = float(inner(u1)), float(outer(u1))
        di, do = float(jax.grad(inner)(u1)), float(jax.grad(outer)(u1))
        assert abs(vi - vo) / max(1.0, abs(vi)) <= 2e-15, rho
        assert abs(di - do) / abs(di) <= 2e-14, rho


def test_table_limit_is_c1():
    # T -> c1(rho) as Z -> inf, in closed form: the end mpmath cannot reach
    for rho in np.exp(np.linspace(-np.log(19), np.log(19), 41)):
        _, tail = _split(jnp.asarray(rho))
        c1 = -(2 * rho + 1) * (rho + 2) / (24 * rho)
        assert abs(float(tail(0.0)) - c1) <= 4e-15, rho


def test_huge_arguments_follow_the_saddle():
    # far past any reference: ln W tracks its leading term to eps relative
    for rho, z in ((1 / 19, 1e300), (1.0, 1e300), (19.0, 1e300), (0.3, 1e150)):
        got = float(chebax.log_wright_bessel0(rho, z))
        t0 = (
            (rho * z) ** (1 / (1 + rho))
            if rho * z < 1e308
            else np.exp((np.log(rho) + np.log(z)) / (1 + rho))
        )
        big_z = (1 + rho) * t0 / rho
        lead = big_z + 0.5 * np.log(t0) - 0.5 * np.log(2 * np.pi * (1 + rho))
        assert np.isfinite(got) and abs(got - lead) / lead <= 1.5e-15, (rho, got, lead)


def test_log_wright_bessel0_edges():
    assert float(chebax.log_wright_bessel0(1.0, 0.0)) == -np.inf
    assert np.isnan(float(chebax.log_wright_bessel0(1.0, -1.0)))
    for rho in (1 / 19 * 0.999, 19.01, 0.0, -1.0):
        assert np.isnan(float(chebax.log_wright_bessel0(rho, 2.0))), rho
    # small z: Phi ~ z / Gamma(rho)
    z = 1e-250
    assert (
        abs(
            float(chebax.log_wright_bessel0(2.5, z))
            - (np.log(z) - float(mp.loggamma(2.5)))
        )
        <= 1e-13
    )


def test_tweedie_values():
    worst = 0.0
    for c in CASES:
        got = float(chebax.tweedie_logpdf(*c))
        assert np.isfinite(got), (c, got)
        worst = max(worst, _rel(got, float(ref_logpdf(*c))))
    assert worst <= 1e-12, worst


@pytest.mark.parametrize(
    "arg, bar", [(0, 1.5e-11), (1, 1.5e-14), (2, 6e-12), (3, 1e-12)]
)
def test_tweedie_gradients(arg, bar):
    worst = 0.0
    for c in CASES:
        if c[3] not in (0.0, 0.05, 3.0, 150.0) or c[1] not in (1.0, 50.0, 3.0):
            continue
        if arg == 3 and c[3] == 0.0:
            continue
        got = float(jax.grad(chebax.tweedie_logpdf, argnums=arg)(*map(float, c)))
        assert np.isfinite(got), (c, got)
        args = list(map(mp.mpf, c))
        ref = float(
            mp.diff(
                lambda v: ref_logpdf(*(args[:arg] + [v] + args[arg + 1 :])), args[arg]
            )
        )
        worst = max(worst, _rel(got, ref))
    assert worst <= bar, worst


def _more_points():
    """chebtorch's fixture points beyond CASES (its tools/export.py, same
    random stream): just below the split at small p, where the density and
    its gradients are worst and CASES does not reach; random points there,
    since the worst errors are rounding spikes a grid samples poorly; the
    spikes chebtorch's parity draws found; random points over the box.
    graded starts from the CASES points test_tweedie_gradients takes, so
    the count, and with it every point, matches chebtorch's."""
    graded = {
        c for c in CASES if c[3] in (0.0, 0.05, 3.0, 150.0) and c[1] in (1.0, 50.0, 3.0)
    }
    for p in (1.05, 1.06, 1.08, 1.15):
        rho = (2 - p) / (p - 1)
        z1 = tt.ZA + tt.S1 * (1 + rho) ** 2
        for phi in (0.01, 0.07, 0.3, 1.5):
            for frac in (0.5, 0.8, 0.9, 0.97):
                y = (frac * z1 * phi * (2 - p) * (p - 1)) ** (1 / (2 - p))
                graded |= {(p, y * r, phi, y) for r in (0.6, 1.2, 1.6)}
    rng = random.Random(0)
    for _ in range(200):
        p = rng.uniform(1.05, 1.1)
        rho = (2 - p) / (p - 1)
        z1 = tt.ZA + tt.S1 * (1 + rho) ** 2
        phi = math.exp(rng.uniform(math.log(0.01), math.log(2.0)))
        y = (rng.uniform(0.6, 1.0) * z1 * phi * (2 - p) * (p - 1)) ** (1 / (2 - p))
        graded.add((p, y * math.exp(rng.uniform(-0.5, 0.5)), phi, y))
    graded |= {
        (1.05, 0.7322062443874073, 0.020028698807062692, 0.6674935636203755),
        (1.05, 14.948103612343962, 0.40139605555805025, 14.238663807946466),
        (1.07, 12.087159758573403, 0.46620545614301584, 11.639439755817165),
        (1.05, 0.8697386997688737, 0.02401901878639002, 0.7764388225849072),
        (1.05, 42.37604975000736, 0.020616521318877278, 41.93810466188303),
    }
    while len(graded) < 650:
        p = rng.uniform(1.05, 1.95)
        mu, phi = (
            math.exp(rng.uniform(math.log(lo), math.log(hi)))
            for lo, hi in ((0.05, 50.0), (0.02, 5.0))
        )
        y = mu * math.exp(rng.uniform(-4.0, 3.0))
        if y ** (2 - p) / (phi * (2 - p) * (p - 1)) <= 1e5:
            graded.add((p, mu, phi, y))
    return sorted(graded - set(CASES))


MORE = _more_points()


def _values_and_log_gradients(points, dtype):
    """Values and (d/dp, d/dln mu, d/dln phi, d/dln y) as float64 arrays."""
    cols = [jnp.asarray([c[k] for c in points], dtype) for k in range(4)]
    f = chebax.tweedie_logpdf
    v = jax.jit(jax.vmap(f))(*cols)
    g = jax.jit(jax.vmap(jax.grad(f, argnums=(0, 1, 2, 3))))(*cols)
    g = [g[0]] + [x * gk for x, gk in zip(cols[1:], g[1:])]
    return np.asarray(v, np.float64), np.asarray(g, np.float64)


def test_tweedie_below_the_split():
    # Gradients in the logarithms of mu, phi and y: df/dphi is 1/phi times
    # a cancellation that does not shrink with phi, so a plain metric grows
    # as 1/phi near a zero of it. Order: density, d/dp, d/dln mu, d/dln phi,
    # d/dln y.
    v, g = _values_and_log_gradients(MORE, jnp.float64)
    assert np.isfinite(v).all() and np.isfinite(g).all()
    worst = np.zeros(5)
    for i, c in enumerate(MORE):
        args = list(map(mp.mpf, c))
        errs = [_rel(v[i], float(ref_logpdf(*c)))]
        for k in range(4):
            ref = mp.diff(
                lambda t: ref_logpdf(*(args[:k] + [t] + args[k + 1 :])), args[k]
            )
            s = 1.0 if k == 0 else c[k]
            errs.append(_rel(g[k, i], s * float(ref)))
        worst = np.maximum(worst, errs)
    assert (worst <= [4e-12, 2e-10, 1.5e-15, 4e-11, 3e-11]).all(), worst


# float32 bars by band of p, in the order above
BARS_X64_OFF = {
    (1.0, 1.1): [2e-3, 7e-2, 7e-7, 2e-2, 2e-2],
    (1.1, 1.3): [1.5e-4, 4e-3, 1e-6, 5e-4, 2.5e-4],
    (1.3, 2.0): [2.5e-5, 1.2e-4, 8e-7, 2.5e-5, 1.2e-5],
}


def test_tweedie_below_the_split_x64_off(tmp_path):
    # float32 against float64 at the same float32 inputs, which separates
    # the arithmetic from the rounding of the inputs: at p = 1.05 df/dp is in
    # the thousands, so rounding p alone moves the density by ~1e-4. p is
    # nudged into the box where it rounds out of it.
    pts = []
    for c in MORE:
        p = np.float32(c[0])
        if not tt.P_LO <= float(p) <= tt.P_HI:
            p = np.nextafter(p, np.float32(1.5))
        pts.append((float(p), *(float(np.float32(x)) for x in c[1:])))
    v, g = _values_and_log_gradients(pts, jnp.float64)
    ref = np.vstack([v, g])
    np.save(tmp_path / "pts.npy", np.asarray(pts))
    code = (
        "import sys, numpy as np, jax.numpy as jnp\n"
        "assert jnp.empty(()).dtype == jnp.float32\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from test_tweedie import _values_and_log_gradients\n"
        "pts = [tuple(r) for r in np.load(sys.argv[1])]\n"
        "v, g = _values_and_log_gradients(pts, jnp.float32)\n"
        "np.save(sys.argv[1], np.vstack([v, g]))\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "pts.npy"), str(HERE)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-1500:]
    got = np.load(tmp_path / "pts.npy")
    assert np.isfinite(got).all()
    err = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
    p = np.asarray(pts)[:, 0]
    for (lo, hi), bars in BARS_X64_OFF.items():
        worst = err[:, (p >= lo) & (p < hi)].max(axis=1)
        assert (worst <= bars).all(), (lo, hi, worst)


def test_half_deviance_against_mpmath():
    # d/2 and its y-partial in each form, relative error. The density's
    # tests cannot see a form's error under the direct sum's: XLA's float64
    # log1p, up to 128 ulp off below the mean, put 1.5e-12 into the mid form
    # at a = 0.95 and no density test moved. Measured worst (value,
    # y-partial): near 4.0e-16, 9.0e-16; mid 7.6e-14 (a = 0.95), 9.7e-16;
    # far 1.3e-14, 2.7e-15.
    val = jax.jit(jax.vmap(_half_deviance, in_axes=(None, 0, 0)))
    grad = jax.jit(jax.vmap(jax.grad(_half_deviance, argnums=2), in_axes=(None, 0, 0)))
    bars = {"near": (1.6e-15, 4e-15), "mid": (3e-13, 4e-15), "far": (5e-14, 1.1e-14)}
    worst = {lane: [0.0, 0.0] for lane in bars}
    rng = np.random.default_rng(3)
    for a in (0.05, 0.5, 0.95):
        for lane, lo, hi in (
            ("near", -0.25, 0.25),
            ("mid", math.exp(-1) - 1, -0.25),
            ("mid", 0.25, math.e - 1),
            ("far", -0.999, math.exp(-1) - 1),
            ("far", math.e - 1, 50.0),
        ):
            mu = np.exp(rng.uniform(-10, 10, 400))
            y = mu * (1 + rng.uniform(lo, hi, 400))
            h, h_y = np.asarray(val(a, mu, y)), np.asarray(grad(a, mu, y))
            s = mp.mpf(a)
            for m, v, hv, gv in zip(mu, y, h, h_y):
                m, v = mp.mpf(m), mp.mpf(v)
                d = (v - m) / m
                ref = m**s * ((1 + d) ** s - 1 - s * d) / (s * (s - 1))
                ref_y = (v ** (s - 1) - m ** (s - 1)) / (s - 1)
                w = worst[lane]
                w[0] = max(w[0], float(abs(hv - ref) / abs(ref)))
                w[1] = max(w[1], float(abs(gv - ref_y) / abs(ref_y)))
    for lane, (bar_h, bar_y) in bars.items():
        assert worst[lane][0] <= bar_h and worst[lane][1] <= bar_y, (lane, worst)


def test_deviance_partials_match_finite_differences():
    # the mu and y partials of the half-deviance are written by hand, and so
    # is _over's in phi: each against central differences to second order,
    # in each of the deviance's forms (near, mid and far, both sides)
    hd, over = jax.jit(_half_deviance), jax.jit(_over)
    for a in (0.05, 0.5, 0.95):
        for y in (2.3, 1.6, 4.5, 0.9, 30.0, 0.1):
            args = tuple(map(jnp.asarray, (a, 2.0, y)))
            check_grads(hd, args, order=2, modes=("fwd", "rev"))
    for h, phi in ((0.0, 0.3), (2.5, 0.05), (-1.0, 40.0)):
        check_grads(over, tuple(map(jnp.asarray, (h, phi))), order=2)


def test_tweedie_normalizes_with_its_moments():
    # P(0) + int f = 1, E Y = mu, Var Y = phi mu^p: the whole density at once.
    # Integrated in v = ln y: near 0 the density is ~ y^(rho - 1), whose mass
    # spreads over hundreds of decades when rho is small (p near 2), and in v
    # that end is a plain exponential e^(rho v).
    from scipy import integrate

    for p, mu, phi in (
        (1.1, 1.0, 1.0),
        (1.5, 2.0, 0.5),
        (1.9, 0.5, 2.0),
        (1.3, 5.0, 0.2),
    ):
        logf = jax.jit(lambda y: chebax.tweedie_logpdf(p, mu, phi, y))
        sd = np.sqrt(phi * mu**p)
        vhi = np.log(mu + 60 * sd)

        def g(v, k):
            return float(np.exp(logf(np.exp(v)) + (k + 1) * v))

        p0 = float(np.exp(logf(0.0)))
        moments = [
            integrate.quad(
                g,
                -745.0,
                vhi,
                args=(k,),
                points=[np.log(mu)],
                epsabs=1e-14,
                epsrel=1e-13,
                limit=400,
            )[0]
            for k in (0, 1, 2)
        ]
        assert abs(p0 + moments[0] - 1) <= 1e-9, (p, moments)
        assert abs(moments[1] - mu) / mu <= 1e-9, (p, moments)
        var = moments[2] - moments[1] ** 2
        assert abs(var - phi * mu**p) / (phi * mu**p) <= 1e-8, (p, moments)


def test_tweedie_zero_mass_and_invalid():
    p, mu, phi = 1.4, 2.0, 0.7
    lam = mu ** (2 - p) / (phi * (2 - p))
    assert float(chebax.tweedie_logpdf(p, mu, phi, 0.0)) == pytest.approx(
        -lam, rel=1e-15
    )
    for bad in (
        (1.0499, mu, phi, 1.0),
        (1.9501, mu, phi, 1.0),
        (p, 0.0, phi, 1.0),
        (p, mu, -1.0, 1.0),
        (p, mu, phi, -0.5),
    ):
        assert np.isnan(float(chebax.tweedie_logpdf(*bad))), bad


def test_tweedie_broadcast_and_parameter_shape():
    mu = jnp.asarray([0.5, 1.0, 2.0, 4.0])
    y = jnp.asarray([0.0, 0.3, 2.0, 9.0])
    got = np.asarray(chebax.tweedie_logpdf(1.6, mu, 0.8, y))
    ref = [
        float(chebax.tweedie_logpdf(1.6, float(m), 0.8, float(v)))
        for m, v in zip(mu, y)
    ]
    # a batch rounds its transcendentals differently from scalars (XLA
    # vectorizes them), and the O(Z) terms that cancel here are ~7: a few
    # of their ulps, measured 1.8e-15
    np.testing.assert_allclose(got, ref, rtol=0, atol=5e-15)
    np.testing.assert_allclose(
        np.asarray(chebax.tweedie_logpdf(jnp.asarray([1.6]), mu, 0.8, y)),
        got,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="pergroup"):
        chebax.tweedie_logpdf(jnp.asarray([1.3, 1.6]), 1.0, 1.0, 1.0)


def test_tweedie_traced_and_mixed_batch_gradients():
    y = jnp.asarray([0.0, 1e-8, 0.4, 3.0, 40.0])
    mu = jnp.full_like(y, 2.0)

    def total(p, m, phi):
        return jnp.sum(chebax.tweedie_logpdf(p, m, phi, y))

    eager = float(total(1.3, mu, 0.9))
    jitted = float(jax.jit(total)(1.3, mu, 0.9))
    assert eager == pytest.approx(jitted, rel=1e-14)
    gp, gm, gphi = jax.grad(total, argnums=(0, 1, 2))(1.3, mu, 0.9)
    assert np.isfinite(float(gp)) and np.isfinite(float(gphi))
    assert np.all(np.isfinite(np.asarray(gm)))
    # the y = 0 lane is -lambda: d/dmu = -mu^(1-p)/phi exactly
    assert float(gm[0]) == pytest.approx(-(2.0 ** (1 - 1.3)) / 0.9, rel=1e-14)
    gy = np.asarray(
        jax.vmap(jax.grad(lambda v: chebax.tweedie_logpdf(1.3, 2.0, 0.9, v)))(y[1:])
    )
    assert np.all(np.isfinite(gy))


def test_x64_off():
    # float32 end to end, in a child process (conftest forces x64 on here)
    pts_w = [
        (r, z)
        for r in (1 / 19, 0.25, 1.0, 4.0, 19.0)
        for z in (1e-3, 0.7, 30.0, 1e4, 1e9, 1e30)
    ]
    pts_f = [
        (p, mu, phi, y)
        for p in (1.05, 1.2, 1.5, 1.8, 1.95)
        for mu, phi in ((1.0, 1.0), (3.0, 0.3), (0.2, 3.0))
        for y in (0.0, 0.05, 1.0, 6.0)
    ]
    ref_w = [float(chebax.log_wright_bessel0(r, z)) for r, z in pts_w]
    ref_f = [float(chebax.tweedie_logpdf(*c)) for c in pts_f]
    code = (
        "import jax, jax.numpy as jnp, chebax\n"
        "assert jnp.empty(()).dtype == jnp.float32\n"
        f"ref_w, ref_f = {ref_w!r}, {ref_f!r}\n"
        f"for (r, z), ref in zip({pts_w!r}, ref_w):\n"
        "    got = float(chebax.log_wright_bessel0(r, jnp.float32(z)))\n"
        "    assert abs(got - ref) / max(1.0, abs(ref)) <= 3e-6, (r, z, got, ref)\n"
        f"for c, ref in zip({pts_f!r}, ref_f):\n"
        "    got = float(chebax.tweedie_logpdf(*c))\n"
        "    assert abs(got - ref) / max(1.0, abs(ref)) <= 3e-5, (c, got, ref)\n"
        "print('ok')\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-1500:]


def _ref_p15(mu, phi, y):
    """ln f at p = 1.5 from Phi(1, 0; z) = sqrt(z) I1(2 sqrt(z)), 30 digits
    above the working precision (70 at the module's 40): the density's
    large terms cancel, and here they reach 1e16. Relative, so mp.diff's
    own extra precision still reaches it."""
    with mp.extradps(30):
        mu, phi, y = map(mp.mpf, (mu, phi, y))
        lam = 2 * mp.sqrt(mu) / phi
        y_theta = 2 * y / (phi * mp.sqrt(mu))
        z = lam * y_theta
        return (
            -lam
            - y_theta
            - mp.log(y)
            + mp.log(mp.sqrt(z) * mp.besseli(1, 2 * mp.sqrt(z)))
        )


# (mu, phi, y) at p = 1.5, saddle scale Z from 4e3 to 4e15
CONCENTRATED = [
    (1e6, 1e-8, 1000010.0),
    (1e6, 1e-12, 1000000.125),
    (1.0, 1e-10, 1.0 + 2.0**-20),
    (5.0, 1e-9, 4.0),
    (2.0, 1e-7, 2.0),
    (1e-3, 1e-9, 1.2e-3),
    (1.0, 1e-6, 1.001),
]


def test_concentrated_densities():
    # Far past what the series reference reaches. The deviance near the
    # mean came from ln y - ln mu and a difference of two expm1 terms, which
    # lost the relative displacement: float32 gave 2.53 for -5.28 at the
    # second case (review, 2026-10-26). Metric as above; measured worst
    # 2.7e-16 in values and 1.6e-13 in d/dmu, d/dphi, d/dy, bars 1e-15 and
    # 6e-13.
    worst_v = worst_g = 0.0
    for c in CONCENTRATED:
        got = float(chebax.tweedie_logpdf(1.5, *c))
        g = jax.grad(chebax.tweedie_logpdf, argnums=(1, 2, 3))(1.5, *c)
        assert np.isfinite([got, *map(float, g)]).all(), (c, got, g)
        worst_v = max(worst_v, _rel(got, float(_ref_p15(*c))))
        for k in range(3):
            ref = mp.diff(
                lambda v: _ref_p15(*(c[:k] + (v,) + c[k + 1 :])), mp.mpf(c[k])
            )
            worst_g = max(worst_g, _rel(float(g[k]), float(ref)))
    assert worst_v <= 1e-15 and worst_g <= 6e-13, (worst_v, worst_g)


def test_concentrated_x64_off():
    # float32 inputs exact except phi; 1.001 is left out, since rounding it
    # moves this density by 8e-6. Measured worst 1.9e-7, bar 1e-6.
    cases = CONCENTRATED[:-1]
    refs = [float(_ref_p15(*c)) for c in cases]
    code = (
        "import jax.numpy as jnp, chebax\n"
        "assert jnp.empty(()).dtype == jnp.float32\n"
        f"for c, ref in zip({cases!r}, {refs!r}):\n"
        "    got = float(chebax.tweedie_logpdf(1.5, *c))\n"
        "    assert abs(got - ref) / max(1.0, abs(ref)) <= 1e-6, (c, got, ref)\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-1500:]


def test_extreme_deviance_scales():
    # y/mu overflowed at y = 1e10, mu = 1e-300 under a finite density; at
    # p = 1.5 the half-deviance is 2y/sqrt(mu) - 4 sqrt(y) + 2 sqrt(mu),
    # and the rest of ln f is below its last digit
    got = float(chebax.tweedie_logpdf(1.5, 1e-300, 1.0, 1e10))
    assert abs(got / -2e160 - 1) <= 1e-15, got
    # phi = 1e-200 at the mean: ln f = -ln(2 pi phi)/2 + O(1/Z), and the
    # p and phi gradients were nan, from unselected branches whose partials
    # overflow there and from jax squaring phi in d(d/phi)/dphi
    gp, gmu, gphi, gy = map(
        float,
        jax.grad(chebax.tweedie_logpdf, argnums=(0, 1, 2, 3))(1.5, 1.0, 1e-200, 1.0),
    )
    assert abs(gp) <= 1e-150 and gmu == 0.0, (gp, gmu)
    assert gphi == pytest.approx(-0.5e200, rel=1e-15)
    assert gy == pytest.approx(-0.75, rel=1e-15)
    v = float(chebax.tweedie_logpdf(1.5, 1.0, 1e-200, 1.0))
    assert v == pytest.approx(-0.5 * np.log(2 * np.pi * 1e-200), rel=1e-15)


def test_mean_gradient_at_extreme_scales():
    # Tweedie is an exponential dispersion family, so d ln f / dmu is
    # exactly (y - mu) / (phi mu^p), zero at y = mu. AD of the deviance
    # squared mu, which overflows at mu = 1e-220: (1.5, 1e-220, 1e-150,
    # 1e-220) had a nan mean gradient under a finite density. The series
    # branch cancelled two mu^(1-p)/phi terms to rounding instead (1e286 at
    # mu = 1e-300 where the answer is 0), and lost y/theta's share to
    # underflow at phi = 1e200 (review follow-up, 2026-10-26). Measured
    # worst 3.2e-16 over 737 such points, bar 1.5e-15.
    cases = [
        (p, mu, phi, mu * ry)
        for p in (1.05, 1.5, 1.95)
        for mu, phi in (
            (1e-300, 1e-150),
            (1e-300, 1e200),
            (1e-220, 1e-150),
            (1e-220, 1e-200),
            (1e100, 1e-10),
            (1e250, 1e10),
        )
        for ry in (1.0, 1.001, 0.7, 20.0)
    ]
    grad = jax.jit(jax.grad(chebax.tweedie_logpdf, argnums=(0, 1, 2, 3)))
    n = 0
    for c in cases:
        if not np.isfinite(float(chebax.tweedie_logpdf(*c))):
            continue
        n += 1
        g = list(map(float, grad(*c)))
        assert not np.isnan(g).any(), (c, g)
        p, mu, phi, y = map(mp.mpf, c)
        ref = (y - mu) / (phi * mu**p)
        if ref == 0:
            assert g[1] == 0.0, (c, g[1])
        elif abs(ref) > np.finfo(np.float64).max:  # overflows, signed
            assert g[1] == (np.inf if ref > 0 else -np.inf), (c, g[1])
        else:
            assert abs(g[1] - ref) <= 1.5e-15 * abs(ref), (c, g[1], float(ref))
    assert n >= 60, n


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    tweedie_gen.main(tmp_path)
    assert (tmp_path / "tweedie_table.py").read_text() == pathlib.Path(
        tt.__file__
    ).read_text()
