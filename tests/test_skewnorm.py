"""Skew-normal log CDF and log survival function: log_skewnorm_cdf(a, x) and
log_skewnorm_sf(a, x), a and x broadcast.

References are mpmath at 40 dps from K(s, b) = F(-s; b) = (1/pi) int_0^phi0
exp(-s^2 (cot^2 phi + 1)/2) dphi, phi0 = acot(b), with e^(-sigma^2) taken
out and break points at the peak's scale: a different integral from both of
skewnorm_gen's charts. F and 1 - F are each a sum of positive terms (or
erfc - K, K <= erfc/2) of K, erf and erfc, as in the runtime, and the
reference ln F is ln F or log1p(-(1 - F)), whichever reads the smaller.

Metrics: |ln F - ref| / max(1, |ref|), relative in F where |ln F| <= 1 and in
ln F beyond, as for gammainc_array's logs; and the plain relative
|ln F - ref| / |ref|, which near F = 1 is relative in 1 - F. The grid crosses
all four sign cases with a from 1e-8 to 1e8 and x from 1e-12 to 1e3, adds both
sides of every patch cut and of the seam sigma = S1, mirrored, and 400
log-uniform random points (1,047 in all).

Measured worst, float64, jax 0.11 and 0.5.3 (CI's floor) alike: ln F 7.8e-16
(bar 3e-15); relative 9.3e-14 (bar 4e-13), at x = 32 where 1 - F = e^-512, from
rounding x^2 in e^(-x^2/2) (log_skewnorm_sf holds that complement to the first
metric). Slopes, both AD modes against the closed forms: 7.7e-15 (bar 4e-14),
at a = 1e8 where the slope is 6e17, assembled from logs; relative 7.7e-14 (bar
3e-13), only where a slope is itself e^(-x^2/2)-small. Second derivatives (AD
of the rule) 2.9e-14, bar 1.2e-13, at a = 10, x = -1 in the tail chart, whose
erfcx they differentiate through the table. float32 (6 x 6 blocks) at the
float32-rounded points: ln F 4.4e-7 (bar 1.8e-6), slopes relative 5.2e-6 (bar
3e-5).
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import skewnorm_gen
from chebax._src.recipes import skewnorm_table as t

mp = pytest.importorskip("mpmath")

FNS = (chebax.log_skewnorm_cdf, chebax.log_skewnorm_sf)


def _k_ref(s, b):
    """(ln K(s, b), K's exponent sigma^2), s, b >= 0 as mpf."""
    if s == 0:
        return mp.log(mp.acot(b) / mp.pi), mp.mpf(0)
    sig2 = s * s * (1 + b * b) / 2
    phi0 = mp.acot(b)
    # exponent beyond e^(-sigma^2): (s^2/2)(cot^2 phi - b^2), steepest at phi0
    width = 1 / (s * s * b * (1 + b * b) + s)
    pts = sorted({phi0 - width * k for k in (1, 4, 16, 64, 256) if width * k < phi0})
    g = lambda p: mp.exp(-s * s * (mp.cot(p) ** 2 - b * b) / 2)
    return mp.log(mp.quad(g, [mp.mpf(0)] + pts + [phi0]) / mp.pi), sig2


def _ref(a, x):
    """(ln F, F, 1 - F) at float a, x; F and 1 - F as mpf."""
    a, x = mp.mpf(a), mp.mpf(x)
    s, b = abs(x), abs(a)
    lks, sig2 = _k_ref(s, b)
    k = mp.exp(lks - sig2)
    e, p = mp.erf(s / mp.sqrt(2)), mp.erfc(s / mp.sqrt(2))
    if x <= 0:
        f, sf = (k, 1 - k) if a >= 0 else (p - k, e + k)
    else:
        f, sf = (e + k, p - k) if a >= 0 else (1 - k, k)
    return (mp.log(f) if f <= sf else mp.log1p(-sf)), f, sf


def _d_ref(a, x, f):
    """(d ln F/dx, d ln F/da) from the closed forms."""
    a, x = mp.mpf(a), mp.mpf(x)
    dens = 2 * mp.npdf(x) * mp.ncdf(a * x)
    dfa = -mp.exp(-x * x * (1 + a * a) / 2) / (mp.pi * (1 + a * a))
    return dens / f, dfa / f


def _seam_points():
    """Both sides of every patch cut and of sigma = S1, at a few positions
    along the other axis, as (a, x) with x <= 0 <= a (the K lanes)."""
    out = []
    side = (1 - 1e-12, 1 + 1e-12)
    nb, nt_ = t.NPAR[0] + 1, t.NARG[0] + 1
    for d in t.PCUTS[0][:nb]:  # bulk delta cuts
        for sg in (0.1, 0.6 * t.S1, 0.99 * t.S1):
            for r in side:
                dl = min(d * r, 1.0)
                b = dl / np.sqrt(1 - dl * dl) if dl < 1 else 1e8
                out.append((b, -sg * np.sqrt(2) / np.hypot(1, b)))
    for sg in t.YCUTS[0][:nt_]:  # bulk sigma cuts, S1 among them
        for b in (0.0, 0.4, 3.0, 50.0):
            for r in side:
                out.append((b, -sg * r * np.sqrt(2) / np.hypot(1, b)))
    for e in t.PCUTS[1][: t.NPAR[1] + 1]:  # tail eps cuts
        for mu_frac in (0.0, 0.5, 0.95):
            for r in side:
                if e == 0:
                    continue
                sg = 1 / np.sqrt(e * r)
                mu = mu_frac * sg
                s = np.sqrt(2 * (sg * sg - mu * mu))
                out.append((np.sqrt(2) * mu / s, -s))
    for tc in t.YCUTS[1][: t.NARG[1] + 1]:  # tail t cuts
        for sg in (t.S1 * 1.01, 10.0, 300.0):
            for r in side:
                if tc in (0.0, 1.0):
                    continue
                mu = (1 - tc * r) / (tc * r)
                if mu >= sg:
                    continue
                s = np.sqrt(2 * (sg * sg - mu * mu))
                out.append((np.sqrt(2) * mu / s, -s))
    return np.array(out)


def _points():
    av = [0.0, 1e-8, 0.1, 0.5, 1.0, 2.0, 3.7, 10.0, 40.0, 1e3, 1e8]
    xv = [0.0, 1e-12, 1e-3, 0.2, 0.7, 1.4, 2.5, 4.0, 7.0, 12.0, 25.0, 60.0]
    xv += [200.0, 1e3]
    av = sorted({s * v for v in av for s in (1, -1)})
    xv = sorted({s * v for v in xv for s in (1, -1)})
    grid = np.array([(a, x) for a in av for x in xv])
    seams = _seam_points()
    seams = np.concatenate([seams, -seams])  # the mirrored lanes too
    rng = np.random.default_rng(49)
    n = 400
    rand = np.stack(
        [
            rng.choice([-1, 1], n) * 10.0 ** rng.uniform(-3, 3, n),
            rng.choice([-1, 1], n) * 10.0 ** rng.uniform(-3, 2.5, n),
        ],
        axis=1,
    )
    return np.concatenate([grid, seams, rand])


@pytest.fixture(scope="module")
def refs():
    pts = _points()
    a, x = pts[:, 0], pts[:, 1]
    with mp.workdps(40):
        rows = [_ref(ai, xi) for ai, xi in zip(a, x)]
        lf = np.array([float(r[0]) for r in rows])
        d = [_d_ref(ai, xi, r[1]) for ai, xi, r in zip(a, x, rows)]
    dx = np.array([float(v[0]) for v in d])
    da = np.array([float(v[1]) for v in d])
    return a, x, lf, dx, da


def _rel(got, ref, tiny=np.finfo(np.float64).tiny):
    """|got - ref| / |ref|; where ref is below the smallest normal (ln F
    once 1 - F is subnormal, which XLA flushes to 0) got must be too."""
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    assert np.all(np.isfinite(got)), np.flatnonzero(~np.isfinite(got))[:10]
    sub = np.abs(ref) < tiny
    rel = np.abs(got - ref) / np.where(sub, 1.0, np.abs(ref))
    return np.where(sub, np.where(np.abs(got) < tiny, 0.0, np.inf), rel)


def _log_err(got, ref):
    got = np.asarray(got, np.float64)
    assert np.all(np.isfinite(got)), np.flatnonzero(~np.isfinite(got))[:10]
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


def test_log_cdf_against_mpmath(refs):
    a, x, lf = refs[:3]
    got = jax.jit(chebax.log_skewnorm_cdf)(a, x)
    for name, err, bar in (
        ("max(1, |ref|)", _log_err(got, lf), BAR_LOGF),
        ("relative", _rel(got, lf), BAR_LOGF_REL),
    ):
        w = int(np.argmax(err))
        print(f"ln F {name} worst {err[w]:.3g} at a={a[w]!r}, x={x[w]!r}")
        assert err[w] <= bar, name


def test_log_sf_is_the_mirrored_cdf(refs):
    a, x, lf = refs[:3]
    got = jax.jit(chebax.log_skewnorm_sf)(-a, -x)
    assert np.max(_log_err(got, lf)) <= BAR_LOGF
    assert np.max(_rel(got, lf)) <= BAR_LOGF_REL


def test_derivatives_against_closed_forms(refs):
    """Both AD modes against dF/dx = 2 phi(x) Phi(a x) and dF/da =
    -e^(-x^2 (1+a^2)/2)/(pi (1+a^2)), each over F."""
    a, x, _, dx, da = refs
    f = chebax.log_skewnorm_cdf
    ga, gx = jax.jit(jax.grad(lambda a, x: jnp.sum(f(a, x)), (0, 1)))(a, x)
    one, zero = jnp.ones_like(a), jnp.zeros_like(a)
    fa = jax.jvp(f, (a, x), (one, zero))[1]
    fx = jax.jvp(f, (a, x), (zero, one))[1]
    for got, ref, name in (
        (gx, dx, "rev x"),
        (fx, dx, "fwd x"),
        (ga, da, "rev a"),
        (fa, da, "fwd a"),
    ):
        err = _log_err(got, ref)
        rel = _rel(got, ref)
        print(f"{name} worst {np.max(err):.3g}, relative {np.max(rel):.3g}")
        assert np.max(err) <= BAR_D, name
        assert np.max(rel) <= BAR_D_REL, name


def _d2_ref(a, x, f):
    """(d2 ln F/da2, d2 ln F/dadx, d2 ln F/dx2): F_ij/F - F_i F_j/F^2 with
    F_a = g = -e^(-x^2 (1+a^2)/2)/(pi (1+a^2)), F_x = 2 phi(x) Phi(a x),
    F_aa = -g a (x^2 + 2/(1+a^2)), F_ax = -g x (1+a^2),
    F_xx = 2 phi(x) (a phi(a x) - x Phi(a x))."""
    a, x = mp.mpf(a), mp.mpf(x)
    g = -mp.exp(-x * x * (1 + a * a) / 2) / (mp.pi * (1 + a * a))
    fx = 2 * mp.npdf(x) * mp.ncdf(a * x)
    faa = -g * a * (x * x + 2 / (1 + a * a))
    fax = -g * x * (1 + a * a)
    fxx = 2 * mp.npdf(x) * (a * mp.npdf(a * x) - x * mp.ncdf(a * x))
    return (faa / f - (g / f) ** 2, fax / f - g * fx / f**2, fxx / f - (fx / f) ** 2)


def test_second_derivatives():
    """d2/da2, d2/dadx and d2/dx2, AD of the rule, against closed forms,
    |err| / max(1, |ref|)."""
    pts = [(3.0, -2.0), (-1.5, -0.4), (0.7, 0.9), (-4.0, 3.0), (10.0, -1.0)]
    pts += [(0.0, 0.0), (2.0, 1e-3), (-30.0, 0.2), (1.0, -8.0), (0.3, -12.0)]
    pts += [(-0.3, 12.0), (5.0, 0.01), (-5.0, -0.01)]
    f = chebax.log_skewnorm_cdf
    hess = jax.jit(jax.hessian(lambda a, x: f(a, x), (0, 1)))
    worst = 0.0
    with mp.workdps(40):
        for a, x in pts:
            ref = _d2_ref(a, x, _ref(a, x)[1])
            h = hess(a, x)
            for g, r in zip([h[0][0], h[0][1], h[1][1]], ref):
                assert np.isfinite(float(g)), (a, x)
                err = abs(float(g) - float(r)) / max(1.0, abs(float(r)))
                worst = max(worst, err)
    print(f"second derivatives worst {worst:.3g}")
    assert worst <= BAR_D2


def test_normal_and_half_normal_limits():
    """a = 0 is the normal CDF, a = +-inf the half-normal ones."""
    lf = chebax.log_skewnorm_cdf
    worst = 0.0
    with mp.workdps(40):
        for x in (-30.0, -3.0, -0.5, 0.0, 0.5, 3.0, 30.0):
            y = abs(mp.mpf(x)) / mp.sqrt(2)
            lo = x < 0
            # ln Phi(x); ln F(x; inf) = ln erf(y) above 0; ln F(x; -inf) = ln erfc(y) below
            norm = mp.log(mp.erfc(y) / 2) if lo else mp.log1p(-mp.erfc(y) / 2)
            assert float(lf(-np.inf, x)) == 0.0 or lo
            assert float(lf(np.inf, x)) == -np.inf or x > 0
            refs = [(0.0, norm)]
            refs += [(np.inf, mp.log1p(-mp.erfc(y)))] if x > 0 else []
            refs += [(-np.inf, mp.log(mp.erfc(y)))] if lo else []
            for a, ref in refs:
                err = abs(float(lf(a, x)) - float(ref)) / max(1.0, abs(float(ref)))
                worst = max(worst, err)
    print(f"limits worst {worst:.3g}")
    assert worst <= BAR_LOGF


def test_endpoints_and_nan():
    lf = chebax.log_skewnorm_cdf
    assert float(lf(2.0, -jnp.inf)) == -np.inf
    assert float(lf(-2.0, jnp.inf)) == 0.0
    assert float(lf(2.0, -1e200)) == -np.inf
    assert float(lf(2.0, 1e200)) == 0.0
    assert np.isnan(float(lf(jnp.nan, 1.0)))
    assert np.isnan(float(lf(1.0, jnp.nan)))
    # F(0; a) = 1/2 - atan(a)/pi = acot(a)/pi for a > 0, 1 - acot(-a)/pi below
    with mp.workdps(40):
        for a in (-1e300, -3.0, 0.0, 0.5, 1e300):
            am = mp.mpf(a)
            ref = float(
                mp.log(mp.acot(am) / mp.pi)
                if a > 0
                else mp.log1p(-mp.acot(-am) / mp.pi)
            )
            assert abs(float(lf(a, 0.0)) - ref) <= BAR_LOGF * max(1.0, abs(ref)), a


@pytest.mark.parametrize("fn", FNS, ids=lambda f: f.__name__)
def test_broadcast_jit_and_arguments(fn):
    a = jnp.asarray([[-2.0], [0.5], [7.0]])
    x = jnp.asarray([-3.0, -0.1, 0.4, 5.0])
    got = fn(a, x)
    assert got.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(jax.jit(fn)(a, x)), np.asarray(got))
    assert float(fn(7.0, 5.0)) == float(got[2, 3])
    assert float(fn(a=2, x=-3)) == float(fn(2.0, -3.0))
    assert float(fn(np.int64(2), np.float32(-3.0))) == float(fn(2.0, -3.0))
    g = jax.grad(lambda a, x: fn(a, x), (0, 1))(2.5, -3.0)
    gk = jax.grad(lambda a, x: fn(x=x, a=a), (0, 1))(2.5, -3.0)
    assert list(map(float, g)) == list(map(float, gk))


def test_extreme_inputs_stay_finite():
    """Values and both slopes at the ends of the float64 range: no nan, and
    ln F finite wherever F is a normal number."""
    big = [1e-300, 1e-8, 1.0, 1e8, 1e150, 1e300, 1.7e308]
    av = np.array([s * v for v in [0.0] + big for s in (1, -1)])
    xv = np.array(
        [s * v for v in [0.0, 1e-300, 1e-8, 1.0, 30.0, 1e8, 1e150] for s in (1, -1)]
    )
    a, x = (v.ravel() for v in np.meshgrid(av, xv))
    f = chebax.log_skewnorm_cdf
    val = np.asarray(f(a, x))
    assert not np.any(np.isnan(val))
    ga, gx = jax.grad(lambda a, x: jnp.sum(f(a, x)), (0, 1))(a, x)
    # slopes are inf where a x overflows sigma (a = 1.7e308, x = -30: nan)
    small = (np.abs(x) <= 30.0) & (np.abs(a * x) < 1e150)
    assert not np.any(np.isnan(np.asarray(ga)[small]))
    assert not np.any(np.isnan(np.asarray(gx)[small]))


def _x64_off(tmp_path, body, **arrays):
    """Run body in a child process with x64 off (float32): it reads the
    arrays from d and puts its results in out."""
    np.savez(tmp_path / "in.npz", **arrays)
    code = (
        textwrap.dedent(f"""
        import numpy as np, jax, jax.numpy as jnp, chebax
        assert jnp.empty(()).dtype == jnp.float32, "x64 leaked in"
        d = np.load({str(tmp_path / "in.npz")!r})
        out = {{}}
    """)
        + textwrap.dedent(body)
        + textwrap.dedent(f"""
        np.savez({str(tmp_path / "out.npz")!r}, **out)
    """)
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, run.stderr[-3000:]
    return dict(np.load(tmp_path / "out.npz"))


def test_float32(refs, tmp_path):
    """x64 off: the leading KY32 x KP32 block, erfcx and the prefactors in
    float32, on the float32-rounded points (references there), where ln F
    and both slopes are normal float32 numbers."""
    a, x = refs[:2]
    a32, x32 = a.astype(np.float32), x.astype(np.float32)
    got = _x64_off(
        tmp_path,
        """
        f = chebax.log_skewnorm_cdf
        out["v"] = np.asarray(jax.jit(f)(d["a"], d["x"]))
        assert out["v"].dtype == np.float32
        g = jax.jit(jax.grad(lambda a, x: jnp.sum(f(a, x)), (0, 1)))(d["a"], d["x"])
        out["ga"], out["gx"] = (np.asarray(v) for v in g)
        """,
        a=a32,
        x=x32,
    )
    with mp.workdps(40):
        rows = [_ref(float(ai), float(xi)) for ai, xi in zip(a32, x32)]
        lf = np.array([float(r[0]) for r in rows])
        d = [_d_ref(float(ai), float(xi), r[1]) for ai, xi, r in zip(a32, x32, rows)]
    dx = np.array([float(v[0]) for v in d])
    da = np.array([float(v[1]) for v in d])
    tiny, big = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    ok = (np.abs(lf) >= tiny) & (np.abs(lf) <= big)
    err = _log_err(got["v"][ok], lf[ok])
    print(f"float32 ln F worst {np.max(err):.3g}")
    assert np.max(err) <= BAR_LOGF32
    okx = (np.abs(dx) >= tiny) & (np.abs(dx) <= big)
    oka = (np.abs(da) >= tiny) & (np.abs(da) <= big)
    errx = _rel(got["gx"][okx], dx[okx])
    erra = _rel(got["ga"][oka], da[oka])
    print(f"float32 d/dx worst {np.max(errx):.3g}, d/da {np.max(erra):.3g}")
    assert max(np.max(errx), np.max(erra)) <= BAR_D32


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    skewnorm_gen.main(tmp_path)
    assert (tmp_path / "skewnorm_table.py").read_text() == pathlib.Path(
        t.__file__
    ).read_text()


BAR_LOGF = 3e-15
BAR_LOGF_REL = 4e-13
BAR_D = 4e-14
BAR_D_REL = 3e-13
BAR_D2 = 1.2e-13
BAR_LOGF32 = 1.8e-6
BAR_D32 = 3e-5
