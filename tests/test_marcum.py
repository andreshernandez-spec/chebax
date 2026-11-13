"""The Marcum Q-function of order 1 in logs: log_marcum_q1(a, b) and
log_marcum_p1(a, b) = ln(1 - Q_1(a, b)), a and b broadcast. Below, x is the
point b.

References are mpmath at 40 dps, from forms marcum_gen does not use: for
max(a^2, a x) <= 2000 the Poisson mixture P = sum_n Pois(n; a^2/2) P(n+1,
x^2/2) and Q the same with the upper gamma (every term positive, the
regularized gammas from recurrences that only add), and above it rays out of
the mean of N((a, 0), I): Q = (1/pi) int_0^pi exp(-r(t)^2/2) dt with r the exit
distance from the disc of radius x when x > a, P from entry and exit when x <
a. x = a has Q = (1 + e^(-a^2) I_0(a^2))/2. The two agree with marcum_gen's
series and v-integral to 5e-39 on 136 points (a to 3e3, near x = a too).

Metric |ln F - ref| / max(1, |ref|), for ln P and ln Q. The grid crosses a
from 1e-8 to 1e8 with x from 1e-12 to 1e8, adds x = a(1 +- delta) and a +-
k, both sides of every patch cut and of each seam (xi = E1^2, eta = E1, x =
B_C, x = a), and 300 random points: 964 in all.

Measured worst, float64, jax 0.11: ln P 1.0e-15 and ln Q 7.8e-16 (bar
4e-15), at a = 1e3, x = a - 1e-3. Slopes, both AD modes against the closed
forms: 6.7e-15 (bar 3e-14), at a = 1e8 where the slope is 1e8. Second
derivatives (AD of the rule) 2.3e-12 (bar 1e-11), at a = 8, x = 1e-3:
d2 ln P/dadx = 0.004 there is the difference of two terms of size a/x = 8e3.
float32 (6 x 6 blocks) at the float32-rounded points: ln F 7.0e-7 (bar 3e-6),
slopes relative 7.0e-6 (bar 3e-5).
"""

import math
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
from chebax._src.recipes import marcum_gen
from chebax._src.recipes import marcum_table as t

mp = pytest.importorskip("mpmath")

FNS = (chebax.log_marcum_p1, chebax.log_marcum_q1)


def _poisson(a, x):
    lam, y = a * a / 2, x * x / 2
    top = max(lam, a * x / 2)  # where the terms peak
    n_hi = int(top + 12 * mp.sqrt(top) + 60)
    ly = mp.log(y)

    def lterm(n):  # ln(e^-y y^n/n!)
        return -y + n * ly - mp.loggamma(n + 1)

    qn = [mp.exp(-y)]  # Q(n+1, y) = Q(n, y) + e^-y y^n/n!
    for n in range(1, n_hi + 1):
        qn.append(qn[-1] + mp.exp(lterm(n)))
    pn = [mp.mpf(0)] * (n_hi + 2)  # P(n+1, y) = P(n+2, y) + e^-y y^(n+1)/(n+1)!
    pn[n_hi + 1] = mp.gammainc(n_hi + 2, 0, y, regularized=True)
    for n in range(n_hi, -1, -1):
        pn[n] = pn[n + 1] + mp.exp(lterm(n + 1))
    w = [mp.exp(-lam + n * mp.log(lam) - mp.loggamma(n + 1)) for n in range(n_hi + 1)]
    return mp.fsum(u * p for u, p in zip(w, pn)), mp.fsum(u * q for u, q in zip(w, qn))


def _rays(a, x):
    h = mp.pi / 2
    geo = [h * mp.mpf(4) ** -k for k in range(1, 25)]
    if x > a:
        # peaks at 0 and, next to x = a, a step of width ~1/a at pi/2
        def g(s):
            r = mp.sqrt(x * x - (a * mp.sin(s)) ** 2) - a * mp.cos(s)
            return mp.exp(-(r * r - (x - a) ** 2) / 2)

        pts = sorted(set(geo) | {h + s * v for v in geo for s in (-1, 1)})
        q = mp.quad(g, [0] + pts + [mp.pi]) / mp.pi * mp.exp(-((x - a) ** 2) / 2)
        return 1 - q, q

    def f(u):  # direction s with sin s = (x/a) sin u, entry r_- and exit r_+
        st = x / a * mp.sin(u)
        ct = mp.sqrt(1 - st * st)
        root = x * mp.cos(u)
        rm = a * ct - root
        jac = x * mp.cos(u) / (a * ct)
        return (
            mp.exp(-(rm * rm - (a - x) ** 2) / 2) * -mp.expm1(-2 * a * ct * root) * jac
        )

    pts = sorted(set(geo) | {h - v for v in geo})
    p = mp.quad(f, [0] + pts + [h]) / mp.pi * mp.exp(-((a - x) ** 2) / 2)
    return p, 1 - p


def _ref(a, x):
    """(ln P, ln Q) at float a, x >= 0."""
    a, x = mp.mpf(a), mp.mpf(x)
    if x == 0:
        return -mp.inf, mp.mpf(0)
    if a == 0:
        return mp.log(-mp.expm1(-x * x / 2)), -x * x / 2
    if x == a:
        q = (1 + mp.exp(-a * a) * mp.besseli(0, a * a)) / 2
        p = 1 - q
    elif max(a * a, a * x) <= 2000:
        p, q = _poisson(a, x)
    else:
        p, q = _rays(a, x)
    lp = mp.log(p) if p <= q else mp.log1p(-q)
    lq = mp.log(q) if q <= p else mp.log1p(-p)
    return lp, lq


def _d_ref(a, x, lp, lq):
    """(d ln P/da, d ln P/dx, d ln Q/da, d ln Q/dx): dP/dx = x e^(-(a^2+x^2)/2)
    I_0(a x), dP/da = -x e^(-(a^2+x^2)/2) I_1(a x)."""
    a, x = mp.mpf(a), mp.mpf(x)
    if x == 0:
        return (mp.mpf(0),) * 4
    base = mp.log(x) - (a * a + x * x) / 2
    l0 = base + mp.log(mp.besseli(0, a * x))
    l1 = base + mp.log(mp.besseli(1, a * x)) if a > 0 else -mp.inf
    return (-mp.exp(l1 - lp), mp.exp(l0 - lp), mp.exp(l1 - lq), -mp.exp(l0 - lq))


def _from_chart(z, p, y):
    """(a, x) pairs, Q side then P side, at chart z's coordinates (p, y)."""
    if z == "CORNER":
        x = math.sqrt(y)
        return [(math.sqrt(p) * x, x)]
    if z == "BULK":
        rho, xi = p, y
    elif z == "MEDIAN":
        rho, xi = 1 - p * y, 1 / (p * p)
    else:
        eta = (1 - y) / y
        rho, xi = p, (eta / (1 - p)) ** 2
    if not (0 < rho < 1 and xi > 0):
        return []
    lo, hi = math.sqrt(xi * rho), math.sqrt(xi / rho)
    return [(lo, hi), (hi, lo)]


def _seam_points():
    out = []
    side = (1 - 1e-9, 1 + 1e-9)
    for zi, z in enumerate(t.NAMES):
        pc = [float(c) for c in t.PCUTS[zi][: t.NPAR[zi] + 1]]
        yc = [float(c) for c in t.YCUTS[zi][: t.NARG[zi] + 1]]
        pmid = [pc[0] + f * (pc[-1] - pc[0]) for f in (0.03, 0.5, 0.97)]
        ymid = [yc[0] + f * (yc[-1] - yc[0]) for f in (0.03, 0.5, 0.97)]
        for c in pc:
            for y in ymid:
                for r in side:
                    if 0 < c * r <= pc[-1]:
                        out += _from_chart(z, c * r, y)
        for c in yc:
            for p in pmid:
                for r in side:
                    if 0 < c * r <= yc[-1]:
                        out += _from_chart(z, p, c * r)
    # chart seams: xi = E1^2, eta = E1 (via MEDIAN's edge), x = X_C
    for r in side:
        for rho in (0.05, 0.5, 0.95):
            out += _from_chart("BULK", rho, t.E1**2 * r)
        for eps in (0.01, 0.2, 0.45):
            out += _from_chart("MEDIAN", eps, t.E1 * r)
        for f in (0.0, 0.3, 0.99):
            out.append((f * t.B_C * r, t.B_C * r))
    return np.array(out)


def _points():
    av = [0.0, 1e-8, 0.01, 0.3, 1.0, 1.5, 2.0, 3.0, 7.0, 15.0, 40.0, 300.0, 1e4, 1e8]
    xv = [1e-12, 1e-4, 0.1, 0.5, 1.0, 1.18, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 40.0]
    xv += [300.0, 1e4, 1e8]
    grid = [(a, x) for a in av for x in xv]
    for a in (0.5, 2.0, 9.0, 60.0, 1e3, 1e6):
        grid += [(a, a * (1 + s * dl)) for s in (1, -1) for dl in (1e-12, 1e-6, 1e-2)]
        grid += [
            (a, a + k) for k in (-5.0, -2.0, -0.3, 0.0, 0.3, 2.0, 5.0) if a + k > 0
        ]
    rng = np.random.default_rng(51)
    n = 300
    ra = 10.0 ** rng.uniform(-3, 4, n)
    rx = np.abs(
        ra + rng.uniform(-1, 1, n) * np.minimum(ra, 40) * rng.uniform(0, 1.2, n)
    )
    rand = np.stack([ra, np.maximum(rx, 1e-6)], axis=1)
    return np.concatenate([np.array(grid), _seam_points(), rand])


@pytest.fixture(scope="module")
def refs():
    pts = _points()
    a, x = pts[:, 0], pts[:, 1]
    with mp.workdps(40):
        rows = [_ref(ai, xi) for ai, xi in zip(a, x)]
        d = [_d_ref(ai, xi, *r) for ai, xi, r in zip(a, x, rows)]
    lp = np.array([float(r[0]) for r in rows])
    lq = np.array([float(r[1]) for r in rows])
    d = np.array([[float(v) for v in row] for row in d])
    return a, x, lp, lq, d


def _log_err(got, ref):
    got = np.asarray(got, np.float64)
    both_inf = np.isinf(got) & np.isinf(ref) & (np.sign(got) == np.sign(ref))
    assert np.all(np.isfinite(got) | both_inf), np.flatnonzero(~np.isfinite(got))[:10]
    with np.errstate(invalid="ignore"):
        err = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
    return np.where(both_inf, 0.0, err)


@pytest.mark.parametrize("which", ["p", "q"])
def test_against_mpmath(refs, which):
    a, x, lp, lq = refs[:4]
    fn, ref = (chebax.log_marcum_p1, lp) if which == "p" else (chebax.log_marcum_q1, lq)
    err = _log_err(jax.jit(fn)(a, x), ref)
    w = int(np.argmax(err))
    print(f"ln {which.upper()} worst {err[w]:.3g} at a={a[w]!r}, x={x[w]!r}")
    assert err[w] <= BAR_LOG


def test_derivatives_against_closed_forms(refs):
    a, x, lp, lq, d = refs
    worst = {}
    for k, fn in enumerate(FNS):
        ga, gx = jax.jit(jax.grad(lambda a, x: jnp.sum(fn(a, x)), (0, 1)))(a, x)
        one, zero = jnp.ones_like(a), jnp.zeros_like(a)
        fa = jax.jvp(fn, (a, x), (one, zero))[1]
        fx = jax.jvp(fn, (a, x), (zero, one))[1]
        ra, rx = d[:, 2 * k], d[:, 2 * k + 1]
        for got, ref, name in (
            (ga, ra, "rev a"),
            (fa, ra, "fwd a"),
            (gx, rx, "rev x"),
            (fx, rx, "fwd x"),
        ):
            err = _log_err(got, ref)
            w = int(np.argmax(err))
            worst[f"{fn.__name__} {name}"] = (err[w], a[w], x[w])
    for name, (e, aw, xw) in worst.items():
        print(f"{name} worst {e:.3g} at a={aw!r}, x={xw!r}")
    assert max(v[0] for v in worst.values()) <= BAR_D


def test_negative_shape_and_edges():
    f, g = chebax.log_marcum_p1, chebax.log_marcum_q1
    a = np.array([0.3, 2.0, 7.0, 40.0])
    x = np.array([0.5, 1.0, 9.0, 38.0])
    np.testing.assert_array_equal(np.asarray(f(-a, x)), np.asarray(f(a, x)))
    np.testing.assert_array_equal(np.asarray(g(-a, x)), np.asarray(g(a, x)))
    assert float(f(2.0, 0.0)) == -np.inf and float(g(2.0, 0.0)) == 0.0
    assert float(f(2.0, -1.0)) == -np.inf and float(g(2.0, -1.0)) == 0.0
    assert np.isnan(float(f(jnp.nan, 1.0))) and np.isnan(float(f(1.0, jnp.nan)))
    # a = 0: Q_1(0, b) = e^(-b^2/2), the Rayleigh distribution
    for xv in (1e-3, 0.5, 2.0, 30.0):
        assert abs(float(g(0.0, xv)) + xv * xv / 2) <= 1e-15 * max(1.0, xv * xv / 2)


@pytest.mark.parametrize("fn", FNS, ids=lambda f: f.__name__)
def test_broadcast_jit_and_arguments(fn):
    a = jnp.asarray([[0.0], [0.5], [7.0]])
    x = jnp.asarray([0.1, 0.4, 5.0, 12.0])
    got = fn(a, x)
    assert got.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(jax.jit(fn)(a, x)), np.asarray(got))
    assert float(fn(a=2, b=3)) == float(fn(2.0, 3.0))
    g = jax.grad(lambda a, x: fn(a, x), (0, 1))(2.5, 3.0)
    gk = jax.grad(lambda a, x: fn(b=x, a=a), (0, 1))(2.5, 3.0)
    assert list(map(float, g)) == list(map(float, gk))


def _d2_ref(a, x, lp, lq):
    """Hessians of ln P and ln Q in (a, x): F_ij/F - F_i F_j/F^2, with E =
    e^(-(a^2+x^2)/2), P_x = x E I_0, P_a = -x E I_1, P_xx = E ((1 - x^2) I_0 +
    a x I_1), P_ax = x E (x I_1 - a I_0), P_aa = -x E (x I_0 - a I_1 - I_1/a),
    I_k at a x; Q's derivatives are minus P's."""
    a, x = mp.mpf(a), mp.mpf(x)
    e = mp.exp(-(a * a + x * x) / 2)
    i0, i1 = mp.besseli(0, a * x), mp.besseli(1, a * x)
    i1_a = i1 / a if a > 0 else x / 2
    pd = {"a": -x * e * i1, "x": x * e * i0}
    pdd = {
        "aa": -x * e * (x * i0 - a * i1 - i1_a),
        "ax": x * e * (x * i1 - a * i0),
        "xx": e * ((1 - x * x) * i0 + a * x * i1),
    }
    out = []
    for lf, sign in ((lp, 1), (lq, -1)):
        f = mp.exp(lf)
        out.append(
            [
                sign * pdd[k] / f - pd[k[0]] * pd[k[1]] / (f * f)
                for k in ("aa", "ax", "xx")
            ]
        )
    return out


def test_second_derivatives():
    """AD of the rule against closed forms, |err| / max(1, |ref|), at points
    in every chart and both sides."""
    pts = [(0.0, 0.7), (0.4, 0.9), (0.5, 0.2), (1.5, 1.8), (3.0, 0.5), (1.9, 2.1)]
    pts += [(4.0, 6.0), (10.0, 10.5), (10.0, 30.0), (30.0, 5.0), (60.0, 59.0)]
    pts += [(300.0, 290.0), (0.05, 40.0), (8.0, 1e-3)]
    worst = 0.0
    with mp.workdps(40):
        for k, fn in enumerate(FNS):
            hess = jax.jit(jax.hessian(lambda a, x: fn(a, x), (0, 1)))
            for a, x in pts:
                ref = _d2_ref(a, x, *_ref(a, x))[k]
                h = hess(a, x)
                for g, r in zip([h[0][0], h[0][1], h[1][1]], ref):
                    assert np.isfinite(float(g)), (fn.__name__, a, x)
                    err = abs(float(g) - float(r)) / max(1.0, abs(float(r)))
                    worst = max(worst, err)
    print(f"second derivatives worst {worst:.3g}")
    assert worst <= BAR_D2


def test_extreme_inputs_stay_finite():
    """No nan in values or slopes from 1e-300 to 1.7e308, and ln P finite
    at x = 1e-300, where x^2 underflows."""
    v = np.array([0.0, 1e-300, 1e-8, 1.0, 30.0, 1e8, 1e150, 1e300, 1.7e308])
    a, x = (u.ravel() for u in np.meshgrid(v, v))
    for fn in FNS:
        assert not np.any(np.isnan(np.asarray(fn(a, x))))
        ga, gx = jax.grad(lambda a, x: jnp.sum(fn(a, x)), (0, 1))(a, x)
        assert not np.any(np.isnan(np.asarray(ga))), fn.__name__
        assert not np.any(np.isnan(np.asarray(gx))), fn.__name__
    with mp.workdps(40):
        ref = float(_ref(3.0, 1e-300)[0])
    assert abs(float(chebax.log_marcum_p1(3.0, 1e-300)) - ref) <= BAR_LOG * abs(ref)


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
    """x64 off: the KY32 x KP32 blocks and float32 prefactors, at the
    float32-rounded points (references there) where ln F and the slopes are
    normal float32 numbers."""
    a, x = refs[:2]
    keep = (a < 1e30) & (x < 1e30)
    a32, x32 = a[keep].astype(np.float32), x[keep].astype(np.float32)
    got = _x64_off(
        tmp_path,
        """
        for name, f in (("p", chebax.log_marcum_p1), ("q", chebax.log_marcum_q1)):
            out[name] = np.asarray(jax.jit(f)(d["a"], d["x"]))
            assert out[name].dtype == np.float32
            g = jax.jit(jax.grad(lambda a, x: jnp.sum(f(a, x)), (0, 1)))(d["a"], d["x"])
            out[name + "a"], out[name + "x"] = (np.asarray(v) for v in g)
        """,
        a=a32,
        x=x32,
    )
    with mp.workdps(40):
        rows = [_ref(float(ai), float(xi)) for ai, xi in zip(a32, x32)]
        d = [_d_ref(float(ai), float(xi), *r) for ai, xi, r in zip(a32, x32, rows)]
    lp = np.array([float(r[0]) for r in rows])
    lq = np.array([float(r[1]) for r in rows])
    d = np.array([[float(v) for v in row] for row in d])
    tiny, big = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    worst_v, worst_d = 0.0, 0.0
    for k, (name, ref) in enumerate((("p", lp), ("q", lq))):
        ok = (np.abs(ref) >= tiny) & (np.abs(ref) <= big)
        worst_v = max(worst_v, np.max(_log_err(got[name][ok], ref[ok])))
        for j, ax in enumerate("ax"):
            r = d[:, 2 * k + j]
            okd = (np.abs(r) >= tiny) & (np.abs(r) <= big)
            g = np.asarray(got[name + ax][okd], np.float64)
            worst_d = max(worst_d, np.max(np.abs(g - r[okd]) / np.abs(r[okd])))
    print(f"float32 ln F worst {worst_v:.3g}, slopes relative {worst_d:.3g}")
    assert worst_v <= BAR_LOG32
    assert worst_d <= BAR_D32


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    marcum_gen.main(tmp_path)
    assert (tmp_path / "marcum_table.py").read_text() == pathlib.Path(
        t.__file__
    ).read_text()


BAR_LOG = 4e-15
BAR_D = 3e-14
BAR_D2 = 1e-11
BAR_LOG32 = 3e-6
BAR_D32 = 3e-5
