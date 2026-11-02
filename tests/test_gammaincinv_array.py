"""Per-element Gamma quantiles: gammaincinv_array and gammainccinv_array.

The inversion contract is two-sided, as in test_quantiles: the returned x
round-trips through the 40-dps CDF on the target's own side (P for lower
targets, Q for upper ones), relative to that side's target. Measured worst
in float64 over a grid from a = 0.1 to 1e6 and targets from 1e-300 to
1 - 2^-52 on both sides: 6.2e-13 for a <= 1000 (a = 640, p = 1e-30, where
the scalar path gives 9.3e-13: the gammainc tables' own accuracy), bar
2.5e-12; 2.3e-12 above (a = 1e6, p = 1e-300, where one float step of x
moves P by 8e-12), bar 1e-11. mpmath's series stall past a ~ 1e6, so from
there up the check is agreement with gammaincinv_scalar, whose
Temme-variable solve above 1e10 test_gammainc_large checks against the
uniform expansion.

Derivatives come from the implicit function theorem on the same ln P and
ln Q the solve inverted. Against mpmath at the 40-dps root (dx/dp = 1/pdf,
dx/da = -dP/da / pdf, each tail differentiated directly), measured worst
7.9e-15 relative over a from 0.1 to 640 in both tails, bar 3e-14; forward
and reverse mode agree to the last bit. Second derivatives, which the
scalar path cannot take in a: d2x/dp2 1.4e-13 (bar 6e-13) against
-((a-1)/x - 1) (dx/dp)^2, d2x/da2 4.6e-12 (bar 2e-11) against a central
difference of the 60-dps first derivative. Above a = 1e3 dx/dp is the
density in closed form at the unrounded root; its limits are in the two
tests that cover it.

float32 runs in a child process with x64 off (jax's default), and covers
the scalar path too: before the solve moved to t = ln(x/a), every
float32 gammaincinv from a = 1e7 to 1e10 came back nan, and the deep
tails from a = 1e5.
"""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import gammaincinv_array as gia

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

A = [0.1, 0.1003, 0.37, 0.999, 1.0, 1.7, 2.0, 2.0001, 7.3, 9.99, 10.0, 10.01]
A += [37.0, 99.9, 100.1, 640.0, 1e4, 1e5, 1e6]
T = [1e-300, 1e-100, 1e-30, 1e-8, 0.003, 0.2, 0.5, 0.8, 0.999, 1 - 1e-9, 1 - 2**-52]
FUNCS = {"inv": chebax.gammaincinv_array, "cinv": chebax.gammainccinv_array}


def _tails(a, x):
    """(P, Q) at 40 digits, the smaller side directly: mpmath's lower
    series stalls at large a past x = a, and the other side is >= ~1/2."""
    a, x = mp.mpf(a), mp.mpf(x)
    if x < a:
        p = mp.gammainc(a, 0, x, regularized=True)
        return p, 1 - p
    q = mp.gammainc(a, x, mp.inf, regularized=True)
    return 1 - q, q


def _side(name, t):
    """(index into _tails, the target) for target t of inv or cinv."""
    upper = t > 0.5 if name == "inv" else t < 0.5
    if name == "inv":
        return (1, 1 - mp.mpf(t)) if upper else (0, mp.mpf(t))
    return (1, mp.mpf(t)) if upper else (0, 1 - mp.mpf(t))


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_round_trip(name):
    aa, tt = (v.ravel() for v in np.meshgrid(A, T, indexing="ij"))
    xs = np.asarray(FUNCS[name](aa, tt))
    tiny = np.finfo(np.float64).tiny
    for a, t, x in zip(aa, tt, xs):
        assert np.isfinite(x), (name, a, t, x)
        if x == 0.0:
            # the quantile underflows: the lower target is below P(a, tiny)
            target = t if name == "inv" else 1 - t
            assert _tails(a, tiny)[0] >= target * (1 - 1e-12), (name, a, t)
            continue
        i, target = _side(name, t)
        err = float(abs(_tails(a, x)[i] / target - 1))
        assert err <= (2.5e-12 if a <= 1000 else 1e-11), (name, a, t, x, err)


def test_matches_the_scalar_path():
    # the same root of two approximations of the same function: measured
    # worst 2.9e-14 relative (a = 3, deep lower tail), bar 1.2e-13; from
    # a = 1e8 up within one float step, and identical from 3e12, where both
    # land on the float nearest the root
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.random(64), 10 ** rng.uniform(-300, -1, 64)])
    shapes = (0.3, 3.0, 30.0, 3000.0, 1e8, 1e10, 3e12, 1e20, 1e30)
    aa = np.repeat(shapes, p.size)
    pp = np.tile(p, len(shapes))
    for name, scalar in (
        ("inv", jax.jit(chebax.gammaincinv_scalar)),
        ("cinv", jax.jit(chebax.gammainccinv_scalar)),
    ):
        xa = np.asarray(FUNCS[name](aa, pp)).reshape(len(shapes), -1)
        for a, row in zip(shapes, xa):
            xs = np.asarray(scalar(a, p))
            np.testing.assert_array_equal(row == 0.0, xs == 0.0)
            ok = xs > 0.0
            rel = np.abs(row[ok] - xs[ok]) / xs[ok]
            assert rel.max() <= 1.2e-13, (name, a, rel.max())


# the start for _root; jitted, so each a is not a fresh compile
_START = {
    False: jax.jit(chebax.gammaincinv_scalar),
    True: jax.jit(chebax.gammainccinv_scalar),
}


def _root(a, t, upper, dps=40):
    """x with P(a, x) = t, or Q(a, x) = t when upper, at dps digits."""
    with mp.workdps(dps):
        a, t = mp.mpf(a), mp.mpf(t)
        f = lambda v: mp.log(_tails(a, mp.exp(v))[1 if upper else 0] / t)
        v0 = mp.log(mp.mpf(float(_START[upper](float(a), float(t)))))
        return mp.exp(mp.findroot(f, v0, tol=mp.mpf(10) ** (5 - dps)))


def _dxda(a, t, upper):
    """dx/da = -dP/da / pdf = dQ/da / pdf at the 60-dps root, each tail
    differentiated directly (1 - Q has no digits of a tiny Q)."""
    with mp.workdps(60):
        x = _root(a, t, upper, 60)
        pdf = mp.exp((a - 1) * mp.log(x) - x - mp.loggamma(a))
        if x > a:
            return mp.diff(lambda s: _tails(s, x)[1], a) / pdf
        return -mp.diff(lambda s: _tails(s, x)[0], a) / pdf


GRAD_POINTS = [(0.1, 1e-5), (0.5, 0.3), (2.5, 0.9), (2.5, 1e-10), (9.99, 1e-20)]
GRAD_POINTS += [(10.01, 0.6), (37.0, 0.5), (640.0, 1e-50), (640.0, 0.999)]
GRAD_POINTS += [(150.0, 0.02)]


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_gradients_against_mpmath(name):
    f = FUNCS[name]
    a = jnp.asarray([q[0] for q in GRAD_POINTS])
    t = jnp.asarray([q[1] for q in GRAD_POINTS])
    rev_a = np.asarray(jax.grad(lambda a: f(a, t).sum())(a))
    rev_t = np.asarray(jax.grad(lambda t: f(a, t).sum())(t))
    fwd_a = np.asarray(jax.jvp(lambda a: f(a, t), (a,), (jnp.ones_like(a),))[1])
    fwd_t = np.asarray(jax.jvp(lambda t: f(a, t), (t,), (jnp.ones_like(t),))[1])
    np.testing.assert_array_equal(fwd_a, rev_a)
    np.testing.assert_array_equal(fwd_t, rev_t)
    d2t = np.asarray(jax.grad(lambda t: jax.grad(lambda t: f(a, t).sum())(t).sum())(t))
    d2a = np.asarray(jax.grad(lambda a: jax.grad(lambda a: f(a, t).sum())(a).sum())(a))
    for i, (ai, ti) in enumerate(GRAD_POINTS):
        upper = name == "cinv"
        x = _root(ai, ti, upper)
        am = mp.mpf(ai)
        pdf = mp.exp((am - 1) * mp.log(x) - x - mp.loggamma(am))
        ref_t = (-1 if upper else 1) / pdf
        ref_t2 = -((am - 1) / x - 1) * ref_t**2
        h = am * mp.mpf(10) ** -12
        ref_a2 = (_dxda(am + h, ti, upper) - _dxda(am - h, ti, upper)) / (2 * h)
        pt = (name, ai, ti)
        assert abs(rev_a[i] / _dxda(am, ti, upper) - 1) <= 3e-14, pt
        assert abs(rev_t[i] / ref_t - 1) <= 3e-14, pt
        assert abs(d2t[i] / ref_t2 - 1) <= 6e-13, pt
        assert abs(d2a[i] / ref_a2 - 1) <= 2e-11, pt


def test_density_in_closed_form_above_1e3():
    # above a = 1e3 dx/dp is 1/pdf in closed form at the unrounded root (a
    # Halley step from x in ln(x/a)), since L_x at the float x is off by its
    # change over one float step, about eps sqrt(a) (1 + |z|). Against
    # mpmath, measured worst 2.0e-13 (a = 2e3, p = 1e-300), bar 8e-13
    pts = [(1e3, 1e-300), (2e3, 1e-300), (2e3, 0.5), (1e4, 1e-100)]
    pts += [(1e5, 1e-200), (1e5, 0.9)]
    a = jnp.asarray([q[0] for q in pts])
    t = jnp.asarray([q[1] for q in pts])
    for name, f in FUNCS.items():
        g = np.asarray(jax.grad(lambda t: f(a, t).sum())(t))
        for i, (ai, ti) in enumerate(pts):
            x = _root(ai, ti, name == "cinv")
            am = mp.mpf(ai)
            pdf = mp.exp((am - 1) * mp.log(x) - x - mp.loggamma(am))
            ref = (-1 if name == "cinv" else 1) / pdf
            assert abs(g[i] / ref - 1) <= 8e-13, (name, ai, ti, g[i], float(ref))


def test_huge_shape_gradients_match_the_scalar_path():
    # above a = 1e10 the scalar path differentiates in Temme's variable,
    # checked against the uniform expansion in test_gammainc_large. Here
    # the Halley step that reaches the root from the float x is exact to
    # its third-order term, which grows like a^(3/2): against that
    # expansion dx/dp measured within 2.3e-13 up to a = 1e25, and 5.7e-6
    # at a = 1e30, where one float step of x is 0.2 standard deviations.
    # Bars 1e-12 and 2e-5; dx/da (~1) agrees to 2.2e-16
    cases = [(1e12, 1e-12), (1e20, 1e-12), (1e25, 1e-12), (1e30, 2e-5)]
    aa = np.repeat([c[0] for c in cases], 3)
    for name, scalar, ts in (
        ("inv", chebax.gammaincinv_scalar, [1e-20, 0.3, 0.9]),
        ("cinv", chebax.gammainccinv_scalar, [1e-300, 1e-20, 0.3]),
    ):
        tt = np.tile(ts, len(cases))
        f = FUNCS[name]
        ga = np.asarray(jax.grad(lambda a: f(a, tt).sum())(jnp.asarray(aa)))
        gt = np.asarray(jax.grad(lambda t: f(aa, t).sum())(jnp.asarray(tt)))
        grad_a, grad_t = jax.jit(jax.grad(scalar)), jax.jit(jax.grad(scalar, 1))
        for i, (a, t) in enumerate(zip(aa, tt)):
            bar = dict(cases)[a]
            sa, st = float(grad_a(a, t)), float(grad_t(a, t))
            assert abs(ga[i] / sa - 1) <= 1e-15, (name, a, t, ga[i], sa)
            if np.isinf(st):  # the density underflows at the root
                assert gt[i] == st, (name, a, t, gt[i], st)
            else:
                assert abs(gt[i] / st - 1) <= bar, (name, a, t, gt[i], st)


def test_endpoints_and_domain():
    inv, cinv = chebax.gammaincinv_array, chebax.gammainccinv_array
    a = np.full(6, 2.5)
    p = np.array([0.0, 1.0, np.nan, -0.1, 1.1, 0.5])
    np.testing.assert_array_equal(
        np.asarray(inv(a, p))[:5], [0.0, np.inf, np.nan, np.nan, np.nan]
    )
    np.testing.assert_array_equal(
        np.asarray(cinv(a, p))[:5], [np.inf, 0.0, np.nan, np.nan, np.nan]
    )
    # the array tables start at a = 0.1; inf, nan and nonpositive a are nan
    bad = np.array([0.09, 0.0, -1.0, np.inf, np.nan])
    for f in (inv, cinv):
        assert np.isnan(np.asarray(f(bad, np.full(5, 0.3)))).all()
        assert np.isnan(np.asarray(f(bad, np.zeros(5)))).all()
    # quantiles below the smallest normal float are 0, as in the scalar path
    assert (
        float(inv(0.1, 1e-300)) == 0.0 == float(chebax.gammaincinv_scalar(0.1, 1e-300))
    )
    # broadcasting: a per row, p per column
    x = np.asarray(inv(np.array([[0.5], [5.0]]), np.array([0.2, 0.7])))
    assert x.shape == (2, 2)
    np.testing.assert_allclose(
        x[1],
        np.asarray(chebax.gammaincinv_scalar(5.0, np.array([0.2, 0.7]))),
        rtol=1e-13,
    )


def test_masked_lanes_keep_clean_gradients():
    # a lane's endpoint, nan or bad shape must not leak into its
    # neighbours, and its own derivative is 0 at the endpoints and nan
    # where the quantile is nan, in both AD modes
    a = jnp.asarray([2.5, 2.5, 2.5, 0.05, 2.5, 30.0, 1e12])
    p = jnp.asarray([0.0, 1.0, 0.3, 0.3, np.nan, 0.9, 1e-30])
    for f in FUNCS.values():
        x = np.asarray(f(a, p))
        for argnum, prim in ((0, a), (1, p)):
            g = lambda v: f(v, p) if argnum == 0 else f(a, v)
            rev = np.asarray(jax.grad(lambda v: g(v).sum())(prim))
            fwd = np.asarray(jax.jvp(g, (prim,), (jnp.ones_like(prim),))[1])
            for d in (rev, fwd):
                assert np.isnan(d[np.isnan(x)]).all(), (argnum, d)
                interior = np.isfinite(x) & (x > 0)
                assert np.isfinite(d[interior]).all(), (argnum, d)
                assert (d[~np.isnan(x) & ~interior] == 0.0).all(), (argnum, d)
            # alone, a lane gives the derivative it gives among others (to
            # rounding: the two shapes compile to different code)
            for i in np.where(np.isfinite(x) & (x > 0))[0]:
                one = jax.grad(lambda v: f(v, p[i]) if argnum == 0 else f(a[i], v))(
                    prim[i]
                )
                assert abs(float(one) / rev[i] - 1) <= 1e-13, (argnum, i)


def test_iterations_have_a_margin(monkeypatch):
    # half the steps land on the same tail probability (_ITERS is twice
    # the most any of 1e5 random lanes needed; this checks a fresh set).
    # x itself can differ where it barely moves P: deep in the lower tail
    # at small a, x ~ p^(1/a). In the probability, measured 8.9e-12 (two
    # solves inside the acceptance bar, 1.6e-10 wide at |ln p| = 690),
    # bar 4e-11
    rng = np.random.default_rng(11)
    a = np.exp(rng.uniform(np.log(0.1), np.log(1e30), 4096))
    p = np.where(
        rng.random(4096) < 0.5, rng.random(4096), 10 ** rng.uniform(-300, 0, 4096)
    )
    full = {k: np.asarray(f(a, p)) for k, f in FUNCS.items()}
    monkeypatch.setitem(gia._ITERS, "float64", gia._ITERS["float64"] // 2)
    for k, f in FUNCS.items():
        # the count is read at trace time, so the cached trace must go
        jax.clear_caches()
        half = np.asarray(jax.jit(lambda a, p: f(a, p))(a, p))
        np.testing.assert_array_equal(np.isnan(half), np.isnan(full[k]))
        upper = p > 0.5 if k == "inv" else p < 0.5
        side = lambda x: np.where(
            upper,
            np.asarray(chebax.log_gammaincc_array(a, x)),
            np.asarray(chebax.log_gammainc_array(a, x)),
        )
        ok = np.isfinite(full[k]) & (full[k] > 0)
        d = np.abs(np.expm1(side(half) - side(full[k])))[ok]
        assert d.max() <= 4e-11, (k, d.max())


F32_CHILD = """
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
import chebax

assert jnp.empty(()).dtype == jnp.float32, "x64 must be off in this process"
a, t = (np.asarray(v, np.float32) for v in json.load(open(sys.argv[1])))
out = {}
for name in ("gammaincinv", "gammainccinv"):
    out[name + "_array"] = np.asarray(getattr(chebax, name + "_array")(a, t)).tolist()
    # jitted, so a is traced: called eagerly, every a was a fresh compile
    # with its tables baked in (17 minutes and 15 GB)
    scalar = jax.jit(getattr(chebax, name + "_scalar"))
    out[name + "_scalar"] = [float(scalar(ai, ti)) for ai, ti in zip(a, t)]
print(json.dumps(out))
"""


def test_float32_in_a_child(tmp_path):
    # x64 off, both paths, a from 0.1 to 1e15: no x is nan, and every x is
    # one float32 step from the root or within a bar of its target (the
    # tail probability at x, judged in float64 here). Measured worst
    # 3.3e-5 on the array path, bar 1.3e-4; the scalar path is held to its
    # own residual bar, solver_limits' 1e-3 (measured 3.6e-4 outside the
    # wedge below)
    a32 = [0.1, 0.37, 1.0, 2.5, 9.99, 10.01, 37.0, 640.0, 1e4, 1e5, 1e6, 1e7, 1e9]
    a32 += [5e9, 1e11, 1e13, 1e15]
    t32 = [1e-37, 1e-20, 1e-10, 1e-4, 0.03, 0.3, 0.5, 0.9, 0.999, 1 - 2**-20]
    aa, tt = (
        v.ravel().astype(np.float32) for v in np.meshgrid(a32, t32, indexing="ij")
    )
    ref = tmp_path / "cases.json"
    ref.write_text(json.dumps([aa.tolist(), tt.tolist()]))
    runner = tmp_path / "child32.py"
    runner.write_text(F32_CHILD)
    src = os.path.dirname(os.path.dirname(os.path.abspath(chebax.__file__)))
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": src}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, str(runner), str(ref)], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    a64, t64 = aa.astype(np.float64), tt.astype(np.float64)
    tiny = float(np.finfo(np.float32).tiny)
    worst = {}
    for key, xs in res.items():
        x = np.asarray(xs, np.float64)
        inv = key.startswith("gammaincinv")
        upper = t64 > 0.5 if inv else t64 < 0.5
        target = np.where(upper == inv, 1 - t64, t64)
        lt = np.log(target)

        def side(v):
            v = np.where(np.isfinite(v) & (v > 0), v, 1.0)
            lp = np.asarray(chebax.log_gammainc_array(a64, v))
            lq = np.asarray(chebax.log_gammaincc_array(a64, v))
            return np.where(upper, lq, lp) - lt

        assert not np.isnan(x).any(), (key, a64[np.isnan(x)], t64[np.isnan(x)])
        zero = x == 0.0
        lower = np.where(inv, t64, 1 - t64)
        lp_tiny = np.asarray(chebax.log_gammainc_array(a64, np.full_like(a64, tiny)))
        assert (lp_tiny[zero] >= np.log(lower[zero]) - 1e-6).all(), key
        f = side(x)
        x32 = x.astype(np.float32)
        fd = side(np.nextafter(x32, np.float32(0)).astype(np.float64))
        fu = side(np.nextafter(x32, np.float32(np.inf)).astype(np.float64))
        sgn = np.where(upper, -1.0, 1.0)
        step = (sgn * fd <= 0) & (sgn * fu >= 0)
        rel = np.abs(np.expm1(f))
        live = ~zero & ~step
        if key.endswith("_scalar"):
            # the scalar path's small-a tables give Q below x = 8 as 1 - P,
            # which float32 resolves to ~6e-8 absolute (Q 1e-2 off at
            # a = 0.37, y = 1e-4, as before this solve); the array path
            # computes Q there directly
            live &= ~((a64 <= 10.0) & (x <= 8.0) & upper)
        worst[key] = float(rel[live].max()) if live.any() else 0.0
    for key, w in worst.items():
        assert w <= (1e-3 if key.endswith("_scalar") else 1.3e-4), worst
