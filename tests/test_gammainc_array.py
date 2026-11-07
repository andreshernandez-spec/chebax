"""Per-element gammainc: gammainc_array, gammaincc_array and their logs, with
a different a on every element, a >= 0.1, x in [0, inf).

References are mpmath at 40 dps on 2,300 points over every zone of
gammainc_array_gen: a from 0.1 to 1e4, the bulk, both tails (ln P down to
-1.2e5, ln Q to -3.0e5), and the seams a = 1, a = 10, x = a + 1 and
lambda = LAM_LO, LAM_HI.
The logs use |ln F - ref| / max(1, |ref|): relative in F where |ln F| <= 1,
relative in ln F beyond. P and Q are absolute (the CDF contract).

Measured worst, float64, the larger of jax 0.11 and 0.5.3 (CI's floor):
ln P 8.9e-16, ln Q 8.4e-15, bars 3.5e-15 and 3.5e-14. ln Q's floor is the
far side of A0 (a < 1, x <= a + 1), where Q = 1 - P and Q can be as small
as 0.024 (a = 0.1, x = 1.1), so P's rounding shows up ~40x in Q; outside
A0 ln Q is 7.8e-16. P, Q absolute 8.4e-16, bar 3.5e-15. float32, against
references at the float32-rounded inputs: ln P 4.5e-7, ln Q 2.9e-6 (the
same far side), bars 2e-6 and 1.2e-5; P, Q absolute 4.0e-7, bar 1.6e-6.

d/dx against the Gamma density (d ln P/dx = density/P, d ln Q/dx =
-density/Q, formed in mpmath), relative: 2.0e-13, bar 8e-13. d/da against
mp.diff, 10 points per zone, |g - ref| / max(1, |ref|): 1.3e-13, bar
5e-13. The far tails, past the random points, have their own check
against mpmath with the precision raised for |ln F| (x up to 1e300, and
x/a underflowing): float64 2.2e-16, bar 1e-15; float32 1.0e-7, bar 4e-7,
in forward and reverse mode. The ends of each dtype's range are checked
for nan in both modes. The IEEE endpoints and the x = 0 slopes are exact,
so those tests assert equality.
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
from chebax import domains
from chebax._src.recipes import gammainc_array_gen
from chebax._src.recipes import gammainc_array_table as t

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

ALLFN = [
    chebax.gammainc_array,
    chebax.gammaincc_array,
    chebax.log_gammainc_array,
    chebax.log_gammaincc_array,
]


def _points():
    """(a, x) over every zone, both tails and the seams."""
    rng = np.random.default_rng(31)
    n = 500
    a = np.concatenate(
        [
            np.exp(rng.uniform(np.log(0.1), 0.0, n)),
            np.exp(rng.uniform(0.0, np.log(10.0), n)),
            np.exp(rng.uniform(np.log(10.0), np.log(1e4), n)),
        ]
    )
    kind = rng.integers(0, 4, a.size)
    seam = np.where(
        a < 1.0,
        a + 1.0,
        a * np.where(rng.uniform(size=a.size) < 0.5, t.LAM_LO, t.LAM_HI),
    )
    x = np.select(
        [kind == 0, kind == 1, kind == 2],
        [
            rng.gamma(a),
            a * np.exp(rng.uniform(np.log(1e-6), np.log(t.LAM_LO), a.size)),
            a + np.sqrt(a) * rng.uniform(0.0, 30.0, a.size),
        ],
        seam * np.exp(rng.normal(0.0, 1e-3, a.size)),
    )
    # the a = 1 and a = 10 seams, a wide log-uniform sweep, and the upper
    # tail at large a, which the draws above rarely reach
    a1 = rng.choice([1.0, 10.0], 300) * np.exp(rng.normal(0.0, 1e-9, 300))
    a2 = np.exp(rng.uniform(np.log(0.1), np.log(100.0), 400))
    x2 = np.exp(rng.uniform(np.log(1e-3), np.log(500.0), 400))
    a3 = np.exp(rng.uniform(np.log(10.0), np.log(1e4), 100))
    x3 = a3 * np.exp(rng.uniform(np.log(t.LAM_HI), np.log(50.0), 100))
    return (
        np.concatenate([a, a1, a2, a3]),
        np.concatenate([x, rng.gamma(a1), x2, x3]),
    )


def _zone(a, x):
    lam = x / a
    return np.where(
        a < 1.0,
        np.where(x <= a + 1.0, 0, 1),
        np.where(
            lam <= t.LAM_LO,
            np.where(a > t.ASPLIT, 4, 2),
            np.where(lam >= t.LAM_HI, np.where(a > t.ASPLIT, 5, 3), 6),
        ),
    )


def _mp_logs(a, x, slopes=False):
    """ln P, ln Q and, with slopes, d ln P/dx = density/P and d ln Q/dx =
    -density/Q, formed in mpmath: from float64 logs near -1e5 the ratio
    would carry their 1e-11 rounding."""
    out = np.empty((4 if slopes else 2, a.size))
    for k, (ak, xk) in enumerate(zip(a, x)):
        am, xm = mp.mpf(ak), mp.mpf(xk)
        lp = mp.log(mp.gammainc(am, 0, xm, regularized=True))
        lq = mp.log(mp.gammainc(am, xm, mp.inf, regularized=True))
        out[:2, k] = float(lp), float(lq)
        if slopes:
            ld = (am - 1) * mp.log(xm) - xm - mp.loggamma(am)
            out[2:, k] = float(mp.exp(ld - lp)), -float(mp.exp(ld - lq))
    return out


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    assert np.all(np.isfinite(got)), "unexpected nonfinite values"
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


@pytest.fixture(scope="module")
def refs():
    """a, x, ln P, ln Q, d ln P/dx, d ln Q/dx."""
    a, x = _points()
    return (a, x, *_mp_logs(a, x, slopes=True))


def test_points_cover_every_zone(refs):
    a, x = refs[:2]
    counts = np.bincount(_zone(a, x), minlength=len(t.NAMES))
    assert np.all(counts >= 60), dict(zip(t.NAMES, counts))


def test_logs_against_mpmath(refs):
    a, x, lp, lq = refs[:4]
    f = jax.jit(
        lambda a, x: (chebax.log_gammainc_array(a, x), chebax.log_gammaincc_array(a, x))
    )
    gp, gq = f(a, x)
    assert np.max(_err(gp, lp)) <= 3.5e-15
    assert np.max(_err(gq, lq)) <= 3.5e-14


def test_values_absolute(refs):
    a, x, lp, lq = refs[:4]
    p = np.asarray(jax.jit(chebax.gammainc_array)(a, x))
    q = np.asarray(jax.jit(chebax.gammaincc_array)(a, x))
    assert np.max(np.abs(p - np.exp(lp))) <= 3.5e-15
    assert np.max(np.abs(q - np.exp(lq))) <= 3.5e-15


def _x64_off(tmp_path, body, **arrays):
    """Run body in a child process with x64 at its default (off): it reads
    the arrays from d and puts its results in the dict out. x64 is
    process-global, so float32 cannot be checked in this process."""
    np.savez(tmp_path / "in.npz", **arrays)
    code = (
        textwrap.dedent(f"""
        import sys
        import numpy as np, jax, jax.numpy as jnp, chebax
        assert jnp.empty(()).dtype == jnp.float32, "x64 leaked in"
        sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})
        from test_gammainc_array import ALLFN, _slopes
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
    """float32 is the default runtime (x64 off): the leading 7 x 7 block of
    the same patches, with erfcx and every prefactor in float32 too, on the
    float32-rounded points."""
    a, x = refs[:2]
    a32, x32 = a.astype(np.float32), x.astype(np.float32)
    got = _x64_off(
        tmp_path,
        """
        for f in ALLFN:
            out[f.__name__] = np.asarray(jax.jit(f)(d["a"], d["x"]))
            assert out[f.__name__].dtype == np.float32
        """,
        a=a32,
        x=x32,
    )
    lp, lq = _mp_logs(a32.astype(np.float64), x32.astype(np.float64))
    assert np.max(_err(got["log_gammainc_array"], lp)) <= 2e-6
    assert np.max(_err(got["log_gammaincc_array"], lq)) <= 1.2e-5
    assert np.max(np.abs(got["gammainc_array"] - np.exp(lp))) <= 1.6e-6
    assert np.max(np.abs(got["gammaincc_array"] - np.exp(lq))) <= 1.6e-6


def _slopes(a, x):
    """Values and both AD modes' slopes in a and in x, for each function."""
    a, x = jnp.asarray(a), jnp.asarray(x)
    one, zero = jnp.ones_like(a), jnp.zeros_like(a)
    out = {}
    for f in ALLFN:
        n = f.__name__
        ga, gx = jax.jit(jax.grad(lambda a, x: jnp.sum(f(a, x)), (0, 1)))(a, x)
        out[n] = np.asarray(f(a, x))
        out[n + " rev a"], out[n + " rev x"] = np.asarray(ga), np.asarray(gx)
        out[n + " fwd a"] = np.asarray(jax.jvp(f, (a, x), (one, zero))[1])
        out[n + " fwd x"] = np.asarray(jax.jvp(f, (a, x), (zero, one))[1])
    return out


def _in_dtype(dtype, tmp_path, a, x):
    if dtype == "float64":
        return _slopes(a, x)
    body = 'out = _slopes(d["a"], d["x"])\n'
    return _x64_off(tmp_path, body, a=a.astype(np.float32), x=x.astype(np.float32))


# every zone at both ends of each dtype's normal range
_EXTREME_A = [0.1, 0.5, 0.999, 1.0, 2.5, 10.0, 10.5, 30.0, 1e3, 1e6]
_EXTREME_X = {
    "float64": [2.3e-308, 1e-306, 1e-300, 1e-200, 1e-20, 1e-8, 0.3, 1.0, 2.0, 9.0]
    + [40.0, 1e8, 1e100, 1e300, 1.7e308],
    "float32": [1.2e-38, 1e-37, 1e-30, 1e-20, 1e-8, 0.3, 1.0, 2.0, 9.0, 40.0]
    + [1e8, 1e20, 1e37, 3e38],
}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_extreme_inputs_in_both_modes(dtype, tmp_path):
    """No nan in values or in forward- or reverse-mode slopes at the ends
    of the range, and the two modes agree. An unused branch that divides by
    an unmasked x once gave nan through 0 * inf (jax.grad of gammainc_array
    in x at (1, 1e-20) in float32), and differentiating x/a through the
    ratio lost the a-term where x/a^2 left the float range. The one
    infinity is d ln P/dx where its true value, ~ a/x, overflows."""
    a, x = (v.ravel() for v in np.meshgrid(_EXTREME_A, _EXTREME_X[dtype]))
    got = _in_dtype(dtype, tmp_path, a, x)
    rtol = 1e-12 if dtype == "float64" else 1e-5
    # slopes within a few decades of the smallest normal number pass through
    # subnormal cotangents in reverse mode, so there the modes agree only
    # absolutely (dP/da ~ 1e-303 at x = 1e-306, a = 1)
    atol = 1e6 * np.finfo(dtype).tiny
    with np.errstate(over="ignore"):
        big = a / x > np.finfo(dtype).max
    for f in ALLFN:
        n = f.__name__
        assert np.all(np.isfinite(got[n])), n
        for w in ("a", "x"):
            rev, fwd = got[f"{n} rev {w}"], got[f"{n} fwd {w}"]
            assert not np.isnan(rev).any() and not np.isnan(fwd).any(), (n, w)
            np.testing.assert_allclose(
                rev, fwd, rtol=rtol, atol=atol, err_msg=f"{n} {w}"
            )
            want = big if (n, w) == ("log_gammainc_array", "x") else False
            assert np.array_equal(np.isinf(rev), want | np.zeros_like(big)), (n, w)
    k = np.flatnonzero((a == 1.0) & (x == 1e-20))[0]
    assert got["gammainc_array rev x"][k] == pytest.approx(1.0, rel=rtol)


# (a, x): the upper tail far past the table (ln Q ~ -x), the lower tail
# where x/a underflows
_TAILS = {
    "float64": (
        [(a, x) for a in (0.5, 2.5, 30.0, 1e3) for x in (1e4, 1e8, 1e15, 1e30, 1e100)]
        + [(2.5, 1e300)],
        [(0.5, 1e-300), (2.5, 1e-300), (30.0, 1e-200), (1e3, 1e-30), (1e3, 1e-306)]
        + [(1e5, 1e-100)],
    ),
    "float32": (
        [(a, x) for a in (0.5, 2.5, 30.0, 1e3) for x in (1e4, 1e8, 1e10, 1e20, 1e37)],
        [(0.5, 1e-37), (2.5, 1e-37), (30.0, 1e-30), (1e3, 1e-36), (1e5, 1e-20)],
    ),
}


def _mp_tail(a, x, upper):
    """(ln F, d ln F/da, d ln F/dx), F = Q if upper else P, with enough
    digits for |ln F| (~1e100 at x = 1e100) to leave 40 for the slopes."""
    if upper:
        F = lambda s, x: mp.gammainc(s, x, mp.inf, regularized=True)  # noqa: E731
    else:
        F = lambda s, x: mp.gammainc(s, 0, x, regularized=True)  # noqa: E731
    size = abs(mp.log(F(mp.mpf(a), mp.mpf(x))))
    with mp.workdps(50 + int(mp.log10(size + 1))):
        am, xm = mp.mpf(a), mp.mpf(x)
        lf = mp.log(F(am, xm))
        da = mp.diff(lambda s: mp.log(F(s, xm)), am)
        ld = (am - 1) * mp.log(xm) - xm - mp.loggamma(am)
        dx = (-1 if upper else 1) * mp.exp(ld - lf)
        return float(lf), float(da), float(dx)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_tails_against_mpmath(dtype, tmp_path):
    """Values and both modes' slopes in the far tails, against mpmath with
    the precision raised for |ln F|. Differentiating -a phi as a product
    once cancelled two terms of size x/a here: d ln Q/da at (2.5, 1e8) was
    13.6 for 17.7 in float32, 4e83 at x = 1e100 in float64. Forming x/a
    lost ln P where it underflows: log_gammainc_array(1000, 1e-36) was -inf
    in float32 for -88805.19. Measured worst, |err| / max(1, |ref|):
    float64 2.2e-16, bar 1e-15; float32 1.0e-7, bar 4e-7. A slope whose
    true value overflows (d ln P/dx ~ a/x at x = 1e-306) must be inf."""
    bar = 1e-15 if dtype == "float64" else 4e-7
    top = np.finfo(dtype).max
    for pts, upper in zip(_TAILS[dtype], (True, False)):
        a, x = (np.array(v) for v in zip(*pts))
        got = _in_dtype(dtype, tmp_path, a, x)
        n = "log_gammaincc_array" if upper else "log_gammainc_array"
        refs = np.array([_mp_tail(*p, upper) for p in pts]).T
        for key, r in zip(("", " rev a", " rev x"), refs):
            gots = [got[n + key]] + (
                [got[n + key.replace("rev", "fwd")]] if key else []
            )
            for g in gots:
                over = np.abs(r) > top
                assert np.array_equal(g[over], np.sign(r[over]) * np.inf), (n, key)
                assert np.max(_err(g[~over], r[~over]), initial=0.0) <= bar, (n, key)


def test_dx_against_density(refs):
    a, x, _, _, rp, rq = refs

    def dx(f):
        return np.asarray(jax.jit(jax.grad(lambda x: jnp.sum(f(a, x))))(x))

    gp, gq = dx(chebax.log_gammainc_array), dx(chebax.log_gammaincc_array)
    for g, r in ((gp, rp), (gq, rq)):
        assert np.all(np.isfinite(g))
        # the deep tails' slopes over/underflow together with the reference
        live = np.isfinite(r) & (np.abs(r) > 1e-300)
        assert np.max(np.abs(g[live] - r[live]) / np.abs(r[live])) <= 8e-13


def test_da_against_mpmath(refs):
    a, x = refs[:2]
    zone = _zone(a, x)
    rng = np.random.default_rng(5)
    pick = np.concatenate(
        [rng.choice(np.flatnonzero(zone == z), 10, replace=False) for z in range(7)]
    )
    ap, xp = a[pick], x[pick]

    def da(f):
        return np.asarray(jax.jit(jax.grad(lambda a: jnp.sum(f(a, xp))))(ap))

    gp, gq = da(chebax.log_gammainc_array), da(chebax.log_gammaincc_array)
    for k, (ak, xk) in enumerate(zip(ap, xp)):
        am, xm = mp.mpf(ak), mp.mpf(xk)
        rp = mp.diff(lambda s: mp.log(mp.gammainc(s, 0, xm, regularized=True)), am)
        rq = mp.diff(lambda s: mp.log(mp.gammainc(s, xm, mp.inf, regularized=True)), am)
        for g, r in ((gp[k], rp), (gq[k], rq)):
            assert abs(g - float(r)) / max(1.0, abs(float(r))) <= 5e-13, (ak, xk)


def test_endpoints_and_invalid_a():
    xs = jnp.asarray([-3.0, -jnp.inf, 0.0, jnp.inf, jnp.nan])
    a = jnp.full(xs.shape, 2.5)
    for f, want in (
        (chebax.gammainc_array, [0.0, 0.0, 0.0, 1.0]),
        (chebax.gammaincc_array, [1.0, 1.0, 1.0, 0.0]),
        (chebax.log_gammainc_array, [-np.inf, -np.inf, -np.inf, 0.0]),
        (chebax.log_gammaincc_array, [0.0, 0.0, 0.0, -np.inf]),
    ):
        got = np.asarray(f(a, xs))
        assert np.array_equal(got[:4], want) and np.isnan(got[4]), f.__name__
        # below the table, a = inf and a = nan are nan, whatever x is
        bad = np.asarray(f(jnp.asarray([0.05, 0.0, -1.0, jnp.inf, jnp.nan]), 2.0))
        assert np.all(np.isnan(bad)), f.__name__
    assert float(chebax.gammainc_array(0.1, 1.0)) > 0.0


def test_x_zero_slope_is_the_density():
    """dP/dx at x = 0 is the Gamma density there: inf below a = 1, 1 at
    a = 1, 0 above; dQ/dx is its negative, d ln P/dx is inf for every a and
    d ln Q/dx = -density. Exact, bar is equality."""
    a = jnp.asarray([0.1, 0.5, 0.999, 1.0, 1.001, 2.5, 10.0, 50.0])
    an = np.asarray(a)
    d0 = np.where(an < 1.0, np.inf, np.where(an > 1.0, 0.0, 1.0))
    gp, gq, glp, glq = [
        np.asarray(jax.vmap(jax.grad(f, 1), in_axes=(0, None))(a, 0.0)) for f in ALLFN
    ]
    assert np.array_equal(gp, d0) and np.array_equal(gq, -d0)
    assert np.all(np.isposinf(glp)) and np.array_equal(glq, -d0)
    assert np.all(np.asarray(chebax.gammainc_array(a, 0.0)) == 0.0)


def test_masked_lanes_stay_finite():
    """A batch mixing every zone with x < 0, x = 0, x = inf, x = nan and
    invalid a: the masked branches see finite dummies, so no other lane is
    poisoned. x < 0 and x = inf are constant there, so flat in both modes;
    lanes whose value is nan get a 0 cotangent."""
    xs = np.array([-3.0, 0.0, 1e-8, 0.3, 1.0, 1.9, 8.0, 40.0, 1e300, np.inf, np.nan])
    av = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 11.0, 300.0, 0.05])
    an, xn = np.meshgrid(av, xs, indexing="ij")
    got = _slopes(an, xn)
    valid = an >= 0.1
    inner = valid & (xn > 0.0) & np.isfinite(xn)
    flat = valid & ((xn < 0.0) | (xn == np.inf))
    assert flat.sum() == 2 * len(av[av >= 0.1])
    for f in ALLFN:
        n = f.__name__
        for key in (" rev a", " rev x", " fwd a", " fwd x"):
            g = got[n + key]
            assert np.all(np.isfinite(g[inner])), (n, key)
            assert np.all(g[flat] == 0.0), (n, key)
        assert np.all(got[n + " rev a"][~inner & ~flat & (xn != 0.0)] == 0.0), n


def test_broadcast_jit_and_arguments():
    a = jnp.asarray([[0.5], [2.5], [40.0]])
    x = jnp.asarray([0.3, 2.0, 9.0, 60.0])
    for f in ALLFN:
        got = f(a, x)
        assert got.shape == (3, 4), f.__name__
        np.testing.assert_allclose(np.asarray(jax.jit(f)(a, x)), np.asarray(got))
        # one element alone gives the same number as in the batch
        assert float(f(2.5, 9.0)) == pytest.approx(float(got[1, 2]), rel=1e-15)
        # keywords, integers and numpy scalars are the same call
        assert float(f(a=2, x=3)) == float(f(2.0, 3.0)), f.__name__
        assert float(f(np.int64(40), np.float32(9.0))) == float(f(40.0, 9.0))
        g = jax.grad(lambda a, x: f(a, x), (0, 1))(2.5, 3.0)
        gk = jax.grad(lambda a, x: f(x=x, a=a), (0, 1))(2.5, 3.0)
        assert list(map(float, g)) == list(map(float, gk)), f.__name__


def test_lower_end_is_the_family_domain():
    assert t.ALO == domains.GAMMAINC.lo


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    gammainc_array_gen.main(tmp_path)
    assert (tmp_path / "gammainc_array_table.py").read_text() == pathlib.Path(
        t.__file__
    ).read_text()


def test_density_where_the_cdf_underflows():
    # dP/dx was exp(ln P) d ln P/dx with slope 0 wherever P underflows, which
    # lost representable densities: 0 for 1e-25 at a = 2, x = 1e-25 in float32
    # (review, 2026-11-06; float32 in test_x64_off). Below the normal floor it
    # is the density from its logs: measured 6.3e-15 at a = 2, x = 1e-160
    a = jnp.asarray(2.0)
    for f, s in (
        (chebax.gammainc_array, 1.0),
        (chebax.gammaincc_array, -1.0),
        (chebax.log_gammaincc_array, -1.0),
    ):
        g = float(jax.grad(lambda x: f(a, x))(jnp.asarray(1e-160)))
        assert abs(g / (s * 1e-160) - 1) <= 3e-14, (f.__name__, g)
