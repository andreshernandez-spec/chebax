"""Per-element gammainc: gammainc_elem, gammaincc_elem and their logs, with
a different a on every element, a >= 0.1, x in [0, inf).

References are mpmath at 40 dps on 2,300 points over every zone of
gammainc_elem_gen: a from 0.1 to 1e4, the bulk, both tails (ln P down to
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
5e-13. The IEEE endpoints and the x = 0 slopes are exact, so those
tests assert equality.
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
from chebax._src.recipes import gammainc_elem_gen
from chebax._src.recipes import gammainc_elem_table as t

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

ALLFN = [
    chebax.gammainc_elem,
    chebax.gammaincc_elem,
    chebax.log_gammainc_elem,
    chebax.log_gammaincc_elem,
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
        lambda a, x: (chebax.log_gammainc_elem(a, x), chebax.log_gammaincc_elem(a, x))
    )
    gp, gq = f(a, x)
    assert np.max(_err(gp, lp)) <= 3.5e-15
    assert np.max(_err(gq, lq)) <= 3.5e-14


def test_values_absolute(refs):
    a, x, lp, lq = refs[:4]
    p = np.asarray(jax.jit(chebax.gammainc_elem)(a, x))
    q = np.asarray(jax.jit(chebax.gammaincc_elem)(a, x))
    assert np.max(np.abs(p - np.exp(lp))) <= 3.5e-15
    assert np.max(np.abs(q - np.exp(lq))) <= 3.5e-15


def test_float32(refs, tmp_path):
    """float32 is the default runtime (x64 off): the leading 7 x 7 block of
    the same patches, with erfcx and every prefactor in float32 too. x64
    is process-global, so the public functions run in a child process, on
    the float32-rounded points."""
    a, x = refs[:2]
    a32, x32 = a.astype(np.float32), x.astype(np.float32)
    np.savez(tmp_path / "in.npz", a=a32, x=x32)
    code = textwrap.dedent(f"""
        import numpy as np, jax, jax.numpy as jnp, chebax
        assert jnp.empty(()).dtype == jnp.float32, "x64 leaked in"
        d = np.load({str(tmp_path / "in.npz")!r})
        out = {{
            f.__name__: np.asarray(jax.jit(f)(d["a"], d["x"]))
            for f in (chebax.gammainc_elem, chebax.gammaincc_elem,
                      chebax.log_gammainc_elem, chebax.log_gammaincc_elem)
        }}
        assert all(v.dtype == np.float32 for v in out.values())
        np.savez({str(tmp_path / "out.npz")!r}, **out)
    """)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, run.stderr[-3000:]
    got = np.load(tmp_path / "out.npz")
    lp, lq = _mp_logs(a32.astype(np.float64), x32.astype(np.float64))
    assert np.max(_err(got["log_gammainc_elem"], lp)) <= 2e-6
    assert np.max(_err(got["log_gammaincc_elem"], lq)) <= 1.2e-5
    assert np.max(np.abs(got["gammainc_elem"] - np.exp(lp))) <= 1.6e-6
    assert np.max(np.abs(got["gammaincc_elem"] - np.exp(lq))) <= 1.6e-6


def test_dx_against_density(refs):
    a, x, _, _, rp, rq = refs

    def dx(f):
        return np.asarray(jax.jit(jax.grad(lambda x: jnp.sum(f(a, x))))(x))

    gp, gq = dx(chebax.log_gammainc_elem), dx(chebax.log_gammaincc_elem)
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

    gp, gq = da(chebax.log_gammainc_elem), da(chebax.log_gammaincc_elem)
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
        (chebax.gammainc_elem, [0.0, 0.0, 0.0, 1.0]),
        (chebax.gammaincc_elem, [1.0, 1.0, 1.0, 0.0]),
        (chebax.log_gammainc_elem, [-np.inf, -np.inf, -np.inf, 0.0]),
        (chebax.log_gammaincc_elem, [0.0, 0.0, 0.0, -np.inf]),
    ):
        got = np.asarray(f(a, xs))
        assert np.array_equal(got[:4], want) and np.isnan(got[4]), f.__name__
        # below the table, a = inf and a = nan are nan, whatever x is
        bad = np.asarray(f(jnp.asarray([0.05, 0.0, -1.0, jnp.inf, jnp.nan]), 2.0))
        assert np.all(np.isnan(bad)), f.__name__
    assert float(chebax.gammainc_elem(0.1, 1.0)) > 0.0


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
    assert np.all(np.asarray(chebax.gammainc_elem(a, 0.0)) == 0.0)


def test_masked_lanes_stay_finite():
    """A batch mixing every zone with x < 0, x = 0, x = inf, x = nan and
    invalid a: the masked branches see finite dummies, so their gradients
    are 0 (or nan where the value is nan) and no other lane is poisoned."""
    xs = np.array([-3.0, 0.0, 1e-8, 0.3, 1.0, 1.9, 8.0, 40.0, 1e300, np.inf, np.nan])
    av = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 11.0, 300.0, 0.05])
    a, x = (jnp.asarray(v) for v in np.meshgrid(av, xs, indexing="ij"))
    live = (np.asarray(a) >= 0.1) & np.isfinite(np.asarray(x))
    for f in ALLFN:
        ga, gx = jax.jit(jax.grad(lambda a, x: jnp.sum(f(a, x)), (0, 1)))(a, x)
        ga, gx = np.asarray(ga), np.asarray(gx)
        inner = live & (np.asarray(x) > 0.0)
        assert np.all(np.isfinite(ga[inner]) & np.isfinite(gx[inner])), f.__name__
        flat = (np.asarray(x) < 0.0) | (np.asarray(x) == np.inf)
        assert np.all(ga[flat & live] == 0.0) and np.all(gx[flat & live] == 0.0)


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
    gammainc_elem_gen.main(tmp_path)
    assert (tmp_path / "gammainc_elem_table.py").read_text() == pathlib.Path(
        t.__file__
    ).read_text()
