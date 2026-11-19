"""The noncentral Student t log-density: nct_logpdf(nu, mu, x), nu per element
in [0.1, inf), mu and x real.

References are mpmath at 40 dps,

    ln f = ln K + lnGamma(nu + 1) - (nu + 1)/2 ln P - (mu^2 - y^2)/2 + ln Hh_nu(-y),

P = nu + x^2, y = mu x/sqrt(P), K = 2 (nu/2)^(nu/2)/(Gamma(nu/2) sqrt(2 pi)),
with hh_gen's ln Hh (pcfd up to nu = 150, its saddle quadrature above), on
1,500 points: nu log-uniform on [1, 100] with x within 5 of mu, |mu| <= 10;
nu on [0.1, 1] with |x| <= 100; |mu| on [10, 50] with x/mu on [0.8, 1.3];
x opposite to mu; a wide box (nu to 1e4, |mu| <= 50, |x| <= 200); and 60
points with nu log-uniform on [150, 1e6] near the mode. Errors are |ln f -
ref| / max(1, |ref|).

Measured worst, the same on jax 0.11 and 0.5.3 (CI's floor): float64 ln f
4.1e-16 (bar 1e-15); float32 2.3e-7 on the float32-rounded points (bar
1e-6). Slopes relative to max(1, |slope|) against mp.diff: d/dnu 2.9e-15,
d/dmu 1.9e-15, d/dx 4.7e-16 (bar 3e-14, as hh's slopes, which T's carry).
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
from chebax._src.recipes import hh_gen, nct_gen
from chebax._src.recipes import nct_table as t

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40


def _points(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    un = lambda lo, hi: rng.uniform(lo, hi, n)
    kind = rng.integers(0, 5, n)
    mu_core = un(-10, 10)
    mu_big = un(10, 50) * rng.choice([-1, 1], n)
    nu = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [lu(1, 100), lu(0.1, 1), lu(1, 1e3), lu(1, 100)],
        lu(0.1, 1e4),
    )
    mu = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [mu_core, un(-5, 5), mu_big, mu_core],
        un(-50, 50),
    )
    x = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [mu + un(-5, 5), un(-100, 100), mu * un(0.8, 1.3), -np.sign(mu) * un(0, 20)],
        un(-200, 200),
    )
    big = lu(150, 1e6)[:60]
    mb = un(-10, 10)[:60]
    return (
        np.concatenate([nu, big]),
        np.concatenate([mu, mb]),
        np.concatenate([x, mb + un(-4, 4)[:60]]),
    )


def _ref1(nu, mu, x):
    p = nu + x * x
    y = mu * x / mp.sqrt(p)
    lnk = (
        mp.log(2)
        + nu / 2 * mp.log(nu / 2)
        - mp.loggamma(nu / 2)
        - mp.log(2 * mp.pi) / 2
    )
    return (
        lnk
        + mp.loggamma(nu + 1)
        - (nu + 1) / 2 * mp.log(p)
        - (mu * mu - y * y) / 2
        + hh_gen.log_hh(mp, nu, -y)
    )


def _mp_ref(nu, mu, x):
    return np.array(
        [
            float(_ref1(*(mp.mpf(float(v)) for v in row)))
            for row in zip(np.ravel(nu), np.ravel(mu), np.ravel(x))
        ]
    )


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    assert np.all(np.isfinite(got)), "unexpected nonfinite values"
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


@pytest.fixture(scope="module")
def refs():
    nu, mu, x = _points()
    return nu, mu, x, _mp_ref(nu, mu, x)


def test_log_against_mpmath(refs):
    nu, mu, x, ln_f = refs
    assert np.max(_err(jax.jit(chebax.nct_logpdf)(nu, mu, x), ln_f)) <= 1e-15


def _x64_off(tmp_path, body, **arrays):
    """Run body in a child process with x64 off (process-global): it reads
    the arrays from d and puts its results in the dict out."""
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
    """x64 off: B's leading K32 terms, hh's 4 x 4 blocks, every closed form in
    float32, on float32-rounded points."""
    nu, mu, x = (v[::3].astype(np.float32) for v in refs[:3])
    got = _x64_off(
        tmp_path,
        """
        out["l"] = np.asarray(jax.jit(chebax.nct_logpdf)(d["nu"], d["mu"], d["x"]))
        assert out["l"].dtype == np.float32
        """,
        nu=nu,
        mu=mu,
        x=x,
    )
    ln_f = _mp_ref(*(v.astype(np.float64) for v in (nu, mu, x)))
    assert np.max(_err(got["l"], ln_f)) <= 1e-6


def test_slopes_against_mpmath(refs):
    """d/dnu, d/dmu and d/dx against mp.diff, relative to max(1, |slope|)."""
    nu, mu, x = (v[::25] for v in refs[:3])
    g = jax.jit(jax.vmap(jax.grad(chebax.nct_logpdf, (0, 1, 2))))
    got = [np.asarray(v) for v in g(nu, mu, x)]
    worst = [0.0, 0.0, 0.0]
    for k in range(nu.size):
        a, b, c = (mp.mpf(float(v[k])) for v in (nu, mu, x))
        ref = (
            mp.diff(lambda v: _ref1(v, b, c), a),
            mp.diff(lambda v: _ref1(a, v, c), b),
            mp.diff(lambda v: _ref1(a, b, v), c),
        )
        for m, r in enumerate(ref):
            worst[m] = max(worst[m], float(abs(got[m][k] - r) / max(1, abs(r))))
    assert max(worst) <= 3e-14, worst


def test_both_modes_agree(refs):
    args = tuple(jnp.asarray(v[::50]) for v in refs[:3])
    rev = jax.grad(lambda *a: chebax.nct_logpdf(*a).sum(), (0, 1, 2))(*args)
    for k, r in enumerate(rev):
        tan = [jnp.zeros_like(args[0])] * 3
        tan[k] = jnp.ones_like(args[0])
        fwd = jax.jvp(chebax.nct_logpdf, args, tuple(tan))[1]
        np.testing.assert_allclose(fwd, r, rtol=1e-14, atol=0)


def test_central_t():
    """mu = 0 is Student's t: lnGamma((nu+1)/2) - lnGamma(nu/2) - ln(nu pi)/2
    - (nu+1)/2 ln(1 + x^2/nu)."""
    for nu in (0.1, 0.9, 4.0, 30.0, 2e3, 1e7):
        for x in (-40.0, -1.5, 0.0, 0.7, 6.0):
            a, b = mp.mpf(nu), mp.mpf(x)
            ref = (
                mp.loggamma((a + 1) / 2)
                - mp.loggamma(a / 2)
                - mp.log(a * mp.pi) / 2
                - (a + 1) / 2 * mp.log1p(b * b / a)
            )
            got = float(chebax.nct_logpdf(nu, 0.0, x))
            assert abs(got - float(ref)) <= 1e-15 * max(1, abs(float(ref))), (nu, x)


def test_edges_and_limits():
    inf, nan = np.inf, np.nan
    nu = jnp.array([inf, 3.0, 3.0, 3.0, 3.0, 0.05, nan, 3.0])
    mu = jnp.array([1.5, 1.0, inf, -inf, 1.0, 1.0, 1.0, nan])
    x = jnp.array([0.25, -inf, 1.0, 1.0, inf, 1.0, 1.0, 1.0])
    got = np.asarray(chebax.nct_logpdf(nu, mu, x))
    normal = -0.5 * 1.25**2 - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(got[0], normal, rtol=1e-15)
    np.testing.assert_array_equal(got[1:], [-inf, -inf, -inf, -inf, nan, nan, nan])
    g = jax.grad(chebax.nct_logpdf, (0, 1, 2))(inf, 1.5, 0.25)
    np.testing.assert_allclose(np.asarray(g), [0.0, -1.25, 1.25], rtol=1e-15)
    assert np.isnan(jax.grad(chebax.nct_logpdf)(0.05, 1.0, 1.0))


def test_huge_arguments():
    """x and mu past where x^2 or x^2 mu^2 overflow, against mpmath; nu past
    where nu^2 does, against the central t and the normal limit (corrections
    O(1/nu)); nan where nu/S^2 leaves the normal range (S ~ max(|x|, |mu|))."""
    pts = [
        (1.0, 2.0, 1e150),
        (3.0, -1e100, 1e100),
        (0.5, 1e100, 1e-3),
        (2.0, 3.0, -1e80),
    ]
    nu, mu, x = (np.array(v) for v in zip(*pts))
    got = np.asarray(chebax.nct_logpdf(nu, mu, x))
    assert np.max(_err(got, _mp_ref(nu, mu, x))) <= 1e-15
    half_ln2pi = 0.5 * np.log(2 * np.pi)
    got = np.asarray(chebax.nct_logpdf(1e250, jnp.array([0.0, 5.0]), 4.0))
    np.testing.assert_allclose(got, [-8.0 - half_ln2pi, -0.5 - half_ln2pi], rtol=1e-15)
    assert np.isnan(chebax.nct_logpdf(1.0, 2.0, 1e160))


def test_broadcast_and_jit():
    nu = jnp.array([[1.5], [70.0]])
    mu = jnp.array([[-2.0, 0.5, 3.0]])
    x = jnp.array([1.0, -0.5, 4.0])
    got = jax.jit(chebax.nct_logpdf)(nu, mu, x)
    assert got.shape == (2, 3)
    ref = _mp_ref(*np.broadcast_arrays(*(np.asarray(v) for v in (nu, mu, x))))
    np.testing.assert_allclose(np.asarray(got).ravel(), ref, rtol=2e-15, atol=0)


def test_b_table():
    """B(nu) = R(nu) - R(nu/2) from the panels and Stirling's terms, and its
    slope, against mpmath on both sides of NU_S. B adds to ln f, so its error
    is absolute: 2.8e-17 measured (one ulp of B at nu = 0.11), slope 1.1e-16
    relative to max(1, |slope|)."""
    from chebax._src.recipes.nct import _b

    rng = np.random.default_rng(1)
    nu = np.concatenate([np.exp(rng.uniform(np.log(0.1), np.log(1e6), 300)), [t.NU_S]])
    val, slope = (np.asarray(v) for v in _b(jnp.asarray(nu), True))
    for k in range(nu.size):
        a = mp.mpf(nu[k])
        assert abs(val[k] - float(nct_gen.binet_diff(mp, a))) <= 1e-16
        d = float(mp.diff(lambda v: nct_gen.binet_diff(mp, v), a))
        assert abs(slope[k] - d) <= 5e-16 * max(1.0, abs(d))


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    nct_gen.main(out_dir=tmp_path)
    here = pathlib.Path(nct_gen.__file__).parent / "nct_table.py"
    assert (tmp_path / "nct_table.py").read_bytes() == here.read_bytes()
