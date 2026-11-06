"""The DEFAULT runtime: jax with x64 disabled, i.e. float32 everywhere.

conftest.py enables x64 for the whole suite, which is what the accuracy
contract is stated in, but it also means the configuration most users get
by default was never exercised. Four release-blocking nan bugs (betaincinv,
gammaincinv, stdtrit, besseli_ratio) lived only there.

x64 is process-global and cannot be turned off once jax has initialized, so
every check here runs in a child process. References are mpmath at 40 dps.
The bar is float32 grade: 2e-5 relative on values, 3e-5 on quantiles (the
solvers inherit the CDF's own float32 accuracy), which is ~150 eps32 and
about 4x the measured worst case (7.6e-6, on gammaincinv(100, 0.5)).

CI runs this file as its own job (see .github/workflows/tests.yml) so a
regression in the default configuration cannot hide behind the x64 suite.
"""

import os
import subprocess
import sys

import pytest

# value: (expression evaluated in the child, 40-dps reference)
CASES = {
    "besselj": ("chebax.besselj_frozen(2.5)(7.0)", -0.2834366512016992),
    "bessely": ("chebax.bessely_frozen(1.5)(3.0)", 0.087008090720835282),
    "besseli_scaled": (
        "chebax.besseli_frozen(2.5, scaled=True)(4.0)",
        0.087138975806603666,
    ),
    "besselk": ("chebax.besselk_frozen(1.5)(2.0)", 0.17990665795209217),
    "besselk_scalar": ("chebax.besselk_scalar(1.7, 5.0)", 0.004802603310190489),
    "log_besselk_scalar": (
        "chebax.log_besselk_scalar(0.0, 300.0)",
        -302.62651585930441,
    ),
    "besseli_scalar": ("chebax.besseli_scalar(1.3, 2.0)", 1.29081921513588),
    # the two besseli_ratio cases the review reported as nan under x64 off
    "besseli_ratio": ("chebax.besseli_ratio(0.0, 3.0)", 0.80998529395650453),
    "besseli_ratio_small_x": ("chebax.besseli_ratio(4.0, 0.5)", 0.049896203861781465),
    "besseli_ratio_tiny_x": ("chebax.besseli_ratio(8.9, 1e-6)", 5.0505050505050393e-08),
    "besseli_ratio_inf": ("chebax.besseli_ratio(0.0, jnp.inf)", 1.0),
    "betainc_scalar": ("chebax.betainc_scalar(2.0, 3.0, 0.4)", 0.5248),
    "log_betainc_scalar": (
        "chebax.log_betainc_scalar(2.0, 3.0, 0.01)",
        -7.4319532486885614,
    ),
    "gammainc_scalar": ("chebax.gammainc_scalar(2.5, 3.0)", 0.6937810815867216),
    "gammainc_array": ("chebax.gammainc_array(2.5, 3.0)", 0.6937810815867216),
    "log_gammaincc_array": (
        "chebax.log_gammaincc_array(2.5, 40.0)",
        -34.714103456537936,
    ),
    "gammaincc_scalar": ("chebax.gammaincc_scalar(2.5, 9.0)", 0.0029464045878802904),
    "log_gammaincc_scalar": (
        "chebax.log_gammaincc_scalar(2.5, 40.0)",
        -34.714103456537936,
    ),
    "stdtr": ("chebax.stdtr(4.0, 1.2)", 0.85182430334382328),
    "stdtr_tail": ("chebax.stdtr(20.0, -2.0)", 0.029632767723285236),
    "dawsn": ("chebax.dawsn(1.0)", 0.53807950691276842),
    "erfcx": ("chebax.erfcx(2.0)", 0.25539567631050574),
    "lambertw": ("chebax.lambertw(3.0, 0)", 1.04990889496404),
    "spherical_jn": ("chebax.spherical_jn_frozen(2)(4.0)", 0.27628368577135016),
    "gammainc_at_inf": ("chebax.gammainc_scalar(2.0, jnp.inf)", 1.0),
}

# the inverses, which is where the x64-off breakage was concentrated
QUANTILES = {
    "betaincinv": ("chebax.betaincinv(2.0, 3.0, 0.4)", 0.32916650337840787),
    "betaincinv_10_10": ("chebax.betaincinv(10.0, 10.0, 0.5)", 0.5),
    "gammaincinv": ("chebax.gammaincinv(2.5, 0.4)", 1.8277498115707929),
    "gammaincinv_a20": ("chebax.gammaincinv(20.0, 0.1)", 14.525261465272755),
    "gammaincinv_a100": ("chebax.gammaincinv(100.0, 0.5)", 99.666864919315489),
    "chi2inv": ("chebax.chi2inv(3.0, 0.5)", 2.3659738843753383),
    "stdtrit": ("chebax.stdtrit(20.0, 0.1)", -1.3253407069850463),
    "stdtrit_nu4": ("chebax.stdtrit(4.0, 0.25)", -0.74069708411268263),
}


def _run(body):
    """Execute body in a child process with x64 left at its default (off)."""
    code = (
        "import numpy as np, jax, jax.numpy as jnp, chebax\n"
        "assert jnp.empty(()).dtype == jnp.float32, 'x64 leaked in'\n" + body
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


@pytest.mark.parametrize("name", sorted(CASES))
def test_value_under_x64_off(name):
    expr, ref = CASES[name]
    _run(
        f"got = float({expr})\n"
        f"assert np.isfinite(got), ('not finite', got)\n"
        f"ref = {ref!r}\n"
        f"err = abs(got - ref) / max(abs(ref), 1e-30)\n"
        f"assert err <= 2e-5, (got, ref, err)\n"
    )


@pytest.mark.parametrize("name", sorted(QUANTILES))
def test_quantile_under_x64_off(name):
    expr, ref = QUANTILES[name]
    _run(
        f"got = float({expr})\n"
        f"assert np.isfinite(got), ('not finite', got)\n"
        f"ref = {ref!r}\n"
        f"err = abs(got - ref) / max(abs(ref), 1e-30)\n"
        f"assert err <= 3e-5, (got, ref, err)\n"
    )


def test_dtype_is_float32_not_promoted():
    # everything must come back float32, not silently widened
    _run(
        "for v in [chebax.besselk_frozen(1.5)(2.0), chebax.betaincinv(2.0, 3.0, 0.4),\n"
        "          chebax.gammaincinv(2.5, 0.4), chebax.stdtrit(4.0, 0.25),\n"
        "          chebax.besseli_ratio(0.0, 3.0), chebax.stdtr(4.0, 1.2),\n"
        "          chebax.log_gammainc_array(jnp.asarray([0.5, 30.0]), 2.0)]:\n"
        "    assert v.dtype == jnp.float32, v.dtype\n"
    )


def test_gradients_are_finite_under_x64_off():
    _run(
        "gs = [jax.grad(chebax.besselk(1.5))(2.0),\n"
        "      jax.grad(chebax.besselk_scalar, argnums=0)(1.7, 5.0),\n"
        "      jax.grad(chebax.betaincinv, argnums=2)(2.0, 3.0, 0.4),\n"
        "      jax.grad(chebax.gammaincinv, argnums=1)(2.5, 0.4),\n"
        "      jax.grad(chebax.stdtrit, argnums=1)(4.0, 0.25),\n"
        "      jax.grad(chebax.besseli_ratio, argnums=1)(0.0, 0.0),\n"
        "      jax.grad(chebax.vonmises_cdf, argnums=0)(2.0, 0.5),\n"
        "      jax.grad(chebax.log_gammaincc_array, argnums=0)(30.0, 25.0)]\n"
        "for g in gs:\n"
        "    assert np.isfinite(float(g)), g\n"
        "assert abs(float(gs[5]) - 0.5) <= 1e-5, gs[5]\n"
    )


def test_parameter_gradients_across_both_panels_under_x64_off():
    # the panel not taken was evaluated past its range, where its series
    # overflows float32, and reverse mode turned inf * 0 into nan: d/dnu was
    # nan for nu > 3.3 (besselk, matern) and nu > 47 (stdtr). log_stdtr's
    # ln(1 - I) did the same outside its lanes at large nu (t = +-30). Forward
    # mode never was, so it is the reference: measured |rev - fwd| 1.7e-5
    # relative (besselk) and 7e-7 absolute (matern, stdtr, log_stdtr; their
    # nu-slopes are small, so float32 leaves them only absolutely accurate).
    _run(
        "cases = [\n"
        "    (lambda n: chebax.besselk_scalar(n, 2.0), np.linspace(0.05, 10.0, 41)),\n"
        "    (lambda n: chebax.log_besselk_scalar(n, 2.0), np.linspace(0.05, 10.0, 41)),\n"
        "    (lambda n: chebax.matern(n, 0.7, 1.0), np.linspace(0.05, 10.0, 41)),\n"
        "    (lambda n: chebax.stdtr(n, 1.5), np.geomspace(0.2, 200.0, 41)),\n"
        "    (lambda n: chebax.log_stdtr(n, -1.5), np.geomspace(0.2, 200.0, 41)),\n"
        "    (lambda n: chebax.log_stdtr(n, -30.0), np.geomspace(0.2, 200.0, 41)),\n"
        "    (lambda n: chebax.log_stdtr(n, 30.0), np.geomspace(0.2, 200.0, 41)),\n"
        "]\n"
        "for i, (f, ps) in enumerate(cases):\n"
        "    ps = jnp.asarray(ps, jnp.float32)\n"
        "    rev = np.asarray(jax.vmap(jax.grad(f))(ps))\n"
        "    fwd = np.asarray(jax.vmap(lambda p: jax.jvp(f, (p,), (1.0,))[1])(ps))\n"
        "    assert np.all(np.isfinite(rev)), (i, ps[~np.isfinite(rev)])\n"
        "    np.testing.assert_allclose(rev, fwd, rtol=1e-4, atol=4e-6, err_msg=str(i))\n"
    )


def test_matern_close_points_under_x64_off():
    # z^nu underflowed and K_nu overflowed apart at small z: matern(6, 1e-8)
    # was 0 * inf = nan. The reference at the clamp z = 1e-6 rounds to 1
    _run(
        "for nu in (0.5, 2.5, 6.0, 9.9):\n"
        "    for r in (1e-18, 1e-8, 1e-6):\n"
        "        got = float(jax.jit(chebax.matern)(nu, jnp.float32(r)))\n"
        "        assert abs(got - 1.0) <= 2e-5 or nu == 0.5, (nu, r, got)\n"
        "        assert np.isfinite(got), (nu, r, got)\n"
    )


def test_widest_float32_domain_under_jit():
    # float32's version of the reciprocal flush: half-width 1e38 is past
    # 1/tiny = 2^126, so XLA's folded reciprocal was flushed to zero and a
    # jitted vectorized call evaluated at t = 0
    _run(
        "s = chebax.ChebSeries(np.array([0.0, 1.0], np.float32), (-1e38, 1e38))\n"
        "xs = jnp.asarray([-1e38, -3e37, 0.0, 5e37, 1e38], jnp.float32)\n"
        "got = np.asarray(jax.jit(s)(xs))\n"
        "ref = np.array([-1.0, -0.3, 0.0, 0.5, 1.0])\n"
        "assert np.allclose(got, ref, rtol=1e-6, atol=1e-7), got\n"
    )


def test_tables_materialize_in_float32():
    # a cache built under x64 off must hold float32, and must NOT be served
    # to a later x64-on process (the dtype tag); the erf family's no-argument
    # cache is the one that used to miss this
    _run(
        "assert np.asarray(chebax.besselk(1.5).ltil.coef).dtype == np.float32\n"
        "from chebax._src.recipes.erf_family import _series\n"
        "chebax.dawsn(1.0)\n"
        "assert np.asarray(_series()[0].coef).dtype == np.float32\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "assert np.asarray(_series()[0].coef).dtype == np.float64\n"
        "assert abs(float(chebax.dawsn(1.0)) - 0.53807950691276842) <= 1e-15\n"
    )
