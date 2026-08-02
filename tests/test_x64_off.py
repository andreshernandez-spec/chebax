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
    "besselj": ("chebax.besselj(2.5)(7.0)", -0.2834366512016992),
    "bessely": ("chebax.bessely(1.5)(3.0)", 0.087008090720835282),
    "besseli_scaled": ("chebax.besseli(2.5, scaled=True)(4.0)", 0.087138975806603666),
    "besselk": ("chebax.besselk(1.5)(2.0)", 0.17990665795209217),
    "besselk_fn": ("chebax.besselk_fn(1.7, 5.0)", 0.004802603310190489),
    "log_besselk_fn": ("chebax.log_besselk_fn(0.0, 300.0)", -302.62651585930441),
    "besseli_fn": ("chebax.besseli_fn(1.3, 2.0)", 1.29081921513588),
    # the two besseli_ratio cases the review reported as nan under x64 off
    "besseli_ratio": ("chebax.besseli_ratio(0.0, 3.0)", 0.80998529395650453),
    "besseli_ratio_small_x": ("chebax.besseli_ratio(4.0, 0.5)", 0.049896203861781465),
    "besseli_ratio_tiny_x": ("chebax.besseli_ratio(8.9, 1e-6)", 5.0505050505050393e-08),
    "besseli_ratio_inf": ("chebax.besseli_ratio(0.0, jnp.inf)", 1.0),
    "betainc_fn": ("chebax.betainc_fn(2.0, 3.0, 0.4)", 0.5248),
    "log_betainc_fn": ("chebax.log_betainc_fn(2.0, 3.0, 0.01)", -7.4319532486885614),
    "gammainc_fn": ("chebax.gammainc_fn(2.5, 3.0)", 0.6937810815867216),
    "gammaincc_fn": ("chebax.gammaincc_fn(2.5, 9.0)", 0.0029464045878802904),
    "log_gammaincc_fn": ("chebax.log_gammaincc_fn(2.5, 40.0)", -34.714103456537936),
    "stdtr": ("chebax.stdtr(4.0, 1.2)", 0.85182430334382328),
    "stdtr_tail": ("chebax.stdtr(20.0, -2.0)", 0.029632767723285236),
    "dawsn": ("chebax.dawsn(1.0)", 0.53807950691276842),
    "erfcx": ("chebax.erfcx(2.0)", 0.25539567631050574),
    "lambertw": ("chebax.lambertw(3.0, 0)", 1.04990889496404),
    "spherical_jn": ("chebax.spherical_jn(2)(4.0)", 0.27628368577135016),
    "gammainc_at_inf": ("chebax.gammainc_fn(2.0, jnp.inf)", 1.0),
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
    code = ("import numpy as np, jax, jax.numpy as jnp, chebax\n"
            "assert jnp.empty(()).dtype == jnp.float32, 'x64 leaked in'\n"
            + body)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


@pytest.mark.parametrize("name", sorted(CASES))
def test_value_under_x64_off(name):
    expr, ref = CASES[name]
    _run(f"got = float({expr})\n"
         f"assert np.isfinite(got), ('not finite', got)\n"
         f"ref = {ref!r}\n"
         f"err = abs(got - ref) / max(abs(ref), 1e-30)\n"
         f"assert err <= 2e-5, (got, ref, err)\n")


@pytest.mark.parametrize("name", sorted(QUANTILES))
def test_quantile_under_x64_off(name):
    expr, ref = QUANTILES[name]
    _run(f"got = float({expr})\n"
         f"assert np.isfinite(got), ('not finite', got)\n"
         f"ref = {ref!r}\n"
         f"err = abs(got - ref) / max(abs(ref), 1e-30)\n"
         f"assert err <= 3e-5, (got, ref, err)\n")


def test_dtype_is_float32_not_promoted():
    # everything must come back float32, not silently widened
    _run("for v in [chebax.besselk(1.5)(2.0), chebax.betaincinv(2.0, 3.0, 0.4),\n"
         "          chebax.gammaincinv(2.5, 0.4), chebax.stdtrit(4.0, 0.25),\n"
         "          chebax.besseli_ratio(0.0, 3.0), chebax.stdtr(4.0, 1.2)]:\n"
         "    assert v.dtype == jnp.float32, v.dtype\n")


def test_gradients_are_finite_under_x64_off():
    _run("gs = [jax.grad(chebax.besselk(1.5))(2.0),\n"
         "      jax.grad(chebax.besselk_fn, argnums=0)(1.7, 5.0),\n"
         "      jax.grad(chebax.betaincinv, argnums=2)(2.0, 3.0, 0.4),\n"
         "      jax.grad(chebax.gammaincinv, argnums=1)(2.5, 0.4),\n"
         "      jax.grad(chebax.stdtrit, argnums=1)(4.0, 0.25),\n"
         "      jax.grad(chebax.besseli_ratio, argnums=1)(0.0, 0.0),\n"
         "      jax.grad(chebax.vonmises_cdf, argnums=0)(2.0, 0.5)]\n"
         "for g in gs:\n"
         "    assert np.isfinite(float(g)), g\n"
         "assert abs(float(gs[5]) - 0.5) <= 1e-5, gs[5]\n")


def test_tables_materialize_in_float32():
    # a cache built under x64 off must hold float32, and must NOT be served
    # to a later x64-on process (the dtype tag); the erf family's no-argument
    # cache is the one that used to miss this
    _run("assert np.asarray(chebax.besselk(1.5).ltil.coef).dtype == np.float32\n"
         "from chebax._src.recipes.erf_family import _series\n"
         "chebax.dawsn(1.0)\n"
         "assert np.asarray(_series()[0].coef).dtype == np.float32\n"
         "jax.config.update('jax_enable_x64', True)\n"
         "assert np.asarray(_series()[0].coef).dtype == np.float64\n"
         "assert abs(float(chebax.dawsn(1.0)) - 0.53807950691276842) <= 1e-15\n")
