"""M6 (first increment) acceptance: besseli on x >= 0, nu in [0, 10].

References mpmath at 40 dps; errors pointwise relative (I > 0, no zeros).
dI/dnu near nu = 0 needs more than that: it sits log10(x/nu) digits below
I_nu, so those references run at 50 dps.

Measured worst over the orders below: values 4.2e-15 (incl. the traced-nu
path), dI/dx 1.3e-15, dI/dnu 6.5e-13, scaled-vs-unscaled 1.2e-15. From the
2026-08-01 review: I_{nu+1}/I_nu 2.6e-15 (5.8e-16 on the small-x series),
dI/dnu on a logarithmic order grid 8.3e-13, d2I/dnu2 at nu = 0 8.5e-13,
the scaled tail out to 1.7e308 4.0e-16. Float32 worsts are in
test_x64_off.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import besseli_gen
from chebax._src.recipes import besseli_table as it

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

NUS = [0.03, 0.5, 1.0, 1.7, 2.0, 3.0, 5.5, 7.77, 9.97]
XI = np.sort(np.concatenate([
    np.logspace(-3, np.log10(8.0), 16), [8.0, 8.0001],
    np.logspace(np.log10(8.2), np.log10(600.0), 10),
]))


def _iref(v):
    return np.array([float(mp.besseli(mp.mpf(v), mp.mpf(x))) for x in XI])


def test_values_and_traced():
    for v in NUS:
        I = _iref(v)
        assert np.max(np.abs(np.asarray(chebax.besseli(v)(XI)) - I) / I) <= 2e-14, v
        assert np.max(np.abs(np.asarray(chebax.besseli_fn(v, XI)) - I) / I) <= 2e-14, v


def test_dx():
    for v in [0.03, 1.0, 1.7, 5.5, 9.97]:
        Ip = np.array([float(mp.diff(lambda t: mp.besseli(mp.mpf(v), t), mp.mpf(x)))
                       for x in XI])
        g = np.asarray(jax.vmap(jax.grad(chebax.besseli(v)))(jnp.asarray(XI)))
        assert np.max(np.abs(g - Ip) / np.abs(Ip)) <= 2e-14, v


def test_dnu():
    for v in [0.5, 1.0, 1.7, 7.77]:
        In = np.array([float(mp.diff(lambda t: mp.besseli(t, mp.mpf(x)), mp.mpf(v)))
                       for x in XI])
        got = np.asarray(chebax.besseli_dnu(v)(XI))
        assert np.max(np.abs(got - In) / np.abs(In)) <= 5e-12, v


def test_scaled():
    for v in [0.5, 2.5]:
        I = _iref(v)
        sc = np.asarray(chebax.besseli(v, scaled=True)(XI))
        ref = I * np.exp(-XI)
        assert np.max(np.abs(sc - ref) / ref) <= 1e-14, v
    # past the overflow point the scaled form keeps working
    ref720 = float(mp.besseli(mp.mpf(2.5), mp.mpf(720)) * mp.exp(-720))
    assert float(chebax.besseli(2.5)(720.0)) == np.inf
    got720 = float(chebax.besseli(2.5, scaled=True)(720.0))
    assert abs(got720 - ref720) / ref720 <= 1e-13


def test_x_zero():
    assert abs(float(chebax.besseli(0.0)(0.0)) - 1.0) <= 5e-15
    assert float(chebax.besseli(2.5)(0.0)) == 0.0


def test_jit_and_pytree():
    iv = chebax.besseli(2.5)
    xs = jnp.asarray(XI[:10])
    y = jax.jit(lambda f, x: f(x))(iv, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(iv(xs)), rtol=1e-15)


def test_order_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.besseli(-0.5)


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    besseli_gen.main(tmp_path)
    assert (tmp_path / "besseli_table.py").read_text() == pathlib.Path(it.__file__).read_text()


# ---- review 2026-07-30 regressions ------------------------------------------

def test_dnu_zero_order_is_neg_k0():
    # dI_v/dv|_{v=0} = -K_0(x): exponentially small, while the generic
    # I * dlog form amplified table noise by e^2x (returned +2.2e242 at
    # x=600 for a true value of -1.4e-262)
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    d = chebax.besseli_dnu(0.0)
    for x in [0.5, 5.0, 30.0, 100.0, 600.0]:
        ref = float(-mp.besselk(0, x))
        assert abs(float(d(x)) - ref) / abs(ref) <= 1e-13, x
    g = float(jax.grad(chebax.besseli_fn, argnums=0)(0.0, 30.0))
    assert abs(g - float(-mp.besselk(0, 30.0))) / 2.13e-14 <= 1e-12
    gs = float(jax.grad(lambda n: chebax.besseli_fn(n, 30.0, scaled=True))(0.0))
    ref_s = float(-mp.besselk(0, 30.0) * mp.exp(-30.0))
    assert abs(gs - ref_s) / abs(ref_s) <= 1e-12


def test_overflow_window_and_inf():
    # e^x alone overflows at 709.78 but I_v is representable to ~713
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    for v, x in [(0.0, 713.0), (10.0, 714.0)]:
        ref = float(mp.besseli(v, x))
        got = float(chebax.besseli(v)(x))
        assert np.isfinite(got) and abs(got - ref) / ref <= 1e-12, (v, x)
    assert float(chebax.besseli(0.0)(jnp.inf)) == np.inf
    assert float(chebax.besseli(0.0, scaled=True)(jnp.inf)) == 0.0
    assert np.isnan(float(chebax.besseli(2.5)(jnp.nan)))


def test_origin_gradients():
    # the generic power rule is 0 * (x/2)^-1 at v = 0: NaN for a constant
    assert float(jax.grad(chebax.besseli(0.0))(0.0)) == 0.0
    assert abs(float(jax.grad(chebax.besseli(0.0, scaled=True))(0.0)) + 1.0) <= 1e-14
    assert abs(float(jax.grad(chebax.besseli(1.0))(0.0)) - 0.5) <= 1e-14
    assert float(jax.grad(chebax.besseli_fn, argnums=1)(0.0, 0.0)) == 0.0


def test_besseli_ratio():
    # relative errors, measured worst 2.6e-15 over the orders below on
    # x in [1e-8, 5000]; bar 4x. Endpoint: exact 0 at x = 0.
    xs = np.concatenate([np.logspace(-8, 2, 20), [500.0, 5000.0]])
    for nu in [0.0, 0.5, 1.0, 4.5, 8.99]:
        ref = np.array([float(mp.besseli(mp.mpf(nu) + 1, mp.mpf(x))
                              / mp.besseli(mp.mpf(nu), mp.mpf(x))) for x in xs])
        got = np.asarray(chebax.besseli_ratio(nu, jnp.asarray(xs)))
        assert np.max(np.abs(got - ref) / ref) <= 1.5e-14, nu
    assert float(chebax.besseli_ratio(2.0, 0.0)) == 0.0
    assert np.isnan(float(chebax.besseli_ratio(2.0, np.nan)))
    # gradients in both arguments against mp.diff
    g_nu = float(jax.grad(chebax.besseli_ratio, argnums=0)(2.0, 3.0))
    g_x = float(jax.grad(chebax.besseli_ratio, argnums=1)(2.0, 3.0))
    r_nu = float(mp.diff(lambda v: mp.besseli(v + 1, 3.0) / mp.besseli(v, 3.0),
                         mp.mpf(2)))
    r_x = float(mp.diff(lambda x: mp.besseli(3, x) / mp.besseli(2, x),
                        mp.mpf(3)))
    assert abs(g_nu - r_nu) <= 1e-13 and abs(g_x - r_x) <= 1e-13


# ---- review 2026-08-01 regressions ------------------------------------------

def test_besseli_ratio_small_x():
    # below x = 2 the ratio is the ascending series with (x/2)^nu and
    # Gamma(nu+1) cancelled analytically. The old quotient of two scaled
    # besseli_fn values divided two independently underflowing numbers:
    # 0/0 = NaN in float32 already at (nu=8.9, x=1e-6), exactly 0 below
    # x = 1e-30, and 8.1e-15 of prefactor pow error in between. Worst
    # relative 5.8e-16 measured over the grid below, bar 2.5e-15.
    xs = np.concatenate([np.logspace(-38, -3, 24), np.logspace(-3, np.log10(2.0), 16)])
    for nu in [0.0, 0.5, 2.0, 4.5, 8.99, 9.0]:
        ref = np.array([float(mp.besseli(mp.mpf(nu) + 1, mp.mpf(float(x)))
                              / mp.besseli(mp.mpf(nu), mp.mpf(float(x)))) for x in xs])
        got = np.asarray(chebax.besseli_ratio(nu, jnp.asarray(xs)))
        assert np.max(np.abs(got - ref) / ref) <= 2.5e-15, nu
    # the two branches meet at the crossover (measured jump 1.6e-15, bar 4x)
    lo = float(chebax.besseli_ratio(4.5, np.nextafter(2.0, 0.0)))
    hi = float(chebax.besseli_ratio(4.5, np.nextafter(2.0, 3.0)))
    assert abs(hi - lo) / lo <= 6.5e-15


def test_besseli_ratio_endpoints():
    # x = +inf is exactly 1 (it used to be NaN: inf/inf through the tail),
    # and the slope at the origin is the exact 1/(2(nu+1)) the von Mises
    # A(kappa) = besseli_ratio(0, kappa) needs. A hard x <= 1e-30 select
    # used to mask that gradient to 0.
    for nu in [0.0, 1.0, 2.0, 5.0, 9.0]:
        assert float(chebax.besseli_ratio(nu, jnp.inf)) == 1.0, nu
        assert float(jax.grad(chebax.besseli_ratio, argnums=1)(nu, 0.0)) == 0.5 / (nu + 1.0)
    assert abs(float(chebax.besseli_ratio(0.0, 1e308)) - 1.0) <= 1e-15
    assert np.isnan(float(chebax.besseli_ratio(np.nan, 3.0)))


def test_dnu_small_order():
    # the generic I * dlogI/dnu route cannot resolve dlnI/dnu just above
    # nu = 0: the tables' first nu-derivative carries ~3e-16 of ABSOLUTE
    # noise while the true log derivative is ~nu/x. At (1e-16, 30) it
    # returned +2.5e-05 for a true -2.65e-06 (wrong sign), at (1e-14, 100)
    # it was 20% low, and the factory and traced paths disagreed. Both now
    # integrate the second nu-derivative below |dlnI/dnu| = 1e-3. Worst
    # relative 8.3e-13 measured over the grid below, bar 3.5e-12.
    grad = jax.jit(jax.grad(chebax.besseli_fn, argnums=0))
    with mp.workdps(50):
        for x in [0.5, 4.0, 7.0, 8.0, 8.5, 30.0, 100.0]:
            for nu in [1e-16, 1e-12, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
                ref = float(mp.diff(lambda v: mp.besseli(v, mp.mpf(x)), mp.mpf(nu)))
                fac = float(chebax.besseli_dnu(nu)(x))
                tr = float(grad(nu, x))
                assert abs(fac - ref) / abs(ref) <= 3.5e-12, (nu, x)
                assert abs(tr - ref) / abs(ref) <= 3.5e-12, (nu, x)


def test_dnu_second_order():
    # the nu = 0 JVP substituted the exact -K_0 through a jnp.where whose
    # OTHER branch carried the nu dependence, so d2I/dnu2 came out 0. The
    # small-order path is analytic in nu, so it differentiates twice.
    # Worst relative 8.5e-13 measured, bar 3.5e-12.
    for x in [0.5, 1.0, 4.0, 8.0, 20.0, 100.0]:
        ref = float(mp.diff(lambda v: mp.besseli(v, mp.mpf(x)), 0, 2))
        got = float(jax.grad(jax.grad(lambda v: chebax.besseli_fn(v, x)))(0.0))
        assert abs(got - ref) / abs(ref) <= 3.5e-12, x


def test_scaled_tail_to_max_double():
    # exp(lt)/sqrt(2 pi x) took the quotient to 0 once 2 pi x overflowed
    # (x ~ 2.9e307): e^-x I_0(1e308) is 3.99e-155, not 0. Worst relative
    # 4.0e-16 measured, bar 1.6e-15. The unscaled combined tail logs 2 pi
    # and x separately for the same reason, or it returned 0 where the
    # value correctly overflows.
    for x in [1e4, 1e40, 1e150, 1e308, 1.7e308]:
        ref = float(mp.besseli(0, mp.mpf(x)) * mp.exp(mp.mpf(-x)))
        assert abs(float(chebax.besseli(0.0, scaled=True)(x)) - ref) / ref <= 1.6e-15, x
        assert abs(float(chebax.besseli_fn(0.0, x, scaled=True)) - ref) / ref <= 1.6e-15, x
    assert float(chebax.besseli(0.0)(1e308)) == np.inf
    assert float(chebax.besseli_fn(0.0, 1e308)) == np.inf


def test_x64_off():
    # x64 is process-global and conftest turns it on, so float32 needs its
    # own process. All three fixes have a float32 side: the ratio's old
    # quotient underflowed to 0/0, sqrt(2 pi x) overflows at x ~ 5.4e37,
    # and the dnu route switches on |dlnI/dnu| < 3e-2 rather than 1e-3
    # (the tables' first nu-derivative is ~1e-7 noisy there, not 3e-16).
    # References mpmath at 50 dps; measured worst ratio 6.6e-7, scaled tail
    # 6.0e-8, dI/dnu 8.5e-6, d2I/dnu2 7.1e-7. Bars ~4x.
    code = (
        "import numpy as np, jax, jax.numpy as jnp, chebax\n"
        "assert jnp.empty(()).dtype == jnp.float32\n"
        "f = np.float32\n"
        "ratio = {(8.9, 1e-6): 5.0505050505050384e-08,\n"
        "         (9.0, 1e-8): 5.0000000000000001e-10,\n"
        "         (0.0, 1e-20): 4.9999999999999997e-21,\n"
        "         (0.5, 1.0): 0.3130352854993313,\n"
        "         (2.0, 3.0): 0.42746673410489596,\n"
        "         (0.0, 500.0): 0.99899949899686193}\n"
        "for (nu, x), r in ratio.items():\n"
        "    g = float(chebax.besseli_ratio(f(nu), f(x)))\n"
        "    assert abs(g - r) / abs(r) <= 3e-6, (nu, x, g, r)\n"
        "assert float(chebax.besseli_ratio(0.0, jnp.inf)) == 1.0\n"
        "for nu in [0.0, 2.0]:\n"
        "    s = float(jax.grad(chebax.besseli_ratio, argnums=1)(f(nu), f(0.0)))\n"
        "    assert s == float(f(0.5) / f(nu + 1.0)), (nu, s)\n"
        "tail = {1e10: 3.9894228040641946e-06, 1e30: 3.9894228040143267e-16,\n"
        "        3e38: 2.3032943298089031e-20}\n"
        "for x, r in tail.items():\n"
        "    for g in [float(chebax.besseli(0.0, scaled=True)(f(x))),\n"
        "              float(chebax.besseli_fn(0.0, f(x), scaled=True))]:\n"
        "        assert abs(g - r) / r <= 3e-7, (x, g, r)\n"
        "assert float(chebax.besseli(0.0, scaled=True)(jnp.inf)) == 0.0\n"
        "dnu = {(1e-12, 30.0): -0.026506597210496226,\n"
        "       (1e-6, 8.0): -0.00020383868858830207,\n"
        "       (1e-4, 8.0): -0.0058832690307996675,\n"
        "       (1e-3, 30.0): -26506596.755694668,\n"
        "       (0.01, 4.0): -0.044774368030499289,\n"
        "       (0.5, 20.0): -1110509.0710160731}\n"
        "grad = jax.jit(jax.grad(chebax.besseli_fn, argnums=0))\n"
        "for (nu, x), r in dnu.items():\n"
        "    for g in [float(chebax.besseli_dnu(nu)(f(x))), float(grad(f(nu), f(x)))]:\n"
        "        assert abs(g - r) / abs(r) <= 3.5e-5, (nu, x, g, r)\n"
        "k30 = -2.1324774964630563e-14\n"
        "assert abs(float(chebax.besseli_dnu(0.0)(f(30.0))) - k30) / abs(k30) <= 4e-7\n"
        "for x, r in {1.0: -1.4452143424810813, 20.0: -2235578.6187629785}.items():\n"
        "    g = float(jax.grad(jax.grad(lambda v: chebax.besseli_fn(v, f(x))))(f(0.0)))\n"
        "    assert abs(g - r) / abs(r) <= 3e-6, (x, g, r)\n"
        "print('ok')\n")
    import os
    import subprocess
    import sys
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-1500:]
