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
XI = np.sort(
    np.concatenate(
        [
            np.logspace(-3, np.log10(8.0), 16),
            [8.0, 8.0001],
            np.logspace(np.log10(8.2), np.log10(600.0), 10),
        ]
    )
)


def _iref(v):
    return np.array([float(mp.besseli(mp.mpf(v), mp.mpf(x))) for x in XI])


def test_values_and_traced():
    for v in NUS:
        I = _iref(v)
        assert (
            np.max(np.abs(np.asarray(chebax.besseli_frozen(v)(XI)) - I) / I) <= 2e-14
        ), v
        assert (
            np.max(np.abs(np.asarray(chebax.besseli_scalar(v, XI)) - I) / I) <= 2e-14
        ), v


def test_dx():
    for v in [0.03, 1.0, 1.7, 5.5, 9.97]:
        Ip = np.array(
            [float(mp.diff(lambda t: mp.besseli(mp.mpf(v), t), mp.mpf(x))) for x in XI]
        )
        g = np.asarray(jax.vmap(jax.grad(chebax.besseli_frozen(v)))(jnp.asarray(XI)))
        assert np.max(np.abs(g - Ip) / np.abs(Ip)) <= 2e-14, v


def test_dnu():
    for v in [0.5, 1.0, 1.7, 7.77]:
        In = np.array(
            [float(mp.diff(lambda t: mp.besseli(t, mp.mpf(x)), mp.mpf(v))) for x in XI]
        )
        got = np.asarray(chebax.besseli_dnu_frozen(v)(XI))
        assert np.max(np.abs(got - In) / np.abs(In)) <= 5e-12, v


def test_scaled():
    for v in [0.5, 2.5]:
        I = _iref(v)
        sc = np.asarray(chebax.besseli_frozen(v, scaled=True)(XI))
        ref = I * np.exp(-XI)
        assert np.max(np.abs(sc - ref) / ref) <= 1e-14, v
    # past the overflow point the scaled form keeps working
    ref720 = float(mp.besseli(mp.mpf(2.5), mp.mpf(720)) * mp.exp(-720))
    assert float(chebax.besseli_frozen(2.5)(720.0)) == np.inf
    got720 = float(chebax.besseli_frozen(2.5, scaled=True)(720.0))
    assert abs(got720 - ref720) / ref720 <= 1e-13


def test_x_zero():
    assert abs(float(chebax.besseli_frozen(0.0)(0.0)) - 1.0) <= 5e-15
    assert float(chebax.besseli_frozen(2.5)(0.0)) == 0.0


def test_jit_and_pytree():
    iv = chebax.besseli_frozen(2.5)
    xs = jnp.asarray(XI[:10])
    y = jax.jit(lambda f, x: f(x))(iv, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(iv(xs)), rtol=1e-15)


def test_order_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.besseli_frozen(-0.5)


def _all_nan(f, nus, x, dx_lanes=slice(None)):
    val = jax.jit(jax.vmap(f, (None, 0)))
    rev = jax.jit(jax.vmap(jax.grad(f, (0, 1)), (None, 0)))
    fwd = jax.jit(jax.vmap(jax.jacfwd(f, (0, 1)), (None, 0)))
    for nu in nus:
        assert np.isnan(float(f(nu, 2.0))), nu
        assert np.isnan(np.asarray(val(nu, x))).all(), nu
        for dnu, dx in [rev(nu, x), fwd(nu, x)]:
            assert np.isnan(np.asarray(dnu)).all(), nu
            assert np.isnan(np.asarray(dx)[dx_lanes]).all(), nu
    return val, rev


@pytest.mark.parametrize("scaled", [False, True])
def test_traced_order_out_of_range_is_nan(scaled):
    # The tables used to extrapolate: besseli(15, 2) gave -0.0062 (true
    # 8.1e-13), besseli(15, 7) -2e9; unscaled I at x = inf was inf for any
    # order, nan included.
    def f(nu, x):
        return chebax.besseli_scalar(nu, x, scaled)

    x = jnp.asarray([0.0, 0.5, 2.0, 7.0, 20.0, 800.0, jnp.inf])
    val, rev = _all_nan(f, [-0.5, 10.5, 15.0], x)
    # the box is closed
    for nu in [0.0, 10.0]:
        xs = x[1:5]
        assert np.isfinite(np.asarray(val(nu, xs))).all(), nu
        assert all(np.isfinite(np.asarray(g)).all() for g in rev(nu, xs)), nu


def test_besseli_ratio_order_out_of_range_is_nan():
    # The tail reads I at nu + 1, so the box ends at 9: nu = 10.5 was off by
    # 3e-3 and nu = 15 gave 201 at x = 7. A nan order gave zero forward-mode
    # derivatives. At x = inf the x clamp leaves d/dx 0 in reverse mode.
    x = jnp.asarray([0.0, 0.5, 2.0, 7.0, 20.0, jnp.inf])
    f = chebax.besseli_ratio
    _all_nan(f, [-0.5, 9.5, 15.0, np.nan], x, slice(None, -1))
    assert np.isfinite(np.asarray(jax.vmap(f, (None, 0))(9.0, x))).all()


def test_out_of_range_group_stays_in_its_group():
    idx = np.array([0, 1, 0, 1, 1])
    x = jnp.asarray([0.3, 1.0, 4.0, 9.0, 30.0])
    f = chebax.pergroup(chebax.besseli_scalar, idx)
    nu = jnp.asarray([2.5, 15.0])
    y = np.asarray(f(nu, x))
    gnu, gx = jax.grad(lambda n, x: jnp.sum(f(n, x)), (0, 1))(nu, x)
    live = idx == 0
    ref = jax.vmap(lambda t: chebax.besseli_scalar(2.5, t))(x[live])
    np.testing.assert_allclose(y[live], np.asarray(ref), rtol=5e-15)
    assert np.isnan(y[~live]).all()
    assert np.isfinite(float(gnu[0])) and np.isnan(float(gnu[1]))
    assert np.isfinite(np.asarray(gx)[live]).all()
    assert np.isnan(np.asarray(gx)[~live]).all()


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib

    besseli_gen.main(tmp_path)
    assert (tmp_path / "besseli_table.py").read_text() == pathlib.Path(
        it.__file__
    ).read_text()


# ---- review 2026-07-30 regressions ------------------------------------------


def test_dnu_zero_order_is_neg_k0():
    # dI_v/dv|_{v=0} = -K_0(x): exponentially small, while the generic
    # I * dlog form amplified table noise by e^2x (returned +2.2e242 at
    # x=600 for a true value of -1.4e-262)
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    d = chebax.besseli_dnu_frozen(0.0)
    for x in [0.5, 5.0, 30.0, 100.0, 600.0]:
        ref = float(-mp.besselk(0, x))
        assert abs(float(d(x)) - ref) / abs(ref) <= 1e-13, x
    g = float(jax.grad(chebax.besseli_scalar, argnums=0)(0.0, 30.0))
    assert abs(g - float(-mp.besselk(0, 30.0))) / 2.13e-14 <= 1e-12
    gs = float(jax.grad(lambda n: chebax.besseli_scalar(n, 30.0, scaled=True))(0.0))
    ref_s = float(-mp.besselk(0, 30.0) * mp.exp(-30.0))
    assert abs(gs - ref_s) / abs(ref_s) <= 1e-12


def test_overflow_window_and_inf():
    # e^x alone overflows at 709.78 but I_v is representable to ~713
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    for v, x in [(0.0, 713.0), (10.0, 714.0)]:
        ref = float(mp.besseli(v, x))
        got = float(chebax.besseli_frozen(v)(x))
        assert np.isfinite(got) and abs(got - ref) / ref <= 1e-12, (v, x)
    assert float(chebax.besseli_frozen(0.0)(jnp.inf)) == np.inf
    assert float(chebax.besseli_frozen(0.0, scaled=True)(jnp.inf)) == 0.0
    assert np.isnan(float(chebax.besseli_frozen(2.5)(jnp.nan)))


def test_origin_gradients():
    # the generic power rule is 0 * (x/2)^-1 at v = 0: NaN for a constant
    assert float(jax.grad(chebax.besseli_frozen(0.0))(0.0)) == 0.0
    assert (
        abs(float(jax.grad(chebax.besseli_frozen(0.0, scaled=True))(0.0)) + 1.0)
        <= 1e-14
    )
    assert abs(float(jax.grad(chebax.besseli_frozen(1.0))(0.0)) - 0.5) <= 1e-14
    assert float(jax.grad(chebax.besseli_scalar, argnums=1)(0.0, 0.0)) == 0.0
    for nu in (0.0, 1.0):
        for scaled in (False, True):
            expected = (0.5 if nu == 0 else 0.0) + (1.0 if nu == 0 else -1.0) * scaled
            for f in (
                chebax.besseli_frozen(nu, scaled),
                lambda x: chebax.besseli_scalar(nu, x, scaled),
            ):
                second = jax.jit(jax.grad(jax.grad(f)))(0.0)
                assert np.isfinite(second)
                assert abs(float(second) - expected) <= 1e-14


def test_besseli_ratio():
    # relative errors, measured worst 2.6e-15 over the orders below on
    # x in [1e-8, 5000]; bar 4x. Endpoint: exact 0 at x = 0.
    xs = np.concatenate([np.logspace(-8, 2, 20), [500.0, 5000.0]])
    for nu in [0.0, 0.5, 1.0, 4.5, 8.99]:
        ref = np.array(
            [
                float(
                    mp.besseli(mp.mpf(nu) + 1, mp.mpf(x))
                    / mp.besseli(mp.mpf(nu), mp.mpf(x))
                )
                for x in xs
            ]
        )
        got = np.asarray(chebax.besseli_ratio(nu, jnp.asarray(xs)))
        assert np.max(np.abs(got - ref) / ref) <= 1.5e-14, nu
    assert float(chebax.besseli_ratio(2.0, 0.0)) == 0.0
    assert np.isnan(float(chebax.besseli_ratio(2.0, np.nan)))
    # gradients in both arguments against mp.diff
    g_nu = float(jax.grad(chebax.besseli_ratio, argnums=0)(2.0, 3.0))
    g_x = float(jax.grad(chebax.besseli_ratio, argnums=1)(2.0, 3.0))
    r_nu = float(
        mp.diff(lambda v: mp.besseli(v + 1, 3.0) / mp.besseli(v, 3.0), mp.mpf(2))
    )
    r_x = float(mp.diff(lambda x: mp.besseli(3, x) / mp.besseli(2, x), mp.mpf(3)))
    assert abs(g_nu - r_nu) <= 1e-13 and abs(g_x - r_x) <= 1e-13


# ---- review 2026-08-01 regressions ------------------------------------------


def test_besseli_ratio_small_x():
    # below x = 2 the ratio is the ascending series with (x/2)^nu and
    # Gamma(nu+1) cancelled analytically. The old quotient of two scaled
    # besseli_scalar values divided two independently underflowing numbers:
    # 0/0 = NaN in float32 already at (nu=8.9, x=1e-6), exactly 0 below
    # x = 1e-30, and 8.1e-15 of prefactor pow error in between. Worst
    # relative 5.8e-16 measured over the grid below, bar 2.5e-15.
    xs = np.concatenate([np.logspace(-38, -3, 24), np.logspace(-3, np.log10(2.0), 16)])
    for nu in [0.0, 0.5, 2.0, 4.5, 8.99, 9.0]:
        ref = np.array(
            [
                float(
                    mp.besseli(mp.mpf(nu) + 1, mp.mpf(float(x)))
                    / mp.besseli(mp.mpf(nu), mp.mpf(float(x)))
                )
                for x in xs
            ]
        )
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
        assert float(jax.grad(chebax.besseli_ratio, argnums=1)(nu, 0.0)) == 0.5 / (
            nu + 1.0
        )
    assert abs(float(chebax.besseli_ratio(0.0, 1e308)) - 1.0) <= 1e-15
    assert np.isnan(float(chebax.besseli_ratio(np.nan, 3.0)))


def test_besseli_ratio_joint_gradient():
    """value_and_grad in nu and x together never finished compiling on the
    CPU while inf and nan had their own mask (nu or x alone: seconds). At
    64 points it takes ~11 s now; a subprocess fails a stall at 300 s. The
    joint gradient equals the two single ones, and is 0 in x at inf."""
    import os
    import subprocess
    import sys

    code = """
import jax, jax.numpy as jnp, numpy as np, chebax
x=jnp.concatenate([jnp.linspace(0.0,50.0,62),jnp.asarray([1e35,jnp.inf])])
f=lambda n,x:jnp.sum(chebax.besseli_ratio(n,x))
_,(gn,gx)=jax.jit(jax.value_and_grad(f,(0,1)))(2.0,x)
assert float(gn)==float(jax.jit(jax.grad(f,0))(2.0,x))
assert np.array_equal(np.asarray(gx),np.asarray(jax.jit(jax.grad(f,1))(2.0,x)))
assert np.all(np.isfinite(np.asarray(gx))) and float(gx[-1])==0.0
"""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true"}
    try:
        run = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the joint gradient did not compile in 300 s")
    assert run.returncode == 0, run.stdout + run.stderr


def test_dnu_small_order():
    # the generic I * dlogI/dnu route cannot resolve dlnI/dnu just above
    # nu = 0: the tables' first nu-derivative carries ~3e-16 of ABSOLUTE
    # noise while the true log derivative is ~nu/x. At (1e-16, 30) it
    # returned +2.5e-05 for a true -2.65e-06 (wrong sign), at (1e-14, 100)
    # it was 20% low, and the factory and traced paths disagreed. Both now
    # integrate the second nu-derivative below |dlnI/dnu| = 1e-3. Worst
    # relative 8.3e-13 measured over the grid below, bar 3.5e-12.
    grad = jax.jit(jax.grad(chebax.besseli_scalar, argnums=0))
    with mp.workdps(50):
        for x in [0.5, 4.0, 7.0, 8.0, 8.5, 30.0, 100.0]:
            for nu in [1e-16, 1e-12, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
                ref = float(mp.diff(lambda v: mp.besseli(v, mp.mpf(x)), mp.mpf(nu)))
                fac = float(chebax.besseli_dnu_frozen(nu)(x))
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
        got = float(jax.grad(jax.grad(lambda v: chebax.besseli_scalar(v, x)))(0.0))
        assert abs(got - ref) / abs(ref) <= 3.5e-12, x


def test_scaled_tail_to_max_double():
    # exp(lt)/sqrt(2 pi x) took the quotient to 0 once 2 pi x overflowed
    # (x ~ 2.9e307): e^-x I_0(1e308) is 3.99e-155, not 0. Worst relative
    # 4.0e-16 measured, bar 1.6e-15. The unscaled combined tail logs 2 pi
    # and x separately for the same reason, or it returned 0 where the
    # value correctly overflows.
    for x in [1e4, 1e40, 1e150, 1e308, 1.7e308]:
        ref = float(mp.besseli(0, mp.mpf(x)) * mp.exp(mp.mpf(-x)))
        assert (
            abs(float(chebax.besseli_frozen(0.0, scaled=True)(x)) - ref) / ref
            <= 1.6e-15
        ), x
        assert (
            abs(float(chebax.besseli_scalar(0.0, x, scaled=True)) - ref) / ref
            <= 1.6e-15
        ), x
    assert float(chebax.besseli_frozen(0.0)(1e308)) == np.inf
    assert float(chebax.besseli_scalar(0.0, 1e308)) == np.inf


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
        "    for g in [float(chebax.besseli_frozen(0.0, scaled=True)(f(x))),\n"
        "              float(chebax.besseli_scalar(0.0, f(x), scaled=True))]:\n"
        "        assert abs(g - r) / r <= 3e-7, (x, g, r)\n"
        "assert float(chebax.besseli_frozen(0.0, scaled=True)(jnp.inf)) == 0.0\n"
        "dnu = {(1e-12, 30.0): -0.026506597210496226,\n"
        "       (1e-6, 8.0): -0.00020383868858830207,\n"
        "       (1e-4, 8.0): -0.0058832690307996675,\n"
        "       (1e-3, 30.0): -26506596.755694668,\n"
        "       (0.01, 4.0): -0.044774368030499289,\n"
        "       (0.5, 20.0): -1110509.0710160731}\n"
        "grad = jax.jit(jax.grad(chebax.besseli_scalar, argnums=0))\n"
        "for (nu, x), r in dnu.items():\n"
        "    for g in [float(chebax.besseli_dnu_frozen(nu)(f(x))), float(grad(f(nu), f(x)))]:\n"
        "        assert abs(g - r) / abs(r) <= 3.5e-5, (nu, x, g, r)\n"
        "k30 = -2.1324774964630563e-14\n"
        "assert abs(float(chebax.besseli_dnu_frozen(0.0)(f(30.0))) - k30) / abs(k30) <= 4e-7\n"
        "for x, r in {1.0: -1.4452143424810813, 20.0: -2235578.6187629785}.items():\n"
        "    g = float(jax.grad(jax.grad(lambda v: chebax.besseli_scalar(v, f(x))))(f(0.0)))\n"
        "    assert abs(g - r) / abs(r) <= 3e-6, (x, g, r)\n"
        "print('ok')\n"
    )
    import os
    import subprocess
    import sys

    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-1500:]


def test_fractional_order_just_above_tiny():
    # x/2 is subnormal below 2 tiny and XLA flushes it: I was 0, dI/dx inf.
    # (I, dI/dv) at mpmath 40 dps, dI/dx = v I / x this close to 0. Measured
    # worst 1.6e-15 (traced dI/dv), bar 6e-15. d/dx x^v is v x^(v-1), and
    # the rounding of v - 1 times |ln x| ~ 708 gives 2.0e-14, bar 8e-14
    ref = {
        (0.1, 2.3e-308): (1.68938685277697e-31, -1.1971547529062514e-28),
        (0.1, 4e-308): (1.7855101592779915e-31, -1.2642827953681088e-28),
        (0.5, 2.3e-308): (1.2100518486599808e-154, -8.5803921889382945e-152),
        (0.5, 4e-308): (1.5957691216057307e-154, -1.1306655376172815e-151),
    }
    dv = jax.jit(jax.grad(chebax.besseli_scalar))
    dx = jax.jit(jax.grad(chebax.besseli_scalar, argnums=1))
    for (v, x), (val, dval) in ref.items():
        got = [
            (chebax.besseli_frozen(v)(x), val, 6e-15),
            (chebax.besseli_scalar(v, x), val, 6e-15),
            (chebax.besseli_dnu_frozen(v)(x), dval, 6e-15),
            (dv(v, x), dval, 6e-15),
            (dx(v, x), v * val / x, 8e-14),
            (jax.grad(chebax.besseli_frozen(v))(x), v * val / x, 8e-14),
        ]
        for k, (g, r, bar) in enumerate(got):
            assert abs(float(g) / r - 1.0) <= bar, (v, x, k, float(g), r)


@pytest.mark.parametrize("v", [0.0, 0.5, 1.0, 2.5, 9.97, 10.0])
@pytest.mark.parametrize("scaled", [False, True])
def test_factored_values_and_derivatives(v, scaled):
    """The x=8 seam, both sides; 60-dps refs, relative value bar 2e-14,
    first and second x-derivative bars 8e-14 and 4e-13 away from zeros.
    Near derivative zeros use absolute bars 1e-16 and 2e-15. Experiments 38/39
    record the measured floors; every result must be finite.
    """
    seams = np.r_[np.sqrt(np.arange(1, 5) * 16), 8 / (np.arange(1, 5) / 4)]
    xs = np.unique(
        np.r_[
            0.01,
            0.5,
            1.0,
            seams,
            seams * (1 - 1e-7),
            seams * (1 + 1e-7),
            [8.08, 8.32, 9.04, 9.68, 10.32, 13.2],
            1e4,
        ]
    )
    with mp.workdps(60):
        refs = []
        for x in xs:
            x, nu = mp.mpf(float(x)), mp.mpf(v)
            y = mp.besseli(nu, x)
            d1 = mp.besseli(nu + 1, x) + nu * y / x
            d2 = (1 + nu * nu / (x * x)) * y - d1 / x
            if scaled:
                e = mp.exp(-x)
                y, d1, d2 = y * e, (d1 - y) * e, (d2 - 2 * d1 + y) * e
            refs.append([float(y), float(d1), float(d2)])
        refs = np.array(refs).T
    f = chebax.besseli_frozen(v, scaled=scaled)
    x = jnp.asarray(xs)
    # Unscaled I has already overflowed at the last point.
    live = np.isfinite(refs[0])
    x, refs = x[live], refs[:, live]
    first = jax.grad(lambda x: f(x).sum())
    got = (
        jax.jit(f)(x),
        jax.jit(first)(x),
        jax.jit(jax.grad(lambda x: first(x).sum()))(x),
    )

    def check(g, k):
        assert np.all(np.isfinite(g))
        near = (
            ((np.asarray(x) < 16) & (abs(refs[k]) < 0.01 * abs(refs[0])))
            if scaled and k
            else np.zeros(x.size, bool)
        )
        np.testing.assert_allclose(
            np.asarray(g)[~near], refs[k, ~near], rtol=(2e-14, 8e-14, 4e-13)[k], atol=0
        )
        np.testing.assert_allclose(
            np.asarray(g)[near], refs[k, near], rtol=0, atol=1e-16 if k == 1 else 2e-15
        )

    for k, g in enumerate(got):
        check(g, k)
    direct = lambda x: chebax.besseli_scalar(v, x, scaled=scaled)
    rev = jax.jit(jax.grad(lambda x: direct(x).sum()))(x)
    fwd = jax.jit(lambda x: jax.jvp(direct, (x,), (jnp.ones_like(x),))[1])(x)
    for g in (rev, fwd):
        check(g, 1)


def test_value_order_and_argument_tangents():
    """Symbolic-zero shortcuts preserve simultaneous and mixed tangents.
    Independent 60-dps references; first bar 2e-13, mixed bar 2e-12.
    """
    x = jnp.asarray([0.5, 4.0, 8.0, 8.5, 32.0])
    nu = 1.3
    with mp.workdps(60):
        ref = np.array(
            [
                [
                    float(
                        mp.diff(
                            lambda n: mp.exp(-mp.mpf(float(t)))
                            * mp.besseli(n, mp.mpf(float(t))),
                            mp.mpf(nu),
                        )
                    ),
                    float(
                        mp.diff(
                            lambda z: mp.exp(-z) * mp.besseli(mp.mpf(nu), z),
                            mp.mpf(float(t)),
                        )
                    ),
                    float(
                        mp.diff(
                            lambda n: mp.diff(
                                lambda z: mp.exp(-z) * mp.besseli(n, z),
                                mp.mpf(float(t)),
                            ),
                            mp.mpf(nu),
                        )
                    ),
                ]
                for t in x
            ]
        ).T
    f = lambda nu, x: chebax.besseli_scalar(nu, x, scaled=True)
    both = jax.jit(
        lambda n, x: jax.jvp(f, (n, x), (jnp.ones_like(n), jnp.ones_like(x)))[1]
    )(nu, x)
    np.testing.assert_allclose(both, ref[0] + ref[1], rtol=2e-13, atol=0)
    a = jax.jit(jax.jacfwd(jax.jacrev(f, 1), 0))(nu, x)
    b = jax.jit(jax.jacrev(jax.jacfwd(f, 0), 1))(nu, x)
    for mixed in (a, b):
        assert np.all(np.isfinite(mixed))
        np.testing.assert_allclose(mixed, np.diag(ref[2]), rtol=2e-12, atol=0)


def test_float32_overflow_window(tmp_path):
    """I_0/I_1 from x = 80, where exp(x) is split off, to x = 91.5, past
    exp's overflow at 88.7. Values and slopes at 60 dps; measured 2.6e-7 /
    3.1e-7 (7.2e-6 before the compensated exponential), relative bar 1.2e-6.
    """
    import os
    import subprocess
    import sys

    xs = np.array([80.5, 82.4, 84.9, 86.3, 88.0, 88.9, 90.0, 91.5], np.float32)
    with mp.workdps(60):
        ref = np.array(
            [
                [
                    [
                        float(mp.besseli(n, x)),
                        float(mp.besseli(n + 1, x) + n * mp.besseli(n, x) / x),
                    ]
                    for x in (mp.mpf(float(t)) for t in xs)
                ]
                for n in (0, 1)
            ]
        )
    path = tmp_path / "refs.npz"
    np.savez(path, xs=xs, ref=ref)
    code = f"""
import jax, jax.numpy as jnp, numpy as np, chebax
data=np.load({str(path)!r})
x=jnp.asarray(data['xs'])
for n in (0,1):
    for f in (chebax.besseli_frozen(n),lambda x:chebax.besseli_scalar(float(n),x)):
        val=np.asarray(jax.jit(f)(x))
        grad=np.asarray(jax.jit(jax.vmap(jax.grad(f)))(x))
        assert np.all(np.isfinite(val)) and np.all(np.isfinite(grad))
        np.testing.assert_allclose(np.stack([val,grad],1),data['ref'][n],rtol=1.2e-6,atol=0)
        second=jax.jit(jax.grad(jax.grad(f)))(jnp.float32(0))
        assert np.isfinite(second)
        assert abs(float(second)-(0.5 if n==0 else 0.0)) <= 2e-6
"""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "false"}
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_float32_tail_anchor(tmp_path):
    """The direct tail must preserve small values near x=8 at large orders.
    References at 60 dps and rounded inputs; worst 3.3e-6/3.4e-6/7.2e-6
    for values and two x derivatives, bars 5e-6/1e-5/4e-5 for both paths.
    """
    import os
    import subprocess
    import sys

    xs = np.array([8, 8.08, 8.32, 9.04, 9.68, 10.32, 13.2, 16, 32, 1e4], np.float32)
    orders = np.array([0, 1, 9.97, 10], np.float32)
    with mp.workdps(60):
        refs = []
        for n in orders:
            nu = mp.mpf(float(n))
            row = []
            for t in xs:
                x = mp.mpf(float(t))
                y = mp.besseli(nu, x)
                d1 = mp.besseli(nu + 1, x) + nu * y / x
                d2 = (1 + nu * nu / (x * x)) * y - d1 / x
                e = mp.exp(-x)
                row.append(
                    [float(y * e), float((d1 - y) * e), float((d2 - 2 * d1 + y) * e)]
                )
            refs.append(np.array(row).T)
    path = tmp_path / "tail_refs.npz"
    np.savez(path, xs=xs, orders=orders, refs=refs)
    code = f"""
import jax, jax.numpy as jnp, numpy as np, chebax
data=np.load({str(path)!r})
x=jnp.asarray(data['xs'])
for n,ref in zip(data['orders'],data['refs']):
    for f in (chebax.besseli_frozen(float(n),True),
              lambda x:chebax.besseli_scalar(n,x,True)):
        first=jax.grad(lambda x:f(x).sum())
        second=jax.grad(lambda x:first(x).sum())
        for fun,r,tol in zip((f,first,second),ref,(5e-6,1e-5,4e-5)):
            got=np.asarray(jax.jit(fun)(x))
            assert np.all(np.isfinite(got)), (n,got)
            np.testing.assert_allclose(got,r,rtol=tol,atol=0)
"""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "false"}
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("nu", [0.0, 1.0])
def test_unscaled_tail_gradients_near_overflow(nu):
    """Keep finite I0/I1 slopes near the overflow limit, with 60-dps refs.
    Measured 3.7e-16 (1.1e-13 before the compensated exponential); bar
    1.5e-15.
    """
    xs = np.array([710.0, 713.0, 713.9, 713.95])
    with mp.workdps(60):
        refs = np.array(
            [
                [
                    float(mp.besseli(nu, x)),
                    float(mp.besseli(nu + 1, x) + nu * mp.besseli(nu, x) / x),
                ]
                for x in xs
            ]
        ).T
    for f in (
        chebax.besseli_frozen(nu),
        lambda x: chebax.besseli_scalar(nu, x),
    ):
        first = jax.grad(lambda x: f(x).sum())
        for fun, ref in zip((f, first), refs):
            got = np.asarray(jax.jit(fun)(jnp.asarray(xs)))
            assert np.all(np.isfinite(got))
            np.testing.assert_allclose(got, ref, rtol=1.5e-15, atol=0)
