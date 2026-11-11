"""M6 (fourth increment) acceptance: betainc on (a, b) in [0.1, 100]^2.

References mpmath at 40 dps; errors absolute (I is a CDF in [0, 1], and
absolute error is the standard contract for CDFs; near the endpoints the
value itself decays, and users needing the small tail to relative accuracy
evaluate the complement parameter-swapped, as documented). dI/dx uses the
exact Beta density as its oracle; dI/da and dI/db are checked against
mp.diff, including through jax.grad on the traced betainc_scalar -- the
gradients jax itself lacks (jax#38610).
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import betainc_gen
from chebax._src.recipes import betainc_table as bt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

PAIRS = [
    (0.15, 0.15),
    (0.5, 0.5),
    (1.0, 1.0),
    (2.0, 3.0),
    (0.15, 9.9),
    (9.9, 0.15),
    (5.5, 5.5),
    (9.9, 9.9),
    (1.0, 0.3),
]
XB = np.sort(
    np.concatenate(
        [
            [1e-4, 1e-2, 0.5, 0.5 - 1e-9, 0.5 + 1e-9, 1 - 1e-2, 1 - 1e-4],
            np.linspace(0.05, 0.95, 15),
        ]
    )
)


def _iref(a, b):
    return np.array(
        [
            float(mp.betainc(mp.mpf(a), mp.mpf(b), 0, mp.mpf(x), regularized=True))
            for x in XB
        ]
    )


def test_values_and_traced():
    for a, b in PAIRS:
        ref = _iref(a, b)
        got = np.asarray(chebax.betainc_frozen(a, b)(XB))
        assert np.max(np.abs(got - ref)) <= 2e-14, (a, b)
        gott = np.asarray(chebax.betainc_scalar(a, b, XB))
        assert np.max(np.abs(gott - ref)) <= 2e-14, (a, b)


def test_kernel_split_matches_every_table():
    # betainc_kernel evaluates at its own XSPLIT so stdtr can run on it
    # without loading this family's tables; every table of the kernel must
    # be fitted on (0, XSPLIT]
    from chebax._src.recipes import betainc_kernel, betainc_wide_table, stdtr_table

    tables = (bt, betainc_wide_table, stdtr_table)
    assert {t.XSPLIT for t in tables} == {betainc_kernel.XSPLIT} == {0.5}


def test_endpoints_exact():
    f = chebax.betainc_frozen(2.5, 0.7)
    assert float(f(0.0)) == 0.0
    assert float(f(1.0)) == 1.0


def test_dx_against_beta_density():
    xs = np.linspace(0.02, 0.98, 25)
    for a, b in [(0.5, 0.5), (2.0, 3.0), (9.9, 0.15), (1.0, 1.0)]:
        pdf = (
            xs ** (a - 1)
            * (1 - xs) ** (b - 1)
            * math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        )
        g = np.asarray(jax.vmap(jax.grad(chebax.betainc_frozen(a, b)))(jnp.asarray(xs)))
        assert np.max(np.abs(g - pdf) / np.max(pdf)) <= 1e-13, (a, b)


def test_dx_is_the_density_pointwise():
    # dI/dx was e * d ln I/dt: above x = 1/2 at a >> b it carried I's absolute
    # error (3.3 relative at (100, 0.1, 0.7287), experiments/33), and in
    # float32 it was 0 * inf where I underflows (test_x64_off). It is the
    # density from its logs now. Measured worst 4.8e-13 at (100, 3, 0.6),
    # bar 2e-12.
    worst = 0.0
    for a, b, xs in [
        (100.0, 0.1, [0.5, 0.7287, 0.9, 0.99]),
        (100.0, 3.0, [1e-3, 0.3, 0.6]),
        (0.1, 100.0, [1e-6, 0.01, 0.4, 0.6]),
        (2.5, 7.0, [1e-8, 0.2, 0.55, 0.999]),
    ]:
        g = jax.grad(lambda x: chebax.betainc_scalar(a, b, x))
        for x in xs:
            ref = mp.exp(
                (a - 1) * mp.log(x) + (b - 1) * mp.log1p(-x) - mp.log(mp.beta(a, b))
            )
            worst = max(worst, abs(float(g(x)) / float(ref) - 1.0))
    assert worst <= 2e-12, worst


def test_da_db_traced():
    for a, b, x in [(0.5, 0.5, 0.3), (2.0, 3.0, 0.6), (7.7, 1.3, 0.2), (1.0, 1.0, 0.7)]:
        da_ref = float(
            mp.diff(
                lambda t: mp.betainc(t, mp.mpf(b), 0, mp.mpf(x), regularized=True),
                mp.mpf(a),
            )
        )
        db_ref = float(
            mp.diff(
                lambda t: mp.betainc(mp.mpf(a), t, 0, mp.mpf(x), regularized=True),
                mp.mpf(b),
            )
        )
        da, db = jax.grad(chebax.betainc_scalar, argnums=(0, 1))(
            jnp.asarray(a), jnp.asarray(b), jnp.asarray(x)
        )
        scale = max(abs(da_ref), abs(db_ref))
        assert abs(float(da) - da_ref) / scale <= 1e-12, (a, b, x)
        assert abs(float(db) - db_ref) / scale <= 1e-12, (a, b, x)


def test_complement_identity():
    for a, b in [(0.3, 4.4), (6.6, 1.5)]:
        s = np.asarray(chebax.betainc_frozen(a, b)(XB)) + np.asarray(
            chebax.betainc_frozen(b, a)(1.0 - XB)
        )
        assert np.max(np.abs(s - 1.0)) <= 5e-14


def test_jit_and_pytree():
    f = chebax.betainc_frozen(2.0, 3.0)
    xs = jnp.asarray(XB[:8])
    y = jax.jit(lambda q, x: q(x))(f, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(f(xs)), rtol=0, atol=1e-15)


def test_param_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.betainc_frozen(0.05, 1.0)
    with pytest.raises(ValueError, match="table covers"):
        chebax.betainc_frozen(1.0, 101.0)


OUT_OF_BOX = [(150.0, 40.0), (40.0, 150.0), (0.05, 2.0), (2.0, 0.05), (101.0, 3.0)]
X_EDGES = jnp.asarray([0.0, 0.3, 0.8, 1.0])


@pytest.mark.parametrize("f", [chebax.betainc_scalar, chebax.log_betainc_scalar])
def test_traced_out_of_box_is_nan(f):
    # the tables used to extrapolate in silence: betainc(150, 40, .8) gave -inf
    # with -inf/+inf gradients (scipy 0.6276), log_betainc nan; a = 101 and
    # 0.05 happened to be close
    one = jax.vmap(f, (None, None, 0))
    rev = jax.jit(jax.vmap(jax.grad(f, (0, 1, 2)), (None, None, 0)))
    fwd = jax.jit(jax.vmap(jax.jacfwd(f, (0, 1, 2)), (None, None, 0)))
    for a, b in OUT_OF_BOX:
        assert np.isnan(np.asarray(one(a, b, X_EDGES))).all()
        assert np.isnan(np.asarray(jax.jit(one)(a, b, X_EDGES))).all()
        for g in rev(a, b, X_EDGES) + fwd(a, b, X_EDGES):
            assert np.isnan(np.asarray(g)).all()
    # the box is closed
    for a, b in [(100.0, 0.1), (0.1, 100.0), (100.0, 100.0)]:
        x = X_EDGES[1:3]
        assert np.isfinite(np.asarray(one(a, b, x))).all()
        assert all(np.isfinite(np.asarray(g)).all() for g in rev(a, b, x))


def test_out_of_box_group_stays_in_its_group():
    idx = np.array([0, 1, 0, 1, 1])
    x = jnp.asarray([0.2, 0.4, 0.6, 0.8, 0.5])
    f = chebax.pergroup(chebax.betainc_scalar, idx)
    a, b = jnp.asarray([2.0, 150.0]), jnp.asarray([3.0, 40.0])
    y = np.asarray(f(a, b, x))
    ga, gb, gx = jax.grad(lambda a, b, x: jnp.sum(f(a, b, x)), (0, 1, 2))(a, b, x)
    live = idx == 0
    ref = jax.vmap(lambda t: chebax.betainc_scalar(2.0, 3.0, t))(x[live])
    np.testing.assert_allclose(y[live], np.asarray(ref), rtol=5e-15)
    assert np.isnan(y[~live]).all()
    assert np.isfinite([ga[0], gb[0]]).all() and np.isnan([ga[1], gb[1]]).all()
    assert np.isfinite(np.asarray(gx)[live]).all()
    assert np.isnan(np.asarray(gx)[~live]).all()


@pytest.mark.slow
def test_table_regenerates_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib

    betainc_gen.main(tmp_path)
    assert (tmp_path / "betainc_table.py").read_text() == pathlib.Path(
        bt.__file__
    ).read_text()
    assert bt.META["dps"] == betainc_gen.DPS


def test_log_betainc():
    # Direct zone (x <= 1/2): error / max(1, |ln I|) measured worst
    # 8.1e-15 over the pairs below, grid down to x = 1e-100 (ln I ~ -2280;
    # betainc_scalar itself underflows below I ~ 1e-308). Reflected zone
    # (x > 1/2): ln of an absolutely-accurate value, error is the value
    # path's absolute floor over I (measured 23 eps / I worst); the bar
    # carries the value bar 2e-14 divided by I. d ln I/da vs mp.diff measured
    # 3.1e-15. Bars ~4x.
    xl = np.concatenate([[1e-100, 1e-30, 1e-10, 1e-4], np.linspace(0.01, 0.5, 8)])
    xr = np.linspace(0.51, 0.999, 8)
    for a, b in [(0.15, 0.15), (2.0, 3.0), (9.9, 0.15), (9.9, 9.9), (0.15, 9.9)]:
        ref = np.array(
            [
                float(
                    mp.log(
                        mp.betainc(mp.mpf(a), mp.mpf(b), 0, mp.mpf(x), regularized=True)
                    )
                )
                for x in xl
            ]
        )
        got = np.asarray(chebax.log_betainc_scalar(a, b, jnp.asarray(xl)))
        assert np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))) <= 4e-14
        refr = np.array(
            [
                float(
                    mp.log(
                        mp.betainc(mp.mpf(a), mp.mpf(b), 0, mp.mpf(x), regularized=True)
                    )
                )
                for x in xr
            ]
        )
        gotr = np.asarray(chebax.log_betainc_scalar(a, b, jnp.asarray(xr)))
        bar = 4e-14 * np.maximum(1.0, np.abs(refr)) + 2e-14 / np.exp(refr)
        assert np.all(np.abs(gotr - refr) <= bar), (a, b)
    # endpoints, nan, and agreement with the linear form where it lives
    assert np.isneginf(float(chebax.log_betainc_scalar(2.0, 3.0, 0.0)))
    assert float(chebax.log_betainc_scalar(2.0, 3.0, 1.0)) == 0.0
    assert np.isnan(float(chebax.log_betainc_scalar(2.0, 3.0, np.nan)))
    xs = np.linspace(0.05, 0.95, 9)
    lin = np.asarray(chebax.betainc_scalar(2.0, 3.0, jnp.asarray(xs)))
    np.testing.assert_allclose(
        np.exp(np.asarray(chebax.log_betainc_scalar(2.0, 3.0, jnp.asarray(xs)))),
        lin,
        rtol=1e-13,
    )
    # shape gradient through the traced path
    xg = np.array([1e-4, 0.05, 0.3, 0.49])
    for a, b in [(0.15, 9.9), (2.0, 3.0), (9.9, 0.15)]:
        da = np.array(
            [
                float(
                    mp.diff(
                        lambda t: mp.log(
                            mp.betainc(t, mp.mpf(b), 0, mp.mpf(x), regularized=True)
                        ),
                        mp.mpf(a),
                    )
                )
                for x in xg
            ]
        )
        gda = np.asarray(
            jax.vmap(
                jax.grad(chebax.log_betainc_scalar, argnums=0), in_axes=(None, None, 0)
            )(jnp.asarray(a), jnp.asarray(b), jnp.asarray(xg))
        )
        assert np.max(np.abs(gda - da) / np.maximum(1.0, np.abs(da))) <= 1.5e-14


WIDE_PAIRS = [
    (0.15, 55.0),
    (55.0, 0.15),
    (55.0, 55.0),
    (99.9, 99.9),
    (2.0, 99.9),
    (99.9, 2.0),
    (30.0, 70.0),
    (0.1, 100.0),
    (100.0, 100.0),
    (11.0, 11.0),
]


def test_wide_panels_values_and_gradients():
    # increment 23: the box is [0.1, 100]^2 via three additional panels.
    # Measured worst over the pairs below: values 1.8e-13 absolute (the
    # sharp-transition interior (30, 70)), dI/da 8.5e-16 vs mp.diff,
    # betaincinv roundtrip 6.7e-15; bars ~4x. The original box's floor
    # is untouched (the low-low panel IS the original tensor).
    for a, b in WIDE_PAIRS:
        ref = _iref(a, b)
        got = np.asarray(chebax.betainc_frozen(a, b)(XB))
        gott = np.asarray(chebax.betainc_scalar(a, b, jnp.asarray(XB)))
        assert np.max(np.abs(got - ref)) <= 8e-13, (a, b)
        assert np.max(np.abs(gott - ref)) <= 8e-13, (a, b)
    xg = np.array([0.1, 0.5, 0.9])
    for a, b in [(55.0, 0.15), (2.0, 99.9), (70.0, 30.0)]:
        da = np.array(
            [
                float(
                    mp.diff(
                        lambda t: mp.betainc(
                            t, mp.mpf(b), 0, mp.mpf(x), regularized=True
                        ),
                        mp.mpf(a),
                    )
                )
                for x in xg
            ]
        )
        g = np.asarray(
            jax.vmap(
                jax.grad(chebax.betainc_scalar, argnums=0), in_axes=(None, None, 0)
            )(jnp.asarray(a), jnp.asarray(b), jnp.asarray(xg))
        )
        assert np.max(np.abs(g - da)) <= 1e-14, (a, b)
    for a, b in [(55.0, 55.0), (2.0, 99.9), (99.9, 2.0)]:
        ps = np.array([1e-6, 0.05, 0.5, 0.95, 1 - 1e-6])
        x = np.asarray(chebax.betaincinv(a, b, jnp.asarray(ps)))
        rt = np.asarray(chebax.betainc_scalar(a, b, jnp.asarray(x)))
        assert np.max(np.abs(rt - ps)) <= 3e-14, (a, b)


def test_panel_edges_are_ordinary_points():
    # each side of the a = 10 split meets mpmath at its own panel's bar;
    # a naive two-sided jump test would just measure dI/da * 2 eps
    for a in (10.0 - 1e-9, 10.0 + 1e-9):
        ref = _iref(a, 5.0)
        got = np.asarray(chebax.betainc_scalar(a, 5.0, jnp.asarray(XB)))
        assert np.max(np.abs(got - ref)) <= 8e-13, a


@pytest.mark.slow
def test_wide_hilo_panel_regenerates_exact():
    # the CI-affordable regeneration canary (~1.5 min): the smallest wide
    # panel rebuilds bit-exactly through the full generator pipeline.
    # The complete three-panel regeneration takes ~25 min and runs only
    # with CHEBAX_FULL_REGEN=1 (test below); policy in CLAUDE.md.
    import mpmath as mpm
    from chebax._src.recipes import betainc_wide_gen as g
    from chebax._src.recipes import betainc_wide_table as bw
    from chebax._src.recipes._gen_common import DPS

    with mpm.workdps(DPS):
        t = g.generate_panel(mpm, (10.0, 100.0), (0.1, 10.0), 24, 60, 20)
    assert np.array_equal(t, bw.TENSOR_HILO)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CHEBAX_FULL_REGEN"),
    reason="~25 min; set CHEBAX_FULL_REGEN=1 (pre-release)",
)
def test_wide_tables_regenerate_bit_for_bit(tmp_path):
    import pathlib
    from chebax._src.recipes import betainc_wide_gen
    from chebax._src.recipes import betainc_wide_table as bw

    betainc_wide_gen.main(tmp_path)
    assert (tmp_path / "betainc_wide_table.py").read_text() == pathlib.Path(
        bw.__file__
    ).read_text()


def test_endpoint_derivatives_are_the_density():
    # Both endpoint values come from a hard select, so AD read a slope of
    # 0 there even where the density is exact and finite: grad of
    # betainc_scalar(1, 3, .) at 0 gave 0 for a true 3 (review, 2026-08-02).
    # f(0) = 1/B(1, b) = b when a = 1, f(1) = 1/B(a, 1) = a when b = 1;
    # 0 above the 1, infinite below it. Values are unchanged.
    assert float(jax.grad(lambda x: chebax.betainc_scalar(1.0, 3.0, x))(0.0)) == 3.0
    assert float(jax.grad(lambda x: chebax.betainc_scalar(3.0, 1.0, x))(1.0)) == 3.0
    assert float(jax.grad(lambda x: chebax.betainc_scalar(2.0, 3.0, x))(0.0)) == 0.0
    assert float(jax.grad(lambda x: chebax.betainc_scalar(3.0, 2.0, x))(1.0)) == 0.0
    assert np.isposinf(
        float(jax.grad(lambda x: chebax.betainc_scalar(0.5, 3.0, x))(0.0))
    )
    assert np.isposinf(
        float(jax.grad(lambda x: chebax.betainc_scalar(3.0, 0.5, x))(1.0))
    )
    for a, b in ((1.0, 3.0), (2.5, 0.5)):
        assert float(chebax.betainc_scalar(a, b, 0.0)) == 0.0
        assert float(chebax.betainc_scalar(a, b, 1.0)) == 1.0


def test_reverse_mode_leaves_the_panel_switch_primal():
    # reverse mode through the panel switch cost ~120 ms per parameter on the
    # GPU (experiments/29); the rules compute the derivatives inside it, so the
    # compiled gradient has no transposed conditional (betaincinv's second one
    # is the solve's, primal)
    a, b, x = jnp.asarray(2.5), jnp.asarray(3.5), jnp.linspace(0.1, 0.9, 8)
    for f in (chebax.betainc_scalar, chebax.log_betainc_scalar, chebax.betaincinv):
        g = jax.jit(jax.grad(lambda a, b, x: f(a, b, x).sum(), argnums=(0, 1, 2)))
        conds = [
            l
            for l in g.lower(a, b, x).compile().as_text().splitlines()
            if " conditional(" in l
        ]
        assert 1 <= len(conds) <= 2 and not any(
            "transpose(" in l for l in conds
        ), f.__name__


def test_second_derivatives():
    # The rules evaluate their derivative series through chebval_either,
    # which had no AD rule of its own: grad of grad, hessian and jacfwd of
    # betainc, log_betainc, betaincinv and stdtrit raised (0.4.1 had them).
    # Closed forms where there are any: f'(x) of the Beta density, and
    # d2x/dp2 = -f'(x)/f(x)^3 for the quantiles.
    a, b, x = 2.0, 3.0, 0.2
    pdf = (
        x ** (a - 1)
        * (1 - x) ** (b - 1)
        / math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    dpdf = pdf * ((a - 1) / x - (b - 1) / (1 - x))
    I = float(chebax.betainc(a, b, x))
    f = lambda x: chebax.betainc(a, b, x)
    for d2 in (jax.grad(jax.grad(f)), jax.hessian(f), jax.jacfwd(jax.jacrev(f))):
        assert abs(float(d2(x)) - dpdf) <= 1e-12 * abs(dpdf)
    lf = jax.grad(jax.grad(lambda x: chebax.log_betainc(a, b, x)))(x)
    ref = dpdf / I - (pdf / I) ** 2
    assert abs(float(lf) - ref) <= 1e-12 * abs(ref)
    got = jax.hessian(lambda p: chebax.betainc(*p, x))(jnp.asarray([a, b]))
    for i, j, n in ((0, 0, (2, 0)), (0, 1, (1, 1)), (1, 1, (0, 2))):
        h = float(
            mp.diff(lambda u, v: mp.betainc(u, v, 0, x, regularized=True), (a, b), n)
        )
        assert abs(float(got[i, j]) - h) <= 1e-11 * abs(h), (i, j)
    p = 0.3
    xq = float(chebax.betaincinv(a, b, p))
    fq = (
        xq ** (a - 1)
        * (1 - xq) ** (b - 1)
        / math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    dfq = fq * ((a - 1) / xq - (b - 1) / (1 - xq))
    got = float(jax.grad(jax.grad(lambda p: chebax.betaincinv(a, b, p)))(p))
    assert abs(got + dfq / fq**3) <= 1e-11 * abs(dfq / fq**3)
    nu = 4.0
    t = float(chebax.stdtrit(nu, p))
    ft = math.exp(
        math.lgamma((nu + 1) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * math.log(nu * math.pi)
        - (nu + 1) / 2 * math.log1p(t * t / nu)
    )
    dft = -ft * (nu + 1) * t / (nu + t * t)
    got = float(jax.grad(jax.grad(lambda p: chebax.stdtrit(nu, p)))(p))
    assert abs(got + dft / ft**3) <= 1e-11 * abs(dft / ft**3)
    # vmapped, as pergroup runs it: the coefficients are batched
    d2 = jax.grad(jax.grad(lambda a, x: chebax.betainc(a, b, x), 1), 1)
    hv = jax.vmap(d2, (0, None))
    hv = hv(jnp.asarray([a, a]), x)
    assert np.all(np.abs(np.asarray(hv) - dpdf) <= 1e-12 * abs(dpdf))
