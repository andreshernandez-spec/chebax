"""hyp1f1 acceptance: Kummer M(a, b, x) on (a, b) in [0.1, 10]^2,
x in [0, inf).

References mpmath at 40 dps. M is positive on the box, so errors are
pointwise-relative, normalized by max(1, |ln M|): the exact e^x
prefactor carries the dynamic range and the contract degrades as
eps |ln M| ~ eps x (the besselk/gammainc log-table contract). Measured
worst over a denser sweep than the grids below (9x9 pairs x 13 x):
values and log form 3.7e-13 (at a=0.1, b=8, x=15; the 3-D contraction
floor), log form at deep x 1.7e-16 relative, dM/dx 9.5e-14 vs the
exact (a/b) M(a+1, b+1, x) oracle, d/da and d/db 2.9e-15 vs mp.diff;
bars ~4x.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax
from chebax._src.recipes import hyp1f1_gen
from chebax._src.recipes import hyp1f1_table as ht

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

PAIRS = [(0.1, 0.1), (0.15, 9.9), (9.9, 0.15), (9.9, 9.9), (1.0, 1.0),
         (2.0, 3.0), (5.5, 0.2), (0.2, 5.5), (10.0, 10.0), (0.1, 10.0)]
# capped at 300: past x ~ 400 the (9.9, 0.15) corner's M leaves f64
# (ln M ~ x + (a-b) ln x); the log form owns that range (test below)
XG = np.array([1e-6, 1e-3, 0.05, 0.5, 2.0, 8.0, 15.0, 29.9, 30.1, 45.0,
               100.0, 300.0])


def _ref(a, b, xs):
    return np.array([mp.hyp1f1(mp.mpf(a), mp.mpf(b), mp.mpf(x)) for x in xs])


def _rel(got, ref):
    ref = np.array(ref)
    lnm = np.array([float(abs(mp.log(r))) for r in ref])
    err = np.array([float(abs((mp.mpf(float(g)) - r) / r))
                    for g, r in zip(got, ref)])
    return err / np.maximum(1.0, lnm)


def test_values_and_traced():
    for a, b in PAIRS:
        ref = _ref(a, b, XG)
        got = np.asarray(chebax.hyp1f1(a, b)(XG))
        gott = np.asarray(chebax.hyp1f1_fn(a, b, jnp.asarray(XG)))
        assert np.max(_rel(got, ref)) <= 1.5e-12, (a, b)
        assert np.max(_rel(gott, ref)) <= 1.5e-12, (a, b)


def test_log_form_no_overflow():
    xs = np.array([1e-6, 0.5, 29.0, 31.0, 700.0, 5000.0])
    for a, b in [(0.15, 9.9), (9.9, 0.15), (2.0, 3.0)]:
        ref = np.array([float(mp.log(mp.hyp1f1(mp.mpf(a), mp.mpf(b),
                                               mp.mpf(x)))) for x in xs])
        got = np.asarray(chebax.log_hyp1f1_fn(a, b, jnp.asarray(xs)))
        assert np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))) \
            <= 1.5e-12, (a, b)
        # agreement with the linear form where it lives
        lin = np.asarray(chebax.hyp1f1_fn(a, b, jnp.asarray(xs[:4])))
        np.testing.assert_allclose(np.exp(got[:4]), lin, rtol=1e-13)


def test_endpoints_and_nan():
    f = chebax.hyp1f1(2.5, 0.7)
    assert float(f(0.0)) == 1.0
    assert np.isnan(float(f(-1.0)))
    assert np.isnan(float(chebax.hyp1f1_fn(2.5, 0.7, np.nan)))
    assert np.isnan(float(chebax.hyp1f1_fn(np.nan, 0.7, 1.0)))


def test_dx_against_contiguous_oracle():
    # dM/dx = (a/b) M(a+1, b+1, x) exactly; a, b chosen so the shifted
    # parameters stay inside the box
    xs = np.array([0.1, 1.0, 7.0, 20.0, 40.0])
    for a, b in [(0.5, 0.5), (2.0, 3.0), (8.9, 0.2), (0.2, 8.9)]:
        ref = np.array([float(mp.mpf(a) / mp.mpf(b)
                              * mp.hyp1f1(mp.mpf(a) + 1, mp.mpf(b) + 1,
                                          mp.mpf(x))) for x in xs])
        g = np.asarray(jax.vmap(jax.grad(
            lambda x: chebax.hyp1f1_fn(a, b, x)))(jnp.asarray(xs)))
        lnm = np.array([max(1.0, float(abs(mp.log(mp.hyp1f1(
            mp.mpf(a), mp.mpf(b), mp.mpf(x)))))) for x in xs])
        assert np.max(np.abs(g - ref) / (np.abs(ref) * lnm)) <= 4e-13, (a, b)


def test_da_db_traced():
    for a, b, x in [(0.5, 0.5, 2.0), (2.0, 3.0, 10.0), (7.7, 1.3, 0.5),
                    (0.15, 9.9, 40.0)]:
        da_ref = float(mp.diff(lambda t: mp.log(mp.hyp1f1(t, mp.mpf(b),
                                                          mp.mpf(x))),
                               mp.mpf(a)))
        db_ref = float(mp.diff(lambda t: mp.log(mp.hyp1f1(mp.mpf(a), t,
                                                          mp.mpf(x))),
                               mp.mpf(b)))
        da, db = jax.grad(chebax.log_hyp1f1_fn, argnums=(0, 1))(
            jnp.asarray(a), jnp.asarray(b), jnp.asarray(x))
        scale = max(abs(da_ref), abs(db_ref), 1.0)
        assert abs(float(da) - da_ref) / scale <= 1e-13, (a, b, x)
        assert abs(float(db) - db_ref) / scale <= 1e-13, (a, b, x)


def test_seam_is_ordinary_point():
    xs = np.array([ht.XS - 1e-9, ht.XS + 1e-9])
    for a, b in [(0.15, 9.9), (9.9, 0.15), (5.0, 5.0)]:
        ref = _ref(a, b, xs)
        got = np.asarray(chebax.hyp1f1_fn(a, b, jnp.asarray(xs)))
        assert np.max(_rel(got, ref)) <= 1.5e-12, (a, b)


def test_jit_and_pytree():
    f = chebax.hyp1f1(2.0, 3.0)
    xs = jnp.asarray(XG[:6])
    y = jax.jit(lambda q, x: q(x))(f, xs)
    # a few ulp: jit fusion may reassociate the Clenshaw arithmetic
    np.testing.assert_allclose(np.asarray(y), np.asarray(f(xs)),
                               rtol=5e-15)


def test_param_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.hyp1f1(0.05, 1.0)
    with pytest.raises(ValueError, match="table covers"):
        chebax.hyp1f1(1.0, 10.5)


@pytest.mark.slow
def test_tail_table_regenerates_exact():
    # the CI-affordable regeneration canary (~1 min): the tail table
    # rebuilds bit-exactly through the full generator pipeline. The full
    # module takes ~11 min and regenerates only with CHEBAX_FULL_REGEN=1
    # (test below); policy in CLAUDE.md.
    import mpmath as mpm
    from chebax._src.recipes._gen_common import DPS
    with mpm.workdps(DPS):
        t = hyp1f1_gen.generate_table(mpm, hyp1f1_gen.tail_rows,
                                      hyp1f1_gen.NT_TAIL,
                                      hyp1f1_gen.NA_TAIL,
                                      hyp1f1_gen.NB_TAIL)
    assert np.array_equal(t, ht.TABLE_TAIL)


@pytest.mark.slow
@pytest.mark.skipif(not __import__("os").environ.get("CHEBAX_FULL_REGEN"),
                    reason="~11 min; set CHEBAX_FULL_REGEN=1 (pre-release)")
def test_table_regenerates_bit_for_bit(tmp_path):
    import pathlib
    hyp1f1_gen.main(tmp_path)
    assert ((tmp_path / "hyp1f1_table.py").read_text()
            == pathlib.Path(ht.__file__).read_text())
    assert ht.META["dps"] == hyp1f1_gen.DPS
