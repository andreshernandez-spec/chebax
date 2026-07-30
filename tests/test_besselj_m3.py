"""M3 acceptance tests: besselj over the full domain x in [0, 1e4].

References are mpmath at 40 dps; errors sup-normalized per order over the
full grid, which includes both seam points (8, 30) and their 1e-9 neighbors.
Measured worst cases over the 11 orders below: values 1.7e-15 f64, dJ/dx
2.6e-15 f64, dJ/dv ~1e-15.

f32 bars are looser than the [0, 8] ones in test_besselj.py because the
outer region's accuracy is phase-limited: XLA's float32 sin/cos reduce the
argument at ~eps32*x accuracy, so relative-to-envelope error grows like
eps32*x in the tail (measured 4.0e-6 / 7.1e-6 sup-normalized to x = 1e4).
That is a float32 evaluation floor, not a table property.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import besselj_gen
from chebax._src.recipes import besselj_table_ext as ext

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

_rng = np.random.default_rng(7)
NUS = np.concatenate([[0.5, 2.5, 9.97], _rng.uniform(0.05, 9.95, 8)])
XF = np.sort(np.concatenate([
    np.linspace(0.05, 8.0, 16),
    [8.0 - 1e-9, 8.0 + 1e-9, 30.0 - 1e-9, 30.0 + 1e-9],
    np.linspace(8.01, 30.0, 18),
    np.logspace(np.log10(30.5), 4, 16),
]))


@pytest.fixture(scope="module")
def refs():
    out = {}
    for v in NUS:
        J = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x))) for x in XF])
        Jp = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x), 1)) for x in XF])
        out[v] = (J, Jp)
    return out


def worst(err_fn):
    return max(err_fn(v) for v in NUS)


def test_values_f64(refs):
    def err(v):
        J, _ = refs[v]
        return np.max(np.abs(np.asarray(chebax.besselj(float(v))(XF)) - J)) / np.max(np.abs(J))
    assert worst(err) <= 5e-15


def test_dx_f64(refs):
    def err(v):
        _, Jp = refs[v]
        g = jax.vmap(jax.grad(chebax.besselj(float(v))))(jnp.asarray(XF))
        return np.max(np.abs(np.asarray(g) - Jp)) / np.max(np.abs(Jp))
    assert worst(err) <= 1e-14


def test_values_f32(refs):
    x32 = jnp.asarray(XF, jnp.float32)
    def err(v):
        J, _ = refs[v]
        y = chebax.besselj(float(v)).astype(jnp.float32)(x32)
        return np.max(np.abs(np.asarray(y) - J)) / np.max(np.abs(J))
    assert worst(err) <= 1e-5


def test_dx_f32(refs):
    x32 = jnp.asarray(XF, jnp.float32)
    def err(v):
        _, Jp = refs[v]
        g = jax.vmap(jax.grad(chebax.besselj(float(v)).astype(jnp.float32)))(x32)
        return np.max(np.abs(np.asarray(g) - Jp)) / np.max(np.abs(Jp))
    assert worst(err) <= 2e-5


def test_dnu_full_domain():
    xd = np.array([0.3, 2.0, 6.0, 8.0, 12.0, 21.0, 29.0, 31.0, 60.0, 300.0, 4000.0])
    for v in [0.3, 2.5, 7.77, 9.6]:
        ref = np.array([float(mp.diff(lambda w: mp.besselj(w, mp.mpf(x)), mp.mpf(v)))
                        for x in xd])
        got = np.asarray(chebax.besselj_dnu(v)(xd))
        assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) <= 5e-15


def test_seam_jumps():
    # exp-04 criterion: both branches evaluated AT the seam differ by at most
    # the sum of their errors (values at seam +- eps would instead be
    # dominated by the true slope, |J'| * 2eps).
    for v in [0.5, 7.77]:
        jv = chebax.besselj(v)
        x8 = jnp.asarray(8.0)
        assert abs(float(jv._inner(x8)) - float(jv._mid(x8))) <= 1e-13
        x30 = jnp.asarray(30.0)
        xo, P, Q = jv._outer_pq(x30)
        a = P * jv._cphi + Q * jv._sphi
        b = P * jv._sphi - Q * jv._cphi
        outer = jnp.sqrt(2.0 / (jnp.pi * xo)) * (jnp.cos(xo) * a + jnp.sin(xo) * b)
        assert abs(float(jv._mid(x30)) - float(outer)) <= 1e-13


def test_grad_at_seam_points(refs):
    # min/max ties halve tangents if branch inputs are clamped naively; the
    # branch-input arrangement must keep the derivative exact AT 8 and 30.
    v = NUS[4]
    J, Jp = refs[v]
    jv = chebax.besselj(float(v))
    for seam in (8.0, 30.0):
        i = np.nonzero(XF == seam)[0][0]
        g = float(jax.grad(jv)(seam))
        assert abs(g - Jp[i]) / np.max(np.abs(Jp)) <= 1e-14


def test_x_large_smoke():
    ref = float(mp.besselj(mp.mpf(2.5), mp.mpf(1e6)))
    assert abs(float(chebax.besselj(2.5)(1e6)) - ref) <= 1e-16


@pytest.mark.slow
def test_ext_tables_regenerate_bit_for_bit():
    mid = besselj_gen.generate_mid_table()
    ptab, qtab = besselj_gen.generate_outer_tables()
    assert np.array_equal(mid, ext.TABLE_MID)
    assert np.array_equal(ptab, ext.TABLE_P)
    assert np.array_equal(qtab, ext.TABLE_QS)
    assert ext.META["dps"] == besselj_gen.DPS
