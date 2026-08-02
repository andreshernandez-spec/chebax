"""Large-a gammainc acceptance: the Temme-zone path on a > 10,
x in [0, inf), plus the a = 10 dispatch seam and the quantile rewire.

References mpmath at 40 dps, Q computed DIRECTLY (1 - P in mp loses the
tail's digits; the reference artifact recorded in experiments/14). Past
a ~ 1e5 mpmath's gammainc stops converging near lambda ~ 1, so the
large-a tests carry their own side-aware reference: the 1F1 series for
P below the mean, Legendre's continued fraction for Q above it, each
where it converges (experiments/16).

P/Q are CDFs: absolute error. The log forms are relative on
max(1, |ln .|). Measured worst over a denser sweep than the grids
below: P/Q 2.9e-16, log P 4.5e-15, log Q 1.3e-14, dP/da 2.8e-17 and the
log-form shape gradients 8.1e-16 vs mp.diff; at a from 1e4 to 1e8 the
log forms reach 5.0e-14, which is the cancelling bracket
a ln x - x - lnGamma(a+1) at that size, not the tables. Bars ~4x.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import chebax
from chebax._src.recipes import gammainc_large_gen as glgen
from chebax._src.recipes import gammainc_large_table as glt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

AS = [10.001, 15.0, 40.0, 100.0, 500.0, 999.0, 1000.0]


def _grid(a):
    return np.array([0.01 * a, 0.3 * a, 0.499 * a, 0.5 * a, 0.501 * a,
                     0.9 * a, a, 1.1 * a, 2.0 * a, 4.0 * a,
                     6.0 * (a - 1) - 0.01, 6.0 * (a - 1) + 0.01,
                     10.0 * a, 30.0 * a])


def _prefs(a, xs):
    return np.array([float(mp.gammainc(mp.mpf(a), 0, mp.mpf(x),
                                       regularized=True)) for x in xs])


def _qrefs(a, xs):
    return np.array([float(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf,
                                       regularized=True)) for x in xs])


def test_values_factory_and_traced():
    for a in AS:
        xs = _grid(a)
        pref, qref = _prefs(a, xs), _qrefs(a, xs)
        assert np.max(np.abs(np.asarray(chebax.gammainc(a)(xs)) - pref)) \
            <= 1.2e-15, a
        assert np.max(np.abs(np.asarray(chebax.gammaincc(a)(xs)) - qref)) \
            <= 1.2e-15, a
        assert np.max(np.abs(
            np.asarray(chebax.gammainc_fn(a, jnp.asarray(xs))) - pref)) \
            <= 1.2e-15, a
        assert np.max(np.abs(
            np.asarray(chebax.gammaincc_fn(a, jnp.asarray(xs))) - qref)) \
            <= 1.2e-15, a


def test_log_forms_and_deep_tails():
    # the log forms stay finite and relatively accurate past the linear
    # forms' f64 underflow (a w > 745 nats deep in either tail)
    for a in [15.0, 500.0, 1000.0]:
        xs = np.array([0.05 * a, 0.4 * a, 0.9 * a, 1.2 * a, 3.0 * a,
                       8.0 * a, 20.0 * a])
        lp = np.array([float(mp.log(mp.gammainc(mp.mpf(a), 0, mp.mpf(x),
                                                regularized=True)))
                       for x in xs])
        lq = np.array([float(mp.log(mp.gammainc(mp.mpf(a), mp.mpf(x),
                                                mp.inf, regularized=True)))
                       for x in xs])
        glp = np.asarray(chebax.log_gammainc_fn(a, jnp.asarray(xs)))
        glq = np.asarray(chebax.log_gammaincc_fn(a, jnp.asarray(xs)))
        assert np.max(np.abs(glp - lp) / np.maximum(1.0, np.abs(lp))) \
            <= 6e-14, a
        assert np.max(np.abs(glq - lq) / np.maximum(1.0, np.abs(lq))) \
            <= 6e-14, a


def test_dispatch_seam_at_ten():
    # both sides of a = 10 meet mpmath at their own path's bar
    xs = np.linspace(0.5, 60.0, 21)
    for a in (10.0 - 1e-9, 10.0 + 1e-9):
        ref = _prefs(a, xs)
        got = np.asarray(chebax.gammainc_fn(a, jnp.asarray(xs)))
        assert np.max(np.abs(got - ref)) <= 1.2e-15, a


def test_complement_and_endpoints():
    for a in (40.0, 700.0):
        xs = _grid(a)
        s = (np.asarray(chebax.gammainc_fn(a, jnp.asarray(xs)))
             + np.asarray(chebax.gammaincc_fn(a, jnp.asarray(xs))))
        assert np.max(np.abs(s - 1.0)) <= 3e-15, a
    f = chebax.gammainc(123.0)
    assert float(f(0.0)) == 0.0
    assert np.isnan(float(f(np.nan)))
    assert float(chebax.gammaincc(123.0)(0.0)) == 1.0


def test_dPda_and_log_grads():
    for a in [15.0, 100.0, 800.0]:
        for lam in [0.3, 0.9, 1.2, 3.0]:
            x = lam * a
            ref = float(mp.diff(lambda t: mp.gammainc(
                t, 0, mp.mpf(x), regularized=True), mp.mpf(a)))
            g = float(jax.grad(chebax.gammainc_fn)(jnp.float64(a),
                                                   jnp.float64(x)))
            assert abs(g - ref) / max(1.0, abs(ref)) <= 1e-15, (a, lam)
            rlq = float(mp.diff(lambda t: mp.log(mp.gammainc(
                t, mp.mpf(x), mp.inf, regularized=True)), mp.mpf(a)))
            glq = float(jax.grad(chebax.log_gammaincc_fn)(
                jnp.float64(a), jnp.float64(x)))
            assert abs(glq - rlq) / max(1.0, abs(rlq)) <= 4e-15, (a, lam)


def test_dx_against_density():
    for a in (25.0, 400.0):
        xs = np.array([0.6 * a, a, 1.5 * a])
        pdf = np.array([float(mp.exp((mp.mpf(a) - 1) * mp.log(mp.mpf(x))
                                     - mp.mpf(x) - mp.loggamma(mp.mpf(a))))
                        for x in xs])
        g = np.asarray(jax.vmap(jax.grad(chebax.gammainc_fn, argnums=1),
                                in_axes=(None, 0))(
            jnp.float64(a), jnp.asarray(xs)))
        assert np.max(np.abs(g - pdf) / np.max(pdf)) <= 1e-13, a


def _log_q_cf(a, x):
    """ln Q(a, x) by Legendre's continued fraction, for x > a: the
    reference where mpmath's gammainc stops converging."""
    a, x = mp.mpf(a), mp.mpf(x)
    tiny = mp.mpf(10) ** (-2 * mp.mp.dps)
    eps = mp.mpf(10) ** (-mp.mp.dps - 5)
    b, c, d = x + 1 - a, 1 / tiny, 1 / (x + 1 - a)
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    else:
        raise RuntimeError("continued fraction stalled")
    return a * mp.log(x) - x - mp.loggamma(a) + mp.log(h)


def _log_p_series(a, x):
    """ln P(a, x) from 1F1(1; a+1; x), for x < a."""
    a, x = mp.mpf(a), mp.mpf(x)
    s = t = mp.mpf(1)
    n = 0
    while abs(t) > mp.mpf(10) ** (-mp.mp.dps - 10) * abs(s):
        t *= x / (a + 1 + n)
        s += t
        n += 1
    return a * mp.log(x) - x - mp.loggamma(a + 1) + mp.log(s)


def _ref_logs(a, x):
    """(ln P, ln Q) at any a, each side computed where it converges."""
    if x < a:
        lp = _log_p_series(a, x)
        return lp, mp.log(-mp.expm1(lp))
    lq = _log_q_cf(a, x)
    return mp.log(-mp.expm1(lq)), lq


def test_unbounded_shape():
    # The v = 10/a axis reaches v = 0, so the path has no upper end; the
    # old a = 1000 cap was the 40-dps REFERENCE giving out, not the
    # representation. Log metric, since P and Q themselves underflow out
    # here. Measured worst 5.0e-14, bar 2e-13.
    worst = 0.0
    for a in (5e3, 1e4, 1e5, 1e6, 1e8):
        for lam in (0.3, 0.6, 0.95, 1.0, 1.05, 2.0, 9.0):
            x = a * lam
            lp_ref, lq_ref = _ref_logs(a, x)
            for got, ref in ((float(chebax.log_gammainc_fn(a, x)), lp_ref),
                             (float(chebax.log_gammaincc_fn(a, x)), lq_ref)):
                r = float(ref)
                worst = max(worst, abs(got - r) / max(1.0, abs(r)))
    assert worst <= 2e-13, worst
    # and the quantiles still invert out there: measured worst 4.5e-12
    # relative on the round trip, bar 2e-11
    wq = 0.0
    for a in (1e4, 1e6):
        for p in (1e-30, 1e-10, 1e-3, 0.5):
            x = float(chebax.gammaincinv(a, p))
            wq = max(wq, abs(float(mp.exp(_ref_logs(a, x)[0])) / p - 1))
            y = float(chebax.gammainccinv(a, p))
            wq = max(wq, abs(float(mp.exp(_ref_logs(a, y)[1])) / p - 1))
    assert wq <= 2e-11, wq


def test_gammaincinv_and_chi2inv_high_a():
    # the quantile solver's recipe residual serves every a > 10
    for a in (50.0, 500.0, 1000.0):
        ps = np.array([1e-12, 1e-4, 0.05, 0.5, 0.95, 1 - 1e-6])
        x = np.asarray(chebax.gammaincinv(a, jnp.asarray(ps)))
        rt = np.asarray(chebax.gammainc_fn(a, jnp.asarray(x)))
        assert np.max(np.abs(rt - ps) / np.maximum(ps, 1e-300)) <= 2e-13, a
    # chi-squared at 1000 and 2000 dof against mp
    for k in (1000.0, 2000.0):
        for p in (0.01, 0.5, 0.99):
            got = float(chebax.chi2inv(k, p))
            ref = 2.0 * float(mp.findroot(
                lambda x: mp.gammainc(mp.mpf(k) / 2, 0, x, regularized=True)
                - p, mp.mpf(k) / 2))
            assert abs(got / 2 - ref / 2) / ref <= 1e-13, (k, p)
    # dx/da via the JVP's large-path dP/da term
    g = float(jax.grad(lambda a: jnp.sum(chebax.gammaincinv(a, 0.3)))(
        jnp.float64(300.0)))
    h = 1e-4
    fd = (float(chebax.gammaincinv(300.0 + h, 0.3))
          - float(chebax.gammaincinv(300.0 - h, 0.3))) / (2 * h)
    assert abs(g - fd) / abs(fd) <= 1e-6


def test_param_out_of_range():
    # there is no upper end any more; the lower one still bites
    with pytest.raises(ValueError, match="table covers"):
        chebax.gammainc(0.05)
    with pytest.raises(ValueError, match="table covers"):
        chebax.gammaincc(0.0)
    for a in (1001.0, 1e6, 1e12):
        assert float(chebax.gammainc(a)(a)) > 0.0


def test_jit_and_pytree():
    f = chebax.gammainc(77.0)
    xs = jnp.asarray(_grid(77.0)[:6])
    y = jax.jit(lambda q, x: q(x))(f, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(f(xs)),
                               rtol=0, atol=1e-15)


def test_table_regenerates_bit_for_bit(tmp_path):
    # ~1.3k coefficients, seconds: the full byte-for-byte check needs no
    # canary split, unlike the big tensors
    import pathlib
    glgen.main(tmp_path)
    assert ((tmp_path / "gammainc_large_table.py").read_text()
            == pathlib.Path(glt.__file__).read_text())
    assert glt.META["dps"] == glgen.DPS
