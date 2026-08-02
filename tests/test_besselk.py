"""M5 acceptance: besselk over [1e-6, inf), nu in [0, 10].

References are mpmath at 40 dps. K is positive with no zeros, so errors are
pointwise RELATIVE here (not sup-normalized). The f64 floor is the
(x/2)^(-v) prefactor's pow error, ~eps*v*|ln(x/2)| (up to ~3e-14 in the
(v=10, x=1e-6) corner); measured worst over the orders below: values
1.2e-14, dK/dx 1.1e-14, dK/dnu 3.3e-13 (including the traced-nu path).
The order set deliberately includes exact integers (1.0, 2.0, 3.0) and the
panel edge (0.999, 1.0, 1.001): direct log-tabulation makes them ordinary
points, unlike the I_{+-v} connection formula (risk S3).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import besselk_gen
from chebax._src.recipes import besselk_table as kt

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

NUS = [0.03, 0.5, 0.999, 1.0, 1.001, 1.7, 2.0, 3.0, 5.5, 7.77, 9.97]
# grid stops at 600: near the underflow edge (x ~ 690, K ~ 1e-305) platforms
# that flush subnormals (XLA CPU) lose small correction terms - e.g. the
# 1/(2x) term of dK/dx underflows in the product K/(2x) - so relative
# accuracy there is platform-dependent by nature
XK = np.sort(np.concatenate([
    np.logspace(-6, np.log10(8.0), 20), [8.0, 8.0001],
    np.logspace(np.log10(8.2), np.log10(600.0), 12),
]))


def _kref(v):
    return np.array([float(mp.besselk(mp.mpf(v), mp.mpf(x))) for x in XK])


def test_values_and_traced():
    for v in NUS:
        K = _kref(v)
        rel = np.abs(np.asarray(chebax.besselk(v)(XK)) - K) / K
        assert rel.max() <= 5e-14, v
        relt = np.abs(np.asarray(chebax.besselk_fn(v, XK)) - K) / K
        assert relt.max() <= 5e-14, v


def test_dx():
    for v in [0.03, 1.0, 1.7, 5.5, 9.97]:
        Kp = np.array([float(mp.diff(lambda t: mp.besselk(mp.mpf(v), t), mp.mpf(x)))
                       for x in XK])
        g = np.asarray(jax.vmap(jax.grad(chebax.besselk(v)))(jnp.asarray(XK)))
        assert np.max(np.abs(g - Kp) / np.abs(Kp)) <= 5e-14, v


def test_dnu_and_traced_grad():
    for v in [0.5, 1.0, 1.7, 7.77]:
        Kn = np.array([float(mp.diff(lambda t: mp.besselk(t, mp.mpf(x)), mp.mpf(v)))
                       for x in XK])
        got = np.asarray(chebax.besselk_dnu(v)(XK))
        assert np.max(np.abs(got - Kn) / np.abs(Kn)) <= 1e-12, v
        gn = np.asarray(jax.vmap(jax.grad(chebax.besselk_fn, argnums=0),
                                 in_axes=(None, 0))(jnp.asarray(v), jnp.asarray(XK)))
        assert np.max(np.abs(gn - Kn) / np.abs(Kn)) <= 1e-12, v


def test_dnu_vanishes_at_zero_order():
    # K_{-v} = K_v, so dK/dv = 0 at v = 0
    xs = XK[:20]
    d0 = np.abs(np.asarray(chebax.besselk_dnu(0.0)(xs)))
    K0 = np.asarray(chebax.besselk(0.0)(xs))
    assert np.max(d0 / K0) <= 1e-13


def test_clamp_below_xmin():
    kv = chebax.besselk(1.5)
    assert float(kv(1e-8)) == float(kv(kt.XMIN))


def test_underflow_graceful():
    y = float(chebax.besselk(2.5)(800.0))
    assert y == 0.0


def test_nonfinite_x():
    # K(inf) = 0, ln K(inf) = -inf, nan propagates on every path. dK/dx was
    # nan at the top of the range because the tail prefactor formed 2*x
    # first (inf), then 0 * inf in the jvp.
    for f in [chebax.besselk(1.5), chebax.besselk_dnu(1.5),
              lambda t: chebax.besselk_fn(1.5, t)]:
        assert float(f(np.inf)) == 0.0
        assert np.isnan(float(f(np.nan)))
    assert float(chebax.log_besselk_fn(1.5, np.inf)) == -np.inf
    assert np.isnan(float(chebax.log_besselk_fn(1.5, np.nan)))
    assert np.isnan(float(jax.grad(chebax.log_besselk_fn, 1)(1.5, np.nan)))
    assert float(jax.grad(chebax.besselk(1.5))(1e308)) == 0.0
    assert float(jax.grad(chebax.besselk_fn, 1)(1.5, 1e308)) == 0.0


def test_jit_and_pytree():
    kv = chebax.besselk(2.5)
    xs = jnp.asarray(XK[:10])
    y = jax.jit(lambda f, x: f(x))(kv, xs)
    np.testing.assert_allclose(np.asarray(y), np.asarray(kv(xs)), rtol=1e-15)


def test_order_out_of_range():
    with pytest.raises(ValueError, match="table covers"):
        chebax.besselk(10.5)


def test_f32_moderate_domain():
    xs = np.logspace(-2, np.log10(30.0), 24)
    K = np.array([float(mp.besselk(mp.mpf(2.5), mp.mpf(x))) for x in xs])
    y = chebax.besselk(2.5).astype(jnp.float32)(jnp.asarray(xs, jnp.float32))
    assert y.dtype == jnp.float32
    assert np.max(np.abs(np.asarray(y) - K) / K) <= 2e-5


def test_matern_learns_nu():
    # gradient recovery of the Matern smoothness, against mpmath targets
    # (an independent oracle, not chebax's own values)
    r = np.linspace(0.05, 3.0, 25)
    nu_t, ell_t, sig2_t = 1.7, 0.9, 1.3
    z = np.sqrt(2 * nu_t) * r / ell_t
    target = jnp.asarray([
        sig2_t * float(2 ** (1 - nu_t) / mp.gamma(nu_t)
                       * mp.mpf(zz) ** nu_t * mp.besselk(nu_t, mp.mpf(zz)))
        for zz in z])
    rj = jnp.asarray(r)

    def matern(rr, nu, ell, sig2):
        zz = jnp.sqrt(2.0 * nu) * rr / ell
        log_c = (1.0 - nu) * jnp.log(2.0) - jax.scipy.special.gammaln(nu)
        return sig2 * jnp.exp(log_c + nu * jnp.log(zz)) * chebax.besselk_fn(nu, zz)

    def loss(theta):
        k = matern(rj, theta[0], jnp.exp(theta[1]), jnp.exp(theta[2]))
        return jnp.mean((jnp.log(k) - jnp.log(target)) ** 2)

    grad = jax.jit(jax.grad(loss))
    theta = jnp.array([1.0, 0.0, 0.0])
    m = v = jnp.zeros_like(theta)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for i in range(600):
        g = grad(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1 ** (i + 1))) / (
            jnp.sqrt(v / (1 - b2 ** (i + 1))) + eps)
    assert abs(float(theta[0]) - nu_t) <= 1e-3
    assert abs(float(jnp.exp(theta[1])) - ell_t) <= 1e-3
    assert abs(float(jnp.exp(theta[2])) - sig2_t) <= 1e-3


def test_log_besselk():
    # ln K straight from the log tables: no underflow ceiling, so the grid
    # extends far past besselk's x ~ 746 limit. Errors normalized by
    # max(1, |ref|); measured worst over this grid: value 1.4e-15,
    # d/dnu 2.9e-15, d/dx 1.5e-15. Bars ~4x: 1e-14 / 2e-14 / 1e-14.
    xs = [1e-6, 0.01, 1.0, 7.9, 8.0, 8.1, 50.0, 600.0, 746.0, 1e4, 1e8]
    for v in [0.03, 0.5, 1.0, 1.7, 5.5, 9.97]:
        for x in xs:
            ref = float(mp.log(mp.besselk(mp.mpf(v), mp.mpf(x))))
            got = float(chebax.log_besselk_fn(v, x))
            assert abs(got - ref) / max(1.0, abs(ref)) <= 1e-14, (v, x)
    for v in [0.5, 1.7, 9.97]:
        for x in [0.01, 8.0, 600.0, 1e6]:
            dn = float(mp.diff(lambda t: mp.log(mp.besselk(t, mp.mpf(x))), mp.mpf(v)))
            gn = float(jax.grad(chebax.log_besselk_fn, 0)(v, x))
            assert abs(gn - dn) / max(1.0, abs(dn)) <= 2e-14, (v, x)
            dx = float(mp.diff(lambda t: mp.log(mp.besselk(mp.mpf(v), t)), mp.mpf(x)))
            gx = float(jax.grad(chebax.log_besselk_fn, 1)(v, x))
            assert abs(gx - dx) / max(1.0, abs(dx)) <= 1e-14, (v, x)


def test_log_besselk_huge_x():
    # up to the largest finite double, the range where forming 2*x in the
    # tail prefactor used to overflow and return -inf. Errors relative to
    # |ref| (which is ~ x here, never near zero); measured worst over this
    # grid: value 0 (bit exact, the -x term swamps the rest), d/dx 2.2e-16.
    # Bars 1e-15, about 4x the d/dx worst.
    xs = jnp.asarray([1e3, 1e6, 1e20, 1e100, 1e200, 1e300, 1e308,
                      float(np.finfo(np.float64).max)])
    xl = [float(x) for x in xs]
    for v in [0.0, 0.5, 1.7, 9.97]:
        ref = np.array([float(mp.log(mp.besselk(mp.mpf(v), mp.mpf(x)))) for x in xl])
        got = np.asarray(chebax.log_besselk_fn(v, xs))
        assert np.all(np.isfinite(got)), v
        assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-15, v
        # d/dx ln K = -(K_{v-1} + K_{v+1}) / (2 K_v), which tends to -1
        dref = np.array([
            float(-(mp.besselk(mp.mpf(v) - 1, mp.mpf(x))
                    + mp.besselk(mp.mpf(v) + 1, mp.mpf(x)))
                  / (2 * mp.besselk(mp.mpf(v), mp.mpf(x)))) for x in xl])
        g = np.asarray(jax.vmap(jax.grad(chebax.log_besselk_fn, 1),
                                in_axes=(None, 0))(jnp.asarray(v), xs))
        assert np.max(np.abs(g - dref) / np.abs(dref)) <= 1e-15, v


def _gig_log_normalizer(p, a, b):
    # A(p, a, b) = ln 2 + (p/2) ln(b/a) + ln K_p(sqrt(ab)); the GIG
    # exponential-family log-normalizer (see examples/). K_{-p} = K_p.
    z = jnp.sqrt(a * b)
    return (jnp.log(2.0) + 0.5 * p * (jnp.log(b) - jnp.log(a))
            + chebax.log_besselk_fn(jnp.abs(p), z))


def test_gig_log_normalizer_grads():
    # value and all three parameter gradients vs mpmath (mp.diff), errors
    # normalized by max(1, |ref|). Measured worst over this grid: A 2.3e-15,
    # dA/dp 8.6e-16, dA/da 4.9e-16, dA/db 3.1e-14 (the small-z corner, where
    # E[1/x] is large). Bars ~4x: value 1e-14, gradients 2e-13.
    def a_mp(p, a, b):
        return (mp.log(2) + p / 2 * (mp.log(b) - mp.log(a))
                + mp.log(mp.besselk(p, mp.sqrt(a * b))))

    grad = jax.grad(_gig_log_normalizer, (0, 1, 2))
    pts = [(2.5, 1.5, 3.0), (0.3, 4.0, 0.5), (7.5, 2.0, 2.0),
           (2.5, 0.01, 0.01), (2.5, 50.0, 40.0), (9.5, 1.0, 1.0),
           (1.0, 1.0, 1.0), (0.05, 2.0, 2.0)]
    for p, a, b in pts:
        ref = a_mp(mp.mpf(p), mp.mpf(a), mp.mpf(b))
        refs = [mp.diff(lambda t: a_mp(t, mp.mpf(a), mp.mpf(b)), mp.mpf(p)),
                mp.diff(lambda t: a_mp(mp.mpf(p), t, mp.mpf(b)), mp.mpf(a)),
                mp.diff(lambda t: a_mp(mp.mpf(p), mp.mpf(a), t), mp.mpf(b))]
        val = float(_gig_log_normalizer(p, a, b))
        assert abs(val - float(ref)) / max(1.0, abs(float(ref))) <= 1e-14, (p, a, b)
        for gi, ri in zip(grad(p, a, b), refs):
            assert abs(float(gi) - float(ri)) / max(1.0, abs(float(ri))) <= 2e-13, (p, a, b)


def test_gig_exp_to_nat_newton():
    # the mean-to-natural conversion, which needs d ln K/dp inside Newton:
    # recover (p, a, b) from exact mean statistics (E[ln x], E[x], E[1/x]) by
    # Newton in (p, ln a, ln b), Jacobian via jax.jacfwd through besselk_fn
    def a_nat(eta):
        return _gig_log_normalizer(eta[0] + 1.0, -2.0 * eta[1], -2.0 * eta[2])

    mean_stats = jax.jit(jax.grad(a_nat))
    p_t, a_t, b_t = 2.5, 1.5, 3.0
    mu = mean_stats(jnp.array([p_t - 1.0, -0.5 * a_t, -0.5 * b_t]))

    def residual(theta):
        p, la, lb = theta
        eta = jnp.array([p - 1.0, -0.5 * jnp.exp(la), -0.5 * jnp.exp(lb)])
        return mean_stats(eta) - mu

    jac = jax.jit(jax.jacfwd(residual))
    theta = jnp.array([1.0, 0.0, 0.0])
    for _ in range(20):
        r = residual(theta)
        if float(jnp.linalg.norm(r)) < 1e-12:
            break
        step = jnp.linalg.solve(jac(theta), r)
        t, rn = 1.0, float(jnp.linalg.norm(r))
        while t > 1e-8:
            r1 = residual(theta - t * step)
            if bool(jnp.all(jnp.isfinite(r1))) and float(jnp.linalg.norm(r1)) < rn:
                break
            t *= 0.5
        theta = theta - t * step
    assert float(jnp.linalg.norm(residual(theta))) <= 1e-10
    assert abs(float(theta[0]) - p_t) <= 1e-8
    assert abs(float(jnp.exp(theta[1])) - a_t) <= 1e-8
    assert abs(float(jnp.exp(theta[2])) - b_t) <= 1e-8


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit(tmp_path):
    # full-file comparison: coefficients, META (generator hash, mpmath
    # version, dps) and the header must reproduce byte for byte
    import pathlib
    besselk_gen.main(tmp_path)
    assert (tmp_path / "besselk_table.py").read_text() == pathlib.Path(kt.__file__).read_text()


def test_scaled_form_survives_large_arguments():
    # exp(log K + x) cancels its -x against +x and takes the -1/2 ln x
    # with it: pytensor's Kve returned 1.0 at x = 1e20 for a true 1.25e-10
    # (review, 2026-08-02). The tail table already holds e^x K, so the
    # scaled form is the one without the e^-x factor rather than a product
    # that puts it back. Reference is mpmath below x = 500 and the
    # asymptotic series above; measured worst 2.1e-15 relative, bar 1e-14.
    def ref_scaled(v, x):
        v, x = mp.mpf(v), mp.mpf(x)
        if x < 500:
            return mp.besselk(v, x) * mp.e ** x
        s, t = mp.mpf(1), mp.mpf(1)
        for k in range(1, 30):
            t *= (4 * v * v - (2 * k - 1) ** 2) / (8 * k * x)
            s += t
            if abs(t) < mp.mpf(10) ** -35:
                break
        return mp.sqrt(mp.pi / (2 * x)) * s

    worst = 0.0
    for v, x in ((2.5, 1e20), (2.5, 5.0), (0.0, 1e6), (3.0, 800.0),
                 (2.5, 1e-3), (1.0, 1e300), (10.0, 1e4)):
        got = float(chebax.besselk_fn(v, x, scaled=True))
        worst = max(worst, abs(got / float(ref_scaled(v, x)) - 1))
    assert worst <= 1e-14, worst
    # and it is exactly e^x K where K itself is representable
    for v, x in ((2.5, 5.0), (0.0, 3.0)):
        k = float(chebax.besselk_fn(v, x))
        assert abs(float(chebax.besselk_fn(v, x, scaled=True))
                   / (k * np.exp(x)) - 1) <= 1e-14
