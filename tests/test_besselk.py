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


@pytest.mark.slow
def test_tables_regenerate_bit_for_bit():
    lo, hi = besselk_gen.generate_inner_tables()
    tail = besselk_gen.generate_tail_table()
    assert np.array_equal(lo, kt.TABLE_IN_LO)
    assert np.array_equal(hi, kt.TABLE_IN_HI)
    assert np.array_equal(tail, kt.TABLE_TAIL)
    assert kt.META["dps"] == besselk_gen.DPS
