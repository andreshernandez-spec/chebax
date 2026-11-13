"""Rice log-CDF and log-SF: where the existing float64 implementations fail,
and chebax.log_marcum_p1/log_marcum_q1 on the same points.

The standard Rice distribution with shape a = nu/sigma has CDF P(a, x) =
1 - Q_1(a, x), Q_1 the Marcum Q-function. scipy.stats.rice computes the CDF
as chndtr(x^2, 2, a^2) and the SF as 1 - CDF; scipy.stats.ncx2 (x^2 with 2
degrees of freedom and noncentrality a^2) has Boost's CDF and complement.
TFP's NoncentralChi2 (JAX substrate) sums Abramowitz and Stegun 26.4.25 in a
while_loop, returns 0 where a Chernoff bound puts the CDF below machine
epsilon, and has no survival function of its own (1 - CDF). jax and pytensor
have none. Measured here against mpmath at 50 dps: ln P and ln Q from
rice.logcdf/logsf, ncx2.logcdf/logsf and TFP's log_cdf/log_survival_function
(float64, jit), on 3,000 points: a
log-uniform on [1e-2, 1e3], x = a + z with z uniform on [-min(a, 40), 40]
(Gaussian units, both tails), and 300 points with x log-uniform on
[1e-8, 1e-1] (the lower tail at small x). Metric |ln F - ref|/max(1, |ref|),
nonfinite counted apart.

Reference: with xi = a x, d = x - a, and on the side x >= a rho = a/x,

  Q = e^(-d^2/2) S_Q,  S_Q = e^(-xi) sum_{k>=0} rho^k I_k(xi),

and for x < a, rho = x/a, P = e^(-d^2/2) S_P, S_P = e^(-xi) sum_{k>=1}
rho^k I_k(xi): every term positive. The sums use Miller's backward
recurrence below xi = 400, and above it the same S as an integral,
v = sqrt(xi) 2 sin(theta/2) in the generating function of e^(-xi) I_k(xi):

  S_Q = (1/pi) int_0^(2/eps) e^(-v^2/2) (eta + rho eps v^2/2)/(eta^2 + rho v^2)
        (1 - eps^2 v^2/4)^(-1/2) dv,
  S_P/rho: the same with (eta - eps v^2/2) in the numerator,

eps = xi^(-1/2), eta = (1 - rho)/eps. The two agree to 6e-48 at xi = 26 to
900 and rho = 0.05 to 1, and match the closed form Q(a, a) = (1 + e^(-a^2)
I_0(a^2))/2. The other probability is 1 - this one at 50 dps.

It does NOT establish: derivatives, float32, or speed.

    PYTHONPATH=<tfp-nightly> python experiments/51_rice_logcdf.py      # ~4 min
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats as st  # noqa: E402
from tensorflow_probability.substrates import jax as tfp  # noqa: E402

import chebax  # noqa: E402

mp.mp.dps = 50


def _ie(xi, n):
    """e^(-xi) I_k(xi) for k = 0..n, Miller's recurrence normalized by
    e^xi = I_0 + 2 sum I_k (every k with e^(-k^2/(2 xi)) above 1e-60)."""
    top = int(n + 60 + 17 * mp.sqrt(xi))
    y = [mp.mpf(0)] * (top + 2)
    y[top] = mp.mpf(10) ** -30
    for k in range(top, 0, -1):
        y[k - 1] = y[k + 1] + 2 * k / xi * y[k]
        if y[k - 1] > mp.mpf(10) ** 30:
            y = [v * mp.mpf(10) ** -60 for v in y]
    norm = y[0] + 2 * mp.fsum(y[1:])
    return [v / norm for v in y[: n + 1]]


def _terms(xi, rho):
    k = 10
    while k * -mp.log(rho) + k * k / (2 * (xi + 1)) < 150:
        k += 10
    return k


def _v_integral(eps, eta, rho, c):
    """S_Q (c = rho) or S_P/rho (c = -1)."""

    def g(v):
        num = eta + c * eps * v * v / 2
        return (
            mp.exp(-v * v / 2)
            * num
            / (eta**2 + rho * v * v)
            / mp.sqrt(1 - (eps * v / 2) ** 2)
        )

    top = 2 / eps
    cut = min(top, mp.mpf(15))  # e^(-v^2/2) < 1e-48 past 15
    pts = {mp.mpf(c) for c in (eta / 8, eta / 2, eta, 2 * eta, 0.5, 1, 2, 4, 8)}
    return mp.quad(g, [0] + sorted(c for c in pts if 0 < c < cut) + [cut]) / mp.pi


def log_pq(a, x):
    """(ln P, ln Q) at 50 dps."""
    a, x = mp.mpf(a), mp.mpf(x)
    if a == 0:
        return mp.log(-mp.expm1(-x * x / 2)), -x * x / 2
    xi, d = a * x, x - a
    upper = x >= a
    rho = a / x if upper else x / a
    if xi <= 400:
        ie = _ie(xi, _terms(xi, rho) + 1)
        k0 = 0 if upper else 1
        s = mp.fsum(rho**k * v for k, v in enumerate(ie) if k >= k0)
    else:
        eps = 1 / mp.sqrt(xi)
        eta = (1 - rho) / eps
        s = _v_integral(eps, eta, rho, rho if upper else -1)
        s = s if upper else rho * s
    small = -d * d / 2 + mp.log(s)
    other = mp.log(-mp.expm1(small))
    return (other, small) if upper else (small, other)


def main():
    rng = np.random.default_rng(51)
    n = 3000
    a = 10 ** rng.uniform(-2, 3, n)
    x = a + rng.uniform(-np.minimum(a, 40), 40)
    a2 = 10 ** rng.uniform(-2, 3, 300)
    x2 = 10 ** rng.uniform(-8, -1, 300)
    a, x = np.concatenate([a, a2]), np.concatenate([x, x2])
    refs = np.array([[float(v) for v in log_pq(ai, xi)] for ai, xi in zip(a, x)])
    with np.errstate(all="ignore"):
        got = {
            "rice.logcdf": st.rice.logcdf(x, a),
            "rice.logsf": st.rice.logsf(x, a),
            "ncx2.logcdf": st.ncx2.logcdf(x * x, 2, a * a),
            "ncx2.logsf": st.ncx2.logsf(x * x, 2, a * a),
        }
        nc2 = tfp.distributions.NoncentralChi2(2.0, a * a)
        got["tfp.log_cdf"] = np.asarray(jax.jit(nc2.log_cdf)(x * x))
        got["tfp.log_sf"] = np.asarray(jax.jit(nc2.log_survival_function)(x * x))
        got["chebax p1"] = np.asarray(jax.jit(chebax.log_marcum_p1)(a, x))
        got["chebax q1"] = np.asarray(jax.jit(chebax.log_marcum_q1)(a, x))
    print(
        f"{len(a)} points; ln P from {refs[:, 0].min():.3g} to 0, ln Q from {refs[:, 1].min():.3g} to 0"
    )
    print(f"jax {jax.__version__}")
    print(
        f"{'':12} {'worst':>9} {'> 1e-12':>8} {'> 1e-6':>7} {'nonfinite':>10}   worst finite point"
    )
    for name, v in got.items():
        ref = refs[:, 0 if ("cdf" in name or "p1" in name) else 1]
        bad = ~np.isfinite(v)
        with np.errstate(all="ignore"):
            err = np.abs(v - ref) / np.maximum(1.0, np.abs(ref))
        ok = ~bad
        i = np.flatnonzero(ok)[np.argmax(err[ok])]
        print(
            f"{name:12} {err[ok].max():9.1e} {int((err[ok] > 1e-12).sum()):8} "
            f"{int((err[ok] > 1e-6).sum()):7} {int(bad.sum()):10}   "
            f"a={a[i]:.3g}, x={x[i]:.3g}, ref {ref[i]:.4g}, got {v[i]:.4g}"
        )
        # where each fails, by the reference's size
        for lo, hi in ((-1e-300, 0), (-30, -1e-300), (-700, -30), (-np.inf, -700)):
            sel = (ref > lo) & (ref <= hi) if np.isfinite(lo) else ref <= hi
            if sel.any():
                fail = bad[sel] | (err[sel] > 1e-6)
                print(
                    f"{'':14}ref in ({lo:g}, {hi:g}]: {int(fail.sum())} of {int(sel.sum())} off by > 1e-6 or nonfinite"
                )


if __name__ == "__main__":
    main()
