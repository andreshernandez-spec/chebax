#!/usr/bin/env python3
"""Tweedie log-likelihood: chebax against scipy's log_wright_bessel.

WHAT THIS MEASURES
------------------
The Tweedie log-density of N = 2^20 observations drawn from the law itself
(so the arguments are the ones a fit actually meets), float64, at three
powers. Two implementations:

- scipy: the same formula with scipy.special.log_wright_bessel(rho, 0, z)
  for ln W, vectorized numpy on one CPU core. This is the existing Python
  route; it has no gradients.
- chebax.tweedie_logpdf under jit, on CPU and on GPU, values and
  value_and_grad in (p, mu, phi), the shape a sampler needs.

Also: agreement between the two, and the number of while loops in chebax's
compiled program (0 means branchless: no data-dependent trip counts).

FINDINGS (2026-10-26, RTX 3080 Laptop, reproduced by this script)
-----------------------------------------------------------------
- Values: chebax 89-130x scipy on the same CPU (44-48 ms against
  4.3-5.7 s), 254-331x on the GPU (17 ms). value_and_grad in (p, mu, phi):
  254-276 ms CPU, 131-132 ms GPU; scipy has no gradient to race.
- No while loops in the compiled program.
- Agreement 3e-15 to 6e-15 at p = 1.5 and 1.8, but 2e-6 at p = 1.2, and
  there SCIPY is the one that is wrong: against mpmath, chebax is within
  6e-15 on the worst points while scipy's log_wright_bessel is off by 4.7e-6
  (y ~ 3.86, saddle scale Z ~ 18.4), and 14% of the p = 1.2 sample differs
  by more than 1e-12. scipy loses accuracy in a band around Z ~ 15-30 at
  intermediate rho (7e-8 at p = 1.2, Z = 20; 1e-11 at p = 1.3, Z = 20),
  apparently the region of scipy/xsf#291; elsewhere it is at 1e-16-1e-14.
- An earlier version of this script closed over the data inside the
  jitted gradient. jit then treats it as a constant, folds the y-only work
  at compile time and ignores its device, which printed 35 ms on "CPU" for
  what ran on the GPU. The data goes in as an argument now.

WHAT IT DOES NOT ESTABLISH
--------------------------
Anything about other hardware (RTX 3080 Laptop, consumer fp64 at 1:64), or
about scipy's other code paths: on this data its log_wright_bessel takes
whichever of its series / asymptotic / integral branches the arguments
select, element by element. Compile time is reported, not raced.

Run:  python experiments/21_tweedie_race.py   (~1 min)
"""

import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import chebax  # noqa: E402
from chebax.numpyro import Tweedie  # noqa: E402

N = 2 ** 20
CASES = ((1.2, 2.0, 1.0), (1.5, 2.0, 1.0), (1.8, 2.0, 1.0))   # (p, mu, phi)


def scipy_logpdf(p, mu, phi, y):
    from scipy.special import log_wright_bessel
    lam = mu ** (2 - p) / (phi * (2 - p))
    theta = phi * (p - 1) * mu ** (p - 1)
    rho = (2 - p) / (p - 1)
    out = np.full_like(y, -lam)
    pos = y > 0
    yp = y[pos]
    z = np.exp(np.log(lam) + rho * (np.log(yp) - np.log(theta)))
    out[pos] = -lam - yp / theta - np.log(yp) + log_wright_bessel(rho, 0.0, z)
    return out


def timed(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def main():
    devices = {"cpu": jax.devices("cpu")[0]}
    try:
        devices["gpu"] = jax.devices("gpu")[0]
    except RuntimeError:
        pass
    print(f"N = {N}, float64, devices: {', '.join(f'{k} ({v.device_kind})' for k, v in devices.items())}")
    for p, mu, phi in CASES:
        y = np.asarray(Tweedie(mu, phi, p).sample(jax.random.PRNGKey(0), (N,)), dtype=float)
        ref = scipy_logpdf(p, mu, phi, y)
        t_sp = timed(lambda: scipy_logpdf(p, mu, phi, y), reps=3)
        row = [f"p={p}: zeros {np.mean(y == 0):.3f}", f"scipy cpu {t_sp:8.1f} ms"]
        for name, dev in devices.items():
            yd = jax.device_put(jnp.asarray(y), dev)
            f = jax.jit(lambda pp, m, s, yy: chebax.tweedie_logpdf(pp, m, s, yy))
            t0 = time.perf_counter()
            got = np.asarray(f(p, mu, phi, yd).block_until_ready())
            t_compile = time.perf_counter() - t0
            agree = np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))
            tv = timed(lambda: f(p, mu, phi, yd).block_until_ready())
            # the data goes in as an argument: a closed-over array is a
            # constant, and jit would not follow its device
            vg = jax.jit(jax.value_and_grad(
                lambda pp, m, s, yy: jnp.sum(chebax.tweedie_logpdf(pp, m, s, yy)),
                argnums=(0, 1, 2)))
            tg = timed(lambda: jax.block_until_ready(vg(p, mu, phi, yd)))
            hlo = f.lower(p, mu, phi, yd).compile().as_text()
            row.append(f"chebax {name} {tv:7.2f} ms ({t_sp / tv:5.1f}x), +grad {tg:7.2f} ms, "
                       f"agree {agree:.1e}, while {hlo.count(' while(')}, compile {t_compile:.1f} s")
        print("  " + " | ".join(row))


if __name__ == "__main__":
    main()
