#!/usr/bin/env python3
"""Race chebax's betainc and stdtr against jax's, on GPU and one CPU point.

WHAT THIS MEASURES
------------------
The jax#28547 scenario: jax.scipy.special.betainc lowers to a Lentz
continued-fraction while_loop (whole-array re-execution to the worst
element's trip count, fusion barrier), and Student-t CDFs built on it
inherit the cost. The competitor is chebax's REAL betainc_fn (fixed-degree
tables, branchless) and stdtr, values and all, so each row carries an
agreement check (max |chebax - jax| over the row's arrays, f64). Timed on
device-resident arrays, interleaved reps, medians, f64 and f32, N = 2^24
per row, plus one CPU row at N = 500k (the reporter's CPU case size) and a
GPU size sweep at the base case. jax's stdtr is composed from its betainc
the way scipy composes it; chebax's is its own table path. The per-call
coefficient reconstruction of the traced chebax functions is inside the
timed region.

WHAT IT DOES NOT MEASURE
------------------------
Accuracy contracts (tests own those; the agreement column only guards
against racing wrong values); per-element (a, b) arrays, which chebax does
not serve (a, b are uniform per call here, the case the reporter's
benchmark exercises with scalar parameters); gradient paths (jax has no
betainc a,b gradient to race; chebax's is one more polynomial); stdtrit
(no jax counterpart to race). The stdtr agreement column reads ~1e-9, not
~1e-14: that error belongs to the composed-through-betainc side, which
cancels near x -> 1 (small |t|); chebax's stdtr is tested against mpmath
to 5e-14 absolute (tests/test_quantiles.py), which is why scipy also
ships stdtr as a dedicated routine.

Run:  python experiments/06_betainc_stdtr_race.py   (~2 min, needs GPU)
"""

import time

import numpy as np

N = 1 << 24
N_CPU = 500_000
REPS = 25


def bench(fns, warmup=3, reps=REPS):
    for launch, sync in fns.values():
        for _ in range(warmup):
            launch()
        sync()
    out = {k: [] for k in fns}
    for _ in range(reps):
        for k, (launch, sync) in fns.items():
            t0 = time.perf_counter()
            launch()
            sync()
            out[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in out.items()}


def race(jx_fn, cb_fn, x_host, device, dtypes, jnp, jax):
    row = []
    agree = None
    for dt in dtypes:
        x = jax.device_put(jnp.asarray(x_host, dt), device)
        jf = jax.jit(jx_fn)
        cf = jax.jit(cb_fn)
        yj, yc = [None], [None]

        def lj():
            yj[0] = jf(x)

        def lc():
            yc[0] = cf(x)

        t = bench({"jx": (lj, lambda: yj[0].block_until_ready()),
                   "cb": (lc, lambda: yc[0].block_until_ready())})
        row.append((t["jx"], t["cb"]))
        if dt == jnp.float64:
            agree = float(jnp.max(jnp.abs(yj[0] - yc[0])))
    return row, agree


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax

    gpu = jax.devices()[0]
    cpu = jax.devices("cpu")[0]
    print(f"jax {jax.__version__} on {gpu}, N = {N} (GPU rows), "
          f"N = {N_CPU} (CPU row), reps = {REPS} (medians)")
    rng = np.random.default_rng(0)

    def fmt(tj, tc):
        return f"{tj * 1e3:8.2f}ms {tc * 1e3:7.2f}ms {tj / tc:6.1f}x"

    # betainc rows: x ~ U(0,1) and x ~ Beta(a,b) (mass where the CF works)
    cases = [
        ("a=2.5 b=3.5, x~U", 2.5, 3.5, lambda r, n: r.uniform(1e-6, 1 - 1e-6, n)),
        ("a=2.5 b=3.5, x~Beta", 2.5, 3.5, lambda r, n: r.beta(2.5, 3.5, n)),
        ("a=0.5 b=0.5, x~Beta", 0.5, 0.5, lambda r, n: r.beta(0.5, 0.5, n)),
        ("a=9.5 b=0.2, x~Beta", 9.5, 0.2, lambda r, n: r.beta(9.5, 0.2, n)),
    ]
    hdr = (f"{'betainc':24s} {'jax64':>10s} {'chebax64':>9s} {'ratio':>6s}"
           f" {'jax32':>10s} {'chebax32':>9s} {'ratio':>6s}  max|diff| (f64)")
    print(hdr)
    print("-" * len(hdr))
    for name, a, b, sampler in cases:
        x_host = np.clip(sampler(rng, N), 1e-6, 1 - 1e-6)
        row, agree = race(
            lambda x, a=a, b=b: jax.scipy.special.betainc(a, b, x),
            lambda x, a=a, b=b: chebax.betainc_fn(a, b, x),
            x_host, gpu, (jnp.float64, jnp.float32), jnp, jax)
        print(f"{name:24s} {fmt(*row[0])} {fmt(*row[1])}  {agree:.1e}")

    # the reporter's CPU scenario: 500k elements on the host backend
    x_host = np.clip(rng.beta(2.5, 3.5, N_CPU), 1e-6, 1 - 1e-6)
    row, agree = race(
        lambda x: jax.scipy.special.betainc(2.5, 3.5, x),
        lambda x: chebax.betainc_fn(2.5, 3.5, x),
        x_host, cpu, (jnp.float64,), jnp, jax)
    print(f"{'a=2.5 b=3.5, CPU 500k':24s} {fmt(*row[0])} "
          f"{'':28s} {agree:.1e}")

    # stdtr: jax composed from its betainc the way scipy does
    def jax_stdtr(nu, t):
        x = nu / (nu + t * t)
        p = 0.5 * jax.scipy.special.betainc(nu / 2.0, 0.5, x)
        return jnp.where(t < 0, p, 1.0 - p)

    print()
    hdr = (f"{'stdtr':24s} {'jax64':>10s} {'chebax64':>9s} {'ratio':>6s}"
           f" {'jax32':>10s} {'chebax32':>9s} {'ratio':>6s}  max|diff| (f64)")
    print(hdr)
    print("-" * len(hdr))
    for name, nu, scale in [("nu=5.5, t~2N", 5.5, 2.0), ("nu=1.5, t~4N", 1.5, 4.0)]:
        t_host = rng.normal(0.0, scale, N)
        row, agree = race(
            lambda t, nu=nu: jax_stdtr(nu, t),
            lambda t, nu=nu: chebax.stdtr(nu, t),
            t_host, gpu, (jnp.float64, jnp.float32), jnp, jax)
        print(f"{name:24s} {fmt(*row[0])} {fmt(*row[1])}  {agree:.1e}")

    # size sweep at the base case, f64, GPU (the reporter swept 2^16..2^25)
    print(f"\nsize sweep, betainc a=2.5 b=3.5 x~Beta, f64, GPU "
          f"(jax ms / chebax ms / ratio)")
    for logn in (16, 20, 22, 24, 25):
        n = 1 << logn
        x_host = np.clip(rng.beta(2.5, 3.5, n), 1e-6, 1 - 1e-6)
        row, _ = race(
            lambda x: jax.scipy.special.betainc(2.5, 3.5, x),
            lambda x: chebax.betainc_fn(2.5, 3.5, x),
            x_host, gpu, (jnp.float64,), jnp, jax)
        tj, tc = row[0]
        print(f"  2^{logn}: {tj * 1e3:8.3f} / {tc * 1e3:7.3f} / {tj / tc:5.1f}x")


if __name__ == "__main__":
    main()
