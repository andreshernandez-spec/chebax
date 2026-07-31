#!/usr/bin/env python3
"""Race chebax's besselj against jax's own bessel_jn at integer order.

WHAT THIS MEASURES
------------------
The only Bessel J jax ships is jax.scipy.special.bessel_jn(z, v, n_iter):
INTEGER order, Miller backward recurrence with a static iteration count,
returning the whole ladder J_0..J_v. This race gives the direct
chebax-vs-jax number that the B3 study (../bessel/experiments/07, vs
cephes::jv) does not: same device, same arrays, f64 and f32, N = 2^24,
device-resident, interleaved reps, medians, max|jax - chebax| agreement
per row at f64.

n_iter is chosen PER DISTRIBUTION as the smallest of {50, 100, 200} that
is correct on that x-range (accuracy prefaces printed first). Three
structural findings shape the race and are part of the result:

- validity is a WINDOW that narrows from both sides as n_iter grows:
  n_iter=50 nans below x ~ 1e-4 (backward-recurrence overflow) and is
  wrong past x ~ 25; n_iter=100 nans below x ~ 0.1 and fails by x ~ 100.
  No single jitted call covers [1e-6, 100].
- peak memory scales with n_iter * N: at N = 2^24, n_iter=100 attempts a
  13 GiB allocation and OOMs a 16 GB card, so n_iter=100 rows run at
  N = 2^22 (the betainc race's size sweep showed ratios flat in N from
  2^20 up).
- cost grows with n_iter, so jax's time is range-dependent by
  construction; the fixed-degree kernel's is not.

WHAT IT DOES NOT MEASURE
------------------------
Non-integer order (bessel_jn cannot do it at all; that comparison has no
jax side); the ladder amortization (bessel_jn's one call yields all
orders 0..v, and the timing charges it to the single order raced -
callers who need the whole ladder get the rest free, callers who need
one order pay for the ladder); accuracy contracts (chebax's tests own
those; the preface and agreement column only guard the race).

Run:  python experiments/08_besselj_jax_race.py   (~2 min, needs GPU)
"""

import time

import numpy as np

N = 1 << 24
N_HIGH_ITER = 1 << 22   # n_iter=100 at 2^24 attempts a 13 GiB allocation
REPS = 25
PROBE_NITERS = (50, 100, 200)


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


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy import special as sps

    import chebax

    dev = jax.devices()[0]
    print(f"jax {jax.__version__} on {dev}, N = {N}, reps = {REPS} (medians)")
    rng = np.random.default_rng(0)

    # accuracy prefaces: where each n_iter is usable (abs err vs scipy jv;
    # scipy is a yardstick for picking n_iter, not an accuracy reference)
    probe_x = np.array([0.5, 5.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0])
    print(f"\nbessel_jn(v=2) abs err vs scipy at x = {[float(v) for v in probe_x]}:")
    for ni in PROBE_NITERS:
        got = np.asarray(jax.scipy.special.bessel_jn(
            jnp.asarray(probe_x), v=2, n_iter=ni))[2]
        with np.errstate(all="ignore"):
            err = np.abs(got - sps.jv(2, probe_x))
        print(f"  n_iter={ni:3d}: " + " ".join(f"{e:8.1e}" for e in err))
    small_x = np.array([1e-6, 1e-4, 1e-3, 0.01, 0.03, 0.1])
    print(f"bessel_jn(v=2) nan onset at small x = {[float(v) for v in small_x]}:")
    for ni in PROBE_NITERS:
        got = np.asarray(jax.scipy.special.bessel_jn(
            jnp.asarray(small_x), v=2, n_iter=ni))[2]
        print(f"  n_iter={ni:3d}: "
              + " ".join("nan" if np.isnan(g) else " ok" for g in got))
    print("  (a validity WINDOW narrowing from both sides: large n_iter"
          "\n   overflows the backward recurrence at small x, small n_iter"
          "\n   is wrong at large x; no single call covers [1e-6, 100])")

    def pick_niter(x_host, v):
        sub = np.random.default_rng(1).choice(x_host, 4096, replace=False)
        ref = sps.jv(v, sub)
        for ni in PROBE_NITERS:
            got = np.asarray(jax.scipy.special.bessel_jn(
                jnp.asarray(sub), v=v, n_iter=ni))[v]
            with np.errstate(all="ignore"):
                if np.nanmax(np.abs(got - ref)) < 1e-12 and not np.isnan(got).any():
                    return ni
        return None

    # lower bounds sit above the relevant nan onset so every raced call
    # is one jax runs correctly
    cases = [
        ("x~U(1e-3,20)", 2, lambda r, n: r.uniform(1e-3, 20.0, n)),
        ("x~U(1e-3,8) (inner)", 2, lambda r, n: r.uniform(1e-3, 8.0, n)),
        ("x~U(8,30)   (mid)", 2, lambda r, n: r.uniform(8.0, 30.0, n)),
        ("x~U(0.2,50) (wide)", 2, lambda r, n: r.uniform(0.2, 50.0, n)),
        ("x~U(1e-3,20)", 9, lambda r, n: r.uniform(1e-3, 20.0, n)),
    ]

    hdr = (f"{'distribution':20s} {'v':>2s} {'n_iter':>6s} {'N':>4s} {'jax64':>9s}"
           f" {'chebax64':>9s} {'ratio':>6s} {'jax32':>9s} {'chebax32':>9s}"
           f" {'ratio':>6s}  max|diff| (f64)")
    print("\n" + hdr)
    print("-" * len(hdr))

    for name, v, sampler in cases:
        ni = pick_niter(sampler(rng, 65536), v)
        if ni is None:
            print(f"{name:20s} {v:2d} {'-':>6s}  no probed n_iter is correct "
                  f"on this range")
            continue
        n = N if ni <= 50 else N_HIGH_ITER
        x_host = sampler(rng, n)
        row = [f"{name:20s} {v:2d} {ni:6d} 2^{int(np.log2(n)):2d}"]
        agree = None
        for dt in (jnp.float64, jnp.float32):
            x = jnp.asarray(x_host, dt)
            jj = jax.jit(lambda z, v=v, ni=ni:
                         jax.scipy.special.bessel_jn(z, v=v, n_iter=ni)[v])
            cb = jax.jit(chebax.besselj(float(v)))
            yj, yc = [None], [None]

            def lj():
                yj[0] = jj(x)

            def lc():
                yc[0] = cb(x)

            t = bench({"jj": (lj, lambda: yj[0].block_until_ready()),
                       "cb": (lc, lambda: yc[0].block_until_ready())})
            row.append(f"{t['jj'] * 1e3:7.2f}ms {t['cb'] * 1e3:7.2f}ms "
                       f"{t['jj'] / t['cb']:5.1f}x")
            if dt == jnp.float64:
                agree = float(jnp.max(jnp.abs(yj[0] - yc[0])))
        row.append(f" {agree:.1e}")
        print(" ".join(row))

    print("\nbessel_jn's one call yields the whole ladder J_0..J_v (charged"
          "\nhere to the single order raced); chebax serves one real order per"
          "\ntable instance, any order in [0, 10], no n_iter to choose.")


if __name__ == "__main__":
    main()
