#!/usr/bin/env python3
"""Race XLA's iterative igamma against a mock fixed-degree kernel; count its
iterations.

WHAT THIS MEASURES
------------------
The queued PROJECT.md gammainc benchmark, part (1) of two: size the prize
before building the recipe. XLA lowers jax.scipy.special.gammainc to two
while loops (Cephes power series for x < a+1, continued fraction beyond),
re-executing the WHOLE array each trip until the worst element converges,
and the loops are fusion barriers. The competitor here is a MOCK with the
op profile of a betainc-runtime-class fixed-degree kernel: exp + log
prefactors plus one degree-27 Clenshaw on a mapped argument. Its VALUES ARE
MEANINGLESS; only its cost is representative (same primitive mix as
chebax's betainc eval). Timed on device-resident arrays, interleaved reps,
medians, f64 and f32.

Iteration statistics come from a host-side numpy transcription of the two
Cephes loops (the same technique as bessel experiments/01): per-element
trip counts, whose MAX is what every element pays inside XLA's lockstep
while loop.

WHAT IT DOES NOT MEASURE
------------------------
Accuracy of anything (the mock is a cost model, not an implementation);
whether a real Kummer/Temme recipe reaches gammainc accuracy at degree ~27
(that is part (2), gated on this prize being real); gradient-path costs
(jax's igamma_grad_a adds a third looped series; noted, not timed).

Run:  python experiments/05_igamma_race.py   (~1 min, needs GPU)
"""

import time

import numpy as np

N = 1 << 24
REPS = 25
DEG = 27  # betainc-runtime class: its x-direction table is degree 23; round up


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


def cephes_trip_counts(a, x, max_iter=2000):
    """Per-element iteration counts of the two Cephes igamma loops (host)."""
    eps = np.finfo(np.float64).eps
    n_series = np.zeros(a.shape, dtype=int)
    n_cf = np.zeros(a.shape, dtype=int)
    lower = x < a + 1.0

    al, xl = a[lower], x[lower]
    r, c, ans = al.copy(), np.ones_like(al), np.ones_like(al)
    active = np.ones(al.shape, bool)
    for i in range(max_iter):
        r = np.where(active, r + 1, r)
        c = np.where(active, c * xl / r, c)
        ans = np.where(active, ans + c, ans)
        active &= c / ans > eps
        n_series[np.flatnonzero(lower)[active]] = i + 1
        if not active.any():
            break

    au, xu = a[~lower], x[~lower]
    if au.size:
        big = 4.503599627370496e15
        biginv = 2.22044604925031308085e-16
        y = 1.0 - au
        z = xu + y + 1.0
        c = np.zeros_like(au)
        pkm2, qkm2 = np.ones_like(au), xu.copy()
        pkm1, qkm1 = xu + 1.0, z * xu
        ans = pkm1 / qkm1
        active = np.ones(au.shape, bool)
        for i in range(max_iter):
            c = c + 1.0
            y = y + 1.0
            z = z + 2.0
            yc = y * c
            pk = pkm1 * z - pkm2 * yc
            qk = qkm1 * z - qkm2 * yc
            with np.errstate(divide="ignore", invalid="ignore"):
                r = pk / qk
                t = np.where(qk != 0, np.abs((ans - r) / r), 1.0)
            ans = np.where(qk != 0, r, ans)
            pkm2, pkm1 = pkm1, pk
            qkm2, qkm1 = qkm1, qk
            fix = np.abs(pk) > big
            pkm2 = np.where(fix, pkm2 * biginv, pkm2)
            pkm1 = np.where(fix, pkm1 * biginv, pkm1)
            qkm2 = np.where(fix, qkm2 * biginv, qkm2)
            qkm1 = np.where(fix, qkm1 * biginv, qkm1)
            newly_done = active & (t <= eps)
            n_cf[np.flatnonzero(~lower)[newly_done]] = i + 1
            active &= t > eps
            if not active.any():
                break
    return n_series[lower], n_cf[~lower]


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    print(f"jax {jax.__version__} on {jax.devices()[0]}, N = {N}, mock degree {DEG}")
    rng = np.random.default_rng(0)
    coefs = rng.standard_normal(DEG + 1) * (0.5 ** np.arange(DEG + 1))

    def mock(a, x):
        # betainc-runtime op profile: prefactor exps/logs + one Clenshaw
        t = 2.0 * x / (x + a) - 1.0
        b1 = jnp.zeros_like(t)
        b2 = jnp.zeros_like(t)
        for ck in coefs[:0:-1]:
            b1, b2 = 2 * t * b1 - b2 + ck, b1
        ser = t * b1 - b2 + coefs[0]
        return jnp.exp(a * jnp.log(x) - x - jax.scipy.special.gammaln(a) + ser)

    dists = [
        ("a=3.5,  x~U(0,20)", 3.5, lambda r: r.uniform(1e-6, 20.0, N)),
        ("a=3.5,  x~Gamma(3.5)", 3.5, lambda r: r.gamma(3.5, 1.0, N)),
        ("a=20,   x~Gamma(20)", 20.0, lambda r: r.gamma(20.0, 1.0, N)),
        ("a=100,  x~Gamma(100)", 100.0, lambda r: r.gamma(100.0, 1.0, N)),
        ("a=0.5,  x~Gamma(0.5)", 0.5, lambda r: r.gamma(0.5, 1.0, N)),
    ]

    hdr = (f"{'distribution':22s} {'igamma64':>9s} {'mock64':>8s} {'ratio':>6s}"
           f" {'igamma32':>9s} {'mock32':>8s} {'ratio':>6s}"
           f"  iter med/p99/max (series | CF)")
    print(hdr)
    print("-" * len(hdr))

    for name, a, sampler in dists:
        x_host = sampler(rng)
        row = [f"{name:22s}"]
        for dt in (jnp.float64, jnp.float32):
            x = jnp.asarray(x_host, dt)
            av = jnp.asarray(a, dt)
            ig = jax.jit(lambda xx, aa=av: jax.scipy.special.gammainc(aa, xx))
            mk = jax.jit(lambda xx, aa=av: mock(aa, xx))
            yi = [None]
            ym = [None]

            def li():
                yi[0] = ig(x)

            def lm():
                ym[0] = mk(x)

            t = bench({"ig": (li, lambda: yi[0].block_until_ready()),
                       "mk": (lm, lambda: ym[0].block_until_ready())})
            row.append(f"{t['ig'] * 1e3:7.2f}ms {t['mk'] * 1e3:6.2f}ms "
                       f"{t['ig'] / t['mk']:5.1f}x")
        sub = np.random.default_rng(1).choice(x_host, 200_000, replace=False)
        ns, nc = cephes_trip_counts(np.full(sub.shape, a), sub)
        stat = lambda v: (f"{np.median(v):.0f}/{np.percentile(v, 99):.0f}/{v.max():.0f}"
                          if v.size else "-")
        row.append(f"  {stat(ns)} | {stat(nc)}")
        print(" ".join(row))

    print("\nThe mock's values are meaningless; its COST is the betainc-class op"
          "\nprofile. XLA's while loops run the whole array to the worst element's"
          "\ntrip count; the max column is what every lane pays.")


if __name__ == "__main__":
    main()
