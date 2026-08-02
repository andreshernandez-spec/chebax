#!/usr/bin/env python3
"""Race gammaincinv's two Newton residuals: chebax tables vs jax's gammainc.

WHAT THIS MEASURES
------------------
gammaincinv solves P(a, e^v) = p with 40 fixed safeguarded-Newton
iterations. The residual CDF used to be jax.scipy.special.gammainc,
whose XLA lowering re-runs a whole-array while_loop EVERY Newton
iteration; since 2026-07-31 the residual for a inside [0.1, 10] is the
chebax gammainc recipe (fixed-degree polynomials, reconstructed once per
call), with jax's gammainc kept as the out-of-box fallback behind one
lax.cond. Since 2026-08-01 a > 10 has a recipe residual too, the
Temme-zone tables ("large" below), so the fallback only serves a < 0.1.
This script times the paths through the same solver
(_gammaincinv_solve with the static mode), same arrays, GPU,
device-resident, interleaved reps, medians, f64, N = 2^20 (the betainc
race's size sweep showed ratios flat in N from 2^20 up), and checks the
two paths agree on the returned quantile.

WHAT IT DOES NOT MEASURE
------------------------
Accuracy contracts (tests/test_quantiles.py owns them and passes
unchanged on the recipe path); the full gradient path end to end (the
custom_jvp is evaluated once at the solution; its dP/da term is raced
in its own section below, since 2026-08-01 it too dispatches to the
recipe in-box); out-of-box a > 10 (no chebax side to race).

Run:  python experiments/09_gammaincinv_race.py   (~2 min, needs GPU)
"""

import time

import numpy as np

N = 1 << 20
REPS = 15


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

    from chebax._src.recipes.quantiles import _gammaincinv_solve

    dev = jax.devices()[0]
    print(f"jax {jax.__version__} on {dev}, N = {N}, reps = {REPS} (medians), "
          f"40 Newton iterations per solve")
    rng = np.random.default_rng(0)

    hdr = (f"{'case':26s} {'jax-resid':>10s} {'recipe':>9s} {'ratio':>6s}"
           f"  max rel diff (x)")
    print("\n" + hdr)
    print("-" * len(hdr))

    cases = [
        ("a=0.5,  p~U(1e-6,1-..)", 0.5, rng.uniform(1e-6, 1 - 1e-6, N)),
        ("a=3.5,  p~U(1e-6,1-..)", 3.5, rng.uniform(1e-6, 1 - 1e-6, N)),
        ("a=9.9,  p~U(1e-6,1-..)", 9.9, rng.uniform(1e-6, 1 - 1e-6, N)),
        ("a=3.5,  p~U(1e-12,1e-2)", 3.5, 10 ** rng.uniform(-12, -2, N)),
        ("a=50,   p~U(1e-6,1-..)", 50.0, rng.uniform(1e-6, 1 - 1e-6, N)),
        ("a=500,  p~U(1e-6,1-..)", 500.0, rng.uniform(1e-6, 1 - 1e-6, N)),
        ("a=500,  p~U(1e-12,1e-2)", 500.0, 10 ** rng.uniform(-12, -2, N)),
    ]
    for name, a, p_host in cases:
        mode = "small" if a <= 10.0 else "large"
        p = jax.device_put(jnp.asarray(p_host, jnp.float64), dev)
        av = jnp.asarray(a, jnp.float64)
        fj = jax.jit(lambda pp, aa=av: _gammaincinv_solve(aa, pp, None)[0])
        fc = jax.jit(lambda pp, aa=av, m=mode: _gammaincinv_solve(aa, pp, m)[0])
        yj, yc = [None], [None]

        def lj():
            yj[0] = fj(p)

        def lc():
            yc[0] = fc(p)

        t = bench({"jx": (lj, lambda: yj[0].block_until_ready()),
                   "cb": (lc, lambda: yc[0].block_until_ready())})
        xj, xc = jnp.exp(yj[0]), jnp.exp(yc[0])
        agree = float(jnp.max(jnp.abs(xj - xc) / jnp.maximum(xj, 1e-300)))
        print(f"{name:26s} {t['jx'] * 1e3:8.2f}ms {t['cb'] * 1e3:7.2f}ms "
              f"{t['jx'] / t['cb']:5.1f}x  {agree:.1e}")

    # the JVP's dP/da term: jax's looped igamma_grad_a series vs the
    # recipe's a-directional polynomial (the in-box dispatch added with
    # the solver rewire's follow-up)
    from chebax._src.recipes.quantiles import (_dPda, _dPda_recipe,
                                                _dPda_recipe_large)

    hdr2 = f"{'dP/da term':26s} {'igamma_grad_a':>13s} {'recipe':>9s} {'ratio':>6s}  max|diff|"
    print("\n" + hdr2)
    print("-" * len(hdr2))
    for a in (0.5, 3.5, 9.9, 50.0, 500.0):
        x_host = rng.gamma(a, 1.0, N)
        x = jax.device_put(jnp.asarray(x_host, jnp.float64), dev)
        av = jnp.asarray(a, jnp.float64)
        rec = _dPda_recipe if a <= 10.0 else _dPda_recipe_large
        fj = jax.jit(lambda xx, aa=av: _dPda(aa, xx))
        fc = jax.jit(lambda xx, aa=av, r=rec: r(aa, xx))
        yj, yc = [None], [None]

        def lj():
            yj[0] = fj(x)

        def lc():
            yc[0] = fc(x)

        t = bench({"jx": (lj, lambda: yj[0].block_until_ready()),
                   "cb": (lc, lambda: yc[0].block_until_ready())})
        agree = float(jnp.max(jnp.abs(yj[0] - yc[0])))
        print(f"a={a:5}, x~Gamma(a)         {t['jx'] * 1e3:11.2f}ms {t['cb'] * 1e3:7.2f}ms "
              f"{t['jx'] / t['cb']:5.1f}x  {agree:.1e}")

    print("\nBoth columns run the SAME solver; only the residual CDF differs."
          "\nchebax.gammaincinv end-to-end takes the recipe column for every"
          "\na >= 0.1 (one lax.cond, small tables to 10 and the Temme zone"
          "\nabove) and the jax column below that.")


if __name__ == "__main__":
    main()
