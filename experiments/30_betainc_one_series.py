#!/usr/bin/env python3
"""Experiment 30: betainc's kernel with one series per point.

WHAT IS MEASURED
----------------
Below x = 1/2 betainc evaluates the direct orientation's series, above it
the reflected one's, and the two-series kernel ran both for every element
and selected. The kernel now runs one Clenshaw per point on per-point
coefficients (series.chebval_either), for the value and for each
derivative the custom_jvp rule needs, and one exp and two logs per point
instead of two and four. Values are the same ops per lane, so they agree
with the two-series form eagerly bit for bit (jit differs at the ulp
level either way, from XLA's fusion). This times betainc_scalar,
log_betainc_scalar and betaincinv (whose solve evaluates the kernel) for
values, grad in x (or p), grad in (a, b) and value_and_grad, the
one-series kernel against the two-series one (copied below from main
aaa6d46 and run through the same panel switch and rules), per (a, b)
across the panels, on the GPU at N = 2^20 and one CPU thread at N = 2^16,
float64, medians of interleaved reps in alternating order.

WHAT IT MEANS
-------------
two / one is the speedup; the panels with b > 10 have 76-row series where
a, b <= 10 have 24, so the per-point form helps most where the series
dominate the element's cost.

WHAT THIS DOES NOT ESTABLISH
----------------------------
Accuracy (tests/test_betainc.py, test_quantiles.py against mp.diff);
stdtr, whose kernel is separate and still runs both orientations.

Run:  python experiments/30_betainc_one_series.py   (~15 min; CPU rows
      pinned with taskset)
"""

import functools
import time

import numpy as np

PAIRS = ((2.5, 3.5), (0.5, 0.5), (9.5, 0.2), (30.0, 70.0), (0.3, 50.0))
REPS = 9


def bench(fns, warmup=1, reps=REPS):
    for launch, sync in fns.values():
        for _ in range(warmup):
            sync(launch())
    out = {k: [] for k in fns}
    for rep in range(reps):
        for k in list(fns) if rep % 2 == 0 else reversed(list(fns)):
            launch, sync = fns[k]
            t0 = time.perf_counter()
            sync(launch())
            out[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in out.items()}


def two_series_kernel():
    """The two-series kernel as of main aaa6d46 (entry 42), verbatim."""
    import jax
    import jax.numpy as jnp

    from chebax._src.recipes._common import canon_tag, newton_bisect, solver_limits
    from chebax._src.recipes.betainc_kernel import (
        XSPLIT,
        _finish,
        _split,
        ln_beta,
        log_direct,
    )

    def _log_direct_parts(a, b, x, lser, lser_1, lser_2, need):
        psi = jax.scipy.special.digamma
        n1, n2, nx = need
        psi_ab = psi(a + b) if n1 or n2 else None
        L = log_direct(a, b, x, lser)
        L1 = jnp.log(x) - 1.0 / a - psi(a) + psi_ab + lser_1(x) if n1 else None
        L2 = jnp.log1p(-x) - psi(b) + psi_ab + lser_2(x) if n2 else None
        Lx = a / x - b / (1.0 - x) + lser.deriv()(x) if nx else None
        return L, L1, L2, Lx

    def _both_parts(a, b, x, cab, cba, dcab, dcba, need):
        na, nb, nx = need
        lower, xd, yd = _split(x)
        direct = _log_direct_parts(a, b, xd, cab, *dcab, (na, nb, nx))
        R, Rb, Ra, Ry = _log_direct_parts(b, a, yd, cba, *dcba, (nb, na, nx))
        return lower, direct, (R, Ra, Rb, Ry)

    def _combine(x, a, b, lower, lo, hi, pieces):
        out = []
        for i, (d, r) in enumerate(pieces):
            if d is None:
                out.append(None)
                continue
            v = jnp.where(lower, d, r)
            out.append(
                _finish(x, a, b, v, lo, hi) if i == 0 else _finish(x, a, b, v, 0.0, 0.0)
            )
        return tuple(out)

    def _times(u, v):
        return None if u is None else u * v

    def eval_betainc_parts(a, b, x, cab, cba, dcab, dcba, need):
        x = jnp.asarray(x)
        lower, (L, La, Lb, Lx), (R, Ra, Rb, Ry) = _both_parts(
            a, b, x, cab, cba, dcab, dcba, need
        )
        d, r = jnp.exp(L), jnp.exp(R)
        pieces = (
            (d, 1.0 - r),
            (_times(La, d), _times(Ra, -r)),
            (_times(Lb, d), _times(Rb, -r)),
            (_times(Lx, d), _times(Ry, r)),
        )
        return _combine(x, a, b, lower, 0.0, 1.0, pieces)

    def eval_log_betainc_parts(a, b, x, cab, cba, dcab, dcba, need):
        x = jnp.asarray(x)
        lower, (L, La, Lb, Lx), (R, Ra, Rb, Ry) = _both_parts(
            a, b, x, cab, cba, dcab, dcba, need
        )
        r = jnp.exp(R)
        s = -r / (1.0 - r)
        pieces = (
            (L, jnp.log1p(-r)),
            (La, _times(Ra, s)),
            (Lb, _times(Rb, s)),
            (Lx, _times(Ry, -s)),
        )
        return _combine(x, a, b, lower, -jnp.inf, 0.0, pieces)

    def eval_betainc(a, b, x, cab, cba):
        x = jnp.asarray(x)
        interior = (x > 0.0) & (x < 1.0)
        xi = jnp.where(interior, x, 0.25)
        xd = jnp.where(xi <= XSPLIT, xi, 0.25)
        yd = jnp.where(xi <= XSPLIT, 0.25, 1.0 - xi)
        direct = jnp.exp(log_direct(a, b, xd, cab))
        reflected = 1.0 - jnp.exp(log_direct(b, a, yd, cba))
        core = jnp.where(xi <= XSPLIT, direct, reflected)
        out = jnp.where(x <= 0.0, 0.0, jnp.where(x >= 1.0, 1.0, core))
        return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)

    def eval_log_betainc(a, b, x, cab, cba):
        interior = (x > 0.0) & (x < 1.0)
        xi = jnp.where(interior, x, 0.25)
        xd = jnp.where(xi <= XSPLIT, xi, 0.25)
        yd = jnp.where(xi <= XSPLIT, 0.25, 1.0 - xi)
        direct = log_direct(a, b, xd, cab)
        refl = jnp.log1p(-jnp.exp(log_direct(b, a, yd, cba)))
        core = jnp.where(xi <= XSPLIT, direct, refl)
        out = jnp.where(x <= 0.0, -jnp.inf, jnp.where(x >= 1.0, 0.0, core))
        return jnp.where(jnp.isnan(x) | jnp.isnan(a) | jnp.isnan(b), jnp.nan, out)

    def betaincinv_core(a, b, p, cab, cba):
        lim = solver_limits(canon_tag())
        interior = (p > 0.0) & (p < 1.0)
        pc = jnp.where(interior, p, 0.5)
        ln_b = ln_beta(a, b)
        swap = pc > 0.5
        ps2 = jnp.where(swap, 1.0 - pc, pc)
        aa = jnp.where(swap, b, a)

        def f_and_df(u):
            x = jax.nn.sigmoid(u)
            f = (
                jnp.where(
                    swap,
                    eval_betainc(b, a, x, cba, cab),
                    eval_betainc(a, b, x, cab, cba),
                )
                - ps2
            )
            ls, l1s = -jax.nn.softplus(-u), -jax.nn.softplus(u)
            dfdu = jnp.exp(
                jnp.where(swap, b, a) * ls + jnp.where(swap, a, b) * l1s - ln_b
            )
            return f, dfdu

        u_generic = jnp.where(swap, jnp.log(b / a), jnp.log(a / b)) + jnp.zeros_like(pc)
        u_asym = (jnp.log(ps2) + jnp.log(aa) + ln_b) / aa
        u0 = jnp.clip(jnp.where(ps2 < 0.01, u_asym, u_generic), lim.lo, lim.hi)
        u = newton_bisect(
            f_and_df,
            u0,
            jnp.full_like(pc, lim.lo),
            jnp.full_like(pc, lim.hi),
            lim.n_beta,
        )
        f_fin, _ = f_and_df(u)
        f_lo, _ = f_and_df(jnp.full_like(u, lim.lo))
        at_floor = f_lo > 0.0
        x_low = jnp.where(at_floor, 0.0, jax.nn.sigmoid(u))
        bad = ~at_floor & (jnp.abs(f_fin) > lim.rtol * ps2)
        x = jnp.where(swap, 1.0 - x_low, x_low)
        x = jnp.where(bad, jnp.nan, x)
        x = jnp.where(p <= 0.0, 0.0, jnp.where(p >= 1.0, 1.0, x))
        oob = jnp.isnan(p) | (p < 0.0) | (p > 1.0) | jnp.isnan(a) | jnp.isnan(b)
        return jnp.where(oob, jnp.nan, x)

    return (
        eval_betainc,
        eval_log_betainc,
        eval_betainc_parts,
        eval_log_betainc_parts,
        betaincinv_core,
    )


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import chebax
    from chebax._src.recipes import betainc as bi
    from chebax._src.recipes.betainc_kernel import ln_beta

    ev2, lg2, evp2, lgp2, inv2 = two_series_kernel()
    sync = lambda y: jax.tree_util.tree_map(lambda a: a.block_until_ready(), y)

    @jax.custom_jvp
    def two_betainc_cj(a, b, x):
        return bi.with_panel_series(a, b, lambda cab, cba: ev2(a, b, x, cab, cba))

    @functools.partial(two_betainc_cj.defjvp, symbolic_zeros=True)
    def _(primals, tangents):
        return bi._parts_rule(evp2, primals, tangents)

    @jax.custom_jvp
    def two_log_cj(a, b, x):
        return bi.with_panel_series(a, b, lambda cab, cba: lg2(a, b, x, cab, cba))

    @functools.partial(two_log_cj.defjvp, symbolic_zeros=True)
    def _(primals, tangents):
        return bi._parts_rule(lgp2, primals, tangents)

    two_betainc = lambda a, b, x: bi.with_edge_slopes(
        a, b, x, two_betainc_cj(a, b, jnp.asarray(x))
    )
    two_log = lambda a, b, x: two_log_cj(a, b, jnp.asarray(x))

    @jax.custom_jvp
    def two_inv(a, b, p):
        return bi.with_panel_series(a, b, lambda cab, cba: inv2(a, b, p, cab, cba))

    @two_inv.defjvp
    def _(primals, tangents):
        a, b, p = primals
        da, db, dp = tangents
        x = two_inv(a, b, p)
        interior = (x > 0.0) & (x < 1.0)
        xs = jnp.where(interior, x, 0.5)
        pdf = jnp.exp((a - 1) * jnp.log(xs) + (b - 1) * jnp.log1p(-xs) - ln_beta(a, b))
        _, d_cdf = jax.jvp(lambda aa, bb: two_betainc(aa, bb, xs), (a, b), (da, db))
        dx = (dp - d_cdf) / pdf
        return x, jnp.where(jnp.isnan(x), jnp.nan, jnp.where(interior, dx, 0.0))

    def rows(f):
        s = lambda a, b, x: f(a, b, x).sum()
        return {
            "values": jax.jit(f),
            "grad x": jax.jit(jax.grad(s, argnums=2)),
            "grad a,b": jax.jit(jax.grad(s, argnums=(0, 1))),
            "v&g a,b,x": jax.jit(jax.value_and_grad(s, argnums=(0, 1, 2))),
        }

    fams = {
        "betainc_scalar": (chebax.betainc_scalar, two_betainc),
        "log_betainc_scalar": (chebax.log_betainc_scalar, two_log),
        "betaincinv": (chebax.betaincinv, two_inv),
    }
    devs = [d for d in jax.devices() if d.platform == "gpu"] + jax.devices("cpu")[:1]
    print(f"jax {jax.__version__}, devices {devs}")
    for dev in devs:
        n = 1 << (20 if dev.platform == "gpu" else 16)
        x = jax.device_put(jnp.asarray(np.random.default_rng(0).uniform(0, 1, n)), dev)
        print(
            f"\n{dev.platform}, N = {n}, float64, ms (median of {REPS}); one / two series, ratio"
        )
        for fam, (one, two) in fams.items():
            j1, j2 = rows(one), rows(two)
            print(f"  {fam}")
            print(f"  {'a, b':>10s} | " + " | ".join(f"{k:>24s}" for k in j1))
            for a, b in PAIRS:
                a_, b_ = jax.device_put(jnp.asarray(a), dev), jax.device_put(
                    jnp.asarray(b), dev
                )
                fns = {}
                for k in j1:
                    fns[k + " 1"] = (lambda f=j1[k]: f(a_, b_, x), sync)
                    fns[k + " 2"] = (lambda f=j2[k]: f(a_, b_, x), sync)
                t = bench(fns)
                cells = [
                    f"{t[k + ' 1'] * 1e3:8.2f} / {t[k + ' 2'] * 1e3:8.2f} {t[k + ' 2'] / t[k + ' 1']:4.1f}x"
                    for k in j1
                ]
                print(f"  {a:4g}, {b:4g} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
