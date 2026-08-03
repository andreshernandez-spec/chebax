#!/usr/bin/env python3
"""Registering pytensor's forward GammaInc/GammaIncC/BetaInc on chebax: what it
buys, and what the out-of-box fallback costs.

WHAT THIS MEASURES
------------------
`chebax.pytensor` lowers the inverse CDFs but not the forward pair, so every
censored or truncated Gamma, Poisson, ChiSquared or NegativeBinomial model on
PyMC's JAX path keeps jax's looped `igamma`. Registering the pair is ten lines.
BetaInc was half-registered in the same way: a custom_jvp supplied the a/b
gradients pytensor has never had, but the VALUES stayed jax's own on every
domain, so the betainc speedup experiments/06 measured reached no PyMC user at
all. The same cond puts the tables under the values.

The functions timed here are the shipped ones, imported from chebax.pytensor,
so these numbers cannot drift away from what the registration actually does.

The question this answers is not "is chebax faster" (experiments/05 and 06
measured that) but the two things that decide HOW to register it:

  (1) The fallback. These are plain forward functions a user may call at any
      shape, so answering nan below chebax's a >= 0.1 box is not defensible the
      way it is for an inverse CDF. The fallback must therefore be a `lax.cond`
      and not a `lax.select`: select evaluates both sides, which would put
      jax's while_loop back on every lane and hand back exactly what the
      registration removes. cond executes one branch. It is available here
      because the dispatch already requires a scalar (uniform-per-call) shape,
      so the predicate is scalar and no lane can disagree with another.

      What that costs is measured below, per shape, for values and for d/da.

  (2) Whether the loop actually leaves the compiled program. With a CONCRETE
      shape XLA folds the constant predicate and drops jax's branch entirely.
      With a TRACED shape, which is the case this change exists for (a sampled
      alpha in a hierarchical model), it cannot fold, so the loop stays in the
      module even though the conditional never executes it. Both counts are
      reported.

WHAT IT DOES NOT ESTABLISH
--------------------------
Not an end-to-end model number. `docs/pymc-gpu-directions.md` measured 2.4-2.9x
on a truncated Gamma with a sampled alpha, where Amdahl applies: the prior, the
Gamma logpdf and the truncation normalizer are all still there. The ratios here
are per-op and larger, and the two should not be quoted interchangeably.

The value speedup is strongly a-dependent for a reason that has nothing to do
with chebax: jax's igamma runs its loop to the WORST lane's trip count, so a
distribution of x concentrated near the mode is the case that flatters it most.
Both a favourable and an unfavourable x distribution are run.

The two gradient tables do NOT measure the same comparison, and the ratios
should not be read side by side. For gammainc, both columns go through the same
`gammainc_fn` rule and differ only by the cond wrapper, so `cond/bare` below 1
is the unexplained part. For betainc, `bare` is `jax.grad` of `betainc_fn`
itself, which transposes the whole panel reconstruction in reverse mode, while
the registration's custom_jvp pushes a single forward-mode tangent through it.
A scalar parameter against a 4.2M-point output is exactly the case forward mode
wins, so the 6-7x there is expected rather than mysterious.

The f32 rows are not an f32 kernel. chebax's tables are f64, so an f32 graph
evaluates in f64 and the registration casts the result back down to keep the
model's dtype. That is a real cost on a consumer part (1:64 fp64), and the
question the row answers is whether the fixed-degree f64 polynomial still beats
jax's f32 loop. It is the honest number for a floatX="float32" PyMC model.

Run: python -u experiments/19_pytensor_forward_cdf.py   (~20 min, GPU)

Most of that wall clock is jax's side, not chebax's: one betainc call at
N = 4.2M costs it seconds, and the betainc gradient takes 35 s to compile.
Run it unbuffered, or a kill part way through leaves you with nothing.
"""

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import chebax  # noqa: E402
from chebax.pytensor import _betainc_scalar_ab, _forward_gamma  # noqa: E402

N = 1 << 22
SHAPES = (0.5, 3.0, 12.0, 60.0)
# the same (a, b) pairs experiments/06 raced, so the two are comparable.
# Large pairs are left out on purpose: jax's betainc costs seconds per call
# at this N and dominates the wall clock without changing the conclusion.
PAIRS = ((2.5, 3.5), (0.5, 0.5), (9.5, 0.2))
REPEATS = 7

# the shipped registration, not a copy of it
_cond = _forward_gamma(None, "igamma", chebax.gammainc_fn,
                       jax.scipy.special.gammainc)


def _bare(a, x):
    return chebax.gammainc_fn(a, x)


def _jax(a, x):
    return jax.scipy.special.gammainc(jnp.broadcast_to(a, x.shape), x)


def _bare_beta(a, b, x):
    return chebax.betainc_fn(a, b, x)


def _jax_beta(a, b, x):
    return jax.scipy.special.betainc(jnp.broadcast_to(a, x.shape),
                                     jnp.broadcast_to(b, x.shape), x)


def _time(fn, params, xs, grad):
    """Time fn(*params, xs); with grad=True, d/d(first param)."""
    if grad:
        def call(*args):
            ps, x = args[:-1], args[-1]
            return jax.grad(lambda t: jnp.sum(fn(t, *ps[1:], x)))(ps[0])
        f = jax.jit(call)
    else:
        f = jax.jit(fn)
    out = f(*params, xs)
    jax.block_until_ready(out)
    best = np.inf
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = f(*params, xs)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    whiles = f.lower(*params, xs).compile().as_text().count("while(")
    return best, np.asarray(out), whiles


def main():
    print(f"device: {jax.devices()[0]}   N = {N}   f64   best of {REPEATS}")
    rng = np.random.default_rng(0)

    for label, draw in (("x ~ Gamma(a, 1), the favourable case", True),
                        ("x ~ Uniform(0, 4a), a wider spread", False)):
        print(f"\n=== gammainc, {label}")
        print(f"{'a':>6} {'chebax':>9} {'+cond':>9} {'jax':>9} "
              f"{'cond/bare':>10} {'jax/cond':>9} {'agree':>9}")
        for a in SHAPES:
            xs = jnp.asarray(rng.gamma(a, 1.0, N) if draw
                             else rng.uniform(0.0, 4.0 * a, N))
            av = jnp.float64(a)
            t_b, v_b, _ = _time(_bare, (av,), xs, False)
            t_c, v_c, _ = _time(_cond, (av,), xs, False)
            t_j, v_j, _ = _time(_jax, (av,), xs, False)
            print(f"{a:6g} {1e3*t_b:8.2f}m {1e3*t_c:8.2f}m {1e3*t_j:8.2f}m "
                  f"{t_c/t_b:10.2f} {t_j/t_c:9.2f} "
                  f"{np.max(np.abs(v_c - v_j)):9.1e}")

    print("\n=== gammainc d/da (igamma_grad_a is jax's expensive path)")
    print(f"{'a':>6} {'chebax':>9} {'+cond':>9} {'jax':>9} "
          f"{'cond/bare':>10} {'jax/cond':>9}")
    for a in SHAPES:
        xs = jnp.asarray(rng.gamma(a, 1.0, N))
        av = jnp.float64(a)
        t_b, g_b, _ = _time(_bare, (av,), xs, True)
        t_c, g_c, _ = _time(_cond, (av,), xs, True)
        t_j, g_j, _ = _time(_jax, (av,), xs, True)
        assert float(g_b) == float(g_c), (float(g_b), float(g_c))
        print(f"{a:6g} {1e3*t_b:8.2f}m {1e3*t_c:8.2f}m {1e3*t_j:8.2f}m "
              f"{t_c/t_b:10.2f} {t_j/t_c:9.2f}")

    # betainc: the half-registration meant these ratios reached nobody
    for label, draw in (("x ~ Beta(a, b), the favourable case", True),
                        ("x ~ Uniform(0, 1), a wider spread", False)):
        print(f"\n=== betainc, {label}")
        print(f"{'a':>6} {'b':>6} {'chebax':>9} {'+cond':>9} {'jax':>9} "
              f"{'cond/bare':>10} {'jax/cond':>9} {'agree':>9}")
        for a, b in PAIRS:
            xs = jnp.asarray(rng.beta(a, b, N) if draw
                             else rng.uniform(1e-6, 1 - 1e-6, N))
            av, bv = jnp.float64(a), jnp.float64(b)
            t_b, v_b, _ = _time(_bare_beta, (av, bv), xs, False)
            t_c, v_c, _ = _time(_betainc_scalar_ab, (av, bv), xs, False)
            t_j, v_j, _ = _time(_jax_beta, (av, bv), xs, False)
            print(f"{a:6g} {b:6g} {1e3*t_b:8.2f}m {1e3*t_c:8.2f}m "
                  f"{1e3*t_j:8.2f}m {t_c/t_b:10.2f} {t_j/t_c:9.2f} "
                  f"{np.max(np.abs(v_c - v_j)):9.1e}")

    print("\n=== betainc d/da (jax has NO a-gradient: this is the new capability)")
    print(f"{'a':>6} {'b':>6} {'chebax':>9} {'+cond':>9} {'cond/bare':>10}")
    for a, b in PAIRS:
        xs = jnp.asarray(rng.beta(a, b, N))
        av, bv = jnp.float64(a), jnp.float64(b)
        t_b, g_b, _ = _time(_bare_beta, (av, bv), xs, True)
        t_c, g_c, _ = _time(_betainc_scalar_ab, (av, bv), xs, True)
        print(f"{a:6g} {b:6g} {1e3*t_b:8.2f}m {1e3*t_c:8.2f}m {t_c/t_b:10.2f}")

    # f32: chebax still evaluates in f64 and casts back, so this is the
    # question a floatX="float32" model actually faces
    print("\n=== f32 graph (chebax computes f64 and casts down)")
    print(f"{'op':>18} {'chebax':>9} {'jax':>9} {'jax/chebax':>11}")
    xs32 = jnp.asarray(rng.gamma(3.0, 1.0, N), jnp.float32)
    t_c, _, _ = _time(_cond, (jnp.float32(3.0),), xs32, False)
    t_j, _, _ = _time(_jax, (jnp.float32(3.0),), xs32, False)
    print(f"{'gammainc a=3':>18} {1e3*t_c:8.2f}m {1e3*t_j:8.2f}m {t_j/t_c:11.2f}")
    xb32 = jnp.asarray(rng.beta(2.5, 3.5, N), jnp.float32)
    t_c, _, _ = _time(_betainc_scalar_ab,
                      (jnp.float32(2.5), jnp.float32(3.5)), xb32, False)
    t_j, _, _ = _time(_jax_beta,
                      (jnp.float32(2.5), jnp.float32(3.5)), xb32, False)
    print(f"{'betainc 2.5,3.5':>18} {1e3*t_c:8.2f}m {1e3*t_j:8.2f}m "
          f"{t_j/t_c:11.2f}")

    print("\n=== while loops in the COMPILED program")
    xs = jnp.asarray(rng.gamma(3.0, 1.0, N))
    for tag, a in (("gammainc concrete in box  ", jnp.float64(3.0)),
                   ("gammainc concrete out box ", jnp.float64(0.05))):
        f = jax.jit(lambda x, a=a: _cond(a, x))
        print(f"  {tag} {f.lower(xs).compile().as_text().count('while(')}")
    f = jax.jit(_cond)
    n_tr = f.lower(jnp.float64(3.0), xs).compile().as_text().count("while(")
    print(f"  gammainc traced shape      {n_tr}   <- the sampled-alpha case")
    f = jax.jit(_jax)
    print(f"  plain jax igamma           "
          f"{f.lower(jnp.float64(3.0), xs).compile().as_text().count('while(')}")

    xb = jnp.asarray(rng.beta(2.5, 3.5, N))
    for tag, (a, b) in (("betainc concrete in box   ", (2.5, 3.5)),
                        ("betainc concrete out box  ", (150.0, 3.5))):
        f = jax.jit(lambda x, a=jnp.float64(a), b=jnp.float64(b):
                    _betainc_scalar_ab(a, b, x))
        print(f"  {tag} {f.lower(xb).compile().as_text().count('while(')}")
    ab = (jnp.float64(2.5), jnp.float64(3.5))
    n_tr = jax.jit(_betainc_scalar_ab).lower(*ab, xb).compile().as_text()
    print(f"  betainc traced shapes      {n_tr.count('while(')}"
          f"   <- the sampled-(a,b) case")
    n_pj = jax.jit(_jax_beta).lower(*ab, xb).compile().as_text()
    print(f"  plain jax betainc          {n_pj.count('while(')}")


if __name__ == "__main__":
    main()
