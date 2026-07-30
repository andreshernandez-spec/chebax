# chebax

Differentiable Chebyshev approximants for special functions, aimed at JAX and GPUs.
A build-time generator (mpmath precision) produces fixed-degree branchless polynomial
kernels with exact derivative series, including gradients with respect to function
parameters (e.g. the Bessel order). Prebuilt recipes give out-of-the-box special
functions; a bake step emits self-contained pure-JAX modules and xsf-style C++ headers.

Early development. What's in: the generic core (fit, jax evaluation, exact derivative
series, segmentation); `besselj(v)` for any real order in [0, 10] on x >= 0 (validated
against mpmath to x = 1e4) with dJ/dx via jax.grad and the order gradient via
`besselj_dnu(v)`; `besselk(v)` on [1e-6, inf) likewise, plus `besselk_fn(nu, x)` with
nu as a traced jax scalar — so a Matern kernel can learn its smoothness by gradient
descent (see `examples/matern_learn_nu.py`); `besseli(v)` on x >= 0 with a
`scaled=True` variant (scipy's ive) that stays finite past the e^x overflow, plus
`besseli_fn`; `bessely(v)` on [1e-6, inf) completing the Bessel quartet; `dawsn` and `erfcx`
(the erf-family members jax lacks); `betainc(a, b)` with `betainc_fn(a, b, x)` taking
both shape parameters as traced scalars, so jax.grad gives dI/da and dI/db (which jax
itself lacks, jax#38610); the differentiable quantile toolkit `betaincinv`,
`gammaincinv`, `stdtr`, `stdtrit` (jax#2399/#5350/#20358 — Beta, Gamma and Student-t
quantiles with implicit-function-theorem gradients in every argument);
`spherical_jn`/`spherical_yn` (n in [0, 9], via the half-integer tables); `lambertw`
(both real branches, jax#13680); a truncated-distribution sampling example
(`examples/truncated_sampling.py`); `vonmises_cdf`/`vonmises_icdf` (kappa in [0, 50],
traced concentration, the circular cdf/icdf numpyro lacks); and bake
emitters
(`chebax.bake.jax_module`, `chebax.bake.xsf_header`) that turn an instance into a
standalone pure-jax module or a self-contained C++17 header. No mpmath at use time
anywhere. Read
`PROJECT.md` for the plan and evidence, `CLAUDE.md` for how to work here. Grown out
of a private research project (the `../bessel/` references in `PROJECT.md` point
there); the two load-bearing measurements are reproduced here in `experiments/`.
BSD-3-Clause.
