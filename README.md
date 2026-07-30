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
descent (see `examples/matern_learn_nu.py`); and bake emitters
(`chebax.bake.jax_module`, `chebax.bake.xsf_header`) that turn an instance into a
standalone pure-jax module or a self-contained C++17 header. No mpmath at use time
anywhere. Y/I and non-Bessel families are not yet. Read
`PROJECT.md` for the plan and evidence, `CLAUDE.md` for how to work here. Grown out
of a private research project (the `../bessel/` references in `PROJECT.md` point
there); the two load-bearing measurements are reproduced here in `experiments/`.
BSD-3-Clause.
