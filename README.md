# chebax

Differentiable Chebyshev approximants for special functions, aimed at JAX and GPUs.
A build-time generator (mpmath precision) produces fixed-degree branchless polynomial
kernels with exact derivative series, including gradients with respect to function
parameters (e.g. the Bessel order). Prebuilt recipes give out-of-the-box special
functions; a bake step emits self-contained pure-JAX modules and xsf-style C++ headers.

Planning stage. Read `PROJECT.md` for the plan and evidence, `CLAUDE.md` for how to
work here. Grown out of `../bessel/` Track B. BSD-3-Clause.
