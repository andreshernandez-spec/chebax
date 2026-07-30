# chebax

Differentiable special functions for JAX and GPUs, built from Chebyshev
approximants. A build-time generator (mpmath precision) produces fixed-degree
branchless polynomial kernels with exact derivative series — including gradients
with respect to function *parameters* (Bessel order, Beta shape, von Mises
concentration), which mainstream ML libraries lack. No mpmath at use time; the
runtime imports only jax and numpy.

```python
import jax, jax.numpy as jnp, chebax

jv = chebax.besselj(2.5)              # any real order in [0, 10]; cached, table-backed
jv(x), jax.grad(jv)(x)                # values and dJ/dx, jit/vmap-safe

chebax.besselk_fn(nu, x)              # nu as a traced scalar: jax.grad w.r.t. the ORDER
chebax.betainc_fn(a, b, x)            # dI/da, dI/db via jax.grad  (jax#38610)
chebax.betaincinv(a, b, p)            # differentiable Beta quantile  (jax#2399)
```

## What's in

| family | factory | traced params | param derivative | notes |
|---|---|---|---|---|
| `besselj` | `besselj(v)` | — | `besselj_dnu(v)` | x ≥ 0, validated to 1e4 |
| `bessely` | `bessely(v)` | — (structural) | `bessely_dnu(v)` | x ≥ 1e-6 |
| `besseli` | `besseli(v, scaled=)` | `besseli_fn` | `besseli_dnu(v, scaled=)` | `scaled` = scipy's ive |
| `besselk` | `besselk(v)` | `besselk_fn` | `besselk_dnu(v)` | x ≥ 1e-6; Matérn demo |
| `betainc` | `betainc(a, b)` | `betainc_fn` | via `grad` of `_fn` | (a, b) ∈ [0.1, 10]² |
| spherical | `spherical_jn/yn(n)` | — | — | n ∈ [0, 9], via half-integer tables |
| quantiles | — | `betaincinv`, `gammaincinv`, `stdtr`, `stdtrit` | via `grad` (IFT) | jax#2399/#5350/#20358 |
| von Mises | — | `vonmises_cdf/icdf` | via `grad` | κ ∈ [0, 50] |
| erf family | — | `dawsn`, `erfcx` | — (no params) | the two jax lacks |
| Lambert W | — | `lambertw(x, k)` | — | both real branches, jax#13680 |

Plus the generic core (`fit`, `ChebSeries`, `PiecewiseCheb`) and bake emitters
(`chebax.bake.jax_module`, `chebax.bake.xsf_header`: self-contained pure-jax
modules and C++17 headers from an instance).

`notebooks/` holds themed, executed walkthroughs: the Bessel family with a
learnable Matérn kernel, differentiable quantiles, truncated and circular
distributions in numpyro, Gaussian tails / Lambert W / binomial reliability,
and baking. The numpyro and plotting dependencies install with
`pip install chebax[examples]`. The same material in script form lives in
`examples/`.

Parameters come first, evaluation point last (opposite of scipy). Orders/shapes
are uniform per call; per-element parameter arrays are out of scope by design.
Accuracy is validated against mpmath at 40 dps — measured worst cases and the
per-family error metrics are in each test file's docstring; tables regenerate
bit-for-bit from checked-in generators.

## Docs

`docs/adding-a-recipe.md` is the workflow for new functions;
`docs/increments.md` the design log of every recipe (what was measured, which
traps were found); `PROJECT.md` the plan and evidence; `CLAUDE.md` how to work
in the repo. Grown out of a private research project on GPU Bessel functions;
its two load-bearing measurements are reproduced here in `experiments/`.
BSD-3-Clause.
