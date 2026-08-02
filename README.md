# chebax

Differentiable special functions for JAX and GPUs, built from Chebyshev
approximants. A build-time generator (mpmath precision) produces fixed-degree
branchless polynomial kernels with exact derivative series — including gradients
with respect to function *parameters* (Bessel order, Beta shape, von Mises
concentration), which mainstream ML libraries lack. No mpmath at use time; the
runtime imports only jax and numpy.

```sh
pip install chebax
```

```python
import jax
jax.config.update("jax_enable_x64", True)   # before building instances: tables
                                            # materialize in the active precision
import jax.numpy as jnp
import chebax

jv = chebax.besselj(2.5)                    # any real order in [0, 10]; cached
x = jnp.linspace(0.1, 30.0, 200)
jv(x), jax.vmap(jax.grad(jv))(x)            # values and dJ/dx, jit/vmap-safe

chebax.besselk_fn(1.7, x)                   # order as a traced scalar:
jax.grad(chebax.besselk_fn)(1.7, 5.0)       #   jax.grad w.r.t. the ORDER
chebax.betainc_fn(2.0, 3.0, 0.4)            # dI/da, dI/db via jax.grad (jax#38610)
chebax.betaincinv(2.0, 3.0, 0.05)           # differentiable Beta quantile (jax#2399)
```

## What's in

| family | factory | traced params | param derivative | notes |
|---|---|---|---|---|
| `besselj` | `besselj(v)` | — | `besselj_dnu(v)` | x ≥ 0, validated to 1e4 |
| `bessely` | `bessely(v)` | — (structural) | `bessely_dnu(v)` | x ≥ 1e-6 |
| `besseli` | `besseli(v, scaled=)` | `besseli_fn`, `besseli_ratio` | `besseli_dnu(v, scaled=)` | `scaled` = scipy's ive; the ratio I_{v+1}/I_v is the circular-statistics workhorse |
| `besselk` | `besselk(v)` | `besselk_fn`, `log_besselk_fn` | `besselk_dnu(v)` | x ≥ 1e-6; the log form has no underflow ceiling |
| Matérn | — | `matern(nu, r, lengthscale)` | via `grad` | unit-variance correlation, LEARNABLE smoothness ν ∈ (0, 10] |
| `betainc` | `betainc(a, b)` | `betainc_fn`, `log_betainc_fn` | via `grad` of `_fn` | (a, b) ∈ [0.1, 100]² (panels, split at 10); the log form resolves the lower tail with no underflow floor |
| `gammainc` | `gammainc(a)`, `gammaincc(a)` | `gammainc_fn`, `gammaincc_fn`, `log_gammainc_fn`, `log_gammaincc_fn` | via `grad` of `_fn` | a ≥ 0.1 (above 10 via Temme-zone tables, whose v = 10/a axis reaches a = ∞), x ≥ 0; branchless (no while_loop), measured 10–27x vs jax's on GPU f64 at a ≤ 10 (`experiments/05`) |
| `hyp1f1` | `hyp1f1(a, b)` | `hyp1f1_fn`, `log_hyp1f1_fn` | via `grad` of `_fn` | Kummer M, (a, b) ∈ [0.1, 10]², x ≥ 0; jax's is documented-unstable (jax#21503); the log form has no overflow ceiling |
| spherical | `spherical_jn/yn(n)` | — | — | n ∈ [0, 9], via half-integer tables |
| quantiles | — | `betaincinv`, `gammaincinv`, `gammainccinv`, `chi2inv`, `stdtr`, `stdtrit` | via `grad` (IFT) | jax#2399/#5350/#20358; chi-squared at real dof to 2000; Student-t ν ∈ [0.2, 200] via dedicated slice tables |
| von Mises | — | `vonmises_cdf/icdf` | via `grad` | κ ∈ [0, 50] |
| erf family | — | `dawsn`, `erfcx` | — (no params) | recent jax ships both; kept as a worked parameter-free recipe (plain functions, so not `bake` inputs: bake takes Recipe instances) |
| Lambert W | — | `lambertw(x, k)` | — | both real branches, jax#13680 |

Plus the generic core (`fit`, `ChebSeries`, `PiecewiseCheb`) and bake emitters
(`chebax.bake.jax_module`, `chebax.bake.xsf_header`: self-contained pure-jax
modules and C++17 headers from an instance; `dtype="float"` emits an f32
CUDA-shaped kernel body, and `truncate_tol=1e-7` drops the converged f64
tail — measured 1.4–2.2x faster f32 evaluation, `experiments/12`).

For pymc users: `import chebax.pytensor` (installs with
`pip install chebax[pytensor]`) registers JAX-backend lowerings for the
pytensor ops that otherwise need tfp-nightly under
`pm.sample(nuts_sampler="numpyro"|"blackjax")` — betaincinv, gammaincinv,
gammainccinv, erfcx, erfcinv, ive, kve — and adds betainc gradients in its
shape parameters, so censored or truncated StudentT/Beta likelihoods with a
latent scalar shape sample instead of raising. Shape parameters must be
scalar (batched ones fall back to tfp or fail loudly); out-of-domain values
return nan, never silently wrong numbers. `gammainccinv` takes its
upper-tail probability directly (it used to be solved at `1 - p`, which
collapsed everything below ~1e-17 onto one point), so both tails resolve to
the same depth.

For numpyro users: `from chebax.numpyro import TruncatedGamma, TruncatedBeta,
TruncatedStudentT` (installs with `pip install chebax[numpyro]`) gives the
truncated distributions numpyro's location-scale machinery cannot cover
(numpyro#969, numpyro#1365): full Distribution classes with reparameterized
inverse-CDF sampling (`has_rsample`), truncation-normalized `log_prob`, and
gradients through every parameter including the shapes, so NUTS and SVI work
with a latent concentration or df. The truncation is normalized in log space,
so an interval whose two ends land in the same tail still works: `Gamma(3)`
truncated to [50, 51] has mass ~3e-20 and returns an ordinary density.
Shape parameters are uniform per call; domain boxes are in the module
docstring and checked at construction.

`notebooks/` holds themed, executed walkthroughs: why Chebyshev nodes work,
the Bessel family with a learnable Matérn kernel, differentiable quantiles,
truncated and circular distributions in numpyro, Gaussian tails / Lambert W /
binomial reliability, copulas, and baking. The numpyro and plotting
dependencies install with `pip install chebax[examples]`. `examples/` holds
plain scripts: a few of the notebook workflows in copy-paste form, plus
integration examples with no notebook counterpart (a GIG distribution for
efax, learned Matérn smoothness, truncated sampling, `chebax.numpyro` with
latent shape parameters under NUTS).

Parameters come first, evaluation point last (opposite of scipy). Orders/shapes
are uniform per call; per-element parameter arrays are out of scope by design.
The middle ground is `chebax.pergroup(fn, group_idx)`: a static integer array
assigns each element to a group, each group gets its own (traceable) parameter
set — one group per chain, mixture component, or plate level.
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
