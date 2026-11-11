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

x = jnp.linspace(0.1, 30.0, 200)
chebax.besselk(1.7, x)                      # parameters first, then x
jax.grad(chebax.besselk)(1.7, 5.0)          # jax.grad w.r.t. the ORDER
chebax.betainc(2.0, 3.0, 0.4)               # dI/da, dI/db via jax.grad (jax#38610)
chebax.gammainc(jnp.asarray([0.5, 2.5, 40.0]), 3.0)  # a per element
chebax.betaincinv(2.0, 3.0, 0.05)           # differentiable Beta quantile (jax#2399)

jv = chebax.besselj_frozen(2.5)             # a cached callable for one order
jv(x), jax.vmap(jax.grad(jv))(x)            # values and dJ/dx, jit/vmap-safe
```

Every function has one name that takes any parameter shape, `name(params, x)`,
and one name per path for callers who know which path they want:

| suffix | the parameters are | example |
|---|---|---|
| `_scalar` | one value for the whole call, traceable | `gammainc_scalar(2.5, x)` |
| `_array` | one value per element, broadcast with x | `gammainc_array(a, x)` |
| `_frozen` | concrete; returns a cached callable of x, what `bake` takes | `gammainc_frozen(2.5)(x)` |

The plain name picks `_scalar` for a parameter holding one value and `_array`
for an array, from the shape, so the choice is fixed at trace time. Only
gammainc and its inverses have an `_array` path so far; for the other
families an array of parameters raises and names `chebax.pergroup`. Log forms are `log_name...`.
The 0.4 spellings (`besselk_fn`, the one-argument `besselk(nu)`) still work
for now, with a DeprecationWarning naming the replacement.

## What's in

| family | functions | paths | param derivative | notes |
|---|---|---|---|---|
| `besselj` | `besselj(v, x, domain=)` | `_frozen` (order concrete) | `besselj_dnu` | x ≥ 0, validated to 1e4; `domain=` keeps only the regions covering the window and drops the rest of the evaluation, measured 2.5x f64 / 2.6x f32 (`experiments/04`), nan outside it |
| `bessely` | `bessely(v, x)` | `_frozen` (order structural) | `bessely_dnu` | x ≥ 1e-6 |
| `besseli` | `besseli(nu, x, scaled=)`, `besseli_ratio(nu, x)` | `_scalar`, `_frozen` | `besseli_dnu`, or `grad` of `besseli` | ν ∈ [0, 10] (the ratio [0, 9]), nan outside; `scaled` = scipy's ive; the ratio I_{v+1}/I_v is the circular-statistics workhorse |
| `besselk` | `besselk(nu, x)`, `log_besselk(nu, x)` | `_scalar`, `_frozen` | `besselk_dnu`, or `grad` of `besselk` | ν ∈ [0, 10], nan outside; x ≥ 1e-6; the log form has no underflow ceiling |
| Matérn | `matern(nu, r, lengthscale)` | — | via `grad` | unit-variance correlation, LEARNABLE smoothness ν ∈ (0, 10] |
| `betainc` | `betainc(a, b, x)`, `log_betainc(a, b, x)` | `_scalar`, `_frozen` | via `grad` | (a, b) ∈ [0.1, 100]² (panels, split at 10), nan outside; for x ≤ 1/2 the log form resolves the lower tail with no underflow floor; above 1/2 (a ≫ b) it carries I's absolute error, ~eps/I relative |
| `gammainc` | `gammainc(a, x)`, `gammaincc`, `log_gammainc`, `log_gammaincc` | `_scalar`, `_array`, `_frozen` (P and Q) | via `grad` | a ≥ 0.1 (above 10 via Temme-zone tables, whose v = 10/a axis reaches a = ∞), x ≥ 0; branchless (no while_loop). `_scalar`: 10–27x vs jax's on GPU f64 at a ≤ 10 (`experiments/05`). `_array`: one small 2-D patch per element (144 coefficients, 49 in float32), 2.3–18x vs jax's gammainc on values and 2.8–23x on value_and_grad on GPU f64, 4.5–66x / 6.6–95x f32 (`experiments/22`) |
| `hyp1f1` | `hyp1f1(a, b, x)`, `log_hyp1f1(a, b, x)` | `_scalar`, `_frozen` | via `grad` | Kummer M, (a, b) ∈ [0.1, 10]², x ≥ 0; jax's is documented-unstable (jax#21503); the log form has no overflow ceiling |
| spherical | `spherical_jn/yn(n, x)` | `_frozen` | — | n ∈ [0, 9], via half-integer tables |
| Gamma quantiles | `gammaincinv(a, p)`, `gammainccinv(a, y)`, `chi2inv(k, p)` | `_scalar`, `_array` | via `grad` (IFT) | jax#5350; `_scalar` any a > 0 (jax's gammainc below 0.1), `_array` a ≥ 0.1 with second derivatives, d²/da² included; both cover every finite a. `_array` vs TFP's `igammainv` on GPU: 0.93–7.3x f64 and 2.6–34x f32 on values, growing with the spread of a (`experiments/23`) |
| quantiles | `betaincinv`, `stdtr`, `stdtrit` | — | via `grad` (IFT) | jax#2399/#20358; betaincinv on betainc's box, nan outside; Student-t ν ∈ [0.2, 200] via dedicated slice tables |
| von Mises | `vonmises_cdf/icdf` | — | via `grad` | κ ∈ [0, 50] |
| erf family | `dawsn`, `erfcx` | — | — (no params) | recent jax ships both; kept as a worked parameter-free recipe (plain functions, so not `bake` inputs: bake takes Recipe instances) |
| gamma | `gammaln(x)`, `digamma(x)` | — | — (no params) | x ≥ 0 (nan below 0, where jax's reflect); both keep relative accuracy next to their zeros (lgamma's at 1 and 2, digamma's at 1.4616), where jax's are only absolutely accurate; gammaln's derivative is digamma, from its own table. Against jax's on one CPU core in float64: gammaln 3.6x on values and 4.1x on value_and_grad, digamma 2.8x and 3.4x; on the GPU in float64, 2.4x / 2.2x and 2.7x / 6.0x (`experiments/44`) |
| normal quantile | `ndtri_exp(y)` | — | — (no params) | ndtri(e^y) for a log-probability y ≤ 0, what PyMC's TruncatedNormal icdf calls; relative accuracy from the smallest normal y to the largest float, next to the median's zero too, where jax's ndtri(exp(y)) is -inf below y = -745 and inf or off by up to 1% within 1e-3 of 0; the derivative from its own table. Against jax's ndtri(exp(y)): value_and_grad 2.0–2.4x on one CPU core and 3.2–4.0x on the GPU; values 0.92–1.10x on the CPU and 1.1–1.9x on the GPU (`experiments/45`) |
| Lambert W | `lambertw(x, k)` | — | — | both real branches, jax#13680 |
| Tweedie | `tweedie_logpdf(p, mu, phi, y)`, `log_wright_bessel0(rho, z)` | — | via `grad` | power p ∈ [1.05, 1.95] (uniform per call), μ, φ > 0 and y ≥ 0 broadcast, the zero mass included; Wright's Φ(ρ, 0; z) with ρ = (2−p)/(p−1) ∈ [1/19, 19]; 89–130x scipy's `log_wright_bessel` on one CPU core, 254–331x on the GPU (`experiments/21`) |

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
latent scalar shape sample instead of raising. The forward CDFs `gammainc`,
`gammaincc` and `betainc` take their VALUES off the tables too, which is
where the loop-free evaluation shows up for censored and truncated models;
outside the table box they fall back to jax's own, since a forward function
has to answer everywhere. Shape parameters must be
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

`chebax.numpyro.Tweedie(mu, phi, power)` is the compound Poisson-gamma law for
data with exact zeros and positive amounts in one likelihood (claims,
rainfall, biomass). Its `log_prob` is `chebax.tweedie_logpdf`, differentiable
in the power too, so NUTS can sample p instead of taking it fixed: on 1,500
draws at (mu, phi, p) = (3, 0.6, 1.35) the posterior is p = 1.41 +- 0.05.
Shape parameters are uniform per call; domain boxes are in the module
docstring and checked at construction.

`notebooks/` holds themed, executed walkthroughs: why Chebyshev nodes work,
the Bessel family with a learnable Matérn kernel, differentiable quantiles,
truncated and circular distributions in numpyro, Gaussian tails / Lambert W /
binomial reliability, copulas, baking, and per-group shape parameters in a
hierarchical model under NUTS. The numpyro and plotting
dependencies install with `pip install chebax[examples]`. `examples/` holds
plain scripts: a few of the notebook workflows in copy-paste form, plus
integration examples with no notebook counterpart (a GIG distribution for
efax, learned Matérn smoothness, truncated sampling, `chebax.numpyro` with
latent shape parameters under NUTS).

Parameters come first, evaluation point last (opposite of scipy). Outside
gammainc, orders and shapes are uniform per call, meaning one value whatever
its shape: a scalar or any size-1 array, which is what numpyro and pytensor
hand back after broadcasting. Per-element parameter arrays are out of scope
for those families by design and say so by name; gammainc takes them on its
`_array` path. The middle ground is `chebax.pergroup(fn, group_idx)`: a static integer array
assigns each element to a group, each group gets its own (traceable) parameter
set — one group per chain, mixture component, or plate level.
Accuracy is validated against mpmath at 40 dps — measured worst cases and the
per-family error metrics are in each test file's docstring; tables regenerate
bit-for-bit from checked-in generators.

## Docs

[Architecture](docs/architecture.md) defines module boundaries, numerical
contracts, and validation; [Coding style](docs/coding-style.md) defines naming,
formatting, and documentation conventions. [Adding a recipe](docs/adding-a-recipe.md)
is the workflow for new functions. The [October review](docs/review-2026-10-26.md)
records the latest review, how each finding was resolved, and what is open.

`docs/increments.md` is the design log of every recipe (what was measured,
which traps were found); `PROJECT.md` holds the plan and evidence; `CLAUDE.md`
has agent operating instructions and local test commands.
Grown out of a private research project on GPU Bessel functions;
its two load-bearing measurements are reproduced here in `experiments/`.
BSD-3-Clause.
