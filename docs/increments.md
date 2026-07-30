# M6 increment log

The per-recipe design record: what was built, what was measured, and which
traps were found. Moved out of PROJECT.md's milestone table (where seven
increments had accumulated in one cell); M0–M5 remain as rows there. Each
recipe's generator docstring holds the same design in condensed form;
`experiments/03_degree_measurement.py` reproduces every degree claim.

## 1 — besseli (2026-07-29)

Log-tables (factored entire part in z = x² inner; e⁻ˣ-scaled tail in
t = 8/x; a single ν-panel suffices, no Γ-pole feature), a `scaled=` flag
(scipy's `ive`, keeps working past the x ≈ 709 overflow, matches mpmath to
the last digit at x = 720), and traced `besseli_fn`. The inner z-degree
(~64 at ν = 0) is set by I₀'s complex zeros at z = −j₀ₖ² entering the log.
Measured pointwise-relative: values 4.2e-15 (incl. traced), dI/dx 1.3e-15,
dI/dν 6.5e-13.

## 2 — bessely (2026-07-29)

The hardest recipe; two design traps found and fixed. Design: reduce to
μ = ν − ⌊ν⌋ ∈ [0, 1), tabulate the scaled pair T_a = (x/2)^μ·Y_μ,
T_b = (x/2)^{μ+1}·Y_{μ+1} in u = ln x on [1e-6, 5], lift by ⌊ν⌋ unrolled
steps of the upward recurrence (Temme's strategy with tables; integer
orders are ordinary points, retiring R6/S3 for Y); mid [5, 30] direct in
x; tail ≥ 30 reuses J's P,Q tables verbatim (Y = √(2/πx)(sin x·A −
cos x·B), same A, B). Trap 1: the round-to-nearest decomposition
(μ ∈ [−½, ½]) breaks — for μ < 0 the (x/2)^μ scaling amplifies the
divergent branch to ~1e3 and leaks ~1e-11 into every μ-reconstruction;
floor fixes it (measured, three orders). Trap 2: the upward lift is only
stable once k ≳ x — running it to x = 8 cost 1.4e-10 in accumulated
oscillatory-regime cancellation; the seam moved to x = 5 where the
measured lift error is ≤ 4e-16. Errors modulus-relative (err/√(J²+Y²)):
values 2.7e-14, dY/dx 2.0e-14, ∂Y/∂ν 2.7e-14 worst over 10 orders incl.
integers and half-integers. No traced-ν variant: the recurrence depth ⌊ν⌋
is structural.

## 3 — dawsn and erfcx (2026-07-29)

The two erf-family members jax lacks (erf/erfc/erfinv are XLA-native).
Parameter-free plain functions: dawsn as x·E(x²) (oddness and the x = 0
gradient exact by construction) with tail G((6/x)²)/x; erfcx fitted on
[0, 6] with tail and the negative-x reflection 2e^{x²} − erfcx(−x)
(correctly inf past −26.6). 116 table doubles total; the seam sits at 6,
not 4, because the e^{−x²} Gevrey fuzz measurably slows the tail fit at 4.
Derivative tests use the exact ODE identities as oracles (D′ = 1 − 2xD,
erfcx′ = 2x·erfcx − 2/√π). Three runtime gotchas worth remembering:
masked-branch exp overflow poisons gradients via 0·inf (clamp the
reflection's argument); `jnp.abs` has a zero derivative at 0 by tie
convention, so |x| for branch inputs is built from a select; and
module-level ChebSeries construction initializes the jax backend as an
import side effect (caught by the import-hygiene test under GPU
contention) — series are built lazily.

## 4 — betainc (2026-07-30)

The library's first two-parameter recipe: the ν-table machinery
generalized to an (x, a, b) tensor (24×72×28, three-stage mp fit, each
x-coefficient a 2-D Chebyshev series over (a, b) ∈ [0.1, 10]²).
Log-tabulated F = ₂F₁(a+b, 1; a+1; x) with the reflection
I_x(a,b) = 1 − I_{1−x}(b,a) for x > ½ (both parameter orders
reconstructed; 1 − x is exact in f64 on [½, 1]). `betainc_fn(a, b, x)`
takes both shape parameters traced: `jax.grad` w.r.t. a and b works — the
jax#38610 gap — measured 2.0e-14 against mp.diff. Values 7.5e-15 absolute
(the CDF contract); dI/dx 5.9e-15 against the exact Beta density as
oracle.

## 5 — the quantile toolkit (2026-07-30)

`betaincinv`, `gammaincinv`, `stdtr`, `stdtrit`, answering jax#2399,
jax#5350 and jax#20358. Fixed-count safeguarded Newton, jittable, with
gradients from the implicit function theorem (never through the
iteration): ∂x*/∂θ = −(∂CDF/∂θ)/pdf, the CDF gradients coming from
`betainc_fn` (or jax's `igamma_grad_a`). Design points earned by
measurement: solve in logit space (beta) / log space (gamma) so tail
quantiles to ~1e-300 stay resolvable (x-space bisection saturated at
3.5e-2 residuals); a symmetry swap solves every element in the mirrored
lower tail (1 − p exact on [½, 1]); and one genuinely subtle solver bug —
the strict-inequality safeguard catapulted converged iterates: a one-sided
approach never tightens the far bound, and a Newton step stalling exactly
at the root got rejected into the midpoint of the half-open bracket
(inclusive test fixes it). Measured: betaincinv 4.9e-15 worst under the
two-sided contract (p-roundtrip lower half, distance-from-1 vs 40-dps
references upper half; quantiles within ε of 1 round to 1.0, the
scipy-shared representation limit, covered by the mirrored call);
gammaincinv 4.0e-14; stdtr/stdtrit roundtrips ~5e-16; IFT gradients
9.3e-15 / 2.6e-16. `gammaincinv` needs no tables at all — jax's own
`gammainc` supplies values and the a-gradient.

## 6 — spherical Bessel, truncated sampling, Lambert W (2026-07-30)

`spherical_jn(n)`/`spherical_yn(n)` for n ∈ [0, 9] are one-line wrappers
over the half-integer cylindrical tables (jₙ = √(π/2x)·J_{n+½}; answers
the spherical half of jax#18119), verified against scipy as an independent
oracle to 1e-12 modulus-relative, with j₀′ = −j₁ as a gradient identity
test. `examples/truncated_sampling.py` + a deterministic test
(midpoint-rule u makes the reparameterized mean a quadrature) demonstrate
the numpyro#1365-class use: truncated Beta/Student-t sampling
differentiable in the shape parameters, gradient verified to 1e-4 against
mp finite differences of the analytic truncated mean. `lambertw(x, k)`
(jax#13680), both real branches, is a table-free fixed-count Halley solver
(branch-point series in p = √(2(1+ex)) + log asymptotics as initializers;
cubic convergence) with the implicit gradient dW/dx = 1/(eᵂ(1+W)) as
custom_jvp: measured 8.6e-16 (k=0) and 6.4e-16 (k=−1) worst away from the
branch point, inside the √ε conditioning allowance near it, gradients 1–2
ulp.

## 7 — von Mises CDF and quantile (2026-07-30)

The circular-distribution gap (numpyro has the sampler, no cdf/icdf).
Structure: F = ½ + θ/2π + θ·H(θ²; κ) with H from a 104×80 table in
(w = θ², r = √κ) on κ ∈ [0, 50] — the dawsn move makes oddness and the
θ = 0 gradient exact, puts the κ→∞ boundary layer (width ~1/√κ) at the
w-endpoint where Chebyshev clusters, and the √κ axis keeps the layer's
motion uniform (measured: raw κ didn't converge by 64, r needs 68; w needs
92 at κ = 50). Samples from mp.quad of the defining integral.
`vonmises_cdf(kappa, theta)` is traced in κ (learnable concentration);
dF/dθ = the density (test oracle, 1e-12); dF/dκ vs mp.diff 1e-11;
`vonmises_icdf` reuses the safeguarded solver + IFT jvp, with I₀(κ)
supplied by chebax's own `besseli_fn`. Roundtrip ≤ 5e-13; all seven tests
passed on first run.

## 8 — review hardening (2026-07-30)

A deep maintainability/onboarding review (author pass + fresh-eyes
subagent) followed by this refactor: shared `_common`/`_gen_common`
modules replacing 17 duplicated helpers and 12 private cross-module
imports; a `Recipe` pytree base retiring ~300 lines of boilerplate; one
table emitter with axis-semantics headers and repr-stable floats (tables
regenerated, coefficients bit-identical); the degree-measurement harness
checked in as experiments/03; and this document plus
`docs/adding-a-recipe.md`. Also surfaced: a plaintext PyPI token in
`.env` (never committed; moved to Andres's credential handling) and the
bake module's per-class scaling problem (the besselj formula lives in four
places), recorded as a design constraint for the next bake consumer.
