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

The erf-family members beyond XLA-native erf/erfc/erfinv. (Correction
2026-07-30: recent jax ships dawsn and erfcx too; the draw here is the
C++ bake path and the uniform recipe form.)
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

2026-08-01: that last sentence is stale. Increment 17 rewired the Newton
residual onto the gammainc recipe tables for a ∈ [0.1, 10], keeping jax's
`gammainc` only as the out-of-box fallback.

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
motion uniform (measured: raw κ didn't converge by 64; correction
2026-07-30, review finding 23: the original 80-node r probe was
self-truncating; remeasured at 128 nodes, r needs 87 and w needs 90, and
the table was upsized 104×80 → 112×108 for ~20% margin). Samples from
mp.quad of the defining integral.
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
places), resolved in the next increment.

## 9 — generic bake via jaxpr emission (2026-07-30)

The bake per-class scaling problem, resolved by Andres's Option-B call:
trace the instance's own `__call__` to a jaxpr and emit source from that,
so the runtime is the single source of truth and recipes stay plain, typed
jnp code — authors keep the type checker and autocomplete, unlike an
abstract-ops shim (his pandas-vs-pydantic argument). Two structural facts,
probed before building and now load-bearing: every ChebSeries evaluation
is ONE `custom_jvp_call` equation whose unrolled Clenshaw stays contained
in its call_jaxpr — the emitter folds it to `clenshaw(C_FIELD, affine(x))`
by value-matching the coefficient constant against the instance's series
fields (recursing into nested recipes like spherical) — and the residual
vocabulary is ~15 primitives (`lgamma` arrives as a direct primitive and
maps to `std::lgamma`). Any Recipe instance now bakes: J, K, I(scaled), Y,
betainc, spherical verified, Python artifacts **bitwise identical** to the
runtime, C++ within 1e-12 (glibc-vs-XLA ulps). Solver-based callables
raise NotImplementedError (`fori_loop` → for-loop is the natural v2). The
two hand-written besselj templates are deleted; the emitter runs at dev
time only, so jax-internal drift fails loudly at regeneration without
invalidating committed artifacts.

## 10 — log_besselk_fn (2026-07-30)

`ln K_nu` exposed straight from the log tables that besselk already stores:
inner `Ltil(ln x) - nu ln(x/2)`, tail `Lt(8/x) + (1/2) ln(pi/(2x)) - x`. No
new tables, no new floors. The point is the underflow ceiling: besselk_fn
returns K itself and hits 0 past x ~ 746, while the log form is valid for
arbitrarily large x (the tail variable 8/x -> 0 is an ordinary point).
Motivated by GIG-class log-normalizers (examples/efax_gig.py), which need
ln K_p and its order derivative inside jax AD; same shape as TFP's
log_bessel_kve, which has no order gradient. Measured vs mpmath (60 dps),
errors normalized by
max(1, |ref|), grid nu in [0.03, 9.97] x in [1e-6, 1e8] including both
seams and the old underflow edge: value 1.4e-15, d/dnu 2.9e-15,
d/dx 1.5e-15 (bars 1e-14 / 2e-14 / 1e-14). Traced-nu contract identical to
besselk_fn (uniform per call, [0, 10] unchecked under trace).

## 11 — chebax.pytensor, the opt-in pymc/JAX dispatch module (2026-07-31)

`import chebax.pytensor` registers `jax_funcify` lowerings for the seven
scalar ops pytensor otherwise routes through tfp-nightly on the JAX
backend, plus a grad-enabled BetaInc. Design points: per-class
singledispatch registration silently replaces pytensor's tfp stubs (no
upstream change needed); Erfcx/Erfcinv are pure jax (native erfcx; ndtri
composition), so the pm.Truncated-Normal breakage class needs no tables
at all; BetaInc keeps jax's own values on every domain and adds a
custom_jvp whose a/b partials come from forward mode through
`betainc_fn`, nan outside the [0.1, 10]^2 box; Ive/Kve/inverse-CDF ops
guard on trace-time parameter SHAPE (scalar -> chebax, batched -> tfp
fallback or a NotImplementedError naming this module) and on runtime
parameter VALUES (nan outside table domains, never silent extrapolation).
Kve goes through `log_besselk_fn` so it stays finite at any x. Verified
end to end with pymc 6.2 and no tfp installed: pm.Truncated Normal under
nuts_sampler="numpyro" (the pymc#7980 class), and a censored StudentT
with a LATENT dof, which stock jax rejects on the betainc gradient.
Wiring tests in tests/test_pytensor_plugin.py (jax_funcify-then-jax.grad,
the numpyro path, plus one mode="JAX" end-to-end); accuracy stays owned
by the per-recipe suites. pytensor imports only inside the module: the
core runtime rule (jax + numpy) is untouched, `pip install
chebax[pytensor]` pins the floor.

## 12 — pergroup, the per-group parameter wrapper (2026-07-31)

`chebax.pergroup(fn, group_idx, num_groups=None)` relaxes uniform-per-call
to per-group: a STATIC integer array assigns every element to one of G
groups, each group gets its own traceable parameter set, and the wrapped
function is jit/grad/vmap-compatible. Built only after
`experiments/07_pergroup_crossover.py` measured that plain vmap over
groups already serves the regime (reconstruction overhead within a few
percent of the uniform floor for n/group >= 16k; vs jax's betainc the
per-group path won every measured cell, worst 7.3x, 44-98x at MCMC
scale), so the wrapper is index bookkeeping around vmap, not new
arithmetic: numpy at wrap time sorts elements by group, pads each row
with the group's own first element (padding never leaves fn's domain and
is never read back), and gathers results back to the input shape. Empty
groups are allowed (pass num_groups) and contribute zero gradient.
Ragged groups cost padding: the padded matrix is (G, max group size).
Contract tested as exactness against the elementwise uniform path at
5e-15 relative (tests/test_pergroup.py); accuracy stays owned by the
per-recipe suites, cost by experiments/07. Per-element parameters stay
out of scope (PROJECT.md scope boundary).

## 13 — gammainc, the incomplete gamma pair (2026-07-31)

`gammainc(a)` / `gammaincc(a)` factories and `gammainc_fn` /
`gammaincc_fn` traced variants: P(a, x) and Q = 1 - P for a in
[0.1, 10], x in [0, inf), from two log-tables (the besselk move applied
to a CDF). Inner x in [0, 8] tabulates L = ln 1F1(1; a+1; x) (DLMF
8.5.1), entire in x so no transform is needed; the x^a branch point and
the dynamic range live in the exact prefactor exp(a ln x - x -
lnGamma(a+1)). Tail tabulates T = ln[Gamma(a,x) x^(1-a) e^x] in
t = 8/x, the divergent-asymptotic-series endpoint handled the same way
as besselk's tail. Measured degrees (experiments/03): L 23 in x, 53 in
a (the x -> 0 edge, where L ~ x/(a+1) puts the a = -1 pole 1.1 from the
domain and sets the ellipse); T 20 in both. Q is direct (relative
accuracy) for x > 8, P below the transition; each is 1-minus-the-other
where it is ~1, so both meet the absolute CDF contract everywhere
(measured 4.8e-15 worst) and stay relatively accurate where they decay,
with the one honest hole noted in the runtime docstring (Q relative,
x just under 8 at small a). dP/dx floors split at the saturated wedge:
relative 3.9e-13 where density >= 1e-4 P, absolute 1.5e-16 where the AD
bracket cancels (eps P/density is unavoidable there). dP/da and dQ/da
3-4e-15 vs mp.diff through the traced path, one polynomial where jax
runs the igamma_grad_a looped series. Motivation is jax#28547-class:
jax's gammainc is two whole-array while loops run to the worst
element's trip count. experiments/05 re-run with the real recipe:
10-27x f64 vs jax's gammainc on GPU for in-box a (agreement <= 3.7e-15),
3.0-5.8x f32; the earlier 18-54x was the single-Clenshaw mock ceiling,
and the real kernel evaluates both region branches (~2x the mock cost),
exactly as the cost model predicted. Bakes generically (test_bake);
a = 1 gives P = 1 - e^-x as a built-in exactness check.

## 14 — chebax.numpyro, the truncated distribution module (2026-07-31)

`from chebax.numpyro import TruncatedGamma, TruncatedBeta,
TruncatedStudentT`: the opt-in numpyro integration, mirroring
chebax.pytensor's isolation (submodule never imported by the package
__init__, numpyro behind a friendly-ImportError guard, [numpyro] extra,
tested only on the latest CI job). Unlike the pytensor module it has no
import side effects; it just defines three Distribution classes. Born
from the 2026-07-31 examples/notebooks review: the inverse-CDF
truncation pattern F^{-1}(F(lo) + u (F(hi) - F(lo))) was hand-rolled
seven times across truncated_sampling.py, notebook 03 (twice, once in a
class whose docstring said "safe to copy into real models") and the
numpyro example's classes. The classes implement the full contract:
has_rsample with pathwise gradients in every parameter including the
shapes, truncation-normalized log_prob (-inf outside support), cdf/icdf,
dependent interval support, eager bounds check when bounds are concrete.
Domains inherited from the tables and stated in the docstrings: Beta
shapes in [0.1, 10]^2, StudentT df in [0.2, 20], Gamma concentration
any positive (solves through jax's own gammainc). Shape parameters
uniform per call; per-group via chebax.pergroup. Notebook 03 and the
numpyro example now consume the module instead of defining classes;
wiring tests in tests/test_numpyro_module.py (scipy oracle, KS against
own cdf, pathwise vs finite differences, NUTS smoke with a latent df).
## 15 — log-CDF exposures: log_betainc_fn, log_gammainc_fn, log_gammaincc_fn (2026-07-31)

The log_besselk_fn move applied to the CDF recipes, prompted by the
2026-07-31 review: notebook 07's censored-Beta likelihood computes
log(betainc_fn(...)), which underflows when the censored mass is tiny,
while the tables hold the log all along. Each function is the direct
table log on the side where the function is small (betainc lower tail
x <= 1/2, unbounded below, ln I ~ a ln x; gammainc lower tail x <= 8;
gammaincc upper tail x > 8, ln Q ~ -x with no underflow ceiling past
Q's x ~ 700) and log1p(-exp(.)) on the far side, where the error is
the value path's absolute floor divided by the function value
(measured 12-23 eps over I or Q; fine where the value is O(1), not a
deep-tail path - the docstrings say which call serves which tail, e.g.
deep upper Beta tail = log_betainc_fn(b, a, 1 - x)). Measured floors:
direct zones 8e-15 / 1.1e-14 of max(1, |ln|) down to x = 1e-100
(ln ~ -2300) and out to x = 1e4 (ln Q ~ -1e4); shape gradients 3e-15
(betainc) / 7e-13 (gammaincc) vs mp.diff through the traced path.
Traced-only functions like log_besselk_fn; no factories, no new
tables, no generator changes.

## 16 — besseli_ratio, chi2inv, and the endpoint contract (2026-07-31)

Three smaller outcomes of the same review. besseli_ratio(nu, x) =
I_{nu+1}(x)/I_nu(x) via the scaled forms so the e^x factors cancel
(notebook 03 hand-rolled this for the von Mises mean resultant A(kappa)
= besseli_ratio(0, kappa); vMF uses nu = d/2 - 1): nu in [0, 9],
measured 3.1e-15 relative worst on x in [1e-8, 5000], both gradients
against mp.diff, exact 0 at x = 0 by hard select. chi2inv(k, p) =
2 gammaincinv(k/2, p), the chi-squared quantile at real dof notebook 02
assembled by hand; gammaincinv owns the accuracy, scipy is the wiring
oracle. Endpoint audit: the review flagged six-plus defensive
jnp.clip(p, eps, 1-eps) sites in the notebooks before the inverse CDFs;
measured, the inverses were ALREADY exact at the endpoints (betaincinv
0/1, gammaincinv 0/inf, stdtrit -inf/inf, nan propagation, masked zero
gradients), so the clips guard downstream log-densities, not the
inverses. Nothing to fix; the behavior is now locked as a contract in
tests/test_quantiles.py::test_endpoint_contract so it cannot regress.

## 17 — gammaincinv solves on the gammainc recipe (2026-07-31)

The Newton residual inside gammaincinv used jax.scipy.special.gammainc,
which re-runs a whole-array while_loop on EVERY one of the 40 fixed
iterations. For a inside [0.1, 10] the residual now evaluates the
chebax gammainc tables (reconstructed once per call, closed over by the
loop body); jax's gammainc stays as the out-of-box fallback behind one
lax.cond on the scalar a, so the domain is still all of a > 0 and only
the taken branch executes. Measured (experiments/09, RTX 3080 Laptop,
N = 2^20, f64, same solver both sides): 23.4x at a = 0.5, 8.5x at 3.5,
6.3x at 9.9 on p ~ U(1e-6, 1-1e-6); 1.4x on a pure deep-tail workload
(p in [1e-12, 1e-2]), where jax's series branch converges in a few
terms and the recipe's flat ~100 ms cost meets it. Path agreement at
solver tolerance (2.2e-12 relative worst, at a = 0.5 extreme
quantiles). test_quantiles passes unchanged on the recipe path;
consumers inherit for free: chi2inv, the pytensor plugin's GammaIncInv,
chebax.numpyro's TruncatedGamma sampling, truncated_sampling.py. The
custom_jvp's dP/da still uses jax's igamma_grad_a (evaluated once at
the solution, not per iteration); replacing it in-box is a possible
follow-up if gradient-path profiling warrants.

## 18 — matern, the correlation with learnable smoothness (2026-07-31)

`chebax.matern(nu, r, lengthscale=1.0)`: the unit-variance Matern
correlation 2^(1-nu)/Gamma(nu) z^nu K_nu(z), z = sqrt(2 nu) r / l, with
nu traced through besselk_fn - the ingredient for GP kernels with
smoothness learned by gradient, which no mainstream GP stack offers.
Extracted from the review's strongest duplication signal: notebook 01
and examples/matern_learn_nu.py carried the identical function,
including the r = 0 gradient trap (hard select with safe-dummy masking
so log(0) cannot poison gradients) every user would re-derive. Signal
variance is deliberately not a parameter (multiply the result); both
consumers now call the library (notebook re-executed, example recovers
(nu, ell, sig2) to 6 digits unchanged). Measured (test docstring):
values 1.3e-14 relative worst over nu in [0.05, 9.97]; d/dnu 8e-13 vs
mp.diff with an explicit step (lesson recorded: a float64-truncating
reference under mp.diff produces eps/h quantization noise - full-mpf
references only); half-integer closed forms (1/2, 3/2, 5/2) as
independent oracles; dk/dl checked against the exact identity
-(r/l) dk/dr; pergroup-compatible (one nu per group tested). This is
the prerequisite for the GPJax general-Matern play in the adoption map.

## 19 — the JVP's dP/da term on the recipe tables (2026-08-01)

The follow-up increment 17 left gated on measurement: gammaincinv's
custom_jvp computed dP/da with jax's igamma_grad_a, a third looped
whole-array series. In-box it now dispatches (the same lax.cond
pattern) to the a-directional JVP through the gammainc tables - one
polynomial. Measured on the term in isolation (experiments/09, re-run):
15.3x at a = 0.5, 5.3x at 3.5, 4.0x at 9.9, agreement <= 1.4e-14.
Derivative semantics preserved exactly: mixed p-a and second-p
derivatives work on every path, and the pure d2/da2 raise keeps its
message (the cond traces both branches under an a-tangent, so the
fallback's NotImplementedError fires regardless of the box - which
also means the recipe path cannot silently enable in-box d2/da2; doing
that properly is future work if anyone asks). Out-of-box gradients
unchanged via the fallback.

## 20 — matern clamp fix and the GPJax play (2026-08-01)

Integration with GPJax found a real matern bug: their distance helper
returns ~1e-18 (never exactly 0) for identical points, and below
besselk's 1e-6 argument clamp the z^nu prefactor used the RAW z while
K used the clamped one - the product collapsed and the whole gram
DIAGONAL read ~0. Fixed by clamping z consistently (prefactor and K
together), so tiny nonzero distances read as k at the clamp; the
docstring now also states the honest small-nu behavior (1 - k ~
(z/2)^(2 nu) means the correlation has genuinely dropped ~0.25 by
z = 1e-6 at nu = 0.05 - that is the Matern, not the clamp).
Regression test at r = 1e-18 across orders. The play itself:
`general-matern` branch in the local GPJax clone (kernel class with
trainable PositiveReal smoothness, half-integers reproduce
Matern12/32/52 to 1e-12, spectral density generalised to 2 nu-dof
Student t, 5 tests + their 620 stationary tests passing; never
pushed), examples/gpjax_learned_matern.py against RELEASED gpjax
(recovers nu = 2.29 for true 2.5 from a clustered design - smoothness
is identified by short-range behavior, so the demo pairs every base
point with a close neighbor), and drafts/gpjax-general-matern.md
citing the maintainer's own blocker statement in gpjax#482 (no
AI-contribution policy found; checked 2026-08-01).

## 21 — stdtr slice tables: Student-t to nu = 200 (2026-08-01)

The cheap discovery from the box-widening feasibility study
(experiments/11): stdtr/stdtrit evaluate betainc only at (1/2, nu/2)
and (nu/2, 1/2), so two 2-D slices of the same kernel replace the 3-D
tensor on that path entirely. Four tables (two orientations x two raw-b
panels split at 10, shared x-node count so the panel select acts
elementwise), ~13k coefficients vs the ~650k a full widening needs -
CI regen stays affordable (~4 min). The solver core was extracted
series-agnostic (_betaincinv_core) so betaincinv (3-D tensor) and
stdtrit (slices) share the logit-space Newton verbatim; stdtr's range
goes from [0.2, 20] to [0.2, 200] with the old range's accuracy
preserved (existing tests pass unchanged on the new path). Measured:
absolute worst 4.6e-14 at nu = 199.9 (2.4e-15 at nu <= 20), roundtrip
2.8e-15, d/dnu at nu = 150 matching mp.diff to 7 digits. Downstream:
chebax.numpyro's TruncatedStudentT df range is updated; the robit and
t-copula notebooks' nu priors could now widen (not re-executed here).
The general (a, b) box widening stays queued on the CI-policy decision;
this increment removes its most demanded consumer.

## 22 — the f32 bake line: truncate(tol) and float headers (2026-08-01)

The queued f32 work, parts (1) and (2). truncate(tol) on
ChebSeries/PiecewiseCheb/Recipe drops each series' converged tail
(besselk inner degree 79 -> 26 at 1e-7, betainc x 23 -> 9; concrete
coefficients only, a traced tail has no static length). The gate
measurement REFUTED the roofline guess: on the RTX 3080, truncation
speeds f32 GPU evaluation 1.4-2.2x (experiments/12; agreement at each
family's own metric and f32 floor, including besselk's eps32 |ln K|
exponent-assembly growth - and a benchmark lesson re-learned: a
relative agreement metric misfires at oscillatory J's zeros, the
family metric table exists for a reason). bake.jax_module and
bake.xsf_header take truncate_tol; xsf_header takes dtype="float",
emitting f-suffixed literals, float arrays, float locals and a
scalar-templated Clenshaw - an f32 CUDA-shaped kernel body,
compile-verified against the runtime at f32 grade with the
representability caveat stated (K(100) ~ 5e-45 is below the f32
subnormal floor; that is the format, not the kernel). Tables always
source from the f64 instance and round at emission. Option (3),
fpminimax polish, stays dead per the earlier 2% measurement.

## 23 — betainc box widened to [0.1, 100]^2 (2026-08-01)

The general widening, built on experiments/11's measurements: three
additional tensor panels (LOxHI 76x64x60, HIxLO 24x60x20, HIxHI
76x62x60, raw axes per panel, split at 10) alongside the untouched
original tensor as the low-low panel. Runtime dispatch is one
lax.switch on the scalar (a, b) quadrant: only the taken branch
executes, each panel keeps its own degrees, and in-box users pay
NOTHING - the low-low branch builds the identical series as before.
The reflection needs both orientations, so each branch pulls (a, b)
and (b, a) from their own, possibly different, panels. betaincinv,
log_betainc_fn, the pytensor plugin's box and chebax.numpyro's
TruncatedBeta all inherit the [0.1, 100]^2 domain. Measured: values
1.8e-13 absolute worst at the sharp-transition interior (30, 70)
(old box unchanged at ~7e-15), dI/da 8.5e-16 vs mp.diff, inverse
roundtrips 6.7e-15, panel edges ordinary points (each side meets
mpmath at its own bar; a naive two-sided jump test measures the true
dI/da times the probe gap - recorded so it is not re-learned).
Generation is 25 min single-core, so the CI regen policy changes for
the first time: CI regenerates the smallest panel bit-exactly as a
canary (~40 s), the full three-panel byte-for-byte check runs behind
CHEBAX_FULL_REGEN=1 before releases (CLAUDE.md). Tucker storage was
DECLINED for checked-in tables: float SVDs are BLAS-build-dependent
and the bit-for-bit contract is deterministic today precisely because
every fit stage is pure mpmath (generator docstring). Dense cost:
~600k coefficients, ~11 MB module. stdtr keeps its own slice tables
(increment 21); nothing else touches the 3-D path.

## 24 — hyp1f1: Kummer's M on (a, b) in [0.1, 10]^2 (2026-08-01)

The recipe-candidate with a documented-unstable jax incumbent
(jax#21503: series truncated at tol 1e-8, imprecise parameter grads).
The naive kernel ln M fails twice at once (experiments/13): M's
nearest zero sits at x ~ -b/a, which hugs the origin when b << a
(x-degree past 150 at the (10, 0.1) corner), and 1F1's denominator
poles start at b = 0, only 0.1 from the box edge (raw b-degree
137-172). Both die with one move, the generalization of gammainc's
1F1(1; a+1; x) trick: tabulate ln R with M = 1 + (a/b) x R. R's
nearest zero moves out to -2(b+1)/(a+1), its b-poles start at -1
(the same 1.1 clearance the other recipes enjoy), and the runtime
reassembles ln M = log1p((a/b) x exp(ln R)) at full relative
accuracy. Parameter axes go log (a: 26 vs 43 raw; b: 36 vs 50) except
the tail's b, where raw wins (20 vs 44). The tail is the DLMF 13.7.1
log-remainder T with t = XS/x, and XS must sit at 30, not gammainc's
8: at x = 8 the asymptotic series is not yet asymptotic for
|b - a| ~ 10 and t needs degree ~100. Two measurement artifacts worth
recording: T(a, a, t) = 0 identically (M(a, a, x) = e^x), so
diagonal fits report full degree on pure noise (max |T| = 7e-40),
and near-diagonal strips report degrees on values ~ 1e-13 that are
absolutely invisible in ln M ~ x >= 30 - normalize per-fit degree
claims by the GLOBAL scale before believing them. Tables: inner
136x32x44, tail 40x32x25, ~5.4 MB module, 11 min generation; CI
regenerates the tail table bit-exactly as the canary, the full check
runs behind CHEBAX_FULL_REGEN=1 (the increment-23 policy). Measured
(sweep 9x9 pairs x 13 x, normalized by max(1, |ln M|)): values and
log form 3.7e-13 worst, log form at deep x 1.7e-16 relative (M
itself leaves f64 past x ~ 700; log_hyp1f1_fn is the likelihood
form), dM/dx 9.5e-14 vs the exact (a/b) M(a+1, b+1, x) shift oracle,
d/da and d/db 2.9e-15 vs mp.diff. x < 0 is out of scope in v1 (the
Kummer transform e^x M(b-a, b, -x) leaves the box when b - a does).
No performance claim: no race was run (B3 rule); the pitch is
stability and shape gradients, not speed, until measured.

## 25 — gammainc to a = 1000: the Temme zone (2026-08-01)

The queued large-a extension (PROJECT.md's part (2) sketch), and the
cheapest increment yet: ~1.3k coefficients TOTAL. Three zones of
lambda = x/a on v = 10/a in [0.01, 1] (experiments/14): the existing
kernels reappear in scaled coordinates and go flat (lower
D = ln 1F1(1; a+1; a lambda), degrees 18x28; upper tail in the uniform
variable s = (a-1)/x, 11x8), and the transition is Temme's uniform
asymptotic with the O(1) correction kernel T(v, eta) tabulated at
degree 20x7 (26x10 nodes): the WHOLE incomplete-gamma transition for
three decades of a fits in 260 numbers. Runtime assembles
Q = erfc(eta sqrt(a/2))/2 + e^(-a w)/sqrt(2 pi a) T and P from its own
erfc side, so both tails stay relatively accurate; the log forms
factor through erfcx (y^2 = a w exactly), staying finite past the
linear forms' underflow. gammainc.py dispatches on one scalar cond at
a = 10; gammaincinv's recipe residual and the JVP's dP/da follow
(nested conds), so chi2inv serves real dof to 2000 on fixed-degree
polynomials. Fallout fixed along the way: (1) erf_family's lazily
cached series leaked tracers when first built inside a cond branch
(ensure_compile_time_eval now guards it); (2) the quantile solver's
pc < 0.01 rule always took the power-law init, which at a = 500,
p = 1e-12 starts so far into the flat region that df underflows and
bisection cannot finish (measured 8.5e-7 relative); both inits are
underestimates, so v0 = max(asym, WH) picks the right regime
everywhere, including the recorded a = 20 case. Measured (sweep +
tests): P/Q 2.9e-16 absolute, log forms 4.5e-15/1.3e-14 on
max(1, |ln .|), dP/da 2.8e-17, log-form shape grads 8.1e-16 vs
mp.diff, quantile roundtrips at a = 1000 within 2e-13 relative-p to
p = 1e-12. Generation 3.4 s; the bit-for-bit test covers the whole
module, no canary split. The box stops at 1000 because the 40-dps
REFERENCE does (mpmath's gammainc fails near lambda ~ 1 for a >~ 1e5;
small-side subtraction mandatory past ~92 nats, experiments/14), not
the representation.

## 26 — the second review round (2026-08-01)

Nine release blockers and a dozen smaller findings, dispositioned in
`docs/review-2026-08-01.md`. Four themes were worth more than their bug
reports.

**The alias guard was guarding nothing.** Increment 22 added off-grid
validation because on n first-kind points T_{2n−k} samples exactly as −T_k.
But the validation grid was sized from the CANDIDATE (`cand.size + 11`), and
for T₂₉ that lands on 17, the fitting grid itself: the aliased fit
reproduced the samples bit for bit and was certified. `fit(T29, dps=60)`
returned −T₅ with dense sup error 2.0. The fix passes the fitting
resolution in and checks two grids, `chebpts(2n)` and `chebpts(4n)`, chosen
so disjointness is a property of the construction rather than of the
arithmetic that happened to come out: `chebpts(p)` and `chebpts(q)` share a
point only when p/gcd and q/gcd are both odd, which n, 2n, 4n never are
together. The lesson is that a validation step whose parameters are derived
from the thing being validated is not independent evidence.

**x64-off was an untested configuration, not a degraded one.** conftest
enables x64 for the whole suite, so the DEFAULT jax setting was never
exercised, and four inverses returned nan there: the log-space brackets were
literal float64 edges (±745) that mean nothing in float32, where the
exponent range stops near ±88. Everything dtype-dependent (brackets,
iteration counts, residual bars) now derives from
`finfo(canonical dtype)`. There is a dedicated `tests/test_x64_off.py` and a
CI job, because the failure mode here is silence.

**Absolute accuracy is not relative accuracy, and the difference decides
what a quantile may promise.** Three findings were the same mistake in
different places. A CDF built as 1/2 + θ/(2π) + θ·H(θ²) is accurate to a few
eps ABSOLUTELY, so inverting it can only pin min(p, 1−p) to ~32 eps
relative; `vonmises_icdf` was returning angles whose CDF was 37x the target
and calling it converged. `gammaincinv` judged its residual against p even in
the upper half, where the information is in Q. The numpyro truncations
normalized with `fhi − flo` on ordinary CDF values, which is exactly 0 once
both ends round to the same float. The uniform fix is to work on the side
where the quantity is small (Q or ln Q, log-CDF or log-survival, a
logdiffexp instead of a subtraction) and to state a floor where even that
runs out. A false start is recorded here on purpose: the von Mises floor was
first written in terms of the DENSITY at the solution, which made it κ
dependent and got the ordering backwards, keeping a 4% answer at κ=2 while
rejecting a 1e-7 one at κ=50. The limit is a property of the CDF's
representation, not of the distribution.

**Two fixes can conflict where neither agent can see it.** The gammainc
endpoint fix carries the exact density at x = 0 through a custom_jvp whose
primal is the literal 0. Independently, bake was hardened to refuse any
custom_jvp that is not a series evaluation, since inlining one drops its
derivative rule. Together they made gammainc unbakeable. The resolution is
structural rather than name based: a custom_jvp whose call_jaxpr has no
equations and returns a zero literal folds to 0.0, which cannot change a
value, and bake's docstring now states the one thing it does cost, that
one-sided slopes AT an endpoint do not survive into the artifact.

Also in this round: `besseli_ratio` rebuilt on an ascending series with the
prefactors cancelled analytically (it was dividing two independently
underflowing scaled functions, nan in f32, and its origin slope was forced
to 0 instead of 1/(2(ν+1))); dI/dν near ν = 0 rebuilt from the tables'
second ν-derivative after the `I · dlogI/dν` path was found returning the
wrong SIGN at ν = 1e-16; `2*x` overflow above x ≈ 9e307 fixed in besselk and
besseli; `pergroup` no longer evaluates padded empty groups; Recipe,
ChebSeries and PiecewiseCheb freeze after construction, since the factories
hand the same object to every caller; series.py's O(n²) matrix caches
replaced by O(n) recurrences; `log_stdtr` / `log_stdtr_sf` added and
exported.

The integration with increments 23-25 (merged the same day) was not
mechanical, and three resolutions matter. The gammaincinv log-space
solver gained a third recipe mode: the Temme-zone tables serve
a in (10, 1000] through eval_large's own log forms (relatively
accurate on both sides, so the residual seam sits at the mean), and
the review's dtype-parameterized limits replaced both of the older
floor constants; the max(asym, WH) init became unnecessary once the
iteration was on ln P. The incomplete-gamma endpoint slopes apply
AFTER the a = 10 dispatch, where they are exact for both branches,
and the large path inherited the x = inf mask the review added to the
small one. chebax.numpyro's gamma gate widened to the merged recipe
box, which closes the review's TruncatedGamma deep-upper-tail gap for
concentration in (10, 1000].

## 27 — gammainccinv, the survival quantile (2026-08-02)

The review's gamma survival-inverse was a real solver sitting in
`chebax/numpyro.py`, a wiring module. It is now `chebax.gammainccinv`, the
x with Q(a, x) = y, and `chebax.numpyro` calls it like any other quantile.

Both inverses run on one core, `_gammainv_core`, which solves against the
LOG probability and takes its target already in logs: gammaincinv passes
(ln p, log1p(-p)), gammainccinv passes (log1p(-y), ln y), each side handing
in the exact one. That is the whole point of the second function.
`gammaincinv(a, 1 - y)` cannot serve the upper tail, since 1 - y is exactly
1 for every y below eps, and neither can ndtri(1 - y) for the start, which
saturates the same way; the core takes z from the caller for that reason.

The a-box dispatch (small tables to 10, Temme zone to 1000, jax outside)
and the dP/da dispatch behind the JVP were each duplicated three ways by
this point and are now `_dispatch_solve` and `_dPda_dispatch`.

One defect fell out of writing it. The upper-tail asymptotic start needs
x >> a, and once ln(1/y) < lnGamma(a) its fixed point collapses to the
floor. A start far BELOW the root is the expensive miss: the recipe's logs
are finite everywhere, so the safeguard never bisects and Newton crawls one
e-fold per step (jax's hard -inf used to force bisection instead, which is
why the fallback path never showed it). 40 steps stopped 5.8e-10 short at
a = 50, y = 1e-8, inside the 1e-6 residual bar. Both tails now start from
whichever of Wilson-Hilferty and their own asymptotic is higher; overshoot
costs a few steps, undershoot costs the answer.

Measured against mpmath at 50 dps, a in [0.05, 1200], y down to 1e-300:
worst 4.3e-12 relative on the round trip through Q, all of it at small a
where the root sits below the gammainc tables' x = 8 seam and Q comes back
as 1 - P (bar 2e-11). Gradients in both arguments from the implicit-function
rule, agreeing with central differences to 3e-9.

## 28 — the chop's noise floor (2026-08-02)

`_chop` reads its noise floor from the last sixteenth of the
coefficients. That window holds the TOP MODE when the fit is exact at the
current resolution: T20 on 21 points puts c[20] = 1 there, which read as a
floor of 2 and chopped the exact fit to one coefficient, so
`fit(T20, max_deg=20, dps=40)` raised for want of a doubling. Recorded as
not-addressed in the 2026-08-01 review, with the note that retuning would
move the oscillatory cases the tests pin. It did, and the second half of
this entry is that half of the work.

The floor now applies only when the window sits under 1e-8 of the function
scale. A genuine sampling plateau is decades below it (the settled test
wants 1e-10 relative before it will believe one); anything nearer is
signal, and the requested tol governs there instead.

That alone regressed two pinned cases, both through `prev_rel`. The
settled test also accepts a plateau that stopped improving under node
doubling, and it was comparing chop LEVELS across rounds. A level is a
measurement only when the window held noise; where the window holds signal
it is just tol·scale, and feeding that into the comparison made the next
round look converged (the runge segments settled a doubling early, 6.5e-13
against a 1e-15 bar) or, going the other way, made T34's float64 fit stop
at the first plateau it saw (degree 47 rather than 34, accurate but
padded). `_chop` now returns the measured plateau separately, inf when
there is none, and only that crosses rounds.

Nothing in the recipes moves: the generators fit at fixed measured degrees
through `_gen_common.dct`, never through the adaptive path.

## 29 — the Temme tables to a = inf (2026-08-02)

The large-a box stopped at a = 1000. That was the 40-dps REFERENCE giving
out, not the representation: mpmath's `gammainc` stops converging near
lambda ~ 1 around a ~ 1e5, and increment 25 said so at the time.

v = 10/a = 0 is a = inf, and it is a regular point of all three kernels:
D -> -ln(1 - lambda) and U -> -ln(1 - s) (the geometric sums their series
become once (a+1)_n / a^n -> 1), T -> Temme's leading coefficient. So the
v-axis simply extends to [0, 1] and the box loses its upper end.
experiments/16 measures what that costs: nothing in the Temme table
(eta 20, v 7, unchanged), nothing in the upper zone (s 11, v 8), and one
coefficient in the lower zone (v 28 -> 29). First-kind nodes never sample
v = 0 itself; the smallest is a = 2.0e4 at 36 nodes.

The reference is the piece that had to be built. Q now comes from
Legendre's continued fraction and P from the 1F1 series, each used only
where it converges, with the erfc split still taken on its small side
(the increment 25 rule). It agrees with mpmath to 1e-38 wherever mpmath
still converges.

Measured on the regenerated tables: log forms 5.0e-14 relative on
max(1, |ln|) over a from 1e4 to 1e8, which is the cancelling bracket
a ln x - x - lnGamma(a+1) at that size rather than the tables; quantile
round trips 4.5e-12 at a up to 1e6. Values themselves underflow out
there, as they must, and stay at the CDF absolute contract where they do
not.

AHI moved out of the table module: a ceiling of inf is a policy of the
dispatch, not an axis of the tables, and the emitter has no literal for
it. It lives in gammainc_large.py now, which is where the dispatch reads
it from.

This closes the last of the three gaps the 2026-08-01 review left open.
A TruncatedGamma at concentration 1e5 on a deep upper-tail interval is
1.3e-11 relative; it used to fall back to jax and return +inf.

## 30 — domain-limited besselj (2026-08-02)

`besselj(v, domain=(lo, hi))`, the narrow-domain build PROJECT.md queued
after experiments/04 measured a single-region evaluation at 2.5x (f64) /
3.0x (f32) against the full three-region select.

The instance keeps only the regions covering [lo, hi] and `__call__`
picks its shape from that static tuple: one region means one polynomial
and no selects, two means one select instead of two. Nothing about the
tables or their accuracy changes, and inside the domain a trimmed
instance returns bit-identical numbers to the full one (asserted in
tests, not approximated). Outside it the answer is nan: the regions that
would have answered are the ones that were dropped, and extrapolating a
region's polynomial past its seam is silently wrong rather than loudly
so. The guard masks the OUTPUT, the branch inputs staying clamped as
before, so no gradient reaches a lane it should not.

experiments/04 now times the SHIPPED instance rather than only the
private inner evaluation. On the 3080 at N = 2^24, v = 2.5, x ~ U(0, 8):
full 55.0 ms, domain= 21.7 ms (2.54x) in f64; 1.33 ms vs 0.51 ms (2.59x)
in f32. The nan guard is the difference between that and the raw inner
kernel, about 9% of the f64 narrow time and nothing measurable in f32.

The queue named the Matern demo as first customer. It is not one:
`matern` runs on besselk, a different recipe with its own two regions.
besselk could take the same option, but a Matern kernel's
z = sqrt(2 nu) r / lengthscale window is data-dependent rather than
static, so it would have to be declared by the caller like this one is.

## 31 — the third review round (2026-08-02)

Disposition in `docs/review-2026-08-02.md`. Three findings are worth
carrying here because they say something about the library rather than
about one function.

**A validation grid can have a structural blind spot.** The adaptive fit
checked its candidate on the n, 2n and 4n Chebyshev grids, and T_{16n} is
exactly +1 on all three, so `fit(T272)` returned the constant 1 with sup
error 2. The 2026-08-01 round had closed this with an argument bounding
the SMALLEST common alias, which is true and irrelevant: the family above
the bound is infinite. Validation now uses points that are the nodes of no
Chebyshev grid, `cos(pi (i + PHI)/m)` with PHI irrational, where an alias
would need `2 q PHI` in Z. The lesson is in the docstring now: a finite
point set detects gross errors and proves nothing, and saying so is what
keeps the next argument honest.

**An integer parameter is not a differentiable type.** `stdtr(4, t)` is
how anyone writes it, and jax hands a custom_jvp rule float0 tangents for
integer primals, which the rules' arithmetic cannot touch. Four public
functions failed outright. The fix is one policy rather than seven
patches: the rule lives on a private name, the public entry point promotes
its arguments first (`_common.float_params`). Anything inside the rule is
too late, because by then the tangent exists.

**A domain that lives in an implementation panel drifts.** chebax.numpyro
imported betainc's [0.1, 10] panel constant and kept rejecting
TruncatedBeta(20, 3) long after the tables reached 100, while its own
docstring advertised the wide box. `chebax/domains.py` reads each family's
box off the table module that governs it, and the recipes, the numpyro
constraints and the pytensor gate all consume it.

Also in this round: bake folds only chebax's own endpoint-slope rule, by
name and defining module and after evaluating the primal, and discloses
the dropped slope in the artifact; betainc and hyp1f1 carry their exact
endpoint densities; stdtrit's shape gradient goes through the logs, since
the ratio is finite where both its parts underflow; besselk_fn takes
scaled=True so pytensor's Kve stops forming exp(log K + x); erfc and dead
equations no longer stop the emitter, and every public Recipe class is
covered; ChebSeries and PiecewiseCheb refuse deletion, evaluate on extreme
finite domains through midpoint/half-width, and validate truncate's
tolerance; matern, the spherical Bessels and scaled besseli_dnu return
their limits at +inf, and a negative Matern distance is nan rather than 1.
## 32 — the forward CDFs on the pytensor path (2026-08-03)

`import chebax.pytensor` lowered the inverse CDFs and gave BetaInc the a/b
gradients pytensor has never had, and stopped there. The forward pair was
unregistered and BetaInc's VALUES stayed jax's own on every domain, which
is half a registration: `pm.logcdf` of Gamma and ChiSquared emits
`GammaInc`, Poisson emits `GammaIncC`, and Beta, StudentT and
NegativeBinomial all emit `BetaInc`, so every censored or truncated model
in that list kept jax's while_loop and the betainc and igamma speedups
reached no PyMC user at all. Three things came out of closing it.

**A gradient without its value is half a win.** The a/b derivatives were
the interesting part, so the values were left alone and the omission read
as deliberate ("values stay jax's own betainc on every domain" was in the
docstring for a month). It cost nothing to check and would have cost the
whole speedup to ship.

**cond, never select.** A forward CDF has to answer everywhere, so
out-of-box parameters fall back to jax instead of returning nan the way an
inverse CDF does. `lax.select` evaluates both sides, which puts the loop
back on every lane and hands back exactly what the registration removes.
`lax.cond` executes one branch, and it is available because the dispatch
already requires a scalar parameter, so the predicate is scalar too. The
cost is measured, not assumed (experiments/19).

**A lowering must not move the graph's dtypes.** The tables are f64, so
the recipes answer f64 whatever they are handed, and a cond whose two
branches disagree does not trace AT ALL: the gamma registration raised on
any floatX="float32" model. Widening to f64 would have made it trace and
silently changed the dtype of the user's sample instead. Both cond'd
registrations now answer in the dtype plain jax would have given.

Also here: the in-box branch takes a clipped parameter, since vmap turns a
cond with a batched predicate into a select that evaluates both sides,
where a nan would poison the gradient through the select.


## 33. Tweedie: Wright's function at beta = 0 (2026-10-26)

`tweedie_logpdf(p, mu, phi, y)`, `log_wright_bessel0(rho, z)` and
`chebax.numpyro.Tweedie`. For 1 < p < 2 the Tweedie density is
exp(-lambda - y/theta) W(rho, z) / y with W = Phi(rho, 0; z), so it needs one
slice of the three-argument function experiments/17 sized at 3e5
coefficients: here a 32 x 64 table and a direct sum (experiments/20).

**Split where the sum stops being a sum.** The series' terms peak at
n0 = Z/(1+rho) with variance Z/(1+rho)^2. Below a variance of ~2 the sum is a
few discrete terms and differs from its smooth saddle form by a ripple of
size exp(-2 pi^2 variance), one period per term, which no degree resolves:
a split at fixed n0 saturated 64 nodes at rho = 19. Small rho has the
opposite problem (the large-Z expansion is poor at Z ~ 2), hence
Z1(rho) = 12 + 2 (1+rho)^2, smooth in rho as the table needs. Below it the
series itself, 64 terms, lgamma(rho n) once per call.

**The saddlepoint approximation is the leading term.** In the tail
Z = y^(2-p) / (phi (2-p)(p-1)), independent of mu, and the log-density is
exactly -d(y, mu)/(2 phi) - ln(2 pi phi y^p)/2 + T/Z: the classical
saddlepoint approximation with d the unit deviance, plus the tabulated
correction. Written that way the cancelling Z-sized terms never form, so a
concentrated density (Z ~ 1e4) keeps its digits. The table's s -> 0 limit
is c1(rho) = -(2 rho + 1)(rho + 2)/(24 rho) in closed form, which tests the
end no reference reaches (8.9e-16).

**Reverse mode through an unrolled sum.** The direct sum's rho gradient
took 2.7 s against 20 ms for the value (N = 2^20, CPU): XLA made one
reduction over every point per term. Both derivatives of a logsumexp are
averages over its own weights, so a custom_jvp accumulates them in the same
pass: 124 ms, and d/dp got more accurate (1.1e-11 to 3.7e-12).

**pow, not exp of log.** exp(e ln z) carries eps |ln z| into Z, 5e-14 at
z = 1e300; XLA's pow is ulp-accurate there (2e-16).

**Check the box on the parameter as given.** float32 rounds p = 1.05 to
1.04999995, whose rho is 19.000011, past RHO_MAX: nan at the advertised edge
until the check moved from rho to p.

**scipy is not a reference.** At p = 1.2 scipy's log_wright_bessel is off by
up to 4.7e-6 in a band around Z ~ 15-30, which covered 14% of a sample drawn
from the law (experiments/21). mpmath, as everywhere here.

Measured: ln W to 7.7e-16, the density to 2.4e-13 (lgamma(rho n) rounding
near the split at p = 1.05), gradients to 3.7e-12 (d/dp, where
drho/dp = -400); float32 7.4e-6. Values 89-130x scipy on one CPU core and
254-331x on the GPU; NUTS recovers the power (p = 1.41 +- 0.05 at 1.35 on
1,500 draws).

## 34. Tweedie: what chebtorch's port found (2026-10-28)

chebtorch's Tweedie fixtures reach the region just below the split at small
p, which the grid here missed, and the stated floors were too low there:
float64 density 1.1e-12 (3.3e-13 stated), d/dp 4.7e-11 (3.6e-12), float32
density 4.5e-4 at p < 1.1 (7.4e-6). The cause is the one already named,
lgamma(rho n) rounding in the direct sum; the grid did not sample its worst.
Those 578 points are chebax's tests now (MORE in test_tweedie.py, the same
random stream), float32 checked by band of p against float64 at the same
float32 inputs.

The port also kept a third deviance form, mu^a (expm1(a x) - a delta) /
(a (a-1)) with x = log1p(delta), for 1/4 < |delta| and |ln(y/mu)| <= 1.
The far form's terms cancel to O(delta^2) there, 32 eps / (a (1-a)) at
|delta| = 1/4; the new one cancels only first-order terms, ~9 eps / (1-a).
float64 does not move. In float32 at p >= 1.3, d/dln phi goes from 8.5e-6
to 5.3e-6 and d/dp from 4.8e-5 to 2.7e-5, but the density stays at 5.6e-6:
XLA's float32 lgamma dominates it (chebtorch, on torch's lgamma, reaches
1.5e-6). The deviance's hand-written partials in mu and y, and _over's in
phi, are checked against finite differences in each form.

The middle form as first taken from chebtorch used x = log1p(delta), and XLA's
float64 log1p is up to 128 ulp off for delta in (-0.63, -0.25) (jax 0.11 on
CPU; torch's and numpy's are within an ulp): the form was 1.5e-12 off at
a = 0.95 below the mean, against 8.9e-14 for the three-term form it replaced,
and the density 1.0e-12 against 3.7e-13 there at p = 1.05. The density tests
did not move, since their worst is the direct sum's, below the split. Below
the mean it is log(y/mu) now (the ratio is within [1/e, 3/4]): 7.6e-14, and
3.9e-13 for that density. A test of the half-deviance and its y-partial
against mpmath in each form catches it directly.

## 35. gammainc with a per element (2026-11-01)

Every table here collapses its parameter axis once per call. A Gamma
likelihood with a shape per observation therefore paid that collapse per
element through vmap: about 4,300 coefficients per element, counting both
sides of the a = 10 dispatch. `gammainc_array`, `gammaincc_array` and
their logs take a per element and evaluate one small 2-D patch instead
(`gammainc_array_gen.py`, `gammainc_array.py`).

The pieces, each measured on its own first (generator docstring):

- The parameter is u = ln(1 + a) up to a = 10. 1F1(1; a+1; .)'s pole at
  a = -1 goes to infinity, and the a-axis needs 37 coefficients globally
  instead of 53 in linear a.
- a < 1 splits at x = a + 1, with x scaled by a + 1 on each side, so the
  transition sits at a fixed coordinate. This also removes the uniform
  recipe's x = 8 seam, below which ln Q comes from 1 - P down to Q ~ 1e-6;
  here the far side never sees Q below 0.024.
- a >= 1 keeps gammainc_large's three zones and its Temme function,
  refitted in w = 1/a down to a = 1. The a-only prefactor (lnGamma(a+1),
  or its Stirling remainder above a = 1) is folded into each table, so no
  lgamma runs.
- Each zone is a tensor grid, halved until every patch's coefficients
  outside 12 x 12 sum to at most 1e-15: 65 patches, 9,360 coefficients.
  float32 reads the leading 7 x 7 of each.

Float traps, the last three found in review:

- XLA's float64 log1p (increment 34) again, through phi = d - log1p(d) at
  d ~ -0.4: 5.5e-14 in ln P. The uniform large-a path has the same exposure
  (w_eta and temme_log_pq's far side; its 6e-14 bars hold it). Here it is
  log(1 + x) below x = -1/4. The textbook fix, x log(u)/(u - 1), is undone
  by XLA's simplifier (u - 1 becomes x): 1.4e-13 in float64, 1.7e-4 in
  float32.
- w_eta switches to its series at |d| = 1e-3, a float64 choice: in float32
  the direct form loses about 4 digits next to lambda = 1. phi comes from
  z = d/(2 + d) instead (log1p(d) = 2 atanh z), with no cancellation in
  either dtype.
- An unused branch must not divide by x. A select zeroes its cotangent, but
  a partial that overflows turns that 0 into nan: jax.grad of gammainc_array
  in x at (1, 1e-20) in float32 was nan. Each branch now gets x masked off
  its own lanes.
- Differentiating -a phi as a product cancels two terms of size x/a in the
  upper tail. d ln Q/da at (2.5, 1e8) was 13.6 for 17.7 in float32, and
  4e83 at x = 1e100 in float64. Written as -(x - a) + a ln lambda, d/da
  stays at ln lambda. ln lambda's slope is written out (through the ratio,
  x/a^2 and 1/lambda leave the float range at its ends).
- x/a underflows deep in the lower tail: ln P at (1000, 1e-36) was -inf in
  float32. ln lambda is log(x/a) only where that ratio is a normal number,
  and ln x - ln a below. P = exp(ln P) gets slope 0 where P is 0, since
  forward mode otherwise multiplies 0 by an overflowed a/x.

Measured against mpmath on 2,300 points: float64 ln P 8.9e-16 and ln Q
8.4e-15. The ln Q floor is the far side of a < 1, x <= a + 1, where
Q = 1 - P and Q gets down to 0.024; elsewhere it is 7.8e-16. Gradients are
within 2e-13, and float32 is 4.5e-7 for ln P and 2.9e-6 for ln Q. In the
far tails (x to 1e300, x/a underflowing) values and slopes in both modes
are within 2.2e-16 in float64 and 1.0e-7 in float32.

Speed (`experiments/22`, x drawn from Gamma(a)): the cost does not depend
on how a is spread. float64 runs at 12.3-12.6 ns per value and 23 ns per
value_and_grad on the GPU, and 0.67-0.70 / 0.98-1.06 us on one CPU core.
That is 2.3-18x jax's gammainc on GPU values and 2.8-23x on value_and_grad,
and 54x vmap of gammainc_scalar on value_and_grad. float32 on the GPU is
1.1 / 1.7 ns, 4.5-66x and 6.6-95x jax's. jax's own gammainc is off by 1.4e-11 (float64)
and 7e-3 (float32) near x ~ a ~ 1e4.

## 36. One naming convention (2026-11-01)

The names had grown per call pattern: `besselk(nu)` was a cached factory,
`besselk_fn(nu, x)` its traced twin, and #35 added `gammainc_elem(a, x)`
for a per element. None of the suffixes says what it means, and the
gammainc entry points sat in three files.

Every function now has one name that takes any parameter shape,
`name(params, x)`, and one name per path:

- `_scalar`: one traceable value per parameter for the whole call (the
  old `_fn`);
- `_array`: a value per element, broadcast with x (gammainc only);
- `_frozen`: a cached callable for concrete parameters, the Recipe that
  bake takes (the old one-argument factory).

The plain name picks `_scalar` or `_array` from the parameters' shapes,
which are static, so the choice costs nothing at run time. Families without
`_array` refuse an array of parameters and name `pergroup`; families whose
only path is `_frozen` (besselj, bessely, spherical) call it, and need a
concrete order. Order derivatives follow suit: `besselk_dnu(nu, x)` and
`besselk_dnu_frozen(nu)`.

gammainc's entry points are all in `gammainc.py` now, and each path's
implementation is in its own module (`gammainc_scalar.py`,
`gammainc_array.py`, `gammainc_frozen.py`, with the a <= 10 and a > 10
numerics the scalar and frozen paths share in `gammainc_small.py` and
`gammainc_large.py`). The entry points import a path's module when it is
first called, so calling `gammainc_scalar` loads 104 KB of tables and
`gammainc_array` its own 231 KB, never both (`test_lazy_imports`).

The 0.4 spellings still work for one release: the eleven `*_fn` names warn
through `chebax.__getattr__` and return the `_scalar` function, and the
one-argument factory calls warn and return the `_frozen` object (a bool in
x's place is besseli's 0.4 positional `scaled`). `gammainc_elem` was never
released and has no alias.

## 37. Gamma quantiles with a per element (2026-11-02)

gammaincinv, gammainccinv and chi2inv took one a per call, and a shape
per element went through vmap of that path, whose lax.conds on a then
run every branch for every element, jax's gammainc among them (a
6-element call did not finish compiling in 12 minutes on the CPU). They
now have the two paths gammainc has: `_scalar` (one traced a, any a > 0)
and `_array` (a per element, a >= 0.1), and the plain names pick one from
the shape of a. Both run one Newton solve, `gammaincinv_kernel.py`, on
their own ln P and ln Q: the scalar path on the tables its a-box picks,
the array path on `gammainc_array`'s patches.

The solve, changed for both paths:

- It is in t = ln(x/a) instead of v = ln x. v carries eps ln a of
  absolute error, which in float32 is 1e-6 relative to x at a = 1e5: the
  scalar path returned nan in the deep tails from there and for every p
  from a = 1e7 to 1e10. Near the mode x = a + a expm1(t), which reaches
  every float (a e^t steps two floats at a time), and each Newton step
  starts from the t the rounded x stands for, so the slope and the
  residual describe one point; at a = 5e14 in float32 one float step of x
  is 1.5 standard deviations.
- The lower tail starts from the larger of Wilson-Hilferty and the
  asymptotic x^a / Gamma(a+1) = p, which is never above the root since
  P <= x^a / Gamma(a+1). Below p = 0.01 the asymptotic alone started at
  x ~ a/e at a = 1e17, where ln P is -1e16 and the slope has no digits.
- A solve is accepted when its residual is within the bar or changes
  sign across the next float, which makes x the root to within one float
  step. From a ~ 1e20 in float64 one float step moves ln P by more than
  the bar.
- Everything after the loop (x, both neighbours, and the ceiling for
  gammainccinv) is one stacked call, so the program holds the residual
  twice instead of five times. The floor needs no residual: at the smallest normal x,
  P = x^a / Gamma(a+1) to rounding.

The scalar path keeps judging its bar on the side its tables compute
directly. Below a = 10 they give Q as 1 - P under x = 8, which in float32
is up to 1e-2 off once Q is small; judged on Q, those solves came back
nan. The array path computes Q there directly. Its small-a mode skips the
near-mode x and the Stirling density, which cost 28% of its GPU time; it
is 4-5% slower than the v = ln x solve below a = 10 and 7-11% above
(`experiments/09`).

The array path:

- Halley steps, at the price of a Newton step: the curvature is exact
  and free, since with g = ln(x pdf) either side's log S has
  S'' = S' (g' - S') and g' = -a expm1(t), so newton_bisect takes the
  slope df (1 - f f''/(2 df^2)). 8 steps in float64 and 6 in float32,
  twice the most any of 1e5 random lanes needed for the tail probability
  (a to 1e30, p down to the smallest normal); Newton needed 7 and 4 by a
  looser measure. The scalar path stays on Newton: its 40 steps are
  fixed, so Halley would only add work per step (19% on the GPU).
- A tighter bar, 256 eps |ln target|: both logs are accurate there, so a
  converged solve sits at the rounding of its target's log, and one that
  stopped short returns nan.
- Derivatives from the implicit function theorem on the same ln P and
  ln Q, by AD, in both modes and to second order, d2x/da2 included.
  1/pdf takes its value from exp(ln p) |L_x| and its derivatives from the
  closed-form density, whose x-derivative (a - 1)/x - 1 has no
  cancellation: differentiating ln p + ln L_x puts two terms of size 1/x
  against each other, and d2x/dp2 at a = 1, p = 1e-20 was 98304 for 1
  (-3 at p = 1e-7 in float32).
- Above a = 1e3 both derivatives are taken at the root found again,
  unrounded. L_x at the float x is off by its change over one float
  step, eps sqrt(a) (1 + |z|): 0.3% at a = 1e30, and at a = 1e35,
  p = 1e-20 x rounds to a, nine standard deviations from the root, where
  one Halley step from x gave dx/dp 2e17 times too small. `_log_pq` takes
  d = x/a - 1 exactly instead of x, and Halley steps in t from the
  Wilson-Hilferty point (O(z^3/a) off there) converge in at most 3 on 2e4
  random lanes; the result is nan if they did not. dx/da there is
  (1 + d) - (dL/d ln a)/(dL/dd): slopes in a and x carry tangents of size
  1/a, flushed to zero from a ~ 2e307. With d exact, a phi = y^2: phi
  alone is subnormal for |d| < 2e-154.
- Every finite a. Above a = 1e36 (1e18 in float32) every p in (0, 1) has
  |z|/sqrt(a) < eps/4, so the quantile is a itself; the solve there met
  1/a subnormal from 4.5e307, which jit flushed inside (x - a)/a. The
  lower-tail start takes lnGamma(a + 1)/a - ln a by Stirling above
  a = 100 (lnGamma overflowed, and gammaincinv_array(1e306, 0.3) was
  nan), the bracket's top is at least x = a, and sqrt(2 pi a) is
  sqrt(2 pi) sqrt(a) here and in the scalar path's Temme solve, whose
  dx/dp was 0 from a = 2.9e307 (and from 9e307 through 2/a, flushed).
- The masks sit in the tangent coefficients: a nan or 0 placed on the
  tangent is not linear in it, and reverse mode returned 0 where x is nan.

XLA on the CPU (jax 0.11):

- On AVX-512 CPUs, XLA hoisting the a-only work of `_log_pq` out of the
  Newton loop (while-loop invariant code motion) gave wrong roots, 8e-9
  off at a = 30, p = 0.956 among 128 lanes and depending on the array
  length, and nan on other lanes. Capping the ISA at AVX2, the old CPU
  fusion emitter, or disabling that pass each avoid it; the array path
  hands `_log_pq` a + 0 x, which XLA cannot fold, so nothing is hoisted.
  gammainc_array alone and inside loops agrees between AVX-512 and AVX2.
- gammainc_array's values take ~23 s to compile with the new CPU fusion
  emitter, 0.5 s with the old one (`--xla_cpu_use_fusion_emitters=false`):
  LLVM spends it on one elementwise fusion. Which fusions trip it depends
  on their shape, not on one operation: its gradients compile in 0.8 s,
  the array quantile's values in ~1 s and its value_and_grad in ~42 s
  (26 s of it without the large-a root; 93-99 s on one pinned core).

Accuracy (`tests/test_gammaincinv_array.py`): the float64 round trip
against mpmath is 6.2e-13 for a <= 1000 (the gammainc tables' own floor
there; the scalar path gives 9.3e-13 at the same point) and 2.3e-12 at
a = 1e6, p = 1e-300 (where one float step of x moves P by 8e-12).
Gradients 7.9e-15, d2x/dp2 1.4e-13, d2x/da2 4.6e-12; float32 3.3e-5.
Above a = 1e3 against mpmath dx/dp is within 2.3e-13 and dx/da 3.6e-16,
and against the scalar path's Temme solve to a = 1e35, 1.1e-13 and
2.2e-16. The exponential case a = 1 is exact to 2.4e-14 through
d2x/dp2 (1.9e-6 in float32).

Speed (`experiments/23`, p uniform, against TFP's igammainv, which is
what pytensor's JAX backend falls back to for a per element; TFP 0.25
does not import under jax 0.11, tfp-nightly does): the array path's
values cost the same on every layout of a, and TFP's cost follows the
spread. GPU float64 0.93x / 2.3x / 7.3x on values for a = 2.5 each /
spread / wide, float32 2.6x / 7.0x / 34x; one CPU core float64
0.61x / 1.6x / 5.2x, float32 0.19x / 0.50x / 2.8x. value_and_grad costs
more once some a passes 1e3, where the root is found again (GPU float64
358 ns on "wide" against 229 ns), and is 4.8x TFP there. TFP's round trip
is 2.0e-11 in float64 and 1.1e-2 in float32 on the wide layout, against
1.0e-14 and 8.4e-6 here. When a is one value the scalar path is faster
(GPU float32 2.7 vs 13.3 ns), and the plain name runs it.

## 38. One series per point in the scalar Gamma quantile (2026-11-02)

The scalar path's residual evaluated both regions on every point and
selected, in every one of its 40 Newton steps: the inner and tail series
below a = 10, and erfcx and erfc at both y+ and y- in the Temme zone.

- Below a = 10 the solve now runs `gammainc_small.eval_logs_one_series`:
  `series.chebval_either` picks each point's coefficients per Clenshaw
  step, so a point runs only the series of its side of x = 8, and one log.
  The same operations as the select, so the quantiles are unchanged (bit
  for bit on the GPU). It has no custom rule: under reverse mode the
  coefficient cotangent runs at full length for both series, which made
  the a-gradient of gammainc_scalar 1.6-2.4x slower on the CPU in a
  prototype. So the forward functions keep both series and select.
- `temme_log_pq` forms the side y points to through one erfcx(|y|) and the
  other as log1p(-exp(own side)), as gammainc_array does. Quantiles move
  by at most 3.8e-16, log Q's worst on a dense sweep against mpmath from
  7.7e-15 to 7.8e-15.

`experiments/09`, GPU float64, N = 2^20: the recipe column 128 -> 97 ms
below a = 10 and 417-437 -> 259-278 ms above. On one CPU at N = 2^18,
6-14% faster, and the value_and_grad compiles in 7.5-7.9 s instead of 9.3.

Tried and dropped:

- Dropping table coefficients whose sum is below 2e-16 over the whole
  parameter box (26 of 32 inner, 22 of 26 tail): worst values unchanged,
  but dP/dx multiplies coefficient k by k^2, and its error went from
  6.5e-13 to 2.6e-11 where the density is small against P (bar 1.5e-12).
  Keeping dP/dx at its floor leaves 29-30 of 32, a few percent of speed.
- One erfc per point in the Temme zone's P and Q (not logs): no
  measurable gain.

## 39. erfcx from 32 panels in t = 3/(3 + x) (2026-11-01)

erfcx's [0, 6] series, its (6/x)^2 tail and the select between them are gone.
x >= 0 maps onto t = K/(K + x) in (0, 1] (Schonfelder, Math. Comp. 32, 1978),
erfcx(x) = t h(t), with h smooth on [0, 1] and h(0) = 1/(K sqrt(pi)). One
series in t needs 26 terms above 2^-56 min h at K = 3 (25-26 for K in
[2.5, 5]); 32 equal t-panels need 8 each, picked by floor(32 t), with no
select on x. The table keeps 9 per panel: the ninth takes the derivative's
worst case from 6.8e-15 to 7.0e-16, at the panel edges where a truncated
series' derivative errs most. ERFCX_PL keeps each coefficient's rounding
residue for compiled kernels that carry the leading coefficient past f64.

The single series came first, for pytensor's Numba backend, where a
branch-free kernel vectorizes. Inside PyMC's fused loops it does not (the
gathers and the gradient's scatter-adds share the loop), and one element at
a time 26 dependent steps lost to scipy's erfcx and glibc's erfc; 8 steps
per panel do not. The panels are also faster here than the single series:
one gather per coefficient (rows were 2x slower on XLA's CPU) and 8 steps.

The reflection 2 e^{x^2} - erfcx(-x) put up to x^2 eps into e^{x^2}: 5.7e-14
at x = -22.75, with any series. x^2 is split exactly now, x_hi by a bit mask
(no fma, so no contraction can move the split). lo reaches 2^-25 x^2, so
1 + lo is not e^lo (that cost 8e-14 at x = -4.3 before it was caught), and
e^hi e^lo is no good either: under jit XLA folds a product of exps into the
exp of the sum, which rounds x^2 again, and the eager tests did not see it.
It is e^hi (1 + expm1(lo)), expm1 by three terms of its series (exact to
1e-20 for |lo| <= 2.1e-5; XLA's expm1 cost 2 ms per 2^20 points on the CPU),
and the value tests run jitted too. A custom JVP gives d/dx e^{x^2} =
2x e^{x^2}: through the split, reverse mode met inf * 0 at overflowing x
whose lo is exactly 0 (x = -27) and returned nan where the derivative is
-inf.

Against mpmath on 8702 points: values 4.4e-16 / 5.8e-16 (x < 0 / x >= 0;
the two tables 5.7e-14 / 4.8e-16), derivative 5.9e-16 / 7.0e-16 (5.7e-14 /
1.4e-14), the same on CPU and GPU and on jax 0.5.3. Against the two tables,
values / gradient take 0.52x / 0.46x the time on one CPU thread and 0.42x /
0.52x on the GPU (experiments/27).

## 40. stdtr's orientation B tables keep 25 of 68 x-coefficients (2026-11-04)

The four stdtr slice tables were all fitted and stored on 68 x-nodes,
although experiments/03 measured orientation B's x-tail at 18: the
runtime selects a panel elementwise on the coefficient vectors, so the
two tables of an orientation share a length, and 68 was the simplest way
to give all four one. Only the A pair needs it (its hi panel's tail is
53). B now keeps 25 rows, the cut at which the dropped tail is under
1e-15 in the value, the x-derivatives to second order (sums of k^2 |c_k|
and k^2 (k^2 - 1)/3 |c_k|) and d/db over both panels (experiments/24:
19, 22, 25 and 18). The fit still runs on 68 nodes, so the kept
coefficients are unchanged.

Every stdtr call evaluates both orientations' series per element (the
central and tail branches are selected, and stdtrit's solve uses both),
so the series work per element goes from 136 to 93 Clenshaw steps.
experiments/06, GPU, N = 2^24: stdtr 87.4 -> 71.5 ms in float64 and
88.6 -> 72.0 in float32 (1.22-1.23x), agreement with jax unchanged. The
test bars hold at their measured worst cases.
