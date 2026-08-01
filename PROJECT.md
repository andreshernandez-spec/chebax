# chebax — differentiable Chebyshev approximants for special functions

**Goal:** a standalone Python library that turns smooth functions on real domains into
fixed-degree Chebyshev approximants that are branchless, differentiable (including with
respect to parameters like the Bessel order), GPU-shaped, and reproducible — with
prebuilt "recipes" giving out-of-the-box coverage of common special functions, and a
`bake` step that emits self-contained artifacts for upstream targets (pure-JAX modules,
xsf-style C++ headers).

**No novelty claim.** Chebyshev series for special functions is how the field has worked
since Clenshaw's 1962 NPL tables; Cephes and Boost.Math are full of them. The claim is
availability and form: differentiable, f32-aware, parameter-generic, regenerable from a
checked-in generator, and importable by JAX users with `pip install`.

**Scope boundary, stated once:** function parameters (the Bessel order ν, the betainc
a,b, …) are *uniform per call* — static at trace time, or a call-time scalar. Per-element
parameter arrays are out of scope; that regime belongs to the branchy CPU-style kernels
(bessel Track A / xsf). Finite domains are covered by segmented Chebyshev; infinite tails
need a per-family asymptotic recipe (phase/modulus for oscillatory functions) — the
machinery does not remove that analysis and does not claim to.

**Name:** `chebax` — confirmed 2026-07-29. PyPI and GitHub free (nearest neighbors
chebpy, pychebfun, ChebTools). The spec-* alternatives were rejected: "spec" reads as
spectroscopy in scientific Python (jaxspec, specutils). Reserving the PyPI name is a
publishing action and stays Andres's.

**Status:** live. Public at https://github.com/andreshernandez-spec/chebax;
0.2.0 released on PyPI 2026-07-31 (0.3.0 pending: pergroup, the gammainc
recipe, chebax.numpyro, matern, the log-CDF forms, the quantile-solver
rewire; nineteen increments in `docs/increments.md`). Open: betainc box
widening (feasibility unmeasured), the GPJax Matern play, gammainc a > 10,
upstream outreach (local notes in `drafts/`). Parent evidence lives in
`../bessel/` (Track B); this project generalizes B1 into a library.
**Owner:** Andres
**Last verified:** 2026-08-01, experiments run locally (see `results/`); upstream
references checked against the clones pinned in `../bessel/PROJECT.md`
(xsf @ `ac62f926`, jax @ `f8e48d5`).

---

## 1. Why this library

- `J_v`/`Y_v`/`K_v` at real order are among the most-requested missing functions in JAX
  ([jax#11002], [jax#2466], [jax#27088] is the best map of the gap). JAX core wants
  branchless fixed-degree implementations ("robust and efficient… implementation at a
  lower level", jakevdp on [jax#17038]; see `../bessel/PROJECT.md` §2.6) but will not
  host general polynomial machinery ([jax#11055] — explicitly deprioritized).
- Nobody occupies "ahead-of-time, precision-first approximant generator with a
  differentiable GPU runtime": Chebfun/ApproxFun are adaptive+interactive (MATLAB/Julia),
  chebpy/pychebfun are numpy clones, ChebTools is C++ eval tooling, orthax is
  numpy.polynomial-in-JAX primitives with no generator, no recipes, no precision
  pipeline.
- Parameter gradients are unavailable anywhere in the ML ecosystem: ∂J_ν/∂ν at machine
  precision (established below) is the ingredient for Matérn kernels with *learned*
  smoothness, which no GP library currently offers.

[jax#11002]: https://github.com/jax-ml/jax/issues/11002
[jax#2466]: https://github.com/jax-ml/jax/issues/2466
[jax#27088]: https://github.com/jax-ml/jax/issues/27088
[jax#17038]: https://github.com/jax-ml/jax/pull/17038
[jax#11055]: https://github.com/jax-ml/jax/issues/11055

---

## 2. What is established

Inherited from `../bessel/` (measured there, scripts checked in there — do not redo):

- **Near-minimax property.** Interpolation at Chebyshev points is within a factor
  1+Λ_n of the best possible polynomial, Λ_n ≈ (2/π)ln n + 0.98 < 4 at any practical
  degree. Convergence rate is set by the Bernstein-ellipse analyticity of the
  (transformed) function; entire functions converge super-geometrically. Reference:
  Trefethen, *Approximation Theory and Approximation Practice*.
- **The factorization is mandatory, not an optimization.** `J_v` fitted directly in x
  converges algebraically (branch point at 0); factored `g_v(z) = ₀F₁(;v+1;−z/4)`,
  z = x², needs only degree ~16 (f64) on x ∈ [0,8] (`../bessel/experiments/02`,
  degree table in `../bessel/PROJECT.md` §2.3). This is the proof that the recipe
  layer (per-family analysis) cannot be automated away.
- **Hard `select` at domain seams, never blend** (`../bessel/experiments/04`): the seam
  jump equals the sum of branch errors; blending costs 2× and fixes a non-problem.
- **Roofline budgets** (`../bessel/experiments/03`): analytical, not measured; a
  degree ≤ ~40 polynomial is free on every datacenter GPU. Bessel **B3 is the
  measurement** and gates every performance claim chebax makes. Until B3, chebax
  claims accuracy and gradients, not speed.

Established here (scripts in `experiments/`, output in `results/`, mpmath 40 dps,
errors sup-normalized = absolute-accuracy contract, see `../bessel/PROJECT.md` §4 q4):

### 2.1 The derivative of the approximant is the gradient — free and exact

`experiments/01_derivative_accuracy.py`: fit g_v at degree 20 on x ∈ [0,8], apply the
exact coefficient recurrence (`chebder`), assemble J and J′ via chain/product rule:

| v | pipeline | J err | J′ err |
|---|---|---|---|
| 0.5 | f64 | 6.5e-16 | 2.5e-16 |
| 0.5 | f32 coeffs + f32 Clenshaw | 3.9e-07 | 2.6e-07 |
| 2.5 | f64 | 9.7e-16 | 2.8e-15 |
| 2.5 | f32 | 8.2e-07 | 7.5e-07 |

No degree penalty for the first derivative in practice. `jax.grad` through Clenshaw
gives the same polynomial; binding `custom_jvp` to the chebder series makes the
gradient cost one extra Clenshaw and keeps higher derivatives exact.

### 2.2 Coefficients are smooth in the order — one table serves every ν

`experiments/02_coeff_smoothness_in_nu.py`: fit c_k(ν) on ν ∈ [0,10] (64 Chebyshev
nodes in ν, z-degree 24 on [0,64]), reconstruct the coefficient vector at off-node ν
by Clenshaw in ν, truncating the ν-series at degree d_ν:

| test ν | d_ν=24 | d_ν=32 | d_ν=48 | d_ν=63 |
|---|---|---|---|---|
| 0.3 | 3.9e-07 | 2.2e-09 | 1.8e-14 | 6.3e-16 |
| 1.7 | 1.7e-06 | 1.1e-08 | 1.2e-13 | 1.6e-15 |
| π | 1.2e-06 | 9.6e-09 | 5.9e-13 | 1.3e-15 |
| 7.77 | 3.6e-07 | 2.9e-09 | 1.3e-13 | 9.0e-16 |

Order gradient ∂J/∂ν via chebder along the ν axis, vs `mp.diff`: **3e-16 to 2e-15**
at all test ν. Table: 25×64 = 12.5 KB; instantiation at any ν ≈ 3.2k FLOPs, once.

**Consequences.** (a) No mpmath at use time: the generator bakes the table once,
instantiation is pure float arithmetic, cheap enough to run inside `jit` on constants
(XLA folds it), so "trace-time-constant ν" relaxes toward "call-time-uniform ν".
(b) ∂J/∂ν — declared "genuinely hard" in bessel Track C2 — falls out at machine
precision within the table domain.

**This does not reopen `../bessel/PROJECT.md` §2.5's 2-D dead end.** That verdict was
about per-*point* cost (~3076 FLOPs/pt). Here the 2-D structure is spent once per
*instantiation*; per-point evaluation stays the 1-D polynomial. Per-element ν remains
out of scope either way.

### 2.3 How upstream consumes implementations (settles the packaging question)

| Consumer | Ships | How chebax output arrives |
|---|---|---|
| `scipy.special` (CPU) | C++ ufuncs from the xsf submodule | PR into xsf, when measurably better |
| CuPy `cupyx.scipy.special` (GPU) | same xsf headers via NVRTC | automatically, once in xsf |
| `jax.scipy.special` | pure-Python jax compositions | PR of a baked module, ν static |
| any JAX user | whatever they pip install | chebax itself, day one |

Verified precedent: `jax.scipy.special.bessel_jn` is plain Python over jax ops with a
static integer order keyword (`jax/_src/scipy/special.py:1896` in the pinned clone).
The scipy contract (`jv(v, x)` with per-element v) is exactly what chebax does *not*
serve, so scipy gets Track A + accuracy fixes, not chebax kernels; JAX-facing artifacts
need no FFI at all. Hence: **standalone library = the pipeline; upstreams get its
exhaust**, each as a separate hand-carried PR.

---

## 3. Architecture

```
src/chebax/_src/algorithms.py   numpy references: nodes, DCT fit, Clenshaw, chebder/chebint
src/chebax/_src/series.py       jax runtime: ChebSeries/PiecewiseCheb + custom_jvp
src/chebax/_src/generate.py     fit() for arbitrary smooth functions
src/chebax/_src/pytree.py       Recipe base class for instance pytrees
src/chebax/_src/recipes/        per family: <name>_gen.py (mpmath generator),
                                <name>_table.py (baked, regenerable bit-for-bit),
                                <name>.py (jax runtime); _gen_common.py and
                                _common.py hold the shared machinery
src/chebax/bake/                emitters: pure-jax module, xsf-style C++ header
docs/                           adding-a-recipe.md (workflow), increments.md (design log)
```

Runtime object: a pytree of arrays (segment breaks, coefficient matrix, derivative
coefficients, metadata). Construction is **eager** (weights at instantiation — cheap
once the ν-table exists), with module-level `lru_cache` factories giving the lazy feel.
Genuinely lazy first-call computation is a trap under `jit` when construction touches
mpmath; with baked tables construction is jax-traceable arithmetic, so it may also
happen inside `jit` when the parameter is static. Coefficient conventions follow
`numpy.polynomial` (plain c₀, lowest-first) so interop with numpy/orthax is trivial.

Runtime dependencies: `jax`, `numpy`. Generator dependencies (`mpmath`, `scipy` for
cross-checks) live behind an extra (`chebax[gen]`) and are never imported at use time.

### API sketch

```python
import chebax, jax

# generic core: any smooth f on a finite interval
p = chebax.fit(f, domain=(0.0, 64.0), tol=1e-15)    # build time: numpy + mpmath
p(x)                     # jax runtime: Clenshaw, jit/vmap-safe pytree
p.deriv()(x)             # chebder series (recomputed per deriv() call;
                         #  jit folds it for concrete coefficients)

# recipes: prebuilt families, tables baked into the package
jv = chebax.besselj(2.5)         # any real v in the table domain; no mpmath
jv(x); jax.grad(jv)(x)           # custom_jvp bound to the derivative series
chebax.besselj_dnu(2.5)(x)       # order gradient, chebder along the nu axis

# bake: self-contained artifacts for upstream targets
chebax.bake.jax_module(jv, "besselj_2p5.py")
chebax.bake.xsf_header(jv, "cyl_bessel_j_2p5.h")
```

---

## 4. Milestones

| ID | Milestone | Acceptance criteria |
|---|---|---|
| M0 | Bootstrap: repo, plan, foundation experiments. **DONE 2026-07-29.** Name and license confirmed same day (§6 q1-q2). | Repo exists with correct identity; experiments reproduce. |
| M1 | Generic core on a finite interval: `fit`/`__call__`/`deriv`, numpy+mpmath build path, jax runtime, pytree + custom_jvp, f32 rounding. **DONE 2026-07-29** (`src/chebax/`, 23 tests green on jax 0.11). Two floors found while calibrating: the original ≤5e-16 value bar is infeasible — Clenshaw rounding is eps·Σ\|c\| ≈ 5–8e-16, bar corrected to ≤1e-15 — and float64-built fits carry a ~deg²·eps derivative floor (Markov amplification of sample noise; 3.9e-14 at deg 14, 5.1e-13 at deg 38) which the dps build path removes (3.9e-15 / 4.0e-16 measured). That floor is the measured argument for the mpmath generator. | Measured: exp / cos / runge-segmented values 4.9e-16 / 6.7e-16 / 7.2e-16 (bar 1e-15); f64-build derivatives within deg²·eps bars (1e-13 smooth, 1e-12 for the deg-38 runge segments); dps-build derivative ≤1e-14; `jit(vmap(jax.grad(p)))` equals the derivative series (same computation by construction); pytree/jit/vmap, coefficient gradients, f32 rounding, and numpy.polynomial convention locks all tested. |
| M2 | `besselj` recipe on x ∈ [0,8], ν ∈ [0,10] via the ν-table. **DONE 2026-07-29** (`chebax.besselj` / `chebax.besselj_dnu`; table baked as diffable source in `_src/recipes/besselj_table.py`; the generator keeps both fit stages in mpmath so no f64 sample noise sits under the ν-fit). One floor found: the `(x/2)^ν` prefactor evaluates through pow = exp(ν·log), so its relative error grows like ν·\|log(x/2)\|·eps ≈ 3e-15 at ν=10 — the f64 value and dJ/dx bars were corrected accordingly (value 2e-15 → 5e-15, dJ/dx 1e-14, f32 dJ/dx 1e-6 → 5e-6). | Measured worst over 21 off-node ν vs mpmath(40 dps): values 2.8e-15 f64 / 9.2e-7 f32 (bars 5e-15 / 1e-6); dJ/dx 5.5e-15 f64 / 2.0e-6 f32 (bars 1e-14 / 5e-6); ∂J/∂ν 1.8e-15 (bar 5e-15, as specced); table regenerates bit-for-bit; instantiation under jit with static ν works both as pytree argument and built at trace time. |
| M3 | Full domain: segmentation + oscillatory tail recipe (phase/modulus a la Hankel) for x > 8. **DONE 2026-07-29.** Three regions, hard selects at 8 and 30: the M2 inner table unchanged; J_ν fitted *directly in x* on [8,30] — no x=0 branch point inside, so this is **not** the §2.5-inherited unfactored dead end, and direct fitting is what keeps the table sup-accurate at large ν, where g_ν spans six decades by x=30 and its f64 Clenshaw floor would destroy envelope accuracy; above 30 the exact modulus functions P, Q (via J and Y) tabulated in t=(30/x)², degree 11 suffices. Two floors + one bug found: (a) forming ω = x−(ν/2+¼)π costs ε·x of phase (2e-12 at 10⁴) — eliminated by the angle-addition identity with baked cos φ, sin φ so `sincos(x)` does its own reduction; (b) f32 tails are phase-limited anyway (~ε₃₂·x from XLA's f32 trig reduction; bars 1e-5/2e-5); (c) min/max **ties split tangents**, so naive clamped branch inputs halved dJ/dx at exactly x=8 and 30 — branch inputs rearranged so the active branch sees raw x through its own seam. dJ/dν now covers the full domain. | Measured worst over 11 off-node ν on a grid to x=10⁴ incl. both seams and ±1e-9 neighbors: values 1.7e-15, dJ/dx 2.6e-15 incl. exactly at the seams, ∂J/∂ν ~1e-15 (f64; bars 5e-15 / 1e-14 / 5e-15); f32 4.0e-6 / 7.1e-6 (bars 1e-5 / 2e-5, phase-floor documented); branch-vs-branch jump at each seam ≤1e-13 (exp-04 criterion); x=10⁶ smoke exact to 1e-16 abs; ext tables regenerate bit-for-bit; per-eval work is all three branches + two selects, no data-dependent branching. |
| M4 | Bake emitters: pure-JAX module + xsf-style header. **DONE 2026-07-29** (`chebax.bake.jax_module` / `chebax.bake.xsf_header`, from a `BesselJ` instance). The emitted jax module inlines coefficients as python floats (weak typed, so one emission serves both dtypes) and needs no custom_jvp: plain AD differentiates the same polynomials. The header is C++17, `<cmath>` only, `XSF_HOST_DEVICE` defined empty when absent so it drops into xsf or host code unchanged. Emission is deterministic: same instance, same bytes. | Emitted jax module meets the library bars (values 5e-15 / grad 1e-14 f64, values 1e-5 f32, eager + jit) in a subprocess with chebax poisoned out of `sys.modules`, at ν=2.5 and 9.97; emitted header compiles standalone with g++ -std=c++17 and matches mpmath to 5e-15 on the full-domain grid; emissions byte-identical across calls; non-BesselJ inputs rejected. |
| M5 | Second family: `besselk` by direct tabulation (the Matérn story), including ∂K/∂ν. **DONE 2026-07-29 — the "general library" gate is passed.** Design: log-tables. Inner (x ∈ [1e-6, 8]): L̃ = ln[(x/2)^ν K_ν] in u = ln x — the log kills 15 decades of dynamic range, the factored (x/2)^ν carries the branch exponent, and ν splits into two instantiation-time panels ([0,1], [1,10]) around the Γ-pole cancellation feature of width ~1/\|ln x\| near ν=0 (measured: unsplit needs ν-degree >96, panels need 45). Tail (x > 8): Lt = ln[√(2x/π) eˣ K_ν] in t = 8/x, with exp(−x) kept as its own correctly-rounded factor. Direct tabulation makes integer orders ordinary points, retiring risk S3 (tested at exactly 1.0, 2.0, 3.0). New capability: `besselk_fn(nu, x)` takes ν as a **traced** scalar (reconstruction inside the computation), realizing the call-time-uniform-ν relaxation — `jax.grad` works with respect to ν. The f64 floor is again the pow term, ε·ν·\|ln(x/2)\| ≈ 3e-14 in the (10, 1e-6) corner. | Measured worst over 11 orders (incl. integers and panel edge 0.999/1.0/1.001), pointwise relative: values 1.2e-14, dK/dx 1.1e-14, dK/dν 3.3e-13 incl. the traced path (bars 5e-14 / 5e-14 / 1e-12); ∂K/∂ν at ν=0 vanishes to 7e-15 (evenness); clamp below 1e-6 and underflow past x≈746 graceful; tables regenerate bit-for-bit; **the Matérn demo (`examples/matern_learn_nu.py`) recovers (ν, ℓ, σ²) = (1.7, 0.9, 1.3) to six decimals by Adam from a cold start, and the test version converges against mpmath-computed targets** — gradient-based learning of Matérn smoothness, which no mainstream ML library offers. |
| M6 | Breadth, docs, packaging (ongoing). **Seven recipe increments delivered 2026-07-29/30** — besseli, bessely, dawsn/erfcx, betainc, the quantile toolkit, spherical/truncated-sampling/lambertw, von Mises — plus a review-hardening pass (shared machinery, pytree base, unified emitter, degree harness). The full per-increment design log (measurements, traps, calibrated bars) lives in `docs/increments.md`; the workflow they established in `docs/adding-a-recipe.md`. Fresnel and Wright Bessel remain open. PyPI releases are Andres's action, never autonomous. | — |

**Performance claims: B3 measured 2026-07-31** (`../bessel/experiments/07`, RTX 3080
Laptop, 16.8M points). chebax may now claim, with the measurement cited: besselj f64
is **4.5–6.0× faster than NVRTC-compiled cephes::jv** on mixed/inner/mid/log-wide
input distributions (0.4× on pure-outer, where cephes takes one short path); f32 runs
35× the f64 (~97 GB/s, near-bandwidth). Honest scope: on consumer GPUs fp64 is
compute-bound at the 1:64 ratio, so the roofline "free polynomial" story is
fp32-everywhere / fp64-datacenter; datacenter fp64 numbers remain analytical until
measured there. Never claim "memory-bound" for f64 on GeForce-class parts.

**Narrow-domain race — measured 2026-07-31** (`experiments/04`, was queued 2026-07-29):
the single-region inner evaluation beats the full three-region select besselj **2.5×
(f64) / 3.0× (f32)** on x ~ U(0, 8), matching the a-third-the-arithmetic prediction.
Verdict: worth offering. **Queued build:** a domain-limited `besselj(v, domain=(a, b))`
that trims to the covering region(s); first customer the Matérn demo.

**gammainc work (2026-07-30, Andres): part (1) benchmark DONE 2026-07-31**
(`experiments/05`): XLA's looped `igamma` vs a mock fixed-degree kernel
(betainc-runtime op profile, degree 27) on 16.8M points: **18–54× (f64), 6–16×
(f32)**, worst at a=0.5 (the continued fraction runs to 70 trips for the whole
array) and a=100 (series to 93). Iteration statistics recorded in
`results/05_igamma_race.txt`. The 10–100×-class analytical headroom is confirmed at
its lower half; `gammaincinv` compounds any win ~40×. **The prize is real: part (2)
below is now justified and queued.** (2) **Build**: a Y-class two-region recipe —
moderate a via log-tables of the Kummer part M = ₁F₁(1; a+1; x) (the betainc pattern),
large a via Temme's uniform representation P ≈ ½erfc(−η√(a/2)) + R(1/a, η), which
covers a → ∞ through the 1/a variable with erfc XLA-native. Same fits-both-orders
consideration as betainc does not arise (single parameter). Also likely fixes XLA
igamma's f32 accuracy reputation in passing.

**Queued f32 work (2026-07-30, Andres; (1) and (2) DELIVERED 2026-08-01,
increment 22):** three bake-step options, in order of value.
(1) Degree truncation for f32: `astype(f32)` keeps the full f64 degree, roughly 2× the
terms f32 accuracy needs (exp-02: f32 degrees are about half the f64 ones); a
`truncate(tol)` dropping the converged tail would halve the FLOPs — benchmark under B3
before bothering, the f32 roofline says the extra terms are free. (2) A `dtype` option
for `bake.xsf_header` emitting `float` constants and truncated arrays for CUDA f32
kernels. (3) An fpminimax-style polish (Sollya / LLL lattice reduction over machine
floats, Brisebarre–Chevillard) for jointly optimized f32 coefficients: measured, naive
downcast leaves only ~2% on the table in the Chebyshev basis (coefficient rounding is
8e-9 of a 3.7e-7 f32 budget for J_2.5; evaluation rounding and the pow/trig floors own
the rest), so this only becomes worthwhile if chebax ever targets correctly-rounded-
grade f32 kernels. In the monomial basis the calculus flips — another reason the
runtime stays Chebyshev + Clenshaw.
MEASURED 2026-08-01 (`experiments/12_f32_truncation_race.py`): the roofline
prediction was WRONG on the 3080 — truncate(1e-7) speeds f32 GPU evaluation
1.4–2.2× (besselj 2.0×, besselk 2.2×, betainc 1.4×) at each family's f32
floor. truncate(tol) ships on ChebSeries/PiecewiseCheb/Recipe; the bake
emitters take truncate_tol and xsf_header takes dtype="float" (f-suffixed
literals, float arrays and locals, templated Clenshaw), compile-verified at
f32 grade. Option (3), fpminimax, stays dead per the 2% measurement.

**Queued per-group parameter API (2026-07-30, Andres; strictly after the B3
benchmark study, scheduled 2026-07-31 in `../bessel/`):** a wrapper over the
traced-parameter path serving *per-group* parameters: G unique parameter values
plus a static group-index array; build the G coefficient tables under `vmap`
(~3k FLOPs each, differentiable, inside `jit`), gather per element. This
relaxes uniform-per-call to "per-group, grouping static, values free to change
every iteration" — the hierarchical-model regime (numpyro
`Beta(alpha[group_idx], ...)`, censored likelihoods with per-group shapes,
mixtures, multi-kernel GPs), which is where the adoption map's constraint
caveat actually bites (`docs/adoption-map.md`). Cost is
G × (reconstruction + n_g × polynomial); B3 must measure the crossover n per
group where reconstruction overhead disappears before any API is designed or
claimed. The per-element contract (one parameter per point, scipy's
`jv(v_array, x_array)`) stays out of scope: that is the §2.5 dead end and no
wrapper changes its arithmetic.

MEASURED 2026-07-31 (`experiments/07_pergroup_crossover.py`, RTX 3080 Laptop,
f64, plain vmap over equal groups): reconstruction overhead vs chebax's own
uniform floor is ≤ 1.07x for n/group ≥ 16k and 1.2x at n/group = 4096
(betainc at N = 2^24); besselk stays ≤ 1.08x down to n/group = 1024. Against
jax's native betainc the per-group path wins EVERY measured cell: worst case
7.3x (G = 16384 groups of 64), typically 60–155x, and 44–98x at MCMC scale
(G = 4 chains, n/group 256–16k, GPU and CPU; absolute cost 0.15–0.5 ms per
call vs jax's 14–26 ms). At N = 2^20 the vmapped lowering pays a flat
1.5–3 ms that the launch-bound uniform path does not; it vanishes into the
compute at 2^24 and does not change any conclusion. Design consequence: plain
`vmap` already serves the hierarchical regime (G ≤ ~1024), so the API is a
thin grouping convenience (static `group_idx` → reshape/segment + vmap), not
new arithmetic; a gather-based variant would only matter for G ≥ 4096 with
tiny groups, which is the per-element regime that stays out of scope.
DELIVERED 2026-07-31: `chebax.pergroup(fn, group_idx, num_groups=None)`,
increment 12 in `docs/increments.md`, tests in `tests/test_pergroup.py`.

**Queued betainc box widening (2026-08-01, feasibility measured in
`experiments/11_betainc_widening_feasibility.py`; design decision is
Andres's):** degree growth beyond [0.1, 10]^2 is tractable but not free.
Measured: the a-axis carries the pole structure (raw degree 129-175 over
[0.1, 100], log-a 40-95, per-panel raw ~50); b is mild (26-63, log WORSE);
x grows 19 -> 62 at sharp interior transitions. Priced options: 2x2 panels
at [0.1, 100]^2 ~650k coefficients (~13x table, ~1 h generation); single
[0.1, 50]^2 log-a tensor ~252k. Both break the CI bit-for-bit regen budget,
so widening waits on a policy call (slice-regen in CI + full regen manual,
or parallel generation) and pairs naturally with Tucker storage
(experiments/10: betainc-class tensors are where compression pays). The
reflection needs BOTH parameter orientations, so a one-sided strip is not
cheaper than half the full job. Separately discovered and NOT blocked:
stdtr/stdtrit evaluate betainc at a = 1/2 exactly, so a dedicated fixed-a
slice table for b in [10, 100] (~4.6k coefficients) extends StudentT to
nu = 200 at a thousandth of the widening cost.

**Queued low-rank table compression (2026-07-30, Andres; sequenced with the f32
bake work, FLOP payoffs gated on B3):** Chebfun2-style separated representation
c(ν,k) ≈ Σ_j σ_j u_j(ν) v_j(k) (Townsend & Trefethen 2013; Hashemi & Trefethen
for 3-D). No cross-approximation needed: the generator already materializes the
full tensor, so this is a truncated SVD at bake time (Tucker/HOSVD for the 3-D
betainc tensor), plus a `compress(tol)` emitter option. Measured 2026-07-30 on
the baked tables (SVD of the checked-in coefficient matrices; sup function
error verified ≈ σ_{r+1} on a dense grid): besselj inner 25×64 has rank 10 at
1e-15 / 7 at 1e-7; besselk panels 20-21/56; bessely mid (oscillatory) 20/48 —
phase/modulus structure is inherently low-rank; betainc 24×72×28 has Tucker
mode ranks (18,21,18) at 1e-15 and (8,10,9) at 1e-7. Every baked family is
strongly rank-deficient. What it buys, honestly: does NOT rescue per-element
parameters (rank-r evaluation is r·(d_ν+d_x) per point ≈ 2k FLOPs for besselj
f64, still ~20-40× over the roofline budget — the §2.5 verdict stands);
instantiation collapse drops ~1.8× for the 2-D families (lowers the per-group
crossover n_g, measure under B3 with the per-group item above); the real wins
are betainc (~5× cheaper (a,b) instantiation, 5× smaller table at f64, ~40-60×
at f32-grade) and feasibility of future 3-and-4-parameter recipes, where dense
tensor-product tables scale as the product of degrees and compressed ones as
the sum plus a small core. Also shrinks baked artifacts (headers/modules),
which feeds the vendoring pitch in `docs/adoption-map.md`. Derivative exactness
survives: chebder applies factor-by-factor, ∂f/∂ν = Σ σ_j u_j'(ν) v_j(x).
Composes with f32 degree truncation (the two truncations compound). When
picked up, first action is an `experiments/` script reproducing the rank table
above per convention.
MEASURED 2026-08-01 (`experiments/10_table_rank_structure.py`, all baked
tables): 2-D epsilon-ranks at 1e-15 give flop/storage models of only
1.1–2.2× (gammainc's inner table is nearly full rank at 19/32;
vonmises the best at 2.2×), so a compressed 2-D evaluation path is NOT
worth building at f64. The cases that matter: the betainc tensor
(Tucker (18,21,18) → 5.2× storage at f64, (8,10,9) → 25.7× at
f32-grade truncation) and f32-grade 2-D truncation generally
(2.3–4.8×). Keeps its original sequencing with the f32 bake work;
revisit when a 3-parameter recipe or baked-artifact size actually
hurts.

**External review 2026-07-30:** a full read-only review found 26 issues, several
release-blocking (silent deep-tail quantile saturation, an aliasing hole in the
adaptive fit, besseli_dnu(0) catastrophic in the tail, stdtr flat at the median).
The earlier queued betaincinv deep-tail note was the visible tip of the quantile
finding. All release-blocking and high-severity items are fixed; the disposition
table, including the still-open maintainability items, is
`docs/review-2026-07-30.md`.

---

## 5. Risks

| ID | Risk | Mitigation |
|---|---|---|
| S1 | Scope creep: "most special functions" before one family is solid (bessel R5's twin). | M5 gates the general claim on a second family; M2 gates M5; public naming stays "Bessel for JAX" until M5. |
| S2 | ν-table domain limits (large ν; ν < 0). | Document the domain; panels in ν (halve d_ν); uniform asymptotics are a later recipe, not v1. |
| S3 | `K_ν`/`Y_ν` near-integer-ν cancellation in connection formulas (bessel R6's lesson). | Recipes tabulate the target function directly; never reconstruct via J_{−ν} combinations. |
| S4 | f32 marketing overstates accuracy near zeros. | State the sup-normalized (absolute) contract everywhere numbers appear; relative-at-zeros is a known impossibility (`../bessel/PROJECT.md` §4 q4). |
| S5 | Overlap/duplication with orthax. | Core stays self-contained (~small); coefficient conventions numpy-compatible so a later orthax dependency is a swap, not a rewrite. |
| S6 | Upstream appetite misjudged (JAX PR stalls, xsf uninterested). | Standalone distribution is the primary channel; upstreaming is discoverability. Every upstream PR is hand-carried by Andres per tree rules. |
| S7 | Baked tables rot or become unverifiable. | Generator checked in; CI check regenerates and diffs bit-for-bit; tables carry generator version + dps + domain in metadata. |
| S8 | The tail recipe (M3) is harder than the core suggests. | It is the one genuinely new numerics in v1; scope M3 to J only, reuse the exp-04 seam methodology, and accept a reduced x-range for a first release if needed. |

---

## 6. Open questions — Andres decides

1. ~~**Name.**~~ **RESOLVED 2026-07-29:** `chebax`, confirmed by Andres. The spec-*
   family was rejected for spectroscopy confusability. Claimed on PyPI the same day
   (`0.1.0.dev0` placeholder, uploaded by Andres).
2. ~~**License.**~~ **RESOLVED 2026-07-29:** BSD-3-Clause, confirmed by Andres;
   `LICENSE` checked in. Rationale: scipy/xsf compatibility for baked artifacts.
3. ~~**Second family** (M5)~~ **RESOLVED 2026-07-29:** `besselk`, confirmed by Andres
   (the Matérn demo is the strongest external motivation). The betainc/hyp2f1 line
   stays a candidate for M6 breadth.
4. ~~**When public.**~~ **RESOLVED 2026-07-29:** after M2, confirmed by Andres (one
   family, full gradient story, honest README). Went public the same day, right after
   M2 landed.
5. ~~**Cross-project updates**~~ **RESOLVED 2026-07-29:** `../bessel/PROJECT.md`
   updated (commit `c6891c2` there) — B1 points at chebax and absorbs the ν-table
   result, C2's order-derivative is re-scoped, C1 notes FFI is unnecessary for
   Track B artifacts, and the §2.5 2-D row records the per-instantiation distinction.

---

## 7. Reference index

| Where | What |
|---|---|
| `../bessel/PROJECT.md` §2.3–2.5 | degree tables, roofline, seams, dead ends (the evidence base) |
| `../bessel/experiments/02,03,04` | the measurements chebax inherits |
| `experiments/01,02` + `results/` | derivative accuracy; ν-smoothness + ∂J/∂ν |
| `jax/_src/scipy/special.py:1896` (pinned clone) | `bessel_jn`: the pure-JAX consumption precedent |
| [orthax](https://github.com/f0uriest/orthax) | numpy.polynomial in JAX (primitives only) |
| [ChebTools](https://github.com/usnistgov/ChebTools), [chebpy](https://pypi.org/project/chebpy/), [pychebfun](https://github.com/alexalemi/pychebfun) | adjacent tooling, no generator/AD/GPU story |
| [Chebfun](https://www.chebfun.org/), ApproxFun.jl | the adaptive-interactive niche (not this one) |
| Trefethen, *Approximation Theory and Approximation Practice* | the theory |
| Bremer, rapid evaluation of Bessel via phase functions (~2019) | closest prior art for the (ν,x) table idea — **read before claiming anything about M3's tail design** |
