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

**Status:** live (2026-07-29). Public at https://github.com/andreshernandez-spec/chebax;
name claimed on PyPI with a `0.1.0.dev0` placeholder the same day. M0–M2 done; next is
M3 (tails), M4 (bake emitters), M5 (`besselk`). Parent evidence lives in `../bessel/`
(Track B); this project generalizes B1 into a library.
**Owner:** Andres
**Last verified:** 2026-07-29, experiments run locally (see `results/`); upstream
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
core/       nodes, DCT fit (mpmath-backed), clenshaw (numpy + jax), chebder/chebint,
            domain maps, segmentation + predicated select, pytree container,
            custom_jvp binding, f32 rounding policy.        ~300-500 lines, generic.
recipes/    per family: the analysis. transform to an entire/smooth form, variable
            maps, segmentation policy, tail asymptotics, parameter tables,
            validation vs mpmath. besselj first.            This is where effort goes.
tables/     baked artifacts, checked in, regenerable bit-for-bit from the generator
            (acceptance criterion inherited from bessel B1).
bake/       emitters: self-contained pure-JAX module (no chebax import), xsf-style
            C++ header, raw coefficients (npz/json).
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
p.deriv()(x)             # chebder series, computed once, cached

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
| M6 | Breadth (Y, I, erf-family, incomplete gamma/beta — ties into `../betainc/`), docs, packaging. PyPI release is Andres's action, never autonomous. **First increment DONE 2026-07-29: `besseli`** — log-tables (factored entire part in z=x² inner; e⁻ˣ-scaled tail in t=8/x; a single ν-panel suffices, no Γ-pole feature), a `scaled=` flag (scipy's `ive`, keeps working past the x≈709 overflow, matches mpmath to the last digit at x=720), and traced `besseli_fn`. The inner z-degree (~64 at ν=0) is set by I₀'s complex zeros at z=−j₀ₖ² entering the log. Measured pointwise-relative: values 4.2e-15 (incl. traced), dI/dx 1.3e-15, dI/dν 6.5e-13. **Second increment DONE 2026-07-29: `bessely`** — the hardest recipe, two design traps found and fixed on the way. Design: reduce to μ = ν−⌊ν⌋ ∈ [0,1), tabulate the scaled pair T_a=(x/2)^μ Y_μ, T_b=(x/2)^{μ+1} Y_{μ+1} in u=ln x on [1e-6, 5], lift by ⌊ν⌋ unrolled steps of the upward recurrence (Temme's strategy with tables; integer orders are ordinary points, retiring R6/S3 for Y); mid [5,30] direct-in-x; tail ≥30 **reuses J's P,Q tables verbatim** (Y = √(2/πx)(sin x·A − cos x·B), same A,B). Trap 1: the round-to-nearest decomposition (μ ∈ [−½,½]) breaks — for μ<0 the (x/2)^μ scaling amplifies the divergent branch to ~1e3 and leaks ~1e-11 into every μ-reconstruction; floor decomposition fixes it (measured 3 orders). Trap 2: the upward lift is only stable once k ≳ x — running it to x=8 cost 1.4e-10 in accumulated oscillatory-regime cancellation; the seam moved to x=5 where the measured lift error is ≤4e-16. Errors are modulus-relative (err/√(J²+Y²), the standard for oscillatory Bessel): values 2.7e-14, dY/dx 2.0e-14, ∂Y/∂ν 2.7e-14 worst over 10 orders incl. integers and half-integers (bars 1e-13). No traced-ν variant: the recurrence depth ⌊ν⌋ is structural. **Third increment DONE 2026-07-29: `dawsn` and `erfcx`** — the two erf-family members jax lacks (erf/erfc/erfinv are XLA-native). Parameter-free, so plain functions: dawsn as x·E(x²) (oddness and the x=0 gradient exact by construction) with tail G((6/x)²)/x; erfcx fitted on [0,6] with tail and the negative-x reflection 2e^{x²}−erfcx(−x) (correctly inf past −26.6). 116 table doubles total; seam at 6, not 4, because the e^{−x²} Gevrey fuzz measurably slows the tail fit at 4. Derivative tests use the exact ODE identities as oracles (D′=1−2xD, erfcx′=2x·erfcx−2/√π), no mp.diff needed. Three runtime gotchas worth remembering: masked-branch exp overflow poisons gradients via 0·inf (clamp the reflection's argument); `jnp.abs` has a zero derivative at 0 by tie convention, so |x| for branch inputs is built from a select; and module-level ChebSeries construction initializes the jax backend as an import side effect (caught by the import-hygiene test under GPU contention) — series are now built lazily. **`betainc` prototyped and queued:** factored form I_x(a,b) = x^a(1−x)^b/(a·B(a,b))·F with F = ₂F₁(a+b,1;a+1;x), tables on x ∈ [0,½] only (reflection I_x(a,b)=1−I_{1−x}(b,a) covers the rest), identity locked in mp. Log-tabulation flattens everything: ln F has x-degree ≤19, b-degree ≤20, range [0.1, 6.7] (raw F spans 833×); the a-direction is the costly axis (~58 at b=0.1 → ~72 nodes or a-panels). **Fourth increment DONE 2026-07-30: `betainc`** — the library's first two-parameter recipe, the ν-table machinery generalized to a (x, a, b) tensor (24×72×28, three-stage mp fit, each x-coefficient a 2-D Chebyshev series over (a,b) ∈ [0.1, 10]²). Log-tabulated F with the reflection I_x(a,b) = 1−I_{1−x}(b,a) for x > ½ (both parameter orders reconstructed; 1−x is exact in f64 on [½,1]). `betainc_fn(a, b, x)` takes **both shape parameters traced**: `jax.grad` w.r.t. a and b works — the jax#38610 gap — measured 2.0e-14 against mp.diff. Values 7.5e-15 absolute (the CDF contract); dI/dx 5.9e-15 against the exact Beta density as oracle. **Fifth increment DONE 2026-07-30: the quantile toolkit** — `betaincinv`, `gammaincinv`, `stdtr`, `stdtrit`, answering jax#2399, jax#5350 and jax#20358. Fixed-count safeguarded Newton, jittable, with gradients from the implicit function theorem (never through the iteration): ∂x*/∂θ = −(∂CDF/∂θ)/pdf, the CDF gradients coming from `betainc_fn` (or jax's `igamma_grad_a`). Design points earned by measurement: solve in **logit space** (beta) / **log space** (gamma) so tail quantiles to ~1e-300 stay resolvable (x-space bisection saturated at 3.5e-2 residuals); a **symmetry swap** solves every element in the mirrored lower tail (1−p exact on [½,1]); and one genuinely subtle solver bug — the strict-inequality safeguard **catapulted converged iterates**: a one-sided approach never tightens the far bound, and a Newton step stalling exactly at the root got rejected into the midpoint of the half-open bracket (inclusive test fixes it). Measured: betaincinv 4.9e-15 worst under the two-sided contract (p-roundtrip lower half, distance-from-1 vs 40-dps references upper half; quantiles within ε of 1 round to 1.0, the scipy-shared representation limit, covered by the mirrored call); gammaincinv 4.0e-14; stdtr/stdtrit roundtrips ~5e-16; IFT gradients 9.3e-15 / 2.6e-16. `gammaincinv` needs no tables at all — jax's own `gammainc` supplies values and the a-gradient. erf-family beyond dawsn/erfcx (fresnel) and further breadth remain open. | — |

**Performance claims remain gated on bessel B3** (real-GPU measurement). chebax's
accuracy and gradient claims stand alone; its speed claims do not exist until B3.

**Queued benchmark (2026-07-29, Andres):** when GPU measurements start (B3), also race
the three-region select `besselj` against a single-region fit on a user-declared narrow
domain (zero selects, one short polynomial — a third the arithmetic). If the narrow form
is noticeably faster, offer both: the unlimited `besselj(v)` and a domain-limited
`besselj(v, domain=(a, b))` that trims to one region. The natural first customer is the
M5 Matérn demo, where kernel inputs live in a known window.

**Queued f32 work (2026-07-30, Andres):** three bake-step options, in order of value.
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
