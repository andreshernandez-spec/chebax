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

**Status:** planning. Nothing public, nothing published. Parent evidence lives in
`../bessel/` (Track B); this project generalizes B1 into a library.
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
| M1 | Generic core on a finite interval: `fit`/`__call__`/`deriv`, numpy+mpmath build path, jax runtime, pytree + custom_jvp, f32 rounding. | Fit exp/cos on [-1,1] and 1/(1+25x²) segmented to ≤5e-16 sup-normalized; chebder matches analytic derivatives ≤1e-14; `jit(vmap(jax.grad(p)))` runs and equals the derivative series to f64 rounding. |
| M2 | `besselj` recipe on x ∈ [0,8], ν ∈ [0,10] via the ν-table. | Sup-normalized error vs mpmath(40 dps) on dense grids and ≥20 off-node ν: ≤2e-15 f64, ≤1e-6 f32; same bar for dJ/dx; ≤5e-15 for ∂J/∂ν; tables regenerate bit-for-bit; instantiation works under jit with static ν. |
| M3 | Full domain: segmentation + oscillatory tail recipe (phase/modulus a la Hankel) for x > 8. | Errors hold across every seam (method of `../bessel/experiments/04`); f64 bar maintained to x = 10⁴; branch count per eval stays 1 (select, no divergence). |
| M4 | Bake emitters: pure-JAX module + xsf-style header. | Emitted jax module passes the M2 test suite with chebax uninstalled; emitted header compiles standalone and matches to the same bar. |
| M5 | Second family: `besselk` by direct tabulation (the Matérn story), including ∂K/∂ν. **Only after M2 holds.** This gates any "general library" claim. | Same accuracy bars on its domain; a Matérn-kernel demo with gradient-based learning of ν. |
| M6 | Breadth (Y, I, erf-family, incomplete gamma/beta — ties into `../betainc/`), docs, packaging. PyPI release is Andres's action, never autonomous. | — |

**Performance claims remain gated on bessel B3** (real-GPU measurement). chebax's
accuracy and gradient claims stand alone; its speed claims do not exist until B3.

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
   family was rejected for spectroscopy confusability. PyPI reservation remains his.
2. ~~**License.**~~ **RESOLVED 2026-07-29:** BSD-3-Clause, confirmed by Andres;
   `LICENSE` checked in. Rationale: scipy/xsf compatibility for baked artifacts.
3. ~~**Second family** (M5)~~ **RESOLVED 2026-07-29:** `besselk`, confirmed by Andres
   (the Matérn demo is the strongest external motivation). The betainc/hyp2f1 line
   stays a candidate for M6 breadth.
4. ~~**When public.**~~ **RESOLVED 2026-07-29:** after M2, confirmed by Andres (one
   family, full gradient story, honest README). Going public is his action.
5. **Cross-project updates** (pending edits, in `../bessel/`): B1/C2 wording should
   absorb the ν-table result; Track C1's FFI route is likely unnecessary for Track B
   artifacts (pure-JAX suffices). Not edited from this project.

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
