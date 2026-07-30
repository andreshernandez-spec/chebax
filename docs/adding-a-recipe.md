# Adding a recipe

Every special function in chebax is a *recipe*: a build-time generator, a
baked table module, a jax runtime, and a test file. This is the workflow
that produced all of them; follow it in order. `besselk` is the cleanest
worked example of every step (read `_src/recipes/besselk_gen.py` and
`besselk.py` side by side with this document).

## 0. Prototype first — never guess degrees

Before writing any generator, measure the Chebyshev coefficient tails of
the thing you intend to tabulate, in every axis, at the corners you suspect
are worst. `experiments/03_degree_measurement.py` is the harness: add a
family function there, run it, and pick node counts with ~20% margin over
the last coefficient above 1e-15. This step is also where design flaws
surface cheaply — the bessely round-vs-floor trap, the besselk ν-panels,
and the von Mises √κ axis were all prototype findings, each recorded in the
generator docstring it shaped.

Recurring transformations, in the order to try them:

- **Factor out the singular part** so the remainder is entire (besselj's
  `(x/2)^ν`; the branch point verdicts of `../bessel/PROJECT.md` §2.5).
- **Tabulate the log** when the function is positive: kills dynamic range
  and connection-formula cancellation at once (besselk, besseli, betainc).
- **Odd functions**: tabulate `f(x)/x` on `w = x²` — oddness and the x=0
  gradient become exact by construction (dawsn, von Mises).
- **Map infinite tails** to `t = (X_s/x)` or `(X_s/x)²`; keep `exp(±x)` as
  its own correctly-rounded factor, never inside the tabulated exponent.
- **Never form large phases**: for oscillatory tails use the
  angle-addition identity with baked `cos φ`, `sin φ` so `sincos(x)` does
  its own reduction (besselj/bessely outer).
- **Moving boundary layers** (a feature whose location/width depends on
  the parameter): change variables until the layer sits at an interval
  *endpoint* (Chebyshev clusters there) and moves uniformly (√κ for von
  Mises); split parameter panels when a feature near a parameter endpoint
  resists (besselk's [0,1]/[1,10]).

## 1. Generator (`_src/recipes/<name>_gen.py`)

Use `_gen_common` (`nodes`, `dct`, `param_fit`, `to_f64`,
`write_table_module`) — do not hand-roll fits or emitters. Rules:

- **Every fit stage stays in mpmath** at `DPS = 40`; round to float64 only
  at the very end. Rounding between stages puts ε-level sample noise under
  later fits, and differentiation amplifies it by ~degree² (the M1 floor).
- The generator docstring records the design, the measured degrees, and
  any trap found while prototyping — it is the recipe's design document.
- `write_table_module` emits the table with an axis note, META (dps,
  mpmath version, domains, node counts), and repr-stable floats.
  Convention: `TABLE[k, j]` = j-th parameter-direction coefficient of the
  k-th argument-direction coefficient.

## 2. Runtime (`_src/recipes/<name>.py`)

- Runtime imports are **jax + numpy only** (the import-hygiene test
  enforces it, and building `ChebSeries` at module level is a bug — it
  initializes the jax backend as an import side effect; build lazily or
  inside factories).
- Reconstruction: `_common.param_coefs` / `param_coefs_der` (eager, in the
  cached factory) and `_common.traced_coefs` (inside a `*_fn` variant).
- Instances subclass `pytree.Recipe`: declare `_static_fields` (scalars,
  aux) and `_series_fields` (ChebSeries, children), constructor takes
  statics then series, `_post_init` derives cached constants.
- Piecewise domains use hard `where` selects. Two gradient traps, both
  found the hard way: **the active branch must see raw x through its own
  seam** (min/max ties split tangents — clamp only the masked lanes), and
  **masked lanes must get finite dummy inputs** (an `exp` overflow or
  `log(0)` in an inactive branch poisons gradients through 0·inf).
  `jnp.abs` has derivative 0 at 0 by tie convention — build |x| from a
  select when the origin matters.
- Quantiles/inverses: `_common.newton_bisect` (fixed count, safeguarded,
  inclusive bounds — see its docstring for why) in a variable that resolves
  the tails (logit/log space), with a `custom_jvp` from the implicit
  function theorem — never differentiate through the iteration.
- Domain checks via `_common.check_range` in factories; traced variants
  cannot check (say so in the docstring).

## 3. API conventions

Parameters first, evaluation point last (note: opposite of scipy). The
taxonomy:

- `name(params)` — cached factory returning a callable pytree, when eager
  per-parameter reconstruction is worthwhile.
- `name_fn(params, x)` — traced-parameter variant *of a factory*, when the
  reconstruction is jax-traceable (gives `jax.grad` in the parameters).
- plain `name(params, x)` — when there is no factory sibling (parameter-
  free like `dawsn`, or solver-based like the quantiles, or traced-only
  like `vonmises_cdf`).
- `name_dnu(params)` — analytic parameter-derivative instance, only where
  `grad`-of-`_fn` is unavailable (bessely: the recurrence depth ⌊ν⌋ is
  structural, so ν cannot trace).

## 3b. Baking is automatic

Any `Recipe` with a closed-form `__call__` bakes with **no per-family
work**: `bake.jax_module` / `bake.xsf_header` derive the artifact from the
instance's own traced jaxpr (see `bake/_jaxpr_emit.py`), so artifact and
runtime cannot diverge. Add your family to `BAKEABLE` in
`tests/test_bake.py`. Solver-based callables are not bakeable yet.

## 4. Tests (`tests/test_<name>.py`)

Copy the structure of `tests/test_besselk.py`: values (+ traced
consistency), dx, parameter gradient, domain edges/clamps, jit/pytree,
out-of-range raise, f32 where meaningful, and a `@pytest.mark.slow`
bit-for-bit regeneration test (mandatory for every new table).

- References are mpmath at ≥ 40 dps; prefer closed-form oracles when they
  exist (the Beta density for dI/dx, ODE identities for dawsn/erfcx, the
  von Mises density for dF/dθ).
- **Measure floors before setting bars**: run the value/gradient sweep,
  find the worst case, understand *why* it is the floor (pow's
  ε·p·|ln x| term, Clenshaw's ε·Σ|c|, trig reduction in f32...), then set
  the bar at ~4× the measured worst and record both numbers in the test
  docstring.
- **The error metric is chosen per family** and stated in the test
  docstring: sup-normalized for oscillatory J; modulus-relative
  (err/√(J²+Y²)) for Y; pointwise-relative for positive functions (K, I,
  erfcx); absolute for CDFs; two-sided (p-roundtrip below the median,
  distance-from-1 above) for quantiles.

## 5. Docs

Update the README export list and the increment log (`docs/increments.md`),
add the regen command to CLAUDE.md, and if the recipe taught a new numerics
lesson, make sure it lives in the generator or runtime docstring where the
next person will trip over the same thing.
