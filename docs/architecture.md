# Architecture

chebax separates high-precision approximation construction from JAX evaluation.
Generators analyze a family and emit coefficient tables; runtime recipes
reconstruct one coefficient vector for a uniform parameter and evaluate it over
an array. The generic series layer also supports user-fitted smooth functions.
This document owns the dependency and numerical contracts. See
[Coding style](coding-style.md) for names and presentation, and
[Adding a recipe](adding-a-recipe.md) for the implementation workflow.

## Modules and dependency direction

| Component | Responsibility | May depend on |
| --- | --- | --- |
| `_src/algorithms.py` | NumPy reference polynomial algorithms | NumPy |
| `_src/series.py` | JAX series evaluation, coefficient calculus, series pytrees | JAX, NumPy, reference algorithms |
| `_src/generate.py` | Generic finite-interval `fit` | Core series and algorithms; mpmath only for the requested high-precision build path |
| `_src/pytree.py` | Recipe field layout, immutability, pytree reconstruction | Generic object protocol; no family mathematics |
| `_src/recipes/_common.py` | Shared normalization, reconstruction, endpoint slopes, safeguarded Newton | Core and JAX; no individual recipe or integration |
| `_src/recipes/<family>_gen.py`, `_gen_common.py` | High-precision sampling, fitting, deterministic emission | Build dependencies and shared generator helpers |
| `_src/recipes/*_table.py` | Coefficients and their provenance | NumPy; no runtime reconstruction |
| `_src/recipes/<family>.py` | Family transformations, evaluation, derivatives, factories | Core, shared helpers, tables, and explicitly needed related families |
| `_src/pergroup.py` | Static grouping around a uniform-parameter function | JAX and NumPy; no family dispatch |
| `domains.py` | Public parameter intervals | Table metadata only |
| `bake/` | Emit supported recipe computations as JAX or C++ | Recipe/series protocol and traced JAX operations |
| `numpyro.py`, `pytensor.py` | Optional distribution and compiler adapters | Public numerical functions and domains; their respective optional dependency |
| `__init__.py` | Curated public exports and package version | Library entry points; no optional integration |

Lower layers must not import optional adapters. Runtime evaluation must not
import generators or mpmath. JAX and NumPy are the only direct third-party
runtime requirements; standard-library imports are allowed. Build dependencies
belong behind `[gen]`, and each adapter must name its extra when unavailable.
Importing chebax must not construct JAX arrays or initialize a backend.

`domains.py` reads its bounds from table modules only, so runtime modules and
adapters import it at module level without a cycle. A bound that is not a
table axis is derived from the table's own metadata, as gammainc's infinite
upper end is from the Temme-zone axis v = 10/a reaching 0.

Related families may share mathematical building blocks: Matérn uses Bessel K,
spherical Bessel functions wrap cylindrical ones, and quantiles use their
forward distributions. Such dependencies stay within the numerical layer.
Adapters own framework wiring, constraints, sampling, and broadcasting; shared
root solvers and special-function identities belong in the numerical layer.

## Construction and evaluation

The build path is:

```text
high-precision reference -> transformed samples -> Chebyshev coefficients
                        -> deterministic NumPy table module
```

The runtime path is:

```text
uniform parameter -> reconstruct coefficients -> ChebSeries -> array of values
```

All fit stages in a recipe generator stay in mpmath at the recorded precision
before one final float64 conversion. Tables record axes, parameter domains,
node counts, working precision, mpmath version, and the generator/shared-helper
source hash. Slow tests compare emitted modules byte for byte. The wide-beta
and hyp1f1 tables have documented partial CI regeneration checks; use the full
`CHEBAX_FULL_REGEN=1` checks when their omitted panels change.

An eager factory uses NumPy reconstruction and a bounded private cache. Its
key includes normalized parameters, structural options, and canonical dtype.
The public wrapper normalizes concrete cache keys; construction validates
parameters before reconstruction. A traced counterpart
uses JAX reconstruction without inserting traced results into a Python cache.
Constructing an eager factory during tracing is supported where its inputs are
concrete; a traced parameter uses the direct numerical entry point.

`ChebSeries`, `PiecewiseCheb`, and `Recipe` instances are immutable. Recipe
subclasses declare static fields and series children, take them in that order,
and register as pytrees. `_post_init` must derive valid state both during normal
construction and during tree reconstruction, without assuming every child is
a concrete host array. `astype` and `truncate` return new objects.

## Input and dtype contracts

Table parameters are uniform per call: a scalar or an array containing one
value. Normalize them through `canon_param`; reject larger arrays with the
parameter name and guidance to `pergroup`. Evaluation points may be arrays.
Some direct functions, notably Tweedie, additionally broadcast parameters such
as mean and dispersion; their own docstrings state which arguments broadcast.
The NumPyro constructors follow the same rule for shape parameters.

Public entry points must apply the same normalization to positional and
keyword arguments before custom JVP rules receive them. A rule that lives on a
public name cannot do this, because the tangent exists before its body runs;
`_common.float_params` wraps a private rule and binds keywords to its
signature first. Numeric integer inputs must behave like the corresponding
floating inputs, evaluation points included. Structural integers such as
degrees and group indices are validated separately.

Recipe construction and canonicalized direct functions use the active JAX
float dtype: float32 by default, float64 with x64 enabled. Dtype-sensitive
caches include that setting. A series also carries its own coefficient dtype;
`astype` is the explicit conversion path. Do not change the caller's global
JAX precision configuration from library code.

Factories reject invalid concrete parameters with a named `ValueError`.
Traced functions can check shapes statically and values with JAX masks;
individual docstrings specify whether invalid values return NaN or are outside
the accuracy contract. A traced value does not justify silently extrapolating
a table. Define endpoint values, behavior outside an argument interval,
infinities, and NaNs for each family rather than assuming a common clamp.

## Derivatives and numerical stability

Series evaluation has a custom JVP for derivatives in the coordinate and the
coefficients. A recipe may use plain AD through reconstruction and composition,
or an analytic custom JVP where an endpoint or conditioning requires it. State
the supported derivative order; an exact derivative of an approximant still
has approximation and floating-point error relative to the target function.

Every piecewise branch must receive safe inputs before operations that can
produce invalid values or derivatives. An output `where` does not undo
`0 * inf` or `0 * NaN` in AD. Active lanes retain raw coordinates at seams.
Test mixed arrays, since a safe scalar branch can still fail when vectorized.
Padding and inactive groups must not contribute to either values or gradients.

Inverse CDFs solve in a variable that resolves the relevant tail, using the
shared safeguarded fixed-iteration Newton method. Enforce the post-solve
residual policy and return NaN on unresolved solves. Differentiate with the
implicit function theorem, not through the iteration. Preserve log-CDF,
log-survival, and complementary probabilities directly when subtraction from
one would lose them.

Keep singular factors and exponentials separate from smooth tabulands.
Logarithmic and scaled functions must avoid reconstructing an overflowing or
underflowing unscaled function. Use stable coordinate maps and calculus
scalings in both runtime and emitted artifacts. Calibrate each family's error
metric and tolerance against independent high-precision references.

## Baking boundary

Both emitters trace the instance's own computation. They support a deliberate
set of JAX primitives and custom rules; an unsupported operation must raise
instead of emitting a guessed translation. Add every supported recipe class
to the round-trip tests, including relevant structural options and seams.

Artifacts declare a dtype. Float32 emission executes the traced computation
in float32, so dtype-dependent runtime branches can differ. Endpoint slopes
carried by `edge_slope` are deliberately dropped and disclosed in the artifact.
Other primitive-level differences are documented in `bake`'s module docstring.
Tracing a shared source does not establish complete value or AD equivalence;
check generated Python and compiled C++ independently. Output names must avoid
language keywords, reserved identifiers, imports, constants, and helper names.

## Documentation and enforcement

The README owns installation, capabilities, and entry points. This document
owns component responsibilities and numerical contracts; the style guide owns
names and presentation. The recipe guide owns the contribution workflow.
`PROJECT.md`, `docs/increments.md`, and dated reviews retain evidence and
history; their old milestone descriptions do not override current contracts.
`CLAUDE.md` holds agent operating instructions and local execution commands.

CI tests the supported JAX floor and a latest resolution, optional dependency
absence, float32 defaults, table regeneration, integrations, and notebooks.
The endpoint and argument contract matrices live in `test_contracts.py`,
`test_int_params.py`, and `test_x64_off.py`; extend them for shared behavior.
Family tests own accuracy sweeps and formula-specific derivative identities.

`test_layers.py` checks the dependency directions in the table above
statically, function-local imports included, and CI checks black formatting
(see [Coding style](coding-style.md)). There is no linter; review checks
shared-helper reuse, naming, and documentation manually. The
[October review](review-2026-10-26.md) records what was found and how each
item was resolved. Numerical tests are no substitute for architecture review.
