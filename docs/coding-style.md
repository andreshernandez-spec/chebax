# Coding style

These conventions apply to handwritten library code, generators, tests, and
examples. They define the review standard for new work and for code being
changed. [The October review](review-2026-10-26.md) records the gaps found
against them and what is still open. The dependency and numerical contracts
are in [Architecture](architecture.md).

## Formatting and imports

Code is formatted with black, at the version the `lint` extra pins and with
the settings in `pyproject.toml` (88 columns). Run `black src tests examples`
before committing; CI checks the same paths. Black decides layout; where it
splits a formula awkwardly, rename an intermediate or restructure the
expression rather than fighting the formatter. Group imports as standard
library, third party, then chebax, with a blank line between groups. Use `np`
for NumPy and `jnp` for JAX NumPy.

Generated `*_table.py` files follow `_gen_common.write_table_module` exactly.
Do not format, rename symbols in, or edit them by hand; black excludes them.
Changing a generator, including a comment or a reformat, changes its recorded
source hash and requires regenerating its outputs. Changing `_gen_common.py`
requires regenerating every table. Black also leaves `experiments/` alone,
since a changed experiment must be re-run and its result updated, and the
executed notebooks.

There is no linter or type checker. Naming and the rest of this guide are
reviewed manually.

## Names

Use `snake_case` for functions, methods, arguments, and modules; `CapWords` for
classes; uppercase for module constants. A leading underscore marks an internal
implementation detail. A name without an underscore inside `_src` can still be
internal: the supported API is what the public modules export.

Short mathematical names are appropriate inside a formula: `x`, `a`, `b`, `p`,
`nu`, `rho`, `phi`. Define transformed coordinates in the function or module
docstring, including their domains. Prefer `nu` for new Bessel-order arguments;
keep existing public `v` arguments compatible. Use descriptive names for
control flow, domains, caches, and reconstructed series: `in_box`, `inner`,
`log_tail`, `series`, `coefs`, `lower`, `upper`. Avoid adding unrelated meanings
to a short name within one function.

Keep these API conventions:

| Form | Meaning |
| --- | --- |
| `name(params)` | Cached factory returning a callable recipe |
| `name_fn(params, x)` | Traced-parameter counterpart of a factory |
| `name(params, x)` | Direct function with no factory counterpart |
| `log_name...` | A logarithmic value computed without first forming an underflowing or overflowing value |
| `name_dnu(params)` | Callable giving the order derivative |
| `scaled=True` | The family's documented exponential scaling |

Function parameters precede the evaluation point for new numerical APIs.
Existing exceptions, such as `lambertw(x, k=0)` and optional configuration
arguments, keep their signatures. Do not rename public keywords to tidy an
internal formula. Both positional and keyword calls must obey the same input
contract. Internal aliases should disambiguate two implementations, not invent
another name for a shared helper.

## Functions and module organization

Keep a public entry point small: normalize inputs, select the implementation,
and return its result. Separate reconstruction, numerical evaluation, and
derivative rules when they have different invariants. Give shared helpers a
single owner and reuse them before adding another implementation.

Order a recipe module so that its numerical design is readable: module
docstring, imports and constants, evaluation helpers, recipe classes, private
cached construction, and public entry points. Keep custom JVP implementations
and their rules adjacent. A wrapper must preserve the documented signature and
normalize every argument before the differentiation rule receives it.

Keep family-specific mathematics in the family module. Extract a helper only
when it expresses a shared numerical or structural contract. Do not create a
new base class just to save a few lines of a formula. Split a module when its
parts have independent responsibilities, tests, and callers; line count alone
does not determine the boundary.

## Docstrings and comments

A public numerical entry point must state what it computes, parameter and
argument domains, scalar versus broadcasting behavior, dtype policy, endpoint
and invalid-input behavior, and available derivatives. State a limitation next
to the affected API, including unsupported higher derivatives. Add a small
example when the call pattern is otherwise unclear.

A recipe's module docstring explains the factorization, coordinate maps,
panels, seams, and numerical floor. Its generator docstring records measured
degrees and their evidence. Put a formula beside the code whose stability
depends on it, and explain why a tempting rearrangement loses accuracy.

Comments explain a choice, a boundary condition, or an invariant. Use plain
words and short sentences. Avoid narrating assignments, repeating the code,
or retaining a review conversation in a helper docstring. Keep review chronology
and superseded decisions in `docs/increments.md` or a dated review, with a short
link from the implementation when needed. Current degree measurements, floors,
and numerical traps stay beside the relevant generator, runtime, or test.

## Numerical expressions

Use the NumPy polynomial coefficient convention: plain `c[0]`, lowest degree
first, with each coordinate mapped to `[-1, 1]`. Name and document any other
coordinate before using it. Distinguish node counts from polynomial degrees.

An algebraic simplification must preserve behavior in both supported
precisions, at seams, and under differentiation. In particular, mask unsafe
inputs before logarithms, division, powers, and exponentials; selecting their
outputs afterward is insufficient. Preserve the raw input in the active
branch so seam tangents are not split by a clamp. Keep logarithmic and scaled
forms in those representations until the final result.

Use an explicit policy for integer inputs and structural arguments. Promote
numeric inputs before a custom JVP; validate structural integers rather than
silently truncating them. Use tolerances justified by the family's measured
error, never a package-wide unexplained epsilon.

## Tests and review

Test observable contracts. For a numerical change, check the affected values
and derivatives against an independent reference, including the seam or tail
that motivated the change. Include forward and reverse AD when masking or
padding changes, and JIT when shapes, tracing, or dispatch changes. Test integer,
size-one, and keyword inputs when changing argument normalization.

Reject unexpected nonfinite values before aggregating errors. In particular,
`max(worst, error)` can discard a `NaN`; a final bound on `worst` is not a
finiteness check. Test expected infinities and NaNs separately. State the
error metric, measured worst case, and acceptance bar in the test docstring.
Use mpmath at 40 dps or more, or an independent exact identity; float64 SciPy
is useful for cross-checking but does not establish a float64 accuracy floor.

Run the relevant tests during development and the documented acceptance suite
before committing numerical changes. Changes to a generator require its
byte-for-byte regeneration check. Changes to an experiment require rerunning
it and updating the captured result. A documentation-only change needs link,
example, and diff checks, not a numerical suite rerun.
