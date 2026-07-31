# CLAUDE.md — operating instructions for chebax

Read `PROJECT.md` first: goal, scope boundary, evidence, milestones, risks.
This file is how to work here.

> **Read `../CLAUDE.md` before anything else.** It governs: the OSS git identity (§1),
> the `gh` restriction (§1.2), never push / never open a PR / never publish (§2), the
> `open-source` conda env (§3), and writing style for anything upstream-facing (§4).
> "Never publish" includes PyPI: no `twine upload`, no `flit publish`, no reserving
> names. Releases are Andres's action.

## Project-specific rules

- **Performance claims cite B3 (measured 2026-07-31) or nothing.** The citable
  numbers: besselj f64 4.5–6.0× vs cephes::jv (0.4× on pure-outer), f32 35× the f64,
  on an RTX 3080 Laptop (`../bessel/experiments/07`); narrow-domain 2.5–3.0×
  (`experiments/04`); igamma headroom 18–54× f64 against a mock (`experiments/05`,
  a cost model, not an implementation — say so when citing); betainc race
  (`experiments/06`, both sides real implementations, f64 agreement ≤ 4e-14):
  f64 79–133× vs jax's betainc on GPU ((a,b)-dependent), f32 13–16×, the 500k
  CPU case 202×, stdtr 59–60× f64 vs the betainc-composed form, ratio flat in N
  from 2^20 up. Consumer-GPU fp64 is
  compute-bound (1:64): never claim "memory-bound" there; datacenter fp64 stays
  analytical until measured on such a part.
- **The accuracy metric is chosen per family** and stated in each test file's
  docstring: sup-normalized for oscillatory J (relative accuracy at zeros is a known
  impossibility, `../bessel/PROJECT.md` §4 q4), modulus-relative (err/√(J²+Y²)) for Y,
  pointwise-relative for positive functions (K, I, erfcx), absolute for CDFs,
  two-sided for quantile inversions. Test bars sit at ~4× the measured worst case;
  both numbers are recorded in the test docstring.
- **Tables are regenerable bit-for-bit.** Any checked-in coefficient table carries
  mpmath version, dps, domain, degree, and a sha256 of the generator source (plus
  `_gen_common`) in META, and a slow test that regenerates the COMPLETE module and
  compares it byte for byte. Consequence: any edit to a generator or to
  `_gen_common.py`, even a comment, changes the recorded hash — rerun that
  generator (all of them for `_gen_common`) as part of the same change or the
  slow test fails.
- **Runtime imports are `jax` + `numpy` only.** mpmath/scipy live behind the `[gen]`
  extra. If a runtime module imports mpmath, that is a bug.
- **Coefficient conventions follow `numpy.polynomial`** (plain c0, lowest-first,
  domain mapped affinely to [-1,1]). No private conventions.
- **Don't re-derive the bessel evidence.** Degree tables, roofline, seam behavior,
  dead ends: measured in `../bessel/`, linked from `PROJECT.md` §2. Re-run their
  scripts if staleness is suspected; don't rebuild them here.

## Experiments

Same conventions as `../bessel/`: every script in `experiments/` is self-contained and
self-explaining — docstring states what is measured, what it means, and what it does
NOT establish. References use mpmath at ≥40 dps; never validate float64 against float64
scipy. When a script changes, re-run it and update `results/`; a stale results file is
worse than none.

```bash
python experiments/01_derivative_accuracy.py      # ~5 s
python experiments/02_coeff_smoothness_in_nu.py   # ~10 s
```

## Library

```bash
python -m pip install -e . --no-build-isolation   # once, into the open-source env
python -m pytest -q -m "not slow"                 # quick loop, ~3.5 min (skips regen;
                                                  #  mpmath references are the rest)
python -m pytest -q                               # full suite ~10 min; run before commits
                                                  # (bake header test skips without g++;
                                                  #  conftest enables x64; CI runs the
                                                  #  full suite on every push)
python -m chebax._src.recipes.besselk_gen         # regenerate the besselk log-tables
python -m chebax._src.recipes.besseli_gen         # regenerate the besseli log-tables
python -m chebax._src.recipes.bessely_gen         # regenerate the bessely tables
python -m chebax._src.recipes.erf_gen             # regenerate the dawsn/erfcx tables
python -m chebax._src.recipes.betainc_gen         # regenerate the betainc tensor (~5 min)
python -m chebax._src.recipes.vonmises_gen        # regenerate the von Mises table (~5 min)
python -m chebax._src.recipes.besselj_gen         # regenerate the baked nu-table
```

Tests run on whatever backend jax picks (the laptop GPU when present); the
accuracy contract is device-independent, so that is fine.

## Layout

```
PROJECT.md      the plan and evidence
CLAUDE.md       this file
README.md       orientation, quickstart, capability table
docs/           adding-a-recipe.md (the recipe workflow), increments.md (design log)
experiments/    reproducible measurements, incl. 03_degree_measurement.py which
                reproduces every degree claim in the generator docstrings
notebooks/      executed example notebooks, themed (see notebooks/README.md);
                re-execute with nbconvert --execute --inplace after API changes;
                numpyro/matplotlib come from the [examples] extra
results/        captured output, checked in
drafts/         prose for upstream (PR text, issues) — Andres rewrites and
                sends. LOCAL ONLY: never committed (kept out via
                .git/info/exclude, purged from history 2026-07-31); outreach
                strategy does not belong in a public library repo
src/chebax/     the library: _src/{algorithms,series,generate,pytree}.py (core),
                _src/recipes/ (per-family gen + baked table + runtime, with
                _common.py and _gen_common.py as the shared machinery), bake/
tests/          per-recipe acceptance tests; test_besselk.py is the template
```
