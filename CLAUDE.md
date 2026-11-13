# CLAUDE.md — operating instructions for chebax

Read `PROJECT.md` first: goal, scope boundary, evidence, milestones, risks.
This file is how to work here.

Follow `docs/architecture.md` for dependency and numerical contracts, and
`docs/coding-style.md` for naming, formatting, and documentation. Open items
from the latest review are in `docs/review-2026-10-26.md`.

> **Read `../CLAUDE.md` before anything else.** It governs: the OSS git identity (§1),
> the `gh` restriction (§1.2), never push / never open a PR / never publish (§2), the
> `open-source` conda env (§3), and writing style for anything upstream-facing (§4).
> "Never publish" includes PyPI: no `twine upload`, no `flit publish`, no reserving
> names. Releases are Andres's action.

## Project-specific rules

- **Performance claims cite a measured experiment or nothing.** The number
  comes from a script in `experiments/` and its captured output in
  `results/`, named where the claim is made (docstring, increments entry, PR
  text); read those files for the figures instead of repeating them here.
  Consumer-GPU fp64 is compute-bound (1:64): never claim "memory-bound" there;
  datacenter fp64 stays analytical until measured on such a part.
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
- **chebtorch ports chebax** (`../chebtorch`, andreshernandez-spec/chebtorch;
  Tweedie so far). What a function computes is decided here: tables, regions
  and splits, formulas, references, test points. chebtorch exports the tables
  and ports the code, so a change to a ported family's numerics lands here
  first and is ported after. What chebtorch had to do first is listed in its
  `docs/divergences.md` until a PR here takes it, with the test points that
  found it and the floors measured again.
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
JAX_PLATFORMS=cpu python -m pytest -q -m "not slow" -n 4
                                                  # quick loop, ~21 min (57 min in one
                                                  #  process, 2h17m on the laptop GPU;
                                                  #  measured 2026-10-26); four workers
                                                  #  peak at ~15.5 GB together
python -m pytest -q -m slow -n 4                  # table regeneration, ~4.5 min; needs the
                                                  #  [gen] mpmath pin. Run both before
                                                  #  commits; CI runs them as separate
                                                  #  jobs on every push (bake header test
                                                  #  skips without g++; conftest enables x64)
python -m black src tests examples                # format (the lint extra pins black;
                                                  #  CI runs --check). Reformatting a
                                                  #  generator changes its source hash:
                                                  #  regenerate its table too
python -m chebax._src.recipes.besselk_gen         # regenerate the besselk log-tables
python -m chebax._src.recipes.besseli_gen         # regenerate the besseli log-tables
python -m chebax._src.recipes.bessely_gen         # regenerate the bessely tables
python -m chebax._src.recipes.erf_gen             # regenerate the dawsn/erfcx tables
python -m chebax._src.recipes.betainc_gen         # regenerate the betainc tensor (~5 min)
python -m chebax._src.recipes.betainc_wide_gen    # regenerate the wide panels (~25 min;
                                                  #  CI regen-checks only the small HILO
                                                  #  panel; run the full check before a
                                                  #  release with CHEBAX_FULL_REGEN=1 pytest
                                                  #  tests/test_betainc.py -m slow)
python -m chebax._src.recipes.gammainc_gen        # regenerate the gammainc log-tables
python -m chebax._src.recipes.gammainc_large_gen  # regenerate the Temme-zone tables (~4 s)
python -m chebax._src.recipes.gammainc_array_gen  # regenerate the per-element patches (~1 min)
python -m chebax._src.recipes.gammaincinv_array_gen # regenerate the quantile table (~2.5 min)
python -m chebax._src.recipes.gammaln_gen         # regenerate the gammaln tables (~3 s)
python -m chebax._src.recipes.ndtri_exp_gen       # regenerate the ndtri_exp table (~13 s)
python -m chebax._src.recipes.hyp1f1_gen          # regenerate the hyp1f1 log-tables (~11 min;
                                                  #  CI regen-checks only the tail table; full
                                                  #  check behind CHEBAX_FULL_REGEN=1)
python -m chebax._src.recipes.stdtr_gen           # regenerate the stdtr slice tables (~4 min)
python -m chebax._src.recipes.vonmises_gen        # regenerate the von Mises table (~5 min)
python -m chebax._src.recipes.besselj_gen         # regenerate the baked nu-table
python -m chebax._src.recipes.tweedie_gen         # regenerate the Tweedie tail table (~25 s)
python -m chebax._src.recipes.skewnorm_gen        # regenerate the skew-normal patches (~16 min)
```

Tests run on whatever backend jax picks; the accuracy contract is
device-independent. Run the suite on the CPU anyway: it is bound by XLA
compilation, and the laptop GPU compiles the table-heavy programs 3-6x slower
(measured 2026-10-26).

## Releasing

1. A PR sets `version` in `pyproject.toml`, the one place it is written
   (`chebax.__version__` reads the installed metadata).
2. Before tagging: the tests run on main is green for the merge commit, and
   so is the full regeneration CI skips, `CHEBAX_FULL_REGEN=1 python -m pytest
   -m slow tests/test_betainc.py tests/test_hyp1f1.py` (~35 min).
3. `git tag -a vX.Y.Z <merge commit>` and push the tag; that push is the
   release, so it is Andres's. `.github/workflows/release.yml` checks the tag
   against the version and the tests run, builds, checks what the sdist and
   wheel hold, installs the wheel in a clean venv, and publishes through PyPI
   trusted publishing (workflow `release.yml`, environment `pypi`, no token
   anywhere). A PyPI version can never be reused: a broken upload is a new
   patch version.
4. The GitHub release notes go on the tag; they are drafted in `drafts/`.

## Layout

```
PROJECT.md      the plan and evidence
CLAUDE.md       this file
README.md       orientation, quickstart, capability table
docs/           architecture.md (contracts), coding-style.md (conventions),
                adding-a-recipe.md (workflow), increments.md (design log), reviews
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
