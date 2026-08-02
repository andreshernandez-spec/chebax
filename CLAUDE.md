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
  on an RTX 3080 Laptop (`../bessel/experiments/07`); besselj DIRECT vs jax's own
  `bessel_jn` (`experiments/08`, integer order only, agreement ≤ 1.7e-15): f64
  7.4–7.5× at v=2 and 6.0× at v=9 on n_iter=50's valid window, 14.4–14.5× where
  the x-range forces n_iter=100 (those rows at N=2^22: bessel_jn's memory scales
  with n_iter·N, a 13 GiB attempt OOMs at 2^24), f32 3.6–7.1×; bessel_jn is
  integer-order only and valid on an n_iter-dependent x-window that narrows from
  both sides (nans below ~1e-4 at n_iter=50, below ~0.1 at n_iter=100) — cite
  the window when citing the ratio; narrow-domain 2.5–3.0×
  (`experiments/04`); igamma REAL race (`experiments/05`, re-run 2026-07-31 with
  the gammainc recipe, agreement ≤ 3.7e-15): f64 10–27× vs jax's gammainc on GPU
  for a in the [0.1, 10] box (a-dependent; jax's loop count grows with worst-lane
  trips), f32 3.0–5.8×; the 18–54× f64 figures are the MOCK's op-profile ceiling,
  still the only number for a outside the box — say which when citing; gammaincinv
  solver rewire (`experiments/09`, same solver, residual swapped): 6.3–23.4× f64
  for uniform p at a in {9.9, 3.5, 0.5}, 1.4× on pure deep-tail p (jax's series
  branch is cheap there), path agreement ≤ 2.2e-12, and the JVP's dP/da term
  4.0–15.3× vs igamma_grad_a (agreement ≤ 1.4e-14); betainc race
  (`experiments/06`, both sides real implementations, f64 agreement ≤ 4e-14):
  f64 79–133× vs jax's betainc on GPU ((a,b)-dependent), f32 13–16×, the 500k
  CPU case 202×, stdtr 59–60× f64 vs the betainc-composed form, ratio flat in N
  from 2^20 up; per-group crossover (`experiments/07`, f64, vmap over equal
  groups): reconstruction overhead ≤ 1.07× vs the uniform floor for
  n/group ≥ 16k (betainc, N = 2^24), besselk ≤ 1.08× down to n/group = 1024,
  and vs jax's betainc the per-group path wins every measured cell, worst 7.3×
  (16384 groups of 64), 44–98× at MCMC scale (4 groups); f32 truncation race (`experiments/12`):
  truncate(1e-7) speeds f32 GPU evaluation 1.4–2.2× (besselj 2.0×, besselk
  2.2×, betainc 1.4×) at each family's f32 floor — the "extra terms are free"
  roofline guess was wrong on this part. Consumer-GPU fp64 is
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
python -m chebax._src.recipes.betainc_wide_gen    # regenerate the wide panels (~25 min;
                                                  #  CI regen-checks only the small HILO
                                                  #  panel; run the full check before a
                                                  #  release with CHEBAX_FULL_REGEN=1 pytest
                                                  #  tests/test_betainc.py -m slow)
python -m chebax._src.recipes.gammainc_gen        # regenerate the gammainc log-tables
python -m chebax._src.recipes.gammainc_large_gen  # regenerate the Temme-zone tables (~4 s)
python -m chebax._src.recipes.hyp1f1_gen          # regenerate the hyp1f1 log-tables (~11 min;
                                                  #  CI regen-checks only the tail table; full
                                                  #  check behind CHEBAX_FULL_REGEN=1)
python -m chebax._src.recipes.stdtr_gen           # regenerate the stdtr slice tables (~4 min)
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
