# CLAUDE.md — operating instructions for chebax

Read `PROJECT.md` first: goal, scope boundary, evidence, milestones, risks.
This file is how to work here.

> **Read `../CLAUDE.md` before anything else.** It governs: the OSS git identity (§1),
> the `gh` restriction (§1.2), never push / never open a PR / never publish (§2), the
> `open-source` conda env (§3), and writing style for anything upstream-facing (§4).
> "Never publish" includes PyPI: no `twine upload`, no `flit publish`, no reserving
> names. Releases are Andres's action.

## Project-specific rules

- **No performance claims until bessel B3 runs.** chebax claims accuracy and gradients.
  Speed language ("fast", "N×") stays out of code, docs, and drafts until
  `../bessel/PROJECT.md` B3 produces a measured number to cite.
- **Accuracy contract is sup-normalized (absolute).** Relative accuracy at function
  zeros is a known impossibility; see `../bessel/PROJECT.md` §4 q4. Every number in
  docs/results states the metric.
- **Tables are regenerable bit-for-bit.** Any checked-in coefficient table carries
  generator version, mpmath dps, domain, and degree in metadata, and a test that
  regenerates and diffs it.
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

## Layout

```
PROJECT.md      the plan and evidence
CLAUDE.md       this file
README.md       short orientation
experiments/    reproducible measurements
results/        captured output, checked in
drafts/         prose for upstream (PR text, issues) — Andres rewrites and sends
src/chebax/     the library, once M1 starts (does not exist yet)
```
