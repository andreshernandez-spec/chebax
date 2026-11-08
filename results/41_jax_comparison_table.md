# chebax against JAX: speed and accuracy

From results/33_jax_release_race.txt (jax.scipy, jax 0.12.0.1, chebax 349579b) and results/40_bessel_real_order_race.txt (TFP 0.26.0-dev20261107, chebax 8cf5d7c); made by experiments/41_jax_comparison_table.py.

The two commits hold the same src/.

## Speed

chebax's speedup, value / value_and_grad in x; one pinned CPU core at N = 2^16, RTX 3080 Laptop GPU at N = 2^20. Below 1x chebax is slower.

| function | against | f64 CPU | f64 GPU | f32 CPU | f32 GPU | values vs scipy, 1 core (f64 / f32) |
|---|---|---|---|---|---|---|
| erfcx | jax | 1x / 0.99x | 2.4x / 3.2x | 0.58x / 0.5x | 1.1x / 1.8x | - |
| dawsn | jax | 0.72x / 0.53x | 1.5x / 0.97x | 0.68x / 0.56x | 0.87x / 0.96x | - |
| I0, I1 | jax | 0.58-0.6x / 0.48-0.55x | 0.91x / 0.9-0.97x | 0.72-0.76x / 0.53-0.58x | 0.96-1x / 0.69-0.96x | - |
| i0e, i1e | jax | 0.88-0.89x / 0.72x | 0.92-0.93x / 0.84-0.9x | 0.85-0.91x / 0.6-0.63x | 0.91-1.1x / 0.9-1.1x | - |
| J_n, integer n | jax | 4-4.2x / 9.6-11x | 5.9-6x / 7.1-7.2x | 3.1x / 5.9-11x | 95x / 88-99x | - |
| betainc | jax | 9.3-9.5x / 8.7-9.5x | 17-21x / 16-20x | 4.9-7.7x / 5.1-7x | 43-59x / 11-22x | - |
| gammainc(c), one a | jax | 15-26x / 10-19x | 3.5-3.9x / 4-5.4x | 7.9-11x / 6.5-7.8x | 38-62x / 5.8-10x | - |
| gammainc, a per element | jax | 2.7x / 5.1x | 7.4x / 10x | 2.4x / 4.5x | 22x / 16x | - |
| hyp1f1 | jax | 16-17x / 24x | 22x / 32-33x | 6.8-7x / 13-14x | 190-200x / 30-32x | - |
| stdtr (t CDF) | jax | 3.9x / 3.3x | 4.9x / 4.5x | 3x / 2.8x | 41x / 34x | - |
| J_v, real v | none in JAX | - | - | - | - | 9.4-11x / 21-22x |
| Y_v, real v | none in JAX | - | - | - | - | 8.3-15x / 20-34x |
| I_v | TFP | 19x / 18x | 18x / 18-19x | 6.8-6.9x / 6.7x | 150-160x / 160-170x | 6.7-7x / 17-18x |
| I_v scaled | TFP | 21x / 21x | 18-19x / 18-19x | 7.3-7.5x / 6.7-7x | 160-170x / 140-150x | 9.5-9.7x / 23-24x |
| K_v, K_v scaled | TFP | 10-11x / 12x | 11x / 11x | 2.6-2.8x / 3-3.2x | 50-54x / 58-61x | 3.5-5.4x / 7.9-12x |
| log K_v | TFP | 13-14x / 26-29x | 13-14x / 20-21x | 4-4.5x / 7.3-8.8x | 74-77x / 100-110x | 4.2-6x / 11-16x |
| I_{v+1}/I_v | TFP | 1.3-1.4x / 0.78-0.92x | 0.98-1x / 0.97-1x | 0.69-0.76x / 0.55-0.58x | 24-25x / 7.6-8.1x | 11-12x / 24-25x |

## Accuracy

Worst error against mpmath, JAX side / chebax, CPU and GPU.

| function | metric | f64 value | f64 d/dx | f32 value | f32 d/dx | f64 parameter derivative |
|---|---|---|---|---|---|---|
| erfcx | rel | 2.7e-16 / 6.1e-16 | 3.8e-16 / 6.0e-16 | 1.8e-07 / 4.6e-07 | 2.4e-07 / 6.4e-07 | - |
| dawsn | rel | 3.6e-16 / 6.0e-16 | 2.8e-08 / 1.3e-14 | 2.2e-07 / 1.7e-07 | 1.2e+01 / 1.2e-05 | - |
| I0, I1 | rel | 7.4e-16 / 1.2e-15 | 1.6e-15 / 1.3e-15 | 7.1e-07 (nan) / 2.7e-07 | 8.4e-07 (nan) / 3.4e-07 | - |
| i0e, i1e | rel | 7.7e-16 / 1.3e-15 | 4.9e-12 / 6.2e-15 | 6.9e-07 / 3.1e-07 | 2.3e-03 / 6.6e-06 | - |
| J_n, n = 0-10 | sup | 1.7e+02 (nan) / 2.1e-15 | 4.2e+06 (nan) / 3.5e-15 | 1.7e+02 (nan) / 4.9e-07 | 1.9e+01 (nan) / 2.1e-06 | - |
| betainc | abs | 1.0e-13 / 1.4e-13 | 2.8e-13 / 2.8e-13 | 2.1e-05 / 2.3e-05 | 4.5e-05 / 4.5e-05 | none / 2.8e-13 |
| gammainc(c), one a | abs | 3.1e-13 / 3.3e-15 | 1.2e-12 / 6.5e-12 | 2.9e-04 / 1.5e-06 | 9.2e-04 / 5.6e-03 | 8.3e-15 / 4.4e-15 |
| gammainc, a per element | abs | 3.1e-13 / 5.6e-16 | 1.2e-12 / 2.8e-08 | 2.9e-04 / 2.6e-07 | 9.2e-04 / 3.1e-03 | 8.3e-15 / 1.5e-13 |
| hyp1f1 | rel_lnM | 6.4e-16 / 3.6e-13 | 5.8e-15 / 1.1e-12 | 3.7e-07 (nan) / 3.4e-06 | 2.2e-06 (nan) / 8.5e-06 (nan) | 1.0e+00 / 1.4e-10 |
| stdtr | abs | 3.7e-12 / 3.6e-15 | 7.3e-09 (nan) / 5.9e-14 | 1.0e-03 / 3.1e-06 | 1.0e+00 (nan) / 7.4e-06 | none / 3.9e-16 |
| J_v, real v | sup | - / 1.4e-15 | - / 4.9e-15 | - / 5.7e-07 | - / 1.3e-06 | - |
| Y_v, real v | mod | - / 1.2e-14 | - / 1.7e-14 | - / 3.2e-06 | - / 6.2e-06 | - |
| I_v | rel | 2.9e-15 / 5.7e-15 | 3.4e-15 / 5.8e-15 | 1.0e+00 / 8.7e-07 | 1.0e+00 / 9.2e-07 | 0 / 1.8e-13 |
| I_v scaled | rel | 2.9e-15 / 5.8e-15 | 1.3e-11 / 3.7e-14 | 1.0e+00 / 8.5e-07 | 1.0e+00 / 1.7e-05 | 0 / 1.8e-13 |
| K_v | rel | 3.8e-15 / 1.2e-14 | 3.0e-15 / 1.1e-14 | 2.4e-06 / 6.4e-06 | 1.4e-06 / 1.1e-02 (nan) | 0 / 2.1e-09 |
| K_v scaled | rel | 4.0e-15 / 1.2e-14 | 3.4e-11 / 1.2e-14 | 2.5e-06 / 6.5e-06 | 1.2e-02 / 9.9e-06 (nan) | 0 / 6.4e-12 |
| log K_v | lrel | 2.2e-15 / 3.7e-15 | 3.2e-14 / 1.6e-15 | 1.2e-06 / 2.9e-06 | 7.8e-06 / 8.6e-07 | 0 / 6.4e-12 |
| I_{v+1}/I_v | rel | 3.4e-16 / 6.1e-15 | 2.1e-08 / 2.1e-12 | 2.2e-07 / 1.1e-06 | 1.4e+01 / 5.7e-04 | 0 / 1.2e-12 |

rel: relative. sup: error / max |J| of the order. mod: error / sqrt(J^2 + Y^2). abs: absolute (CDFs). lrel: error / max(1, |ln K|). rel_lnM: hyp1f1's relative error / max(1, |ln M|), its derivatives as errors in d ln M. A CDF's d/dx (split) is relative where the density is at least 1e-4 of the CDF, absolute elsewhere. (nan): some points nan or inf where the true value is finite. Parameter derivative: none = jax has no gradient there (betainc in a, b; the t CDF in nu); 0 = TFP returns exactly 0 for the gradient in the order.
