#!/usr/bin/env python3
"""Measure the low-rank structure of every baked coefficient table.

WHAT THIS MEASURES
------------------
The queued low-rank compression item (PROJECT.md): coefficient tables of
smooth bivariate functions are rank-structured (Chebfun2; Townsend &
Trefethen 2013), so c(param, k) ~ sum_j sigma_j u_j(param) v_j(k) with
few terms. For every 2-D table: the singular value spectrum and the
epsilon-ranks at 1e-15 and 1e-7 of sigma_0 (f64-grade and f32-grade
truncation). For the 3-D betainc tensor: the mode-k unfolding ranks
(the Tucker core shape a HOSVD would keep). From the ranks, two COST
MODELS (labeled as such, not measurements): the per-instantiation FLOP
ratio dense/compressed (dense m*n contraction vs r*(m+n) plus the small
core for 3-D) and the storage ratio.

WHAT IT DOES NOT MEASURE
------------------------
Runtime speedups (instantiation is once per call and already cheap at
2-D; the per-group crossover in experiments/07 bounds what faster
reconstruction could buy); accuracy of a compressed evaluation path
(building one is the follow-up this script gates); f32 evaluation error
(the 1e-7 rank is a truncation budget, not an end-to-end error).

Run:  python experiments/10_table_rank_structure.py   (~5 s, CPU)
"""

import sys

import numpy as np

sys.path.insert(0, "src")


def eps_rank(svals, eps):
    return int(np.sum(svals > eps * svals[0]))


def report_2d(tag, table):
    t = np.asarray(table)
    s = np.linalg.svd(t, compute_uv=False)
    m, n = t.shape
    r15, r7 = eps_rank(s, 1e-15), eps_rank(s, 1e-7)
    flops = lambda r: (m * n) / (r * (m + n))
    print(f"  {tag:22s} {str(t.shape):>10s}  rank1e-15 {r15:3d} "
          f"(flop-model x{flops(r15):4.1f}, store x{m * n / (r15 * (m + n)):4.1f})"
          f"  rank1e-7 {r7:3d} (x{flops(r7):4.1f})")
    return s


def report_3d(tag, tensor):
    t = np.asarray(tensor)
    shape = t.shape
    ranks15, ranks7 = [], []
    for mode in range(3):
        unf = np.moveaxis(t, mode, 0).reshape(shape[mode], -1)
        s = np.linalg.svd(unf, compute_uv=False)
        ranks15.append(eps_rank(s, 1e-15))
        ranks7.append(eps_rank(s, 1e-7))
    dense = np.prod(shape)
    for eps, rk in (("1e-15", ranks15), ("1e-7", ranks7)):
        core = np.prod(rk)
        factors = sum(r * n for r, n in zip(rk, shape))
        print(f"  {tag:22s} {str(shape):>12s}  Tucker@{eps} {tuple(rk)}  "
              f"store x{dense / (core + factors):4.1f}  "
              f"(core {core} + factors {factors} vs dense {dense})")


def main():
    from chebax._src.recipes import (besseli_table, besselj_table,
                                     besselk_table, bessely_table,
                                     betainc_table, gammainc_table,
                                     vonmises_table)

    print("2-D tables: SVD epsilon-ranks and dense/compressed cost models")
    report_2d("besselj TABLE", besselj_table.TABLE)
    report_2d("besselk IN_LO", besselk_table.TABLE_IN_LO)
    report_2d("besselk IN_HI", besselk_table.TABLE_IN_HI)
    report_2d("besselk TAIL", besselk_table.TABLE_TAIL)
    report_2d("besseli IN", besseli_table.TABLE_IN)
    report_2d("besseli TAIL", besseli_table.TABLE_TAIL)
    report_2d("bessely A", bessely_table.TABLE_A)
    report_2d("bessely B", bessely_table.TABLE_B)
    report_2d("bessely MID", bessely_table.TABLE_MID)
    report_2d("gammainc IN", gammainc_table.TABLE_IN)
    report_2d("gammainc TAIL", gammainc_table.TABLE_TAIL)
    report_2d("vonmises TABLE", vonmises_table.TABLE)

    print("\n3-D tensor: mode unfoldings (Tucker/HOSVD core shape)")
    report_3d("betainc TENSOR", betainc_table.TENSOR)

    print("\nThe flop model is per INSTANTIATION (once per call), so 2-D"
          "\ncompression mostly buys smaller baked artifacts and a lower"
          "\nper-group crossover; the betainc tensor and any future 3-D+"
          "\nrecipe are where the storage ratio changes what is feasible.")


if __name__ == "__main__":
    main()
