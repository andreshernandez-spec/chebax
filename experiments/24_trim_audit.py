#!/usr/bin/env python3
"""Experiment 24: how far each baked table's argument axis can be cut.

WHAT IS MEASURED
----------------
Every 2-D and 3-D table holds, for each argument-direction coefficient k, a
series over its parameter axes. A static trim keeps the first n argument
coefficients whatever the parameters. Its cost in the tabulated quantity f(t),
t the argument mapped to [-1, 1], at parameters p is bounded by

    value:         sum_{k >= n} |c_k(p)|
    d/dt:          sum_{k >= n} k^2 |c_k(p)|       (|T_k'| <= k^2 on [-1, 1])
    d2/dt2:        sum_{k >= n} k^2 (k^2 - 1)/3 |c_k(p)|   (|T_k''| <= that)
    d/dp (unit):   sum_{k >= n} |dc_k/dp|,        p mapped to [-1, 1]

Each bound is taken at its worst over a grid of parameters covering the box,
ends included, and the table prints the smallest n at which it is under
1e-15, 1e-13, 1e-10 and 1e-7, next to the table's current n. The tabulated
quantity is a log in most families, so an absolute bound on it is a relative
bound on the function; where it is not (besselj, bessely, the betainc-shaped
kernels are logs too), the bound is absolute on the tabulated quantity.
1-D tables (erf) get the value and d/dt columns only; erfcx's panel table
is audited as its worst panel.

WHAT IT MEANS
-------------
A trim that passes the value bound alone can wreck the x-derivative: cutting
gammainc's inner table 32 -> 26 left values at their floor and took dP/dx from
6.5e-13 to 2.6e-11 (found preparing PR #47, 2026-11-03). The d/dt column is
the cut a trim may actually take; d/dp applies where a test measures the
parameter gradient (the _dnu functions, betainc's dI/da, stdtr's d/dnu), and
d2/dt2 where a runtime differentiates a table twice in its argument (besseli
and besselk's integer-order _dnu). The
earlier value-only trims (stdtr B 68 -> 20, besseli 72 -> 57, vonmises
112 -> 93, besselk inner-lo 80 -> 61, measured 2026-10-31) are superseded by
the d/dt and d/dp columns here.

WHAT THIS DOES NOT ESTABLISH
----------------------------
The runtime floors (the family tests measure those, and chain-rule factors
outside the table enter them) and the speed of a trim, which depends on the
device (experiments/12: f32 truncate(1e-7) is 1.4-2.2x on the GPU). Trims of
the parameter axes, and of gammainc_array's patches, which its generator sizes
per patch.

Run:  python experiments/24_trim_audit.py          # ~10 s
"""

import importlib
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, "src")

TOLS = (1e-15, 1e-13, 1e-10, 1e-7)

# table module -> arrays to audit; axis 0 is the argument, the rest parameters
TABLES = {
    "besselj_table": ["TABLE"],
    "besselj_table_ext": ["TABLE_MID", "TABLE_P", "TABLE_QS"],
    "besselk_table": ["TABLE_IN_LO", "TABLE_IN_HI", "TABLE_TAIL"],
    "besseli_table": ["TABLE_IN", "TABLE_TAIL"],
    "bessely_table": ["TABLE_A", "TABLE_B", "TABLE_MID"],
    "betainc_table": ["TENSOR"],
    "gammainc_table": ["TABLE_IN", "TABLE_TAIL"],
    "gammainc_large_table": ["TABLE_TEMME", "TABLE_LOW", "TABLE_UP"],
    "hyp1f1_table": ["TABLE_IN", "TABLE_TAIL"],
    "stdtr_table": ["TABLE_A_LO", "TABLE_A_HI", "TABLE_B_LO", "TABLE_B_HI"],
    "vonmises_table": ["TABLE"],
    "tweedie_table": ["TABLE_TAIL"],
    "erf_table": ["DAWSN_E", "DAWSN_G", "ERFCX_P"],
}
# arrays whose axis 0 indexes panels, each row its own 1-D series
PANELS = {"ERFCX_P"}


def unit_grid(m):
    """m Chebyshev points plus both ends, on [-1, 1]."""
    return np.concatenate([[-1.0], np.cos(np.pi * (np.arange(m) + 0.5) / m), [1.0]])


def collapse(table, grids, der_axis=None):
    """Argument-direction coefficient vectors at every parameter grid point:
    shape (n_arg, *grid sizes). der_axis differentiates along that parameter
    axis (1-based) first, in the unit coordinate."""
    t = np.asarray(table, dtype=float)
    if der_axis is not None:
        t = C.chebder(t, axis=der_axis)
    out = t
    # contract the last parameter axis first, so axis numbering stays valid
    for ax in range(t.ndim - 1, 0, -1):
        moved = np.moveaxis(out, ax, 0)
        out = np.moveaxis(C.chebval(grids[ax - 1], moved), -1, ax)
    return out


def tails(coefs, weights):
    """max over parameters of sum_{k >= n} weights[k] |c_k|, for every n."""
    w = np.abs(coefs) * weights.reshape((-1,) + (1,) * (coefs.ndim - 1))
    flat = w.reshape(w.shape[0], -1)
    rev = np.cumsum(flat[::-1], axis=0)[::-1]
    return np.concatenate([rev.max(axis=1), [0.0]])


def first_under(tail, tol):
    return int(np.argmax(tail <= tol))


def audit(name, table, panels=False):
    table = np.asarray(table, dtype=float)
    shape = table.shape
    if panels:
        table = table.T  # worst panel, no collapse
    n = table.shape[0]
    k = np.arange(n, dtype=float)
    w1, w2 = k * k, k * k * (k * k - 1.0) / 3.0
    if table.ndim == 1 or panels:
        coefs = table.reshape(n, -1)
    else:
        grids = [
            unit_grid(129 if table.ndim == 2 else 33) for _ in range(table.ndim - 1)
        ]
        coefs = collapse(table, grids)
    bounds = {
        "value": tails(coefs, np.ones(n)),
        "d/dt": tails(coefs, w1),
        "d2/dt2": tails(coefs, w2),
    }
    if table.ndim > 1 and not panels:
        for ax in range(1, table.ndim):
            d = collapse(table, grids, der_axis=ax)
            bounds[f"d/dp{ax}"] = tails(d, np.ones(n))
    scale = float(np.max(np.abs(coefs)))
    print(f"\n{name}: shape {shape}, n = {n}, max |c| = {scale:.3g}")
    print(f"  {'tol':>6s} | " + " ".join(f"{b:>7s}" for b in bounds))
    for tol in TOLS:
        print(
            f"  {tol:>6.0e} | "
            + " ".join(f"{first_under(v, tol):>7d}" for v in bounds.values())
        )


def main(which=None):
    print(__doc__.split("Run:")[0].strip().splitlines()[0])
    print("smallest n keeping the bound under tol; columns are the bounds above")
    for mod, arrays in TABLES.items():
        if which and mod.split("_table")[0] not in which:
            continue
        m = importlib.import_module(f"chebax._src.recipes.{mod}")
        for arr in arrays:
            audit(f"{mod}.{arr}", getattr(m, arr), panels=arr in PANELS)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
