"""Generator for the besselk log-tables. Build-time only; mpmath inside.

K_v is positive and non-oscillatory but spans ~15 decades over the (v, x)
rectangle, and the classical route to it (the connection formula through
I_{+-v}) cancels catastrophically near integer v (risk S3, the bessel R6
lesson). Both problems disappear by tabulating logarithms of K directly:

- Inner, x in [XMIN, 8], u = ln x:
      Ltil(v, u) = ln[ (x/2)^v K_v(x) ]
  The (x/2)^v factor removes the v-dependent branch exponent at 0 (the same
  move as besselj's g_v), and the log tames the dynamic range: Ltil spans
  ~[-9, 12] instead of 15 decades, so f64 coefficient noise stays at the
  eps*|Ltil| level. The Gamma(+-v)-pole cancellation compresses into a
  feature of width ~1/|ln x| near v=0, so v is split into two panels,
  [0, 1] and [1, 10]; panels are free at runtime (chosen at instantiation,
  or one select in the traced path). Measured worst tails: degree 66 in u,
  45 in v per panel (reproduce via experiments/03_degree_measurement.py).

- Tail, x >= 8, t = 8/x:
      Lt(v, t) = ln[ sqrt(2x/pi) e^x K_v(x) ]
  Smooth and O(1); degree 19 in t and in v covers everything (K_{1/2} makes
  Lt identically zero). The runtime multiplies exp(Lt) by sqrt(pi/(2x)) and
  a separately computed exp(-x), which is correctly rounded at any x, so no
  eps*x exponent-assembly error appears.

Both fit stages stay in mpmath (the M1 floor rule). Regenerate with:

    python -m chebax._src.recipes.besselk_gen
"""

import math
import pathlib

from chebax._src.recipes._gen_common import (DPS, dct, nodes, param_fit,
                                             to_f64, write_table_module)
from chebax._src.recipes.besselj_gen import VMAX

XMIN = 1e-6
XS_K = 8.0
VSPLIT = 1.0
NU_IN = 80   # u-nodes (degree 79; worst measured tail at 66)
NV_IN = 56   # v-nodes per panel (worst measured 45)
NT_TAIL = 24
NV_TAIL = 32


def _u01(mp):
    return mp.log(mp.mpf(f"{XMIN:.0e}")), mp.log(mp.mpf(XS_K))


def _ltil(mp, v, u):
    x = mp.exp(u)
    return mp.log(mp.besselk(v, x)) + v * mp.log(x / 2)


def generate_inner_tables():
    import mpmath as mp

    out = []
    with mp.workdps(DPS):
        u0, u1 = _u01(mp)
        tu = nodes(mp, NU_IN)
        for lo, hi in ((0, VSPLIT), (VSPLIT, VMAX)):
            lo, hi = mp.mpf(lo), mp.mpf(hi)
            vnodes = [lo + (t + 1) / 2 * (hi - lo) for t in nodes(mp, NV_IN)]
            rows = [dct(mp, [_ltil(mp, v, u0 + (t + 1) / 2 * (u1 - u0)) for t in tu])
                    for v in vnodes]
            out.append(to_f64(param_fit(mp, rows)))
    return out[0], out[1]


def generate_tail_table():
    import mpmath as mp

    with mp.workdps(DPS):
        xs = mp.mpf(XS_K)
        tt = nodes(mp, NT_TAIL)
        vnodes = [(t + 1) / 2 * mp.mpf(VMAX) for t in nodes(mp, NV_TAIL)]
        rows = []
        for v in vnodes:
            samples = []
            for t in tt:
                x = xs / ((t + 1) / 2)
                samples.append(mp.log(mp.besselk(v, x)) + x + mp.log(mp.sqrt(2 * x / mp.pi)))
            rows.append(dct(mp, samples))
        return to_f64(param_fit(mp, rows))


def main():
    lo, hi = generate_inner_tables()
    tail = generate_tail_table()
    write_table_module(
        pathlib.Path(__file__).with_name("besselk_table.py"),
        "chebax._src.recipes.besselk_gen",
        "T[k, j] = j-th v-coefficient of the k-th argument coefficient; "
        "TABLE_IN_LO has v in [0, VSPLIT], TABLE_IN_HI v in [VSPLIT, VMAX], "
        "argument u = ln x in [U0, U1]; TABLE_TAIL has v in [0, VMAX], "
        "argument t = XS/x in [0, 1]",
        {"nu_in": NU_IN, "nv_in": NV_IN, "nt_tail": NT_TAIL, "nv_tail": NV_TAIL,
         "xmin": XMIN, "xs": XS_K, "vsplit": VSPLIT, "vmax": VMAX},
        {"XMIN": XMIN, "XS": XS_K, "VSPLIT": VSPLIT, "VMAX": VMAX,
         "U0": math.log(XMIN), "U1": math.log(XS_K)},
        {"TABLE_IN_LO": lo, "TABLE_IN_HI": hi, "TABLE_TAIL": tail})


if __name__ == "__main__":
    main()
