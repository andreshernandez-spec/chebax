"""Generator for the gammainc log-tables. Build-time only; mpmath inside.

P(a, x) and Q = 1 - P for a in [0.1, 10], as logarithms with the singular
parts in exact prefactors. Inner, x in [0, XS]: L(a, x) = ln 1F1(1; a+1; x),
P = exp(a ln x - x - lnGamma(a+1) + L); degree 23 in x and 53 in a, the pole
at a = -1 setting the a-axis (experiments/03). Tail, x >= XS, t = XS/x:
T(a, t) = ln[Gamma(a, x) x^(1-a) e^x], Q = exp((a-1) ln x - x - lnGamma(a)
+ T); t = 0 is a C-infinity endpoint (the asymptotic series diverges),
degree 20 in t and in a, and T(1, t) = 0 is a built-in check. Each side is
1 minus the other beyond its own zone. Both stages stay in mpmath.
Regenerate with:

    python -m chebax._src.recipes.gammainc_gen
"""

from chebax._src.recipes._gen_common import (
    DPS,
    dct,
    nodes,
    param_fit,
    to_f64,
    table_path,
    write_table_module,
)

XS_G = 8.0
ALO, AHI = 0.1, 10.0
NX_IN = 32  # x-nodes (degree 31; worst measured tail at 23)
NA_IN = 68  # a-nodes (degree 67; worst measured 53, the x -> 0 edge)
NT_TAIL = 26  # t-nodes (worst measured 20)
NA_TAIL = 30  # a-nodes (worst measured 20)


def _lser(mp, a, x):
    return mp.log(mp.hyp1f1(1, a + 1, x))


def _ltail(mp, a, t):
    x = mp.mpf(XS_G) / t
    return mp.log(mp.gammainc(a, x, mp.inf)) + x + (1 - a) * mp.log(x)


def generate_inner_table():
    import mpmath as mp

    with mp.workdps(DPS):
        tx = nodes(mp, NX_IN)
        lo, hi = mp.mpf(ALO), mp.mpf(AHI)
        rows = []
        for ta in nodes(mp, NA_IN):
            a = lo + (ta + 1) / 2 * (hi - lo)
            rows.append(dct(mp, [_lser(mp, a, (t + 1) / 2 * mp.mpf(XS_G)) for t in tx]))
        return to_f64(param_fit(mp, rows))


def generate_tail_table():
    import mpmath as mp

    with mp.workdps(DPS):
        tt = nodes(mp, NT_TAIL)
        lo, hi = mp.mpf(ALO), mp.mpf(AHI)
        rows = []
        for ta in nodes(mp, NA_TAIL):
            a = lo + (ta + 1) / 2 * (hi - lo)
            rows.append(dct(mp, [_ltail(mp, a, (t + 1) / 2) for t in tt]))
        return to_f64(param_fit(mp, rows))


def main(out_dir=None):
    inner = generate_inner_table()
    tail = generate_tail_table()
    write_table_module(
        table_path(out_dir, __file__, "gammainc_table.py"),
        "chebax._src.recipes.gammainc_gen",
        "T[k, j] = j-th a-coefficient of the k-th argument coefficient; "
        "TABLE_IN has argument x in [0, XS], TABLE_TAIL argument t = XS/x "
        "in [0, 1]; a in [ALO, AHI] for both",
        {
            "nx_in": NX_IN,
            "na_in": NA_IN,
            "nt_tail": NT_TAIL,
            "na_tail": NA_TAIL,
            "xs": XS_G,
            "alo": ALO,
            "ahi": AHI,
        },
        {"XS": XS_G, "ALO": ALO, "AHI": AHI},
        {"TABLE_IN": inner, "TABLE_TAIL": tail},
    )


if __name__ == "__main__":
    main()
