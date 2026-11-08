#!/usr/bin/env python3
"""chebax against JAX in one place: the speed and accuracy tables of
experiments/33 (jax.scipy) and experiments/40 (TFP, where jax.scipy has no
real-order Bessel function), as markdown.

WHAT THIS DOES
--------------
Reads results/33_jax_release_race.txt and results/40_bessel_real_order_race.txt
and prints two tables; it measures nothing. Settings of one function (orders,
parameter values) become a range. Speed is chebax's speedup over jax 0.12
(experiments/33) or TFP (experiments/40), value / value_and_grad in x, one
pinned CPU core at N = 2^16 and the GPU at N = 2^20; J and Y of real order
have no JAX counterpart, so only scipy's CPU values are shown for them.
Accuracy is the worst error against mpmath, JAX side / chebax, in each
family's own metric (defined in the two experiments and in the legend).

WHAT IT DOES NOT ESTABLISH
--------------------------
Anything the two experiments do not: see their docstrings. The two runs are
only comparable if they ran on the same library code; the output names both
commits and says whether their src/ trees differ.

Run:

  python experiments/41_jax_comparison_table.py > results/41_jax_comparison_table.md
"""

import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
R33 = os.path.join(HERE, "..", "results", "33_jax_release_race.txt")
R40 = os.path.join(HERE, "..", "results", "40_bessel_real_order_race.txt")
BLOCKS = (("f64", "cpu"), ("f64", "gpu"), ("f32", "cpu"), ("f32", "gpu"))

# (row, setting prefixes)
SPEED33 = (
    ("erfcx", ("erfcx",)),
    ("dawsn", ("dawsn",)),
    ("I0, I1", ("i0,", "i1,")),
    ("i0e, i1e", ("i0e", "i1e")),
    ("J_n, integer n", ("J1", "J10")),
    ("betainc", ("betainc",)),
    ("gammainc(c), one a", ("gammainc a=", "gammaincc")),
    ("gammainc, a per element", ("gammainc a per",)),
    ("hyp1f1", ("hyp1f1",)),
    ("stdtr (t CDF)", ("stdtr",)),
)
SPEED40 = (
    ("J_v, real v", ("J_v",)),
    ("Y_v, real v", ("Y_v",)),
    ("I_v", ("I_v v=",)),
    ("I_v scaled", ("I_v scaled",)),
    ("K_v, K_v scaled", ("K_v v=", "K_v scaled")),
    ("log K_v", ("log K_v",)),
    ("I_{v+1}/I_v", ("I_{v+1}",)),
)
# (row, families, value metric, derivative, its metric, parameter derivatives,
#  chebax column)
C33 = "chebax 0.12.0.1"
ACC33 = (
    ("erfcx", ("erfcx",), "rel", "d_x", "rel", (), C33),
    ("dawsn", ("dawsn",), "rel", "d_x", "rel", (), C33),
    ("I0, I1", ("i0", "i1"), "rel", "d_x", "rel", (), C33),
    ("i0e, i1e", ("i0e", "i1e"), "rel", "d_x", "rel", (), C33),
    ("J_n, n = 0-10", ("j0", "j1", "j2", "j5", "j10"), "sup", "d_x", "sup", (), C33),
    ("betainc", ("betainc",), "abs", "d_x", "split", ("d_a", "d_b"), C33),
    (
        "gammainc(c), one a",
        ("gammainc", "gammaincc"),
        "abs",
        "d_x",
        "split",
        ("d_a",),
        C33,
    ),
    (
        "gammainc, a per element",
        ("gammainc", "gammaincc"),
        "abs",
        "d_x",
        "split",
        ("d_a",),
        "chebax array 0.12.0.1",
    ),
    ("hyp1f1", ("hyp1f1",), "rel_lnM", "d_x", "dlog", ("d_a", "d_b"), C33),
    ("stdtr", ("stdtr",), "abs", "d_t", "split", ("d_nu",), C33),
)
ACC40 = (
    ("J_v, real v", ("jv",), "sup", "d_x", "sup", ()),
    ("Y_v, real v", ("yv",), "mod", "d_x", "dmod", ()),
    ("I_v", ("iv",), "rel", "d_x", "rel", ("d_nu",)),
    ("I_v scaled", ("ive",), "rel", "d_x", "rel", ("d_nu",)),
    ("K_v", ("kv",), "rel", "d_x", "rel", ("d_nu",)),
    ("K_v scaled", ("kve",), "rel", "d_x", "rel", ("d_nu",)),
    ("log K_v", ("logkv",), "lrel", "d_x", "rel", ("d_nu",)),
    ("I_{v+1}/I_v", ("iratio",), "rel", "d_x", "rel", ("d_nu",)),
)
PARAM_METRIC = {"hyp1f1": "dlog"}


def cells(line):
    return re.split(r"\s{2,}", line.strip())


def table(lines, start):
    """Rows of the first table after the line starting with start."""
    i = next(k for k, ln in enumerate(lines) if ln.startswith(start))
    rows = []
    for ln in lines[i + 1 :]:
        if not ln.strip():
            if rows:
                break
            continue
        if ln.startswith(("===", "---")):
            break
        rows.append(ln)
    hdr = cells(rows[0])
    return [dict(zip(hdr, cells(r)), indent=r.startswith("  ")) for r in rows[1:]]


def fmt_x(v):
    if v >= 100:
        return f"{round(v, -1):.0f}"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


def span(vals):
    a, b = fmt_x(min(vals)), fmt_x(max(vals))
    return f"{a}x" if a == b else f"{a}-{b}x"


def speed_rows(lines, groups, col):
    tabs = {b: table(lines, f"--- {b[0]} {b[1]}") for b in BLOCKS}
    out = []
    for label, prefs in groups:
        sel = {
            b: [
                r for r in tabs[b] if not r["indent"] and r["setting"].startswith(prefs)
            ]
            for b in BLOCKS
        }
        row = [label]
        for b in BLOCKS:
            assert sel[b], (label, b)
            got = [r.get(col, "-") for r in sel[b]]
            if all(g == "-" for g in got):
                row.append("-")
                continue
            v = [float(g.split("/")[0].strip().rstrip("x")) for g in got]
            g = [float(g.split("/")[1].strip().rstrip("x")) for g in got]
            row.append(f"{span(v)} / {span(g)}")
        sci = [
            [float(r["chebax vs scipy"].rstrip("x")) for r in sel[(d, "cpu")]]
            for d in ("f64", "f32")
            if all("chebax vs scipy" in r for r in sel[(d, "cpu")])
        ]
        row.append(" / ".join(span(s) for s in sci) if len(sci) == 2 else "-")
        out.append(row)
    return out


def parse(c):
    if c in (None, "-"):
        return None
    m = re.match(r"([0-9.e+-]+)(?: \((\d+) nan\))?", c)
    return float(m.group(1)), int(m.group(2) or 0)


def worst(rows, fams, qty, metric, col):
    got = [
        parse(r.get(col))
        for r in rows
        if r["family"] in fams
        and r["quantity"].split(" [")[0] == qty
        and r["metric"] == metric
    ]
    got = [g for g in got if g is not None]
    if not got:
        return None
    return max(g[0] for g in got), sum(g[1] for g in got)


def fmt_e(w, zero=False):
    if w is None:
        return "-"
    if zero:
        return "0"
    return f"{w[0]:.1e}" + (" (nan)" if w[1] else "")


def tfp_zero(lines):
    """experiments/40's check that TFP's order gradient is exactly 0."""
    got = [ln for ln in lines if ln.startswith("  tfp d/dnu exactly 0")]
    assert len(got) == 2, "results/40 lacks the zero-gradient check"
    return all(": True" in ln for ln in got)


def accuracy_rows(tabs, groups, jcol, ccol_of, zero_param=False):
    out = []
    for g in groups:
        label, fams, vm, dq, dm, pq = g[:6]
        ccol = ccol_of(g)
        row = [label, vm]
        for d in ("f64", "f32"):
            for q, m in (("v", vm), (dq, dm)):
                j = fmt_e(worst(tabs[d], fams, q, m, jcol))
                c = fmt_e(worst(tabs[d], fams, q, m, ccol))
                row.append(f"{j} / {c}")
        if pq:
            m = PARAM_METRIC.get(fams[0], "rel" if jcol == "tfp" else "abs")
            js = [worst(tabs["f64"], fams, q, m, jcol) for q in pq]
            cs = [worst(tabs["f64"], fams, q, m, ccol) for q in pq]
            js = [w for w in js if w is not None]
            j = fmt_e(max(js), zero_param) if js else "none"
            row.append(f"{j} / {fmt_e(max(w for w in cs if w is not None))}")
        else:
            row.append("-")
        out.append(row)
    return out


def md(hdr, rows):
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def commit_of(lines, tag):
    runs = [ln for ln in lines if ln.startswith(f"  run {tag}")]
    return sorted({re.search(r"chebax (\w+)", ln).group(1) for ln in runs})


def same_src(commits):
    """True when every commit holds the same src/ tree."""
    trees = {
        subprocess.run(
            ["git", "-C", HERE, "rev-parse", f"{c}:src"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        for c in commits
    }
    return len(trees) == 1 and "" not in trees


def main():
    l33 = open(R33).read().splitlines()
    l40 = open(R40).read().splitlines()
    c33, c40 = commit_of(l33, "0.12"), commit_of(l40, "f")
    print("# chebax against JAX: speed and accuracy\n")
    print(
        f"From results/33_jax_release_race.txt (jax.scipy, jax 0.12.0.1, chebax "
        f"{', '.join(c33)}) and results/40_bessel_real_order_race.txt (TFP "
        f"{l40[1].split('tfp ')[1].split(',')[0]}, chebax {', '.join(c40)}); "
        "made by experiments/41_jax_comparison_table.py."
    )
    if c33 != c40 or len(c33) != 1:
        if same_src(c33 + c40):
            print("\nThe two commits hold the same src/.")
        else:
            print("\n**The two runs are on different chebax code (src/ differs).**")

    print("\n## Speed\n")
    print(
        "chebax's speedup, value / value_and_grad in x; one pinned CPU core at "
        "N = 2^16, RTX 3080 Laptop GPU at N = 2^20. Below 1x chebax is slower.\n"
    )
    hdr = ["function", "against", "f64 CPU", "f64 GPU", "f32 CPU", "f32 GPU"]
    hdr.append("values vs scipy, 1 core (f64 / f32)")
    s33 = speed_rows(l33, SPEED33, "speedup vs jax 0.12.0.1")
    s40 = speed_rows(l40, SPEED40, "chebax vs tfp")
    rows = [[r[0], "jax"] + r[1:] for r in s33]
    rows += [[r[0], "TFP" if r[1] != "-" else "none in JAX"] + r[1:] for r in s40]
    md(hdr, rows)

    print("\n## Accuracy\n")
    print("Worst error against mpmath, JAX side / chebax, CPU and GPU.\n")
    a33 = {d: table(l33, f"=== ACCURACY {d}") for d in ("f64", "f32")}
    a40 = {d: table(l40, f"=== ACCURACY {d}") for d in ("f64", "f32")}
    rows = accuracy_rows(a33, ACC33, "jax 0.12.0.1", lambda g: g[6])
    rows += accuracy_rows(a40, ACC40, "tfp", lambda g: "chebax", tfp_zero(l40))
    hdr = ["function", "metric", "f64 value", "f64 d/dx", "f32 value", "f32 d/dx"]
    md(hdr + ["f64 parameter derivative"], rows)
    print(
        "\nrel: relative. sup: error / max |J| of the order. mod: error / "
        "sqrt(J^2 + Y^2). abs: absolute (CDFs). lrel: error / max(1, |ln K|). "
        "rel_lnM: hyp1f1's relative error / max(1, |ln M|), its derivatives as "
        "errors in d ln M. A CDF's d/dx (split) is relative where the density "
        "is at least 1e-4 of the CDF, absolute elsewhere. (nan): some points "
        "nan or inf where the true value is finite. Parameter derivative: "
        "none = jax has no gradient there (betainc in a, b; the t CDF in nu); "
        "0 = TFP returns exactly 0 for the gradient in the order."
    )


if __name__ == "__main__":
    main()
