"""Bake: emit self-contained artifacts from any Recipe instance.

jax_module(inst, path) writes a pure-jax module (no chebax import) and
xsf_header(inst, path) a standalone C++17 header (<cmath> and <limits>,
XSF_HOST_DEVICE defined empty when absent), both from the instance's own
traced computation, so formula, constants and branches cannot drift from the
runtime. Emission is deterministic and generic: any Recipe with a closed-form
evaluation bakes, solver-based callables raise NotImplementedError, nonfinite
tables are rejected, and the source must be float64 (x64 on).

The trace is taken at float64, so an artifact is that computation in its
declared dtype, not the float32 runtime (besselj(2.5) on [0.05, 100]: 4.4e-8
sup-normalized from the x64-off runtime); a recipe that picks a branch from
the dtype bakes the float64 branch; scalar subexpressions are folded at bake
time (_jaxpr_emit), so values do not depend on the running process's x64
setting. One-sided endpoint slopes carried by edge_slope are dropped: the
baked gammainc has slope 0 at exactly x = 0 where the runtime gives the
density. Each artifact rejects any input dtype but its own. C++ std::fmax and
fmin ignore nan operands where jnp.maximum and minimum propagate them; the
recipes keep nan out of interior min/max.
"""

import keyword
import re

import numpy as np

from chebax import __version__
from chebax._src.pytree import Recipe
from chebax.bake._jaxpr_emit import trace_and_emit

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


# C++17 keywords that are not also python ones, plus the alternative
# operator spellings. A name only python knows about is not enough: both
# emitters take the same name, and name="template" produced a header that
# would not compile (review, 2026-08-02).
_CPP_KEYWORDS = frozenset("""
alignas alignof asm auto bitand bitor bool catch char char16_t char32_t
char8_t compl const consteval constexpr const_cast decltype default delete
do double dynamic_cast enum explicit export extern float friend goto inline
int long mutable namespace new noexcept not_eq operator or_eq private
protected public register reinterpret_cast short signed sizeof static
static_assert static_cast struct switch template this thread_local throw
typedef typeid typename union unsigned using virtual void volatile wchar_t
xor_eq
""".split())


# Names the emitted module itself defines or calls. A valid identifier is
# not enough: name="jnp" replaced the import and name="_clenshaw" the
# helper, and both artifacts failed when called (review, 2026-10-26).
_PY_RESERVED = frozenset({"jax", "jnp", "_clenshaw", "TypeError"})
_CPP_RESERVED = frozenset({"XSF_HOST_DEVICE"})


def _check_free(name, reserved, coefs=()):
    if name in reserved or name in coefs:
        raise ValueError(
            f"name {name!r} is used by the emitted source itself; pick another"
        )


def _ident(value, what):
    """Names go straight into emitted source (and into the C++ guard macro
    as name.upper()), so reject anything but a plain identifier that is
    neither language's keyword and not a form C++ reserves: a leading
    underscore before an uppercase letter, or a double underscore
    anywhere, both of which belong to the implementation."""
    if not isinstance(value, str) or not _IDENT.match(value):
        raise ValueError(
            f"{what} must be an identifier matching [A-Za-z_][A-Za-z0-9_]*, "
            f"got {value!r}"
        )
    if keyword.iskeyword(value) or value in _CPP_KEYWORDS:
        raise ValueError(
            f"{what} must be an identifier that is neither a python nor a "
            f"C++ keyword, got {value!r}"
        )
    if "__" in value or (value[:1] == "_" and value[1:2].isupper()):
        raise ValueError(
            f"{what} must be an identifier C++ does not reserve (no double "
            f"underscore, no leading underscore before a capital), got "
            f"{value!r}"
        )
    return value


def _check_tables(inst, prefix=""):
    for f in inst._series_fields:
        s = getattr(inst, f)
        if hasattr(s, "_series_fields"):
            _check_tables(s, prefix + f + ".")
            continue
        coef = np.asarray(s.coef)
        if coef.dtype != np.float64:
            raise ValueError(
                "bake emits double-precision artifacts, but this instance holds "
                f"{coef.dtype} tables (jax x64 disabled?); enable "
                "jax.config.update('jax_enable_x64', True) before constructing it"
            )
        if not np.all(np.isfinite(coef)):
            k = int(np.argmax(~np.isfinite(coef)))
            raise ValueError(
                f"series {prefix + f} holds a nonfinite coefficient "
                f"({float(coef[k])!r} at index {k}); a table with nan or inf is "
                "broken, and neither backend has a literal for it inside an array"
            )


def _check(inst):
    if not isinstance(inst, Recipe):
        raise TypeError(
            "bake supports Recipe instances (BesselJ, BesselK, BesselI, BesselY, "
            f"BetaInc, ...); got {type(inst).__name__}"
        )
    _check_tables(inst)
    return inst


def _statics(inst):
    return {f: getattr(inst, f) for f in inst._static_fields}


def _statics_text(inst):
    """Statics for the header line. repr keeps it on one line; the triple
    quote is escaped so a string static cannot close the emitted docstring."""
    text = ", ".join(f"{k} = {v!r}" for k, v in _statics(inst).items())
    return text.replace('"""', '\\"\\"\\"')


def _family(inst):
    return type(inst).__name__.lower().lstrip("_")


def _tag(inst):
    """Statics spelled as an identifier: 2.5 -> 2p5, -1 -> m1, and a tuple
    such as besselj's domain (0.0, 8.0) -> 0p0_8p0, whose parentheses and
    comma used to reach the identifier check and fail it."""
    parts = []
    for k, v in _statics(inst).items():
        text = str(v).replace(".", "p").replace("-", "m").replace("+", "")
        part = k + re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        _ident(part, f"the name built from static field {k}={v!r}")
        parts.append(part)
    return "_".join(parts)


def _py_tuple(name, vals):
    lines = [f"{name} = ("]
    vals = [float(v) for v in np.asarray(vals)]
    for i in range(0, len(vals), 4):
        lines.append("    " + ", ".join(repr(x) for x in vals[i : i + 4]) + ",")
    lines.append(")")
    return lines


def _slope_note(dropped):
    """Disclosure for an artifact whose endpoint slope was folded away.

    The artifact is otherwise value-exact, and the emitted docstring says
    plain AD gives the derivative, so where that stops being true it has
    to say so itself: the flag used to be recorded and never surfaced
    (review, 2026-08-02)."""
    if not dropped:
        return []
    return [
        "",
        "One exception: a one-sided slope AT a domain endpoint was folded",
        "to 0 while baking (chebax carries those through a custom rule the",
        "emitter cannot express). Values are unaffected; the derivative",
        "exactly at that endpoint is 0 here and nonzero in the recipe.",
    ]


def jax_module(inst, path, name=None, truncate_tol=None, dtype="float64"):
    """Write a pure-jax module defining <name>(x); returns name. dtype is
    "float64" (the trace) or "float32" (the same computation in float32, see
    the package docstring); truncate_tol drops each series' converged tail
    first (ChebSeries.truncate), which pairs with float32."""
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
    _check(inst)
    name = _ident(_family(inst) if name is None else name, "name")
    if truncate_tol is not None:
        inst = inst.truncate(truncate_tol)
    body, coefs, result, slope_dropped = trace_and_emit(inst, cpp=False, dtype=dtype)
    _check_free(name, _PY_RESERVED, coefs)
    statics = _statics_text(inst)
    dtype_note = (
        ["Baked for float64 input, the dtype of the trace; anything else raises."]
        if dtype == "float64"
        else [
            "Baked for float32 input: the float64 trace carried out in float32, so",
            "it can differ from the float32 runtime by an f32 rounding. Anything",
            "else raises.",
        ]
    )
    lines = [
        f'"""{type(inst).__name__} ({statics}), baked by chebax {__version__} from the',
        "runtime jaxpr. Self-contained: imports jax only. Do not edit.",
        "",
        *dtype_note,
        "Plain jax AD gives the derivative.",
        *_slope_note(slope_dropped),
        '"""',
        "",
        "import jax",
        "import jax.numpy as jnp",
        "",
    ]
    for cname, arr in coefs.items():
        lines.extend(_py_tuple(cname, arr))
    lines += [
        "",
        "",
        "def _clenshaw(t, c):",
        "    b1 = t * 0",
        "    b2 = t * 0",
        "    for ck in c[:0:-1]:",
        "        b1, b2 = 2 * t * b1 - b2 + ck, b1",
        "    return t * b1 - b2 + c[0]",
        "",
        "",
        f"def {name}(x):",
        "    x = jnp.asarray(x)",
        f"    if x.dtype != jnp.{dtype}:",
        f'        raise TypeError("{name}() is baked for {dtype} input, got "',
        f'                        f"{{x.dtype}}; cast x, or re-bake for that dtype")',
    ]
    lines.extend(body)
    lines.append(f"    return {result}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return name


def _cpp_array(name, vals, ctype="double"):
    vals = [float(v) for v in np.asarray(vals)]
    suf = "f" if ctype == "float" else ""
    body = "\n".join(
        "    " + ", ".join(repr(x) + suf for x in vals[i : i + 4]) + ","
        for i in range(0, len(vals), 4)
    )
    return f"inline constexpr {ctype} {name}[{len(vals)}] = {{\n{body}\n}};"


def xsf_header(inst, path, name=None, dtype="double", truncate_tol=None):
    """Write a standalone C++17 header; returns the emitted function name.
    dtype "double" or "float" (f-suffixed literals, float arrays and locals,
    rounded from the f64 instance, the trace still float64); truncate_tol as
    in jax_module, 1e-7 pairs with "float"."""
    if dtype not in ("double", "float"):
        raise ValueError(f"dtype must be 'double' or 'float', got {dtype!r}")
    _check(inst)
    name = _ident(f"{_family(inst)}_{_tag(inst)}" if name is None else name, "name")
    _check_free(name, _CPP_RESERVED)
    if truncate_tol is not None:
        inst = inst.truncate(truncate_tol)
    body, coefs, result, slope_dropped = trace_and_emit(inst, cpp=True, ctype=dtype)
    statics = _statics_text(inst)
    guard = name.upper()
    arrays = "\n\n".join(_cpp_array(cname, arr, dtype) for cname, arr in coefs.items())
    nl = "\n"
    slope_lines = "".join(f"// {ln}{nl}" for ln in _slope_note(slope_dropped))
    src = f"""// {type(inst).__name__} ({statics}), baked by chebax {__version__} from the
// runtime jaxpr. Self-contained C++17, {dtype} precision; do not edit.
{slope_lines}
#ifndef CHEBAX_BAKED_{guard}_H
#define CHEBAX_BAKED_{guard}_H

#include <cmath>
#include <limits>

#ifndef XSF_HOST_DEVICE
#define XSF_HOST_DEVICE
#endif

namespace chebax_baked {{
namespace detail_{name} {{
namespace detail {{

{arrays}

template <typename T, int N>
XSF_HOST_DEVICE inline T clenshaw(const T (&c)[N], T t) {{
    T b1 = T(0), b2 = T(0);
    for (int k = N - 1; k > 0; --k) {{
        T bk = T(2) * t * b1 - b2 + c[k];
        b2 = b1;
        b1 = bk;
    }}
    return t * b1 - b2 + c[0];
}}

}} // namespace detail
}} // namespace detail_{name}

XSF_HOST_DEVICE inline {dtype} {name}({dtype} x) {{
    using namespace detail_{name};
{nl.join(body)}
    return {result};
}}

}} // namespace chebax_baked

#endif
"""
    with open(path, "w") as f:
        f.write(src)
    return name
