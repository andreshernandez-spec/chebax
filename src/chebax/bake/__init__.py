"""Bake: emit self-contained artifacts from any Recipe instance.

jax_module(inst, path)  writes a pure-jax module (no chebax import) whose
function is derived from the instance's own traced computation, so the same
formula, constants and branch structure run in both places and cannot drift
apart as the recipe changes. Plain jax AD differentiates it.

xsf_header(inst, path)  writes a standalone C++17 header (only <cmath> and
<limits>), double precision, from the same trace. XSF_HOST_DEVICE is
defined empty when absent so the header drops into xsf or plain host code
unchanged.

Derived from the trace does not mean bit-identical to every run of the
runtime. The trace is always taken at float64, so:

- An artifact evaluated in float32 is the float64 computation carried out
  in float32, not the float32 runtime. Measured for besselj(2.5) on
  x in [0.05, 100]: 4.4e-8 worst sup-normalized difference against the
  runtime under x64 off. Scalar subexpressions are folded at bake time
  (see _jaxpr_emit), so an artifact's values do not depend on the x64
  setting of the process that runs it.
- A recipe that picks a branch from the dtype (canon_tag(), x.dtype) bakes
  the float64 branch whatever the artifact's dtype. Only the float64
  artifact is then faithful.
- One-sided endpoint slopes do not survive. Where a recipe carries an
  exact derivative AT a domain endpoint through a custom rule whose value
  is zero (gammainc's density at x = 0), the artifact keeps the values and
  loses that slope: jax.grad of the baked gammainc at exactly x = 0 gives
  0, the runtime gives the density. Values agree everywhere, and so do
  derivatives at every interior point.

Both artifacts declare the dtype they were baked for rather than following
their input: jax_module's function rejects any other input dtype (dtype=
selects which one), xsf_header's C++ signature fixes it. Nonfinite
coefficient tables are rejected at bake time.

Both emissions are deterministic (same instance, same bytes) and generic
over recipes: any Recipe subclass with a closed-form evaluation path bakes;
solver-based callables raise NotImplementedError. Baking requires float64
tables (enable jax_enable_x64 before constructing the instance): a float32
source would silently produce a "double" artifact with float32 content.
Known semantic edge of the C++ backend: std::fmax/fmin ignore NaN operands
where jnp.maximum/minimum propagate them; recipes keep NaN out of interior
min/max by construction, but a custom recipe relying on NaN propagation
through min/max will differ. See _jaxpr_emit for the mechanism and its
(dev-time-only) coupling to jaxpr structure.
"""

import keyword
import re

import numpy as np

from chebax import __version__
from chebax._src.pytree import Recipe
from chebax.bake._jaxpr_emit import trace_and_emit


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _ident(value, what):
    """Names go straight into emitted source (and into the C++ guard macro
    as name.upper()), so reject anything but a plain identifier."""
    if not isinstance(value, str) or not _IDENT.match(value) or keyword.iskeyword(value):
        raise ValueError(
            f"{what} must be an identifier matching [A-Za-z_][A-Za-z0-9_]* and not a "
            f"python keyword, got {value!r}")
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
                "jax.config.update('jax_enable_x64', True) before constructing it")
        if not np.all(np.isfinite(coef)):
            k = int(np.argmax(~np.isfinite(coef)))
            raise ValueError(
                f"series {prefix + f} holds a nonfinite coefficient "
                f"({float(coef[k])!r} at index {k}); a table with nan or inf is "
                "broken, and neither backend has a literal for it inside an array")


def _check(inst):
    if not isinstance(inst, Recipe):
        raise TypeError(
            "bake supports Recipe instances (BesselJ, BesselK, BesselI, BesselY, "
            f"BetaInc, ...); got {type(inst).__name__}")
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
    parts = []
    for k, v in _statics(inst).items():
        part = f"{k}{str(v).replace('.', 'p').replace('-', 'm')}"
        _ident(part, f"the name built from static field {k}={v!r}")
        parts.append(part)
    return "_".join(parts)


def _py_tuple(name, vals):
    lines = [f"{name} = ("]
    vals = [float(v) for v in np.asarray(vals)]
    for i in range(0, len(vals), 4):
        lines.append("    " + ", ".join(repr(x) for x in vals[i:i + 4]) + ",")
    lines.append(")")
    return lines


def jax_module(inst, path, name=None, truncate_tol=None, dtype="float64"):
    """Write a self-contained pure-jax module defining <name>(x). Returns name.

    dtype declares what the artifact runs in: "float64" (default) is the
    trace itself, "float32" is that same computation carried out in
    float32, which is not the float32 runtime (see the package docstring).
    Either way the emitted function raises on any other input dtype instead
    of running a computation that was never traced.

    truncate_tol drops each series' converged tail before tracing (see
    ChebSeries.truncate); pair it with dtype="float32", where the f64 fit
    carries roughly twice the terms f32 needs."""
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
    _check(inst)
    name = _ident(_family(inst) if name is None else name, "name")
    if truncate_tol is not None:
        inst = inst.truncate(truncate_tol)
    body, coefs, result = trace_and_emit(inst, cpp=False)
    statics = _statics_text(inst)
    dtype_note = (
        ["Baked for float64 input, the dtype of the trace; anything else raises."]
        if dtype == "float64" else
        ["Baked for float32 input: the float64 trace carried out in float32, so",
         "it can differ from the float32 runtime by an f32 rounding. Anything",
         "else raises."])
    lines = [
        f'"""{type(inst).__name__} ({statics}), baked by chebax {__version__} from the',
        'runtime jaxpr. Self-contained: imports jax only. Do not edit.',
        "",
        *dtype_note,
        'Plain jax AD gives the derivative."""',
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
    body = "\n".join("    " + ", ".join(repr(x) + suf for x in vals[i:i + 4]) + ","
                     for i in range(0, len(vals), 4))
    return f"inline constexpr {ctype} {name}[{len(vals)}] = {{\n{body}\n}};"


def xsf_header(inst, path, name=None, dtype="double", truncate_tol=None):
    """Write a standalone C++17 header. Returns the emitted function name.

    dtype: "double" (default) or "float" - float emits f-suffixed
    literals, float arrays and float locals throughout (a CUDA f32
    kernel body). Tables are always sourced from the f64 instance and
    rounded at emission, and the trace is float64 either way, so "float"
    is that computation in float, not the float32 runtime. truncate_tol
    drops each series' converged tail first; pair dtype="float" with
    truncate_tol=1e-7 (the f64 fit carries roughly twice the terms f32
    needs)."""
    if dtype not in ("double", "float"):
        raise ValueError(f"dtype must be 'double' or 'float', got {dtype!r}")
    _check(inst)
    name = _ident(f"{_family(inst)}_{_tag(inst)}" if name is None else name, "name")
    if truncate_tol is not None:
        inst = inst.truncate(truncate_tol)
    body, coefs, result = trace_and_emit(inst, cpp=True, ctype=dtype)
    statics = _statics_text(inst)
    guard = name.upper()
    arrays = "\n\n".join(_cpp_array(cname, arr, dtype) for cname, arr in coefs.items())
    nl = "\n"
    src = f"""// {type(inst).__name__} ({statics}), baked by chebax {__version__} from the
// runtime jaxpr. Self-contained C++17, {dtype} precision; do not edit.
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
