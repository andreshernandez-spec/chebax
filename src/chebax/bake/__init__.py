"""Bake: emit self-contained artifacts from any Recipe instance.

jax_module(inst, path)  writes a pure-jax module (no chebax import) whose
function is derived from the instance's own traced computation — the
runtime is the single source of truth, so artifact and runtime cannot
diverge. Constants are inlined as python floats (weak typed: the emitted
function follows the dtype of x); plain jax AD differentiates it.

xsf_header(inst, path)  writes a standalone C++17 header (only <cmath> and
<limits>), double precision, from the same trace. XSF_HOST_DEVICE is
defined empty when absent so the header drops into xsf or plain host code
unchanged.

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

import numpy as np

from chebax import __version__
from chebax._src.pytree import Recipe
from chebax.bake._jaxpr_emit import trace_and_emit


def _f64_tables(inst):
    for f in inst._series_fields:
        s = getattr(inst, f)
        if hasattr(s, "_series_fields"):
            _f64_tables(s)
        elif np.asarray(s.coef).dtype != np.float64:
            raise ValueError(
                "bake emits double-precision artifacts, but this instance holds "
                f"{np.asarray(s.coef).dtype} tables (jax x64 disabled?); enable "
                "jax.config.update('jax_enable_x64', True) before constructing it")


def _check(inst):
    if not isinstance(inst, Recipe):
        raise TypeError(
            "bake supports Recipe instances (BesselJ, BesselK, BesselI, BesselY, "
            f"BetaInc, ...); got {type(inst).__name__}")
    _f64_tables(inst)
    return inst


def _statics(inst):
    return {f: getattr(inst, f) for f in inst._static_fields}


def _family(inst):
    return type(inst).__name__.lower().lstrip("_")


def _tag(inst):
    parts = []
    for k, v in _statics(inst).items():
        parts.append(f"{k}{str(v).replace('.', 'p').replace('-', 'm')}")
    return "_".join(parts)


def _py_tuple(name, vals):
    lines = [f"{name} = ("]
    vals = [float(v) for v in np.asarray(vals)]
    for i in range(0, len(vals), 4):
        lines.append("    " + ", ".join(repr(x) for x in vals[i:i + 4]) + ",")
    lines.append(")")
    return lines


def jax_module(inst, path, name=None):
    """Write a self-contained pure-jax module defining <name>(x). Returns name."""
    _check(inst)
    body, coefs, result = trace_and_emit(inst, cpp=False)
    name = name or _family(inst)
    statics = ", ".join(f"{k} = {v!r}" for k, v in _statics(inst).items())
    lines = [
        f'"""{type(inst).__name__} ({statics}), baked by chebax {__version__} from the',
        'runtime jaxpr. Self-contained: imports jax only. Do not edit.',
        "",
        "Constants are python floats (weak typed), so the function follows the",
        'dtype of x; plain jax AD gives the derivative."""',
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
    ]
    lines.extend(body)
    lines.append(f"    return {result}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return name


def _cpp_array(name, vals):
    vals = [float(v) for v in np.asarray(vals)]
    body = "\n".join("    " + ", ".join(repr(x) for x in vals[i:i + 4]) + ","
                     for i in range(0, len(vals), 4))
    return f"inline constexpr double {name}[{len(vals)}] = {{\n{body}\n}};"


def xsf_header(inst, path, name=None):
    """Write a standalone C++17 header. Returns the emitted function name."""
    _check(inst)
    body, coefs, result = trace_and_emit(inst, cpp=True)
    name = name or f"{_family(inst)}_{_tag(inst)}"
    statics = ", ".join(f"{k} = {v!r}" for k, v in _statics(inst).items())
    guard = name.upper()
    arrays = "\n\n".join(_cpp_array(cname, arr) for cname, arr in coefs.items())
    nl = "\n"
    src = f"""// {type(inst).__name__} ({statics}), baked by chebax {__version__} from the
// runtime jaxpr. Self-contained C++17, double precision; do not edit.
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

template <int N>
XSF_HOST_DEVICE inline double clenshaw(const double (&c)[N], double t) {{
    double b1 = 0.0, b2 = 0.0;
    for (int k = N - 1; k > 0; --k) {{
        double bk = 2.0 * t * b1 - b2 + c[k];
        b2 = b1;
        b1 = bk;
    }}
    return t * b1 - b2 + c[0];
}}

}} // namespace detail
}} // namespace detail_{name}

XSF_HOST_DEVICE inline double {name}(double x) {{
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
