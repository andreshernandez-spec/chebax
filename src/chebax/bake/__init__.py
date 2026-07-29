"""Bake: emit self-contained artifacts from a BesselJ instance.

jax_module(jv, path)  writes a pure-jax module (imports jax.numpy only, never
chebax) implementing the same three-region formula with the instance's
coefficients inlined as python floats. Python-float constants are weak typed,
so the emitted function follows the dtype of x; plain jax AD gives dJ/dx (it
differentiates the same polynomial the custom_jvp would).

xsf_header(jv, path)  writes a standalone C++17 header (only <cmath>), double
precision, same formula. XSF_HOST_DEVICE is defined empty when absent so the
header drops into xsf or plain host code unchanged.

Both emissions are deterministic: the same instance produces the same bytes
(repr round-trips floats; no timestamps). Both return the emitted function
name.
"""

import math

import numpy as np

from chebax import __version__
from chebax._src.recipes import besselj_table as _tab
from chebax._src.recipes import besselj_table_ext as _ext
from chebax._src.recipes.besselj import BesselJ


def _check(jv):
    if not isinstance(jv, BesselJ):
        raise TypeError(f"bake supports BesselJ instances for now, got {type(jv).__name__}")
    return {name: [float(c) for c in np.asarray(getattr(jv, attr).coef)]
            for name, attr in (("inner", "g"), ("mid", "jm"), ("p", "p"), ("qs", "q"))}


def _vtag(v):
    return str(float(v)).replace(".", "p").replace("-", "m")


def _py_tuple(vals, per_line=4):
    out = []
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join(repr(x) for x in vals[i:i + per_line]) + ",")
    return "\n".join(out)


def jax_module(jv, path):
    """Write a self-contained pure-jax module defining besselj(x). Returns 'besselj'."""
    c = _check(jv)
    x0, x1, xs = _ext.MID_X0, _ext.MID_X1, _ext.XS
    src = f'''"""J_v at v = {jv.v}, baked by chebax {__version__} (tables: dps {_tab.META["dps"]},
mpmath {_tab.META["mpmath"]}). Self-contained: imports jax.numpy only. Do not edit.

Valid for x >= 0 (validated upstream against mpmath to x = 1e4). dJ/dx via
plain jax AD. Constants are python floats (weak typed), so the function
follows the dtype of x.
"""

import jax.numpy as jnp

V = {jv.v!r}
_INV_GAMMA = {jv._inv_gamma!r}
_CPHI = {jv._cphi!r}
_SPHI = {jv._sphi!r}

_C_INNER = (
{_py_tuple(c["inner"])}
)
_C_MID = (
{_py_tuple(c["mid"])}
)
_C_P = (
{_py_tuple(c["p"])}
)
_C_QS = (
{_py_tuple(c["qs"])}
)


def _clenshaw(t, c):
    b1 = t * 0
    b2 = t * 0
    for ck in c[:0:-1]:
        b1, b2 = 2 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + c[0]


def besselj(x):
    x = jnp.asarray(x)
    # active branches see raw x through their own seam (min/max ties split tangents)
    xi = jnp.where(x <= {x0!r}, x, {x0!r})
    inner = _INV_GAMMA * jnp.power(xi / 2, V) * _clenshaw(2 * (xi * xi) / {_tab.ZMAX!r} - 1, _C_INNER)
    xm = jnp.where(x <= {x0!r}, {x0!r}, jnp.where(x <= {x1!r}, x, {x1!r}))
    mid = _clenshaw((2 * xm - {x0 + x1!r}) / {x1 - x0!r}, _C_MID)
    xo = jnp.where(x <= {x1!r}, {x1!r}, x)
    s = {xs!r} / xo
    t = s * s
    p = _clenshaw(2 * t - 1, _C_P)
    q = s * _clenshaw(2 * t - 1, _C_QS)
    a = p * _CPHI + q * _SPHI
    b = p * _SPHI - q * _CPHI
    outer = jnp.sqrt(2.0 / (jnp.pi * xo)) * (jnp.cos(xo) * a + jnp.sin(xo) * b)
    return jnp.where(x <= {x0!r}, inner, jnp.where(x <= {x1!r}, mid, outer))
'''
    with open(path, "w") as f:
        f.write(src)
    return "besselj"


def _cpp_array(name, vals, per_line=4):
    body = []
    for i in range(0, len(vals), per_line):
        body.append("    " + ", ".join(repr(x) for x in vals[i:i + per_line]) + ",")
    return f"inline constexpr double {name}[{len(vals)}] = {{\n" + "\n".join(body) + "\n};"


def xsf_header(jv, path, name=None):
    """Write a standalone C++17 header. Returns the emitted function name."""
    c = _check(jv)
    name = name or f"cyl_bessel_j_v{_vtag(jv.v)}"
    x0, x1, xs = _ext.MID_X0, _ext.MID_X1, _ext.XS
    guard = name.upper()
    src = f'''// J_v at v = {jv.v}, baked by chebax {__version__} (tables: dps {_tab.META["dps"]},
// mpmath {_tab.META["mpmath"]}). Self-contained C++17, double precision; do not edit.
// Valid for x >= 0 (validated upstream against mpmath to x = 1e4).
#ifndef CHEBAX_BAKED_{guard}_H
#define CHEBAX_BAKED_{guard}_H

#include <cmath>

#ifndef XSF_HOST_DEVICE
#define XSF_HOST_DEVICE
#endif

namespace chebax_baked {{
namespace detail_{name} {{

{_cpp_array("C_INNER", c["inner"])}

{_cpp_array("C_MID", c["mid"])}

{_cpp_array("C_P", c["p"])}

{_cpp_array("C_QS", c["qs"])}

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

}} // namespace detail_{name}

XSF_HOST_DEVICE inline double {name}(double x) {{
    namespace d = detail_{name};
    constexpr double V = {jv.v!r};
    constexpr double INV_GAMMA = {jv._inv_gamma!r};
    constexpr double CPHI = {jv._cphi!r};
    constexpr double SPHI = {jv._sphi!r};
    constexpr double PI = {math.pi!r};

    double xi = x <= {x0!r} ? x : {x0!r};
    double inner = INV_GAMMA * std::pow(xi / 2.0, V)
                 * d::clenshaw(d::C_INNER, 2.0 * (xi * xi) / {_tab.ZMAX!r} - 1.0);
    double xm = x <= {x0!r} ? {x0!r} : (x <= {x1!r} ? x : {x1!r});
    double mid = d::clenshaw(d::C_MID, (2.0 * xm - {x0 + x1!r}) / {x1 - x0!r});
    double xo = x <= {x1!r} ? {x1!r} : x;
    double s = {xs!r} / xo;
    double t = s * s;
    double p = d::clenshaw(d::C_P, 2.0 * t - 1.0);
    double q = s * d::clenshaw(d::C_QS, 2.0 * t - 1.0);
    double a = p * CPHI + q * SPHI;
    double b = p * SPHI - q * CPHI;
    double outer = std::sqrt(2.0 / (PI * xo)) * (std::cos(xo) * a + std::sin(xo) * b);
    return x <= {x0!r} ? inner : (x <= {x1!r} ? mid : outer);
}}

}} // namespace chebax_baked

#endif
'''
    with open(path, "w") as f:
        f.write(src)
    return name
