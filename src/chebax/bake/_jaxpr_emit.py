"""Generic artifact emission: trace a Recipe instance's __call__ to a jaxpr
and print it as self-contained Python-jax or C++17 source.

The runtime is the single source of truth — recipes stay plain, typed jnp
code, and artifacts are derived from the very computation that runs. Two
structural facts (probed, and load-bearing):

- Every ChebSeries evaluation appears as ONE custom_jvp_call equation whose
  unrolled Clenshaw is contained in its call_jaxpr, never opened here; the
  coefficient vector arrives as a jaxpr constant, matched back to the
  instance's series field BY VALUE to recover its name and domain, and the
  whole equation folds to `clenshaw(C_<FIELD>, <affine-mapped arg>)`.
- After that fold, recipe formulas use a tiny primitive vocabulary
  (arithmetic, comparisons, select, exp/log/sqrt/sin/cos/pow), mapped
  one-to-one below. `jit`-wrapped helpers (jnp.where, clip) are recursed
  into, except for a small fold table of named functions with direct
  library equivalents (gammaln -> std::lgamma).

Solver-based callables (anything containing while/scan/cond) are not
bakeable and raise. The emitter runs at development time only: if a jax
upgrade changes jaxpr structure it fails loudly at regeneration, and
already-committed artifacts remain valid.
"""

import numpy as np
import jax
import jax.numpy as jnp

try:
    from jax.extend.core import Literal as _Literal
except ImportError:  # older jax
    from jax.core import Literal as _Literal

_INLINE_CALLS = {"jit", "pjit", "closed_call", "core_call", "named_call"}
_LOOP_PRIMS = {"while", "scan", "cond"}
_PASSTHROUGH = {"broadcast_in_dim", "stop_gradient", "copy", "squeeze", "reshape"}
_BINOP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}

_UNARY = {
    "neg":   ("-({0})",            "-({0})"),
    "exp":   ("jnp.exp({0})",      "std::exp({0})"),
    "log":   ("jnp.log({0})",      "std::log({0})"),
    "log1p": ("jnp.log1p({0})",    "std::log1p({0})"),
    "sqrt":  ("jnp.sqrt({0})",     "std::sqrt({0})"),
    "sin":   ("jnp.sin({0})",      "std::sin({0})"),
    "cos":   ("jnp.cos({0})",      "std::cos({0})"),
    "abs":   ("jnp.abs({0})",      "std::fabs({0})"),
    "sign":  ("jnp.sign({0})",     "((double)(({0}) > 0.0) - (double)(({0}) < 0.0))"),
    "logistic": ("jax.nn.sigmoid({0})", "(1.0 / (1.0 + std::exp(-({0}))))"),
    "is_finite": ("jnp.isfinite({0})", "std::isfinite({0})"),
    "lgamma": ("jax.scipy.special.gammaln({0})", "std::lgamma({0})"),
}

_BINFN = {
    "pow": ("jnp.power({0}, {1})", "std::pow({0}, {1})"),
    "max": ("jnp.maximum({0}, {1})", "std::fmax({0}, {1})"),
    "min": ("jnp.minimum({0}, {1})", "std::fmin({0}, {1})"),
    "and": ("jnp.logical_and({0}, {1})", "(({0}) && ({1}))"),
    "or":  ("jnp.logical_or({0}, {1})", "(({0}) || ({1}))"),
    "atan2": ("jnp.arctan2({0}, {1})", "std::atan2({0}, {1})"),
}

# jit-wrapped library functions folded to a direct equivalent instead of
# being recursed into (their internals would bloat the artifact)
_NAMED_FOLDS = {
    "gammaln": ("jax.scipy.special.gammaln({0})", "std::lgamma({0})"),
    "_lgamma": ("jax.scipy.special.gammaln({0})", "std::lgamma({0})"),
}


class _Emit:
    def __init__(self, cpp):
        self.cpp = cpp
        self.lines = []
        self.counter = 0
        self.coef_names = {}      # id -> (name, series)

    def fresh(self):
        self.counter += 1
        return f"t{self.counter}"

    def assign(self, expr, is_bool=False):
        sym = self.fresh()
        if self.cpp:
            self.lines.append(f"    {'bool' if is_bool else 'double'} {sym} = {expr};")
        else:
            self.lines.append(f"    {sym} = {expr}")
        return sym


def _lit(val, cpp):
    a = np.asarray(val)
    if a.dtype == bool:
        if cpp:
            return ("true" if bool(a) else "false"), True
        return ("True" if bool(a) else "False"), True
    v = float(a)
    if v != v:
        return ("(std::numeric_limits<double>::quiet_NaN())" if cpp
                else "float('nan')"), False
    if v == float("inf") or v == float("-inf"):
        s = ("std::numeric_limits<double>::infinity()" if cpp
             else "float('inf')")
        return (f"(-{s})" if v < 0 else s), False
    return repr(v), False


def _collect_matches(inst, a, prefix=""):
    hits = []
    for f in inst._series_fields:
        s = getattr(inst, f)
        # nested recipes (spherical wraps a cylindrical instance)
        if hasattr(s, "_series_fields"):
            hits += _collect_matches(s, a, prefix + f + "_")
            continue
        coef = np.asarray(s.coef)
        if coef.shape == a.shape and np.array_equal(coef, a):
            hits.append((prefix + f, s))
    return hits


def _match_series(inst, const):
    hits = _collect_matches(inst, np.asarray(const))
    if not hits:
        return None
    # identification is by coefficient VALUE; two fields with equal
    # coefficients but different domains would fold to the wrong affine
    # map while looking valid, so refuse to guess
    domains = {(float(h[1].domain[0]), float(h[1].domain[1])) for h in hits}
    if len(domains) > 1:
        names = [h[0] for h in hits]
        raise NotImplementedError(
            f"a coefficient constant matches series fields {names} whose domains "
            f"differ ({sorted(domains)}); value-matching cannot identify it — "
            "make the coefficient vectors distinct")
    return hits[0]


def _emit_jaxpr(em, jaxpr, env):
    def read(v):
        if isinstance(v, _Literal):
            return _lit(v.val, em.cpp)[0]
        return env[v]

    for eqn in jaxpr.eqns:
        p = eqn.primitive.name
        out = eqn.outvars[0] if eqn.outvars else None
        if p in _LOOP_PRIMS:
            raise NotImplementedError(
                "solver-based callables (while/scan/cond in the jaxpr) are not bakeable yet")
        if p == "custom_jvp_call":
            coef_in = [v for v in eqn.invars
                       if not isinstance(v, _Literal) and env[v] in em.coef_names]
            if len(coef_in) != 1:
                raise NotImplementedError(f"unrecognized custom_jvp_call (coef inputs: {len(coef_in)})")
            name, series = em.coef_names[env[coef_in[0]]]
            (x_in,) = [v for v in eqn.invars if v is not coef_in[0]]
            a, b = series.domain
            t_expr = f"(2.0 * ({read(x_in)}) - {b + a!r}) / {b - a!r}"
            if em.cpp:
                expr = f"detail::clenshaw(detail::{name}, {t_expr})"
            else:
                expr = f"_clenshaw({t_expr}, {name})"
            env[out] = em.assign(expr)
        elif p in _INLINE_CALLS:
            name = str(eqn.params.get("name", ""))
            inner = eqn.params.get("jaxpr")
            if name in _NAMED_FOLDS:
                tmpl = _NAMED_FOLDS[name][1 if em.cpp else 0]
                env[out] = em.assign(tmpl.format(*[read(v) for v in eqn.invars]))
                continue
            inner_jaxpr = inner.jaxpr if hasattr(inner, "jaxpr") else inner
            inner_env = dict(zip(inner_jaxpr.invars, [read(v) for v in eqn.invars]))
            for cv, cval in zip(inner_jaxpr.constvars, getattr(inner, "consts", [])):
                inner_env[cv] = _register_const(em, None, cval)
            _emit_jaxpr(em, inner_jaxpr, inner_env)
            for ov, iv in zip(eqn.outvars, inner_jaxpr.outvars):
                env[ov] = inner_env[iv] if not isinstance(iv, _Literal) \
                    else _lit(iv.val, em.cpp)[0]
        elif p == "convert_element_type":
            nd = np.dtype(eqn.params["new_dtype"])
            src = read(eqn.invars[0])
            if nd == eqn.invars[0].aval.dtype:
                env[out] = src  # weak-type normalization only
            elif nd == np.float64:
                # bool -> double masks and float promotions; multiplying by
                # 1.0 is exact for floats and promotes bools, keeping the
                # emitted python weak-typed
                env[out] = em.assign(f"(double)({src})" if em.cpp
                                     else f"({src}) * 1.0")
            else:
                raise NotImplementedError(
                    f"cast to {nd} in the traced computation; a double artifact "
                    "cannot preserve reduced-precision semantics")
        elif p in _PASSTHROUGH:
            env[out] = read(eqn.invars[0])
        elif p in _BINOP:
            x, y = (read(v) for v in eqn.invars)
            env[out] = em.assign(f"({x}) {_BINOP[p]} ({y})")
        elif p in _CMP:
            x, y = (read(v) for v in eqn.invars)
            env[out] = em.assign(f"({x}) {_CMP[p]} ({y})", is_bool=True)
        elif p in _BINFN:
            x, y = (read(v) for v in eqn.invars)
            env[out] = em.assign(_BINFN[p][1 if em.cpp else 0].format(x, y))
        elif p in _UNARY:
            env[out] = em.assign(_UNARY[p][1 if em.cpp else 0].format(read(eqn.invars[0])),
                                 is_bool=(p == "is_finite"))
        elif p == "integer_pow":
            k = eqn.params["y"]
            x = read(eqn.invars[0])
            env[out] = em.assign(f"std::pow({x}, {float(k)!r})" if em.cpp
                                 else f"({x}) ** {k}")
        elif p == "select_n":
            pred, case0, case1 = (read(v) for v in eqn.invars)
            if em.cpp:
                env[out] = em.assign(f"(({pred}) ? ({case1}) : ({case0}))")
            else:
                env[out] = em.assign(f"jnp.where({pred}, {case1}, {case0})")
        else:
            raise NotImplementedError(f"primitive not supported by the bake emitter: {p}")
    return env


def _register_const(em, name_hint, val):
    a = np.asarray(val)
    if a.ndim == 0:
        return _lit(a, em.cpp)[0]
    sym = name_hint
    if sym is None:
        raise NotImplementedError(f"unnamed array constant of shape {a.shape} in jaxpr")
    return sym


def trace_and_emit(inst, cpp):
    """Returns (lines, coef_arrays{name: np.array}, result_symbol)."""
    closed = jax.make_jaxpr(inst)(jnp.zeros((), jnp.float64))
    em = _Emit(cpp)

    # name jaxpr constants: coefficient vectors by value-match against the
    # instance's series; anything else must be a scalar
    coef_arrays = {}
    env = {}
    for cv, cval in zip(closed.jaxpr.constvars, closed.consts):
        a = np.asarray(cval)
        if a.ndim == 0:
            env[cv] = _lit(a, cpp)[0]
            continue
        hit = _match_series(inst, a)
        if hit is None:
            raise NotImplementedError(f"jaxpr constant of shape {a.shape} matches no series field")
        name = f"C_{hit[0].upper()}"
        coef_arrays[name] = np.asarray(hit[1].coef)
        env[cv] = name
        em.coef_names[name] = (name, hit[1])

    (x_var,) = closed.jaxpr.invars
    env[x_var] = "x"
    env = _emit_jaxpr(em, closed.jaxpr, env)
    (out_var,) = closed.jaxpr.outvars
    result = env[out_var] if not isinstance(out_var, _Literal) \
        else _lit(out_var.val, cpp)[0]
    return em.lines, coef_arrays, result
