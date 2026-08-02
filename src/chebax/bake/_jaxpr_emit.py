"""Generic artifact emission: trace a Recipe instance's __call__ to a jaxpr
and print it as self-contained Python-jax or C++17 source.

The runtime is the single source of truth: recipes stay plain, typed jnp
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

Equations whose inputs are all literals (jnp.sqrt(2/pi) and friends) are
evaluated here and inlined as constants. Left as ops they would be
recomputed in whatever dtype the process running the artifact happens to
use, which for float32 is not what was traced.

Tracing is done at float64 (bake rejects float32 tables), so a recipe's
python-level control flow is resolved at float64 and only the float64
artifact reproduces it exactly; chebax.bake's docstring states the contract.

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
    # returns x itself for +-0 and NaN, which is what jnp.sign does there
    "sign":  ("jnp.sign({0})",
              "(({0}) > 0.0 ? (T)(1) : (({0}) < 0.0 ? (T)(-1) : ({0})))"),
    "logistic": ("jax.nn.sigmoid({0})", "(1.0 / (1.0 + std::exp(-({0}))))"),
    "is_finite": ("jnp.isfinite({0})", "std::isfinite({0})"),
    "not":   ("jnp.logical_not({0})", "(!({0}))"),
    "lgamma": ("jax.scipy.special.gammaln({0})", "std::lgamma({0})"),
    # the large-a gammainc path assembles its Temme correction through
    # erfc, so a public gammainc(a > 10) hit "primitive not supported"
    # (review, 2026-08-02)
    "erf":   ("jax.scipy.special.erf({0})",  "std::erf({0})"),
    "erfc":  ("jax.scipy.special.erfc({0})", "std::erfc({0})"),
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

_BOOL_OUT = {"is_finite", "not"}

# a scalar equation of any of these folds to a literal at bake time (_fold)
_FOLDABLE = (set(_BINOP) | set(_CMP) | set(_UNARY) | set(_BINFN)
             | {"integer_pow", "select_n"})


class _Emit:
    def __init__(self, cpp, ctype="double"):
        self.cpp = cpp
        self.ctype = ctype
        self.lines = []
        self.counter = 0
        self.coef_names = {}      # id -> (name, series)
        self.endpoint_slope_dropped = False
        self.folded = {}          # var -> its bake-time value (scalar subexpressions)

    def fresh(self):
        self.counter += 1
        return f"t{self.counter}"

    def tmpl(self, pair):
        """Template for the active backend, with (T) resolved to the C++ type."""
        return pair[1].replace("(T)", f"({self.ctype})") if self.cpp else pair[0]

    def assign(self, expr, is_bool=False):
        sym = self.fresh()
        if self.cpp:
            self.lines.append(f"    {'bool' if is_bool else self.ctype} {sym} = {expr};")
        else:
            self.lines.append(f"    {sym} = {expr}")
        return sym


def _lit(val, cpp, ctype="double"):
    a = np.asarray(val)
    if a.dtype == bool:
        if cpp:
            return ("true" if bool(a) else "false"), True
        return ("True" if bool(a) else "False"), True
    v = float(a)
    if v != v:
        return (f"(std::numeric_limits<{ctype}>::quiet_NaN())" if cpp
                else "float('nan')"), False
    if v == float("inf") or v == float("-inf"):
        s = (f"std::numeric_limits<{ctype}>::infinity()" if cpp
             else "float('inf')")
        return (f"(-{s})" if v < 0 else s), False
    return (repr(v) + ("f" if cpp and ctype == "float" else "")), False


_MISSING = object()


def _const_val(em, v):
    """v's value if it is a literal or an already-folded scalar, else _MISSING."""
    if isinstance(v, _Literal):
        return v.val
    return em.folded.get(v, _MISSING)


def _fold(em, eqn):
    """Evaluate a literals-only equation now, at the traced float64, and
    return its value. Left in the artifact it would be recomputed in
    whatever dtype the host runs (jnp.sqrt(0.63) is f32 with x64 off), which
    is not what was traced."""
    vals = [_const_val(em, v) for v in eqn.invars]
    if any(w is _MISSING or np.ndim(w) != 0 for w in vals):
        return _MISSING
    return eqn.primitive.bind(*vals, **eqn.params)


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
            f"differ ({sorted(domains)}); value-matching cannot identify it. "
            "Make the coefficient vectors distinct")
    return hits[0]


# The ONE custom_jvp bake is allowed to fold, named explicitly. A
# structural "primal is zero" test is not enough: a rule returning zero
# with derivative 1 has that shape too, and folding it silently turns a
# gradient of 1 into 0 (review, 2026-08-02). Identity by function name AND
# defining module, so nothing outside chebax can present itself as this.
_ENDPOINT_JVP = ("edge_slope", "chebax/_src/recipes/_common.py")


def _is_endpoint_slope_jvp(eqn):
    """True only for chebax's own endpoint-slope helper, and only when it
    really does evaluate to zero.

    Two independent checks, because either alone is unsafe. The name and
    source file say the author INTENDED the fold and accepted what it
    costs (the one-sided slope at the endpoint, disclosed in the emitted
    artifact). Evaluating the primal says the fold cannot change a value:
    if it is not identically zero on probe inputs, refuse regardless of
    what it calls itself."""
    cj = eqn.params.get("call_jaxpr")
    if cj is None:
        return False
    inner = cj.jaxpr if hasattr(cj, "jaxpr") else cj
    info = getattr(inner, "debug_info", None)
    name = getattr(info, "func_name", None)
    src = (getattr(info, "func_src_info", "") or "").replace("\\", "/")
    if name != _ENDPOINT_JVP[0] or _ENDPOINT_JVP[1] not in src:
        return False
    return _evaluates_to_zero(cj)


def _evaluates_to_zero(cj):
    """Run the primal on probe inputs; every output must be exactly 0."""
    closed = cj if hasattr(cj, "jaxpr") else None
    jaxpr = closed.jaxpr if closed is not None else cj
    consts = closed.consts if closed is not None else ()
    try:
        args = [jnp.full(v.aval.shape, p, v.aval.dtype)
                for p in (1.0, -2.5, 7.0) for v in jaxpr.invars]
        n = len(jaxpr.invars)
        for k in range(0, len(args), n):
            outs = jax.core.eval_jaxpr(jaxpr, consts, *args[k:k + n])
            if any(float(np.max(np.abs(np.asarray(o)))) != 0.0 for o in outs):
                return False
    except Exception:
        return False
    return True


def _live_vars(jaxpr):
    """Vars some later equation or the jaxpr's own output reads.

    An equation nothing consumes still has to be emitted-or-refused
    otherwise, and jax leaves such equations behind: besseli_dnu carries a
    dead `empty` whose output goes nowhere, which is why baking it raised
    "primitive not supported" for a primitive with no effect on the result
    (review, 2026-08-02)."""
    live = {id(v) for v in jaxpr.outvars if not isinstance(v, _Literal)}
    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if not isinstance(v, _Literal):
                live.add(id(v))
    return live


def _emit_jaxpr(em, jaxpr, env):
    def read(v):
        if isinstance(v, _Literal):
            return _lit(v.val, em.cpp, em.ctype)[0]
        return env[v]

    live = _live_vars(jaxpr)
    for eqn in jaxpr.eqns:
        if eqn.outvars and not any(id(v) in live for v in eqn.outvars):
            continue        # dead: nothing downstream reads it
        p = eqn.primitive.name
        out = eqn.outvars[0] if eqn.outvars else None
        if p in _LOOP_PRIMS:
            raise NotImplementedError(
                "solver-based callables (while/scan/cond in the jaxpr) are not bakeable yet")
        if p in _FOLDABLE:
            val = _fold(em, eqn)
            if val is not _MISSING:
                em.folded[out] = val
                env[out] = _lit(val, em.cpp, em.ctype)[0]
                continue
        if p == "custom_jvp_call":
            if _is_endpoint_slope_jvp(eqn):
                # chebax's endpoint-slope rule (gammainc's density at
                # x = 0), which is verified to be identically zero, so
                # folding it cannot change a value, only the one-sided
                # slope AT the endpoint. Recorded so the emitted artifact
                # can say so; every other custom_jvp that is not a series
                # evaluation is refused below.
                em.endpoint_slope_dropped = True
                env[out] = _lit(np.zeros(()), em.cpp, em.ctype)[0]
                continue
            coef_in = [v for v in eqn.invars
                       if not isinstance(v, _Literal) and env[v] in em.coef_names]
            if len(coef_in) != 1:
                raise NotImplementedError(
                    "custom_jvp_call that is not a series evaluation (coefficient "
                    f"inputs: {len(coef_in)}); inlining it would drop its derivative "
                    "rule and the artifact would differ from the runtime under AD")
            name, series = em.coef_names[env[coef_in[0]]]
            (x_in,) = [v for v in eqn.invars if v is not coef_in[0]]
            a, b = series.domain
            suf = "f" if em.cpp and em.ctype == "float" else ""
            t_expr = (f"(2.0{suf} * ({read(x_in)}) - {b + a!r}{suf})"
                      f" / {b - a!r}{suf}")
            if em.cpp:
                expr = f"detail::clenshaw(detail::{name}, {t_expr})"
            else:
                expr = f"_clenshaw({t_expr}, {name})"
            env[out] = em.assign(expr)
        elif p in _INLINE_CALLS:
            name = str(eqn.params.get("name", ""))
            inner = eqn.params.get("jaxpr")
            if name in _NAMED_FOLDS:
                tmpl = em.tmpl(_NAMED_FOLDS[name])
                env[out] = em.assign(tmpl.format(*[read(v) for v in eqn.invars]))
                continue
            inner_jaxpr = inner.jaxpr if hasattr(inner, "jaxpr") else inner
            inner_env = dict(zip(inner_jaxpr.invars, [read(v) for v in eqn.invars]))
            for cv, cval in zip(inner_jaxpr.constvars, getattr(inner, "consts", [])):
                inner_env[cv] = _register_const(em, None, cval)
            _emit_jaxpr(em, inner_jaxpr, inner_env)
            for ov, iv in zip(eqn.outvars, inner_jaxpr.outvars):
                env[ov] = inner_env[iv] if not isinstance(iv, _Literal) \
                    else _lit(iv.val, em.cpp, em.ctype)[0]
        elif p == "convert_element_type":
            nd = np.dtype(eqn.params["new_dtype"])
            src = read(eqn.invars[0])
            if nd == eqn.invars[0].aval.dtype:
                env[out] = src  # weak-type normalization only
            elif nd == np.float64:
                # bool -> double masks and float promotions; multiplying by
                # 1.0 is exact for floats and promotes bools, keeping the
                # emitted python weak-typed
                env[out] = em.assign(f"({em.ctype})({src})" if em.cpp
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
            env[out] = em.assign(em.tmpl(_BINFN[p]).format(x, y))
        elif p in _UNARY:
            if p == "not" and eqn.invars[0].aval.dtype != np.dtype(bool):
                # jax's not is bitwise on integers, C's ! is logical
                raise NotImplementedError("not on a non-boolean operand is not bakeable")
            env[out] = em.assign(em.tmpl(_UNARY[p]).format(read(eqn.invars[0])),
                                 is_bool=p in _BOOL_OUT)
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
        return _lit(a, em.cpp, em.ctype)[0]
    sym = name_hint
    if sym is None:
        raise NotImplementedError(f"unnamed array constant of shape {a.shape} in jaxpr")
    return sym


def trace_and_emit(inst, cpp, ctype="double"):
    """Returns (lines, coef_arrays{name: np.array}, result_symbol,
    endpoint_slope_dropped)."""
    closed = jax.make_jaxpr(inst)(jnp.zeros((), jnp.float64))
    em = _Emit(cpp, ctype)

    # name jaxpr constants: coefficient vectors by value-match against the
    # instance's series; anything else must be a scalar
    coef_arrays = {}
    env = {}
    for cv, cval in zip(closed.jaxpr.constvars, closed.consts):
        a = np.asarray(cval)
        if a.ndim == 0:
            env[cv] = _lit(a, cpp, em.ctype)[0]
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
        else _lit(out_var.val, cpp, em.ctype)[0]
    return em.lines, coef_arrays, result, em.endpoint_slope_dropped
