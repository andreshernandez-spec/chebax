"""M4 acceptance: baked artifacts stand alone.

The emitted jax module must meet the library's own bars with chebax made
unimportable (sys.modules poisoning in a subprocess). The emitted C++ header
must compile standalone (g++, C++17) and match to the same value bar.
Emission must be deterministic (same instance, same bytes).

The dtype contract is part of the acceptance: the trace is float64, an
artifact declares the dtype it runs in and refuses any other input, and the
float32 artifact is the float64 computation in float32, not the float32
runtime (measured 4.4e-8 worst sup-normalized against it for besselj(2.5) on
x in [0.05, 100], bar 2e-7) but is the same values whatever the x64 setting
of the process running it (scalar subexpressions are folded at bake time).
Emitted names and coefficient tables that cannot be printed as valid source
are rejected at bake time.
"""

import functools
import importlib.util
import os
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import tempfile

import pytest

import chebax
from chebax import bake
from chebax._src.pytree import Recipe
from chebax._src.series import ChebSeries

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

XS_GRID = np.sort(np.concatenate([
    np.linspace(0.05, 8.0, 10),
    [7.9999, 8.0, 30.0, 30.0001],
    np.linspace(8.1, 29.9, 10),
    np.logspace(np.log10(31.0), 4, 10),
]))

RUNNER = """
import sys
sys.modules["chebax"] = None  # poison: any chebax import now raises
import importlib.util
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

mod_path, mod32_path, npz_path = sys.argv[1], sys.argv[2], sys.argv[3]

def load(p, nm):
    spec = importlib.util.spec_from_file_location(nm, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m = load(mod_path, "baked")
d = np.load(npz_path)
xs, J, Jp = d["xs"], d["J"], d["Jp"]
y = np.asarray(m.besselj(xs))
assert np.max(np.abs(y - J)) / np.max(np.abs(J)) <= 5e-15, "f64 values"
yj = np.asarray(jax.jit(m.besselj)(xs))
assert np.max(np.abs(yj - J)) / np.max(np.abs(J)) <= 5e-15, "jit values"
g = np.asarray(jax.vmap(jax.grad(m.besselj))(jnp.asarray(xs)))
assert np.max(np.abs(g - Jp)) / np.max(np.abs(Jp)) <= 1e-14, "f64 grad"
try:
    m.besselj(jnp.asarray(xs, jnp.float32))
except TypeError as e:
    assert "float64" in str(e) and "float32" in str(e), str(e)
else:
    raise AssertionError("the f64 artifact ran an untraced f32 input")

m32 = load(mod32_path, "baked32")
y32 = m32.besselj(jnp.asarray(xs, jnp.float32))
assert y32.dtype == np.float32, "dtype"
assert np.max(np.abs(np.asarray(y32) - J)) / np.max(np.abs(J)) <= 1e-5, "f32 values"
print("ok")
"""

# x64 stays OFF here, so chebax's canonical dtype is float32 and the f32
# runtime is the thing the f32 artifact is compared against
RUNNER_F32 = """
import sys
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
import chebax
from chebax import bake

mod_path, v, bake_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
assert jnp.empty(()).dtype == jnp.float32, "x64 must be off in this process"
spec = importlib.util.spec_from_file_location("baked32", mod_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

xs = jnp.asarray(np.linspace(0.05, 100.0, 41), jnp.float32)
ref = np.asarray(chebax.besselj(v)(xs))
got = np.asarray(m.besselj(xs))
assert ref.dtype == np.float32 and got.dtype == np.float32
np.save(bake_path + ".npy", got)  # the parent compares these to its own run

# baking here would source float32 tables, which stays rejected
try:
    bake.jax_module(chebax.besselj(v), bake_path)
except ValueError as e:
    assert "float32 tables" in str(e), str(e)
else:
    raise AssertionError("float32 tables accepted by bake")
print(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
"""


@functools.lru_cache(maxsize=None)
def _refs(v):
    J = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x))) for x in XS_GRID])
    Jp = np.array([float(mp.besselj(mp.mpf(v), mp.mpf(x), 1)) for x in XS_GRID])
    return J, Jp


@pytest.mark.parametrize("v", [2.5, 9.97])
def test_jax_module_standalone(tmp_path, v):
    mod = tmp_path / "baked.py"
    name = bake.jax_module(chebax.besselj(v), mod)
    assert name == "besselj"
    src = mod.read_text()
    assert "import chebax" not in src and "from chebax" not in src
    mod32 = tmp_path / "baked32.py"
    bake.jax_module(chebax.besselj(v), mod32, dtype="float32")

    J, Jp = _refs(v)
    npz = tmp_path / "refs.npz"
    np.savez(npz, xs=XS_GRID, J=J, Jp=Jp)
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    # CPU: the parent pytest process holds the GPU memory pool, and the
    # accuracy contract is device-independent anyway
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, str(runner), str(mod), str(mod32), str(npz)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout


def test_jax_module_f32_artifact_vs_f32_runtime(tmp_path):
    # the f32 artifact is the f64 trace carried out in f32, so it is close
    # to but not the same as chebax under x64 off: measured worst 4.4e-8
    # sup-normalized, bar 2e-7
    mod = tmp_path / "baked32.py"
    bake.jax_module(chebax.besselj(2.5), mod, dtype="float32")
    runner = tmp_path / "runner32.py"
    runner.write_text(RUNNER_F32)
    src = os.path.dirname(os.path.dirname(os.path.abspath(chebax.__file__)))
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": src}
    env.pop("JAX_ENABLE_X64", None)
    rejected = tmp_path / "rejected.py"   # baking under x64 off must refuse
    out = subprocess.run([sys.executable, str(runner), str(mod), "2.5", str(rejected)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert not rejected.exists()
    assert float(out.stdout.strip()) <= 2e-7
    # same artifact, x64 on here: folded scalars keep it the same function
    here = np.asarray(_load(mod, "baked32_here").besselj(
        jnp.asarray(np.linspace(0.05, 100.0, 41), jnp.float32)))
    np.testing.assert_array_equal(here, np.load(str(rejected) + ".npy"))


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_jax_module_rejects_untraced_dtype(tmp_path):
    inst = chebax.besselj(2.5)
    xs = np.linspace(0.05, 30.0, 15)
    mods = {}
    for dt in ("float64", "float32"):
        p = tmp_path / f"baked_{dt}.py"
        bake.jax_module(inst, p, dtype=dt)
        mods[dt] = _load(p, f"baked_{dt}")
    for dt, other in (("float64", jnp.float32), ("float32", jnp.float64)):
        good = jnp.asarray(xs, dt)
        assert mods[dt].besselj(good).dtype == np.dtype(dt)
        with pytest.raises(TypeError, match=f"baked for {dt} input"):
            mods[dt].besselj(jnp.asarray(xs, other))
        # an int array is not the traced dtype either
        with pytest.raises(TypeError, match="baked for"):
            mods[dt].besselj(jnp.arange(3))
    with pytest.raises(ValueError, match="dtype"):
        bake.jax_module(inst, tmp_path / "x.py", dtype="float16")


def test_xsf_header_compiles_and_matches(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    hdr = tmp_path / "baked.h"
    name = bake.xsf_header(chebax.besselj(2.5), hdr)
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "baked.h"\n#include <cstdio>\n'
        'int main() { double x; while (std::scanf("%lf", &x) == 1) '
        'std::printf("%.17e\\n", chebax_baked::' + name + "(x)); return 0; }\n")
    exe = tmp_path / "baked_exe"
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(exe), str(main)],
                   check=True, cwd=tmp_path)
    inp = "\n".join(repr(float(x)) for x in XS_GRID)
    out = subprocess.run([str(exe)], input=inp, capture_output=True, text=True, check=True)
    got = np.array([float(t) for t in out.stdout.split()])
    J, _ = _refs(2.5)
    assert np.max(np.abs(got - J)) / np.max(np.abs(J)) <= 5e-15


BAKEABLE = [
    ("besselj", lambda: chebax.besselj(2.5), np.linspace(0.05, 100.0, 20)),
    ("besselk", lambda: chebax.besselk(1.5), np.logspace(-5, 2, 20)),
    ("besseli", lambda: chebax.besseli(2.5, scaled=True), np.linspace(0.0, 100.0, 20)),
    ("bessely", lambda: chebax.bessely(1.5), np.logspace(-5, 3, 20)),
    ("betainc", lambda: chebax.betainc(2.0, 3.0), np.linspace(0.0, 1.0, 21)),
    ("gammainc", lambda: chebax.gammainc(2.5), np.linspace(0.0, 50.0, 21)),
    ("spherical", lambda: __import__("chebax").spherical_jn(2), np.linspace(0.1, 40.0, 20)),
]


@pytest.mark.parametrize("family,make,xs", BAKEABLE, ids=[b[0] for b in BAKEABLE])
def test_generic_bake_python_matches_runtime(tmp_path, family, make, xs):
    # the artifact is derived from the runtime's own trace, so it must agree
    # to reassociation-free precision
    import importlib.util
    inst = make()
    mod = tmp_path / "baked.py"
    name = bake.jax_module(inst, mod)
    spec = importlib.util.spec_from_file_location("baked_gen", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    got = np.asarray(getattr(m, name)(xs))
    ref = np.asarray(inst(xs))
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("family,make,xs", BAKEABLE[1:3], ids=["besselk", "besseli"])
def test_generic_bake_cpp_matches_runtime(tmp_path, family, make, xs):
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    inst = make()
    hdr = tmp_path / "baked.h"
    name = bake.xsf_header(inst, hdr)
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "baked.h"\n#include <cstdio>\n'
        'int main() { double x; while (std::scanf("%lf", &x) == 1) '
        'std::printf("%.17e\\n", chebax_baked::' + name + "(x)); return 0; }\n")
    exe = tmp_path / "exe"
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(exe), str(main)],
                   check=True, cwd=tmp_path)
    out = subprocess.run([str(exe)], input="\n".join(repr(float(x)) for x in xs),
                         capture_output=True, text=True, check=True)
    got = np.array([float(t) for t in out.stdout.split()])
    ref = np.asarray(inst(xs))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-300)


def test_emission_deterministic(tmp_path):
    jv = chebax.besselj(2.5)
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    bake.jax_module(jv, a)
    bake.jax_module(jv, b)
    assert a.read_bytes() == b.read_bytes()
    ha, hb = tmp_path / "a.h", tmp_path / "b.h"
    bake.xsf_header(jv, ha)
    bake.xsf_header(jv, hb)
    assert ha.read_bytes() == hb.read_bytes()


def test_bake_rejects_non_besselj(tmp_path):
    with pytest.raises(TypeError, match="BesselJ"):
        bake.jax_module(chebax.fit(np.exp), tmp_path / "x.py")


def test_truncate_drops_converged_tail():
    # measured: besselk(1.5) inner degree 79 -> 26 at tol 1e-7 with
    # 1.7e-7 worst relative deviation from the full instance
    k = chebax.besselk(1.5)
    kt = k.truncate(1e-7)
    assert kt.ltil.degree < k.ltil.degree // 2
    xs = np.logspace(-5, 2, 40)
    rel = np.abs(np.asarray(kt(xs)) - np.asarray(k(xs))) / np.asarray(k(xs))
    assert rel.max() <= 1e-6
    # a fit-produced series truncates too
    s = chebax.fit(lambda x: np.exp(-x * x), (0.0, 3.0), tol=1e-15)
    st = s.truncate(1e-7)
    assert st.degree < s.degree
    # tol below every coefficient is the identity
    assert k.truncate(1e-30).ltil.degree == k.ltil.degree


def test_jax_module_truncate_tol(tmp_path):
    import importlib.util
    inst = chebax.besselk(1.5)
    mod = tmp_path / "baked32.py"
    name = bake.jax_module(inst, mod, truncate_tol=1e-7)
    spec = importlib.util.spec_from_file_location("baked32_gen", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    xs = np.logspace(-5, 1.5, 30)
    got = np.asarray(getattr(m, name)(xs))
    ref = np.asarray(inst(xs))
    np.testing.assert_allclose(got, ref, rtol=1e-6)


@pytest.mark.parametrize("family,make,xs", BAKEABLE[1:3], ids=["besselk", "besseli"])
def test_bake_cpp_float32(tmp_path, family, make, xs):
    # dtype="float" + truncate_tol=1e-7: an f32 CUDA-shaped kernel body.
    # The accuracy contract is f32-grade AND value-representability: for
    # the log-table recipes the f32 floor is ~eps32 * |ln f| (exponent
    # assembly), so the grid keeps |ln f| <= 40 and the bar sits at
    # 1e-5 (measured worst 3.9e-6 for besselk); values below the f32
    # normal floor underflow by format, not by kernel error.
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    inst = make()
    hdr = tmp_path / "baked32.h"
    name = bake.xsf_header(inst, hdr, dtype="float", truncate_tol=1e-7)
    assert "inline constexpr float" in hdr.read_text()
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "baked32.h"\n#include <cstdio>\n'
        'int main() { float x; while (std::scanf("%f", &x) == 1) '
        'std::printf("%.9e\\n", chebax_baked::' + name + "(x)); return 0; }\n")
    exe = tmp_path / "exe32"
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(exe), str(main)],
                   check=True, cwd=tmp_path)
    ref_full = np.asarray(inst(xs))
    keep = (np.abs(np.log(np.abs(ref_full) + 1e-300)) <= 40.0) & (ref_full != 0)
    xs_k = np.asarray(xs)[keep]
    out = subprocess.run([str(exe)], input="\n".join(repr(float(x)) for x in xs_k),
                         capture_output=True, text=True, check=True)
    got = np.array([float(t) for t in out.stdout.split()])
    ref = np.asarray(inst(xs_k))
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_xsf_header_dtype_validation(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        bake.xsf_header(chebax.besselk(1.5), tmp_path / "x.h", dtype="half")


@jax.tree_util.register_pytree_node_class
class _DtypeBranch(Recipe):
    """Recipe whose python-level branch is picked by the input dtype."""

    _static_fields = ()
    _series_fields = ("s",)

    def __call__(self, x):
        x = jnp.asarray(x)
        if x.dtype == jnp.float64:
            return self.s(x)
        return -self.s(x)


@jax.tree_util.register_pytree_node_class
class _Signy(Recipe):
    """Recipe carrying a sign() into the jaxpr (dawsn's tail shape)."""

    _static_fields = ()
    _series_fields = ("s",)

    def __call__(self, x):
        x = jnp.asarray(x)
        return jnp.sign(x) * self.s(jnp.abs(x))


def _exp_series():
    return chebax.fit(np.exp, (0.0, 3.0))


def test_bake_dtype_branch_recipe(tmp_path):
    # the trace is float64, so a dtype-dependent branch bakes the f64 one;
    # the artifact must refuse f32 input rather than run that branch in f32
    inst = _DtypeBranch(_exp_series())
    mod = tmp_path / "branch.py"
    name = bake.jax_module(inst, mod, name="branchy")
    m = _load(mod, "branchy_mod")
    xs = np.linspace(0.0, 3.0, 11)
    np.testing.assert_allclose(np.asarray(getattr(m, name)(xs)),
                               np.asarray(inst(xs)), rtol=1e-13)
    assert np.all(np.asarray(m.branchy(xs)) > 0)  # the f64 branch, not -s(x)
    with pytest.raises(TypeError, match="baked for float64 input"):
        m.branchy(jnp.asarray(xs, jnp.float32))
    # f32 runtime takes the other branch, which is what the guard protects
    assert np.all(np.asarray(inst(jnp.asarray(xs, jnp.float32))) < 0)


BAD_NAMES = ["foo bar", "1x", "x); evil()", "", "lambda", "a-b", "f\nx", 7]


@pytest.mark.parametrize("bad", BAD_NAMES, ids=[repr(b) for b in BAD_NAMES])
def test_bake_rejects_bad_names(tmp_path, bad):
    inst = chebax.besselk(1.5)
    with pytest.raises(ValueError, match="identifier"):
        bake.jax_module(inst, tmp_path / "x.py", name=bad)
    with pytest.raises(ValueError, match="identifier"):
        bake.xsf_header(inst, tmp_path / "x.h", name=bad)
    assert not (tmp_path / "x.py").exists() and not (tmp_path / "x.h").exists()


@jax.tree_util.register_pytree_node_class
class _StaticTag(Recipe):
    _static_fields = ("lab",)
    _series_fields = ("s",)

    def __call__(self, x):
        return self.s(jnp.asarray(x))


def test_bake_rejects_bad_static_field_name(tmp_path):
    # the C++ name (and its guard macro) is built from the static fields
    inst = _StaticTag("a b", _exp_series())
    with pytest.raises(ValueError, match="static field lab='a b'"):
        bake.xsf_header(inst, tmp_path / "x.h")
    name = bake.xsf_header(inst, tmp_path / "ok.h", name="tagged")
    assert name == "tagged"
    assert "#define CHEBAX_BAKED_TAGGED_H" in (tmp_path / "ok.h").read_text()
    # statics also land in the emitted docstring, which they must not close
    mod = tmp_path / "quoted.py"
    bake.jax_module(_StaticTag('"""x', _exp_series()), mod, name="quoted")
    m = _load(mod, "quoted_mod")
    np.testing.assert_allclose(np.asarray(m.quoted(np.linspace(0.0, 3.0, 5))),
                               np.exp(np.linspace(0.0, 3.0, 5)), rtol=1e-13)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan], ids=["inf", "-inf", "nan"])
def test_bake_rejects_nonfinite_table(tmp_path, bad):
    # repr(nan) / repr(inf) inside a coefficient array is a NameError in the
    # emitted python and an undeclared identifier in C++, so refuse the table
    k = chebax.besselk(1.5)
    coef = np.asarray(k.ltil.coef).copy()
    coef[3] = bad
    broken = type(k)(*[getattr(k, f) for f in k._static_fields],
                     *[ChebSeries(jnp.asarray(coef), k.ltil.domain) if f == "ltil"
                       else getattr(k, f) for f in k._series_fields])
    for fn, p in ((bake.jax_module, tmp_path / "n.py"),
                  (bake.xsf_header, tmp_path / "n.h")):
        with pytest.raises(ValueError, match="nonfinite coefficient"):
            fn(broken, p)
        assert not p.exists()


@pytest.mark.parametrize("ctype,fmt", [("double", "%lf"), ("float", "%f")])
def test_cpp_sign_matches_jnp(tmp_path, ctype, fmt):
    # jnp.sign propagates NaN and keeps the sign of zero; the C++ emission
    # must do the same (and must compile: the old form used an undeclared T)
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    inst = _Signy(_exp_series())
    hdr = tmp_path / "signy.h"
    name = bake.xsf_header(inst, hdr, name="signy", dtype=ctype)
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "signy.h"\n#include <cstdio>\n'
        f'int main() {{ {ctype} x; while (std::scanf("{fmt}", &x) == 1) '
        'std::printf("%.17e\\n", chebax_baked::' + name + "(x)); return 0; }\n")
    exe = tmp_path / "exe"
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(exe), str(main)],
                   check=True, cwd=tmp_path)
    xs = ["nan", "0.0", "-0.0", "1.5", "-1.5", "-2.9", "inf", "-inf"]
    out = subprocess.run([str(exe)], input="\n".join(xs),
                         capture_output=True, text=True, check=True)
    got = np.array([float(t) for t in out.stdout.split()])
    ref = np.array([float(inst(jnp.asarray(float(t), jnp.float64))) for t in xs])
    assert np.isnan(got[0]) and np.isnan(ref[0]), "sign(NaN) must propagate"
    assert np.all(np.isnan(got) == np.isnan(ref)), "NaN lanes must agree with jax"
    assert got[2] == 0.0 and np.signbit(got[2]), "sign(-0.0) keeps its sign"
    # finite lanes against mpmath: sign(x) * exp(|x|), the recipe's closed form
    fin = np.isfinite(ref)
    exact = np.array([float(mp.sign(mp.mpf(t)) * mp.exp(abs(mp.mpf(t))))
                      for t in np.asarray(xs)[fin]])
    rtol = 5e-15 if ctype == "double" else 1e-6
    np.testing.assert_allclose(got[fin], exact, rtol=rtol)
    assert np.all(np.signbit(got[fin]) == np.signbit(ref[fin]))


# ---- review 2026-08-02: bake must not fold a custom_jvp it does not own --

_TMP = pathlib.Path(tempfile.mkdtemp())


def test_bake_refuses_a_foreign_zero_custom_jvp():
    # The fold used to be purely structural ("primal is a zero literal"),
    # which a rule returning zero with derivative 1 satisfies just as well:
    # runtime gradient 1, baked gradient 0, and the artifact still claimed
    # plain AD gave the derivative. Now the fold takes chebax's own
    # endpoint helper by name AND defining module, and only after checking
    # the primal really is zero.
    from chebax._src.pytree import Recipe
    from chebax._src.series import ChebSeries

    @jax.custom_jvp
    def sneaky(x):
        return jnp.zeros_like(x)

    @sneaky.defjvp
    def _sneaky_jvp(primals, tangents):
        return jnp.zeros_like(primals[0]), tangents[0]

    @jax.tree_util.register_pytree_node_class
    class Sneaky(Recipe):
        _static_fields = ()
        _series_fields = ("s",)

        def __call__(self, x):
            return self.s(x) + sneaky(x)

    inst = Sneaky(ChebSeries(np.array([1.0, 0.5]), (0.0, 1.0)))
    assert float(jax.grad(inst)(0.5)) != 0.0
    with pytest.raises(NotImplementedError, match="custom_jvp"):
        bake.jax_module(inst, str(_TMP / "sneaky.py"), name="sneaky")


def test_baked_artifact_discloses_a_dropped_endpoint_slope(tmp_path):
    # the flag was recorded and never reached the generated documentation
    p = tmp_path / "g.py"
    bake.jax_module(chebax.gammainc(1.0), str(p), name="g")
    head = p.read_text()[:1200]
    assert "one-sided slope" in head, head[:400]
    q = tmp_path / "b.py"
    bake.jax_module(chebax.besselj(2.5), str(q), name="b")
    assert "one-sided slope" not in q.read_text()[:1200]
