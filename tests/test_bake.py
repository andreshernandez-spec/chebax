"""M4 acceptance: baked artifacts stand alone.

The emitted jax module must meet the library's own bars with chebax made
unimportable (sys.modules poisoning in a subprocess). The emitted C++ header
must compile standalone (g++, C++17) and match to the same value bar.
Emission must be deterministic (same instance, same bytes).
"""

import functools
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import chebax
from chebax import bake

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

mod_path, npz_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("baked", mod_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

d = np.load(npz_path)
xs, J, Jp = d["xs"], d["J"], d["Jp"]
y = np.asarray(m.besselj(xs))
assert np.max(np.abs(y - J)) / np.max(np.abs(J)) <= 5e-15, "f64 values"
yj = np.asarray(jax.jit(m.besselj)(xs))
assert np.max(np.abs(yj - J)) / np.max(np.abs(J)) <= 5e-15, "jit values"
g = np.asarray(jax.vmap(jax.grad(m.besselj))(jnp.asarray(xs)))
assert np.max(np.abs(g - Jp)) / np.max(np.abs(Jp)) <= 1e-14, "f64 grad"
y32 = np.asarray(m.besselj(jnp.asarray(xs, jnp.float32)))
assert y32.dtype == np.float32, "dtype"
assert np.max(np.abs(y32 - J)) / np.max(np.abs(J)) <= 1e-5, "f32 values"
print("ok")
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

    J, Jp = _refs(v)
    npz = tmp_path / "refs.npz"
    np.savez(npz, xs=XS_GRID, J=J, Jp=Jp)
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    # CPU: the parent pytest process holds the GPU memory pool, and the
    # accuracy contract is device-independent anyway
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, str(runner), str(mod), str(npz)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout


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
