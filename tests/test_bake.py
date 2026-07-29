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
