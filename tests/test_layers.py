"""Import layering: the dependency table in docs/architecture.md, read off
the source with ast, so nothing here imports jax or compiles anything.

test_extras.py checks that chebax imports without the extras installed;
this checks the direction of every import, including function-local ones,
which an import test cannot see until the function runs.
"""

import ast
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "chebax"
ADAPTERS = {"numpyro.py", "pytensor.py"}


def _rel(path):
    return path.relative_to(SRC).as_posix()


def _imports(path):
    """Every module the file imports, anywhere in it. `from chebax import
    domains` counts as chebax.domains, since domains is a submodule."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{_rel(path)}: relative import"
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def _kind(path):
    rel = _rel(path)
    name = path.name
    if rel in ADAPTERS:
        return "adapter"
    if name.endswith("_gen.py") or name == "_gen_common.py":
        return "generator"
    if "_table" in name and rel.startswith("_src/recipes/"):
        return "table"
    return "runtime"


FILES = sorted(p for p in SRC.rglob("*.py"))
RUNTIME = [p for p in FILES if _kind(p) == "runtime"]


def _third_party(mods):
    tops = {m.split(".")[0] for m in mods}
    return tops - set(sys.stdlib_module_names) - {"chebax"}


@pytest.mark.parametrize("path", RUNTIME, ids=_rel)
def test_runtime_imports_jax_and_numpy_only(path):
    mods = _imports(path)
    extra = _third_party(mods) - {"jax", "numpy"}
    if _rel(path) == "_src/generate.py":
        # fit(dps=...) is the one requested high-precision path, imported
        # inside the function that needs it
        extra -= {"mpmath"}
        top = {
            n.names[0].name
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Import)
        }
        assert "mpmath" not in top
    assert not extra, (_rel(path), extra)
    gens = {m for m in mods if m.endswith("_gen") or m.endswith("_gen_common")}
    assert not gens, (_rel(path), gens)
    adapters = {m for m in mods if m in ("chebax.numpyro", "chebax.pytensor")}
    assert not adapters, (_rel(path), adapters)


@pytest.mark.parametrize("path", [p for p in FILES if _kind(p) == "table"], ids=_rel)
def test_tables_hold_data_only(path):
    mods = _imports(path)
    assert _third_party(mods) <= {"numpy"}, (_rel(path), mods)
    assert not {m for m in mods if m.startswith("chebax")}, (_rel(path), mods)


def test_domains_read_tables_only():
    # domains imported gammainc_large for one bound, which made the
    # metadata layer depend on a runtime module (review, 2026-10-26)
    mods = _imports(SRC / "domains.py")
    chebax = {m for m in mods if m.startswith("chebax")}
    bad = {
        m
        for m in chebax
        if m != "chebax._src.recipes"
        and not (m.startswith("chebax._src.recipes.") and "_table" in m)
    }
    assert not bad, bad


def test_shared_helpers_import_no_family():
    mods = _imports(SRC / "_src" / "recipes" / "_common.py")
    fam = {m for m in mods if m.startswith("chebax._src.recipes.")}
    assert not fam, fam
    mods = _imports(SRC / "_src" / "pergroup.py")
    assert not {m for m in mods if m.startswith("chebax")}, mods


def test_kernels_import_no_table():
    # a family's table-free kernel is what other families run on their own
    # tables (stdtr on betainc's); importing a table here would load it for
    # every one of them
    kernels = sorted((SRC / "_src" / "recipes").glob("*_kernel.py"))
    assert kernels
    for path in kernels:
        rec = {
            m
            for m in _imports(path)
            if m.startswith("chebax._src.recipes.")
            and not m.startswith("chebax._src.recipes._common")
        }
        assert not rec, (_rel(path), rec)
