"""Using one family loads that family and nothing else.

The expected program needs one to three special functions. So `import
chebax` imports nothing beyond the package itself (a module __getattr__,
PEP 562, imports each public name on first use), and each name loads only
the tables its own computation reads. Tables are the cost: betainc's are
15 MB of source, and with no bytecode cache compiling them alone takes
~1.2 s and ~640 MB (measured 2026-10-26).

Each check runs in a fresh child process and reads its sys.modules, since
this process imported everything long ago. EXPECTED is the dependency
record: a new public module, or a new table under an existing one, has to
be entered here on purpose.
"""

import ast
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import chebax

# defining module -> the tables accessing its names loads
_GAMMA = {"gammainc_table", "gammainc_large_table", "erf_table"}
_BETA = {"betainc_table", "betainc_wide_table"}
EXPECTED = {
    "chebax._src.generate": set(),
    "chebax._src.series": set(),
    "chebax._src.pergroup": set(),
    "chebax._src.recipes.lambertw": set(),
    "chebax._src.recipes.erf_family": {"erf_table"},
    "chebax._src.recipes.besselk": {"besselk_table"},
    "chebax._src.recipes.matern": {"besselk_table"},
    # dI/dnu at nu = 0 is -K_0, which besseli_fn's order derivative uses
    "chebax._src.recipes.besseli": {"besseli_table", "besselk_table"},
    "chebax._src.recipes.besselj": {"besselj_table", "besselj_table_ext"},
    # Y's tail shares J's modulus and phase tables
    "chebax._src.recipes.bessely": {"bessely_table", "besselj_table_ext"},
    "chebax._src.recipes.spherical": {
        "besselj_table",
        "besselj_table_ext",
        "bessely_table",
    },
    "chebax._src.recipes.betainc": _BETA,
    "chebax._src.recipes.betaincinv": _BETA,
    "chebax._src.recipes.gammainc": _GAMMA,
    "chebax._src.recipes.gammaincinv": _GAMMA,
    "chebax._src.recipes.hyp1f1": {"hyp1f1_table"},
    # the betainc kernel without betainc's tensor (betainc_kernel.py)
    "chebax._src.recipes.stdtr": {"stdtr_table"},
    "chebax._src.recipes.tweedie": {"tweedie_table"},
    "chebax._src.recipes.vonmises": {
        "vonmises_table",
        "besseli_table",
        "besselk_table",
    },
}

DOMAIN_TABLES = {
    "BETAINC": {"betainc_wide_table"},
    "GAMMAINC": {"gammainc_table", "gammainc_large_table"},
    "HYP1F1": {"hyp1f1_table"},
    "STUDENT_T_DF": {"stdtr_table"},
    "TWEEDIE_P": {"tweedie_table"},
    "WRIGHT_RHO": {"tweedie_table"},
    "VONMISES_KAPPA": {"vonmises_table"},
    "BESSELI": {"besseli_table"},
    "BESSELJ": {"besselj_table"},
    "BESSELK": {"besselk_table"},
    "BESSELY": {"bessely_table"},
}


def _loaded(setup):
    """Run setup in a fresh interpreter; return the chebax submodules (and
    jax, if present) left in its sys.modules."""
    code = textwrap.dedent(setup) + textwrap.dedent("""
        import json, sys
        mods = [m for m in sys.modules if m.startswith("chebax.") or m == "jax"]
        print(json.dumps(sorted(mods)))
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return set(json.loads(out.stdout.splitlines()[-1]))


def _tables(mods):
    return {
        m.rsplit(".", 1)[1]
        for m in mods
        if m.startswith("chebax._src.recipes.") and "_table" in m
    }


def test_import_loads_nothing():
    # not even jax: `import chebax` costs the package's own __init__
    assert _loaded("import chebax") == set()


def test_expected_covers_every_public_module():
    assert set(EXPECTED) == set(chebax._LAZY)


@pytest.mark.parametrize("module", sorted(EXPECTED))
def test_names_load_only_their_tables(module):
    names = chebax._LAZY[module]
    mods = _loaded(f"""
        import chebax
        for name in {names!r}:
            getattr(chebax, name)
    """)
    assert _tables(mods) == EXPECTED[module], (module, sorted(_tables(mods)))


@pytest.mark.parametrize("name", sorted(DOMAIN_TABLES))
def test_domain_loads_only_its_table(name):
    mods = _loaded(f"from chebax.domains import {name}")
    assert _tables(mods) == DOMAIN_TABLES[name]
    assert "jax" not in mods  # the metadata layer needs numpy only


@pytest.mark.parametrize("plugin", ["numpyro", "pytensor"])
def test_plugins_import_no_table(plugin):
    pytest.importorskip(plugin)
    assert _tables(_loaded(f"import chebax.{plugin}")) == set()


def test_numpyro_gamma_loads_only_gamma_tables():
    # construction validates against arg_constraints, which TruncatedBeta
    # and Tweedie build from chebax.domains on first access
    pytest.importorskip("numpyro")
    mods = _loaded("""
        from chebax.numpyro import TruncatedGamma
        d = TruncatedGamma(3.0, low=1.0, high=6.0, validate_args=True)
        d.log_prob(2.0), d.cdf(2.3), d.icdf(0.3)
    """)
    assert _tables(mods) == _GAMMA


def test_type_checking_imports_match_the_lazy_map():
    # IDEs and type checkers read the TYPE_CHECKING block, the runtime
    # reads _LAZY; the two must name the same things
    tree = ast.parse(pathlib.Path(chebax.__file__).read_text())
    (block,) = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and getattr(n.test, "id", "") == "TYPE_CHECKING"
    ]
    static = {}
    for n in block.body:
        assert isinstance(n, ast.ImportFrom), ast.dump(n)
        static.setdefault(n.module, set()).update(a.name for a in n.names)
    lazy = {m: set(v) for m, v in chebax._LAZY.items()}
    lazy["chebax"] = set(chebax._SUBMODULES)
    assert static == lazy


def test_module_attributes():
    assert set(chebax.__all__) == set(chebax._WHERE) | {"bake", "__version__"}
    assert set(chebax.__all__) <= set(dir(chebax))
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        chebax.nope
    assert not hasattr(chebax, "nope")
    ns = {}
    exec("from chebax import *", ns)
    assert set(chebax.__all__) <= set(ns)
    assert ns["betainc"] is chebax._src.recipes.betainc.betainc
    assert chebax.domains.BETAINC.hi == 100.0
    assert callable(chebax.bake.jax_module)
