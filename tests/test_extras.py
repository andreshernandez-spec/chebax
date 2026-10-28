"""Optional extras stay optional.

pip install chebax brings jax and numpy and nothing else. The extras are
imported only where they are used, so import chebax works without any of
them, and the entry points that do need one say which extra to install
rather than failing with a bare ModuleNotFoundError.

Every check runs in a child process with the optional packages made
unimportable (sys.modules[name] = None makes `import name` raise
ImportError), so the main suite checks this wherever it runs. The
no-extras CI job also runs this file on a plain pip install, where they
are really absent.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# the extras plus tfp (chebax.pytensor's batched fallback). scipy is not
# here: jax depends on it, so it is never missing.
_OPTIONAL = ("mpmath", "numpyro", "pytensor", "matplotlib", "tensorflow_probability")


def _child(body):
    """Run body with the optional packages unimportable; return stdout."""
    code = (
        f"import sys\n"
        f"for name in {_OPTIONAL!r}:\n"
        f"    sys.modules[name] = None\n" + textwrap.dedent(body)
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_import_needs_no_extra():
    _child("""
        import chebax, chebax.bake, chebax.domains
    """)


@pytest.mark.parametrize(
    "module, extra", [("chebax.numpyro", "numpyro"), ("chebax.pytensor", "pytensor")]
)
def test_plugin_names_its_extra(module, extra):
    msg = _child(f"""
        try:
            import {module}
        except ImportError as e:
            print(e)
    """)
    assert f"pip install chebax[{extra}]" in msg


def test_fit_with_dps_names_its_extra():
    msg = _child("""
        import chebax
        try:
            chebax.fit(lambda x: x, dps=30)
        except ImportError as e:
            print(e)
    """)
    assert "pip install chebax[gen]" in msg
