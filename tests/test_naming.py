"""The naming convention: one name per function, and one per path.

name(params, x) takes any parameter shape: one value runs name_scalar,
an array runs name_array where the family has a per-element path
(gammainc and its inverses) and raises naming chebax.pergroup where it
does not.
name_frozen(params) is the cached callable. The checks here are that the
plain name and the path it picks give the same numbers, and that the 0.4
spellings (the *_fn names, the one-argument factory calls) still work
with a DeprecationWarning naming their replacement.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax

X = jnp.asarray([0.3, 2.0, 9.0, 60.0])
A = jnp.asarray([0.5, 2.5, 7.0, 40.0])

# plain name, its scalar (or frozen) path, the parameters
SAME_AS_SCALAR = [
    ("gammainc", lambda p, x: chebax.gammainc_scalar(*p, x), (2.5,)),
    ("gammaincc", lambda p, x: chebax.gammaincc_scalar(*p, x), (2.5,)),
    ("log_gammainc", lambda p, x: chebax.log_gammainc_scalar(*p, x), (2.5,)),
    ("log_gammaincc", lambda p, x: chebax.log_gammaincc_scalar(*p, x), (2.5,)),
    ("besselk", lambda p, x: chebax.besselk_scalar(*p, x), (1.5,)),
    ("log_besselk", lambda p, x: chebax.log_besselk_scalar(*p, x), (1.5,)),
    ("besseli", lambda p, x: chebax.besseli_scalar(*p, x), (1.5,)),
    ("betainc", lambda p, x: chebax.betainc_scalar(*p, x / 61.0), (2.0, 3.0)),
    ("log_betainc", lambda p, x: chebax.log_betainc_scalar(*p, x / 61.0), (2.0, 3.0)),
    ("hyp1f1", lambda p, x: chebax.hyp1f1_scalar(*p, x), (1.5, 2.5)),
    ("log_hyp1f1", lambda p, x: chebax.log_hyp1f1_scalar(*p, x), (1.5, 2.5)),
    ("besselj", lambda p, x: chebax.besselj_frozen(*p)(x), (2.5,)),
    ("bessely", lambda p, x: chebax.bessely_frozen(*p)(x), (1.5,)),
    ("besselj_dnu", lambda p, x: chebax.besselj_dnu_frozen(*p)(x), (2.5,)),
    ("bessely_dnu", lambda p, x: chebax.bessely_dnu_frozen(*p)(x), (1.5,)),
    ("besselk_dnu", lambda p, x: chebax.besselk_dnu_frozen(*p)(x), (1.5,)),
    ("besseli_dnu", lambda p, x: chebax.besseli_dnu_frozen(*p)(x), (1.5,)),
    ("spherical_jn", lambda p, x: chebax.spherical_jn_frozen(*p)(x), (2,)),
    ("spherical_yn", lambda p, x: chebax.spherical_yn_frozen(*p)(x), (2,)),
    ("gammaincinv", lambda p, x: chebax.gammaincinv_scalar(*p, x / 61.0), (2.5,)),
    ("gammainccinv", lambda p, x: chebax.gammainccinv_scalar(*p, x / 61.0), (2.5,)),
    ("chi2inv", lambda p, x: chebax.chi2inv_scalar(*p, x / 61.0), (3.0,)),
]

# the functions whose argument is a probability, not a point on [0, inf)
_UNIT = ("betainc", "log_betainc", "gammaincinv", "gammainccinv", "chi2inv")


@pytest.mark.parametrize(
    "name,path,params", SAME_AS_SCALAR, ids=[c[0] for c in SAME_AS_SCALAR]
)
def test_one_value_runs_the_scalar_path(name, path, params):
    x = X / 61.0 if name in _UNIT else X
    got = getattr(chebax, name)(*params, x)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(path(params, X)))


@pytest.mark.parametrize(
    "name",
    [
        "gammainc",
        "gammaincc",
        "log_gammainc",
        "log_gammaincc",
        "gammaincinv",
        "gammainccinv",
        "chi2inv",
    ],
)
def test_an_array_runs_the_array_path(name):
    f, arr = getattr(chebax, name), getattr(chebax, name + "_array")
    x = X / 61.0 if name in _UNIT else X
    np.testing.assert_array_equal(np.asarray(f(A, x)), np.asarray(arr(A, x)))
    # a size-one array is one value, whatever its shape
    np.testing.assert_array_equal(
        np.asarray(f(jnp.asarray([[2.5]]), x)),
        np.asarray(getattr(chebax, name + "_scalar")(2.5, x)),
    )
    # gradients follow the path, per element
    g = jax.grad(lambda a: f(a, x).sum())(A)
    ga = jax.grad(lambda a: arr(a, x).sum())(A)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(ga))


@pytest.mark.parametrize(
    "name,call",
    [
        ("besselk", lambda f: f(A, X)),
        ("besseli", lambda f: f(A, X)),
        ("betainc", lambda f: f(A, 3.0, X / 61.0)),
        ("hyp1f1", lambda f: f(1.5, A, X)),
        ("gammainc_scalar", lambda f: f(A, X)),
        ("gammaincinv_scalar", lambda f: f(A, X / 61.0)),
        ("gammainccinv_scalar", lambda f: f(A, X / 61.0)),
        ("chi2inv_scalar", lambda f: f(A, X / 61.0)),
    ],
    ids=[
        "besselk",
        "besseli",
        "betainc",
        "hyp1f1",
        "gammainc_scalar",
        "gammaincinv_scalar",
        "gammainccinv_scalar",
        "chi2inv_scalar",
    ],
)
def test_no_array_path_names_the_way_out(name, call):
    with pytest.raises(ValueError, match="pergroup") as e:
        call(getattr(chebax, name))
    if name.endswith("_scalar"):
        assert name.replace("_scalar", "_array") in str(e.value)


@pytest.mark.parametrize("old,new", sorted(chebax._RENAMED.items()))
def test_renamed_functions_warn(old, new):
    with pytest.warns(DeprecationWarning, match=new):
        f = getattr(chebax, old)
    assert f is getattr(chebax, new)
    with pytest.warns(DeprecationWarning):
        exec(f"from chebax import {old}", {})


FACTORIES = [
    ("gammainc", (2.5,), {}),
    ("gammaincc", (30.0,), {}),
    ("besselj", (2.5,), {"domain": (0.0, 8.0)}),
    ("besselj_dnu", (2.5,), {}),
    ("bessely", (1.5,), {}),
    ("bessely_dnu", (1.5,), {}),
    ("besseli", (1.5,), {"scaled": True}),
    ("besseli_dnu", (1.5,), {}),
    ("besselk", (1.5,), {}),
    ("besselk_dnu", (1.5,), {}),
    ("betainc", (2.0, 3.0), {}),
    ("hyp1f1", (1.5, 2.5), {}),
    ("spherical_jn", (2,), {}),
    ("spherical_yn", (2,), {}),
]


@pytest.mark.parametrize("name,params,kw", FACTORIES, ids=[f[0] for f in FACTORIES])
def test_one_argument_factory_call_warns(name, params, kw):
    frozen = getattr(chebax, name + "_frozen")(*params, **kw)
    with pytest.warns(DeprecationWarning, match=f"{name}_frozen"):
        old = getattr(chebax, name)(*params, **kw)
    # the frozen callables are cached, so it is the very same object
    assert old is frozen


def test_positional_scaled_is_the_old_factory():
    # besseli(v, True) was scaled=True positionally, never x = 1
    with pytest.warns(DeprecationWarning):
        old = chebax.besseli(1.5, True)
    assert old is chebax.besseli_frozen(1.5, scaled=True)


@pytest.mark.parametrize("name", ["besselk", "besseli"])
def test_v_keyword_is_the_old_spelling_of_nu(name):
    f = getattr(chebax, name)
    with pytest.warns(DeprecationWarning, match="nu="):
        old = f(v=1.5)
    assert old is getattr(chebax, name + "_frozen")(1.5)
    with pytest.warns(DeprecationWarning, match="nu="):
        got = f(v=1.5, x=X)
    np.testing.assert_array_equal(got, f(nu=1.5, x=X))
    with pytest.raises(TypeError, match="twice"):
        f(1.5, X, v=1.5)


@pytest.mark.parametrize("name", ["besselj", "besselj_dnu"])
def test_positional_domain_is_the_old_factory(name):
    # besselj(v, (lo, hi)) had the domain second, before x existed
    with pytest.warns(DeprecationWarning, match=f"{name}_frozen"):
        old = getattr(chebax, name)(2.5, (0.0, 8.0))
    assert old is getattr(chebax, name + "_frozen")(2.5, domain=(0.0, 8.0))


@pytest.mark.parametrize(
    "name", ["gammainc", "gammaincc", "gammaincinv", "gammainccinv", "log_gammainc"]
)
def test_empty_parameter_arrays_take_the_array_path(name):
    empty = jnp.zeros((0,))
    got = getattr(chebax, name)(empty, empty)
    assert got.shape == (0,)
    assert got.shape == getattr(chebax, name + "_array")(empty, empty).shape


def test_new_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        chebax.gammainc(2.5, X)
        chebax.gammainc(A, X)
        chebax.besselk(1.5, X)
        chebax.betainc_frozen(2.0, 3.0)(X / 61.0)
