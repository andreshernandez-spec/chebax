"""The Hermite function: hh and log_hh, nu per element in [0.1, inf), z real.

References are mpmath at 40 dps: pcfd, Hh_nu(z) = e^(-z^2/4) D_{-nu-1}(z), on
2,000 points with nu log-uniform on [0.1, 150] and z uniform on [-10, 10], at
sigma = z/sqrt(z^2 + 4(nu + 1)) uniform on (-1, 1), log-uniform in |z| from 10
to 1e4 on both sides, and the noncentral t's nu in [1, 100], z in [-40, 40];
100 points with nu log-uniform on [150, 1e6] against hh_gen's quadrature in
the saddle's scaled variable (equal to pcfd to 1e-38 on nu in [20, 150]); and
the 4 corners of every 4th patch, where every T_k is +-1 and the dropped
coefficients add up. Hh is positive, so errors are |ln Hh - ref| / max(1,
|ref|) for both forms.

Measured worst, the same on jax 0.11 and 0.5.3 (CI's floor): float64 ln Hh
2.7e-16 on the random points and 4.1e-16 at the corners, Hh 4.0e-16 (bars
1e-15, 2e-15, 1.5e-15); float32 2.0e-7 for both forms on the float32-rounded
points and corners (bar 1e-6). Slopes relative to max(1, |slope|), d/dnu
against mp.diff and d/dz against -Hh_{nu-1}/Hh_nu: 3.3e-15 and 6.1e-15 here
(bar 3e-14), 5.3e-14 and 2.5e-14 at every patch corner below nu = 1.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebax
from chebax._src.recipes import hh_gen
from chebax._src.recipes import hh_table as t

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40


def _points(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    nu = np.exp(rng.uniform(np.log(0.1), np.log(150.0), n))
    kind = rng.integers(0, 4, n)
    sg = rng.uniform(-0.999, 0.999, n)
    big = np.exp(rng.uniform(np.log(10.0), np.log(1e4), n)) * rng.choice([-1, 1], n)
    nct = np.exp(rng.uniform(0.0, np.log(100.0), n))
    nu = np.where(kind == 3, nct, nu)
    z = np.select(
        [kind == 0, kind == 1, kind == 2],
        [
            rng.uniform(-10.0, 10.0, n),
            2.0 * np.sqrt(nu + 1.0) * sg / np.sqrt(1.0 - sg * sg),
            big,
        ],
        rng.uniform(-40.0, 40.0, n),
    )
    return nu, z


def _corners(step=4):
    """(nu, z) at the 4 corners of every step-th patch, just inside; zone H's
    corners at tau = 0 (nu = inf) and sigma = +-1 (z = +-inf) left out."""
    box = np.asarray(t.BOX).reshape(-1, 2, 2)
    tree = np.asarray(t.TREE)
    high = np.zeros(len(box), bool)
    for zone, rows in enumerate(t.ROOT):
        for row in rows:
            for root in row:
                stack = [root]
                while stack:
                    d, _, left, right = tree[stack.pop()]
                    if d >= 0:
                        stack += [int(left), int(right)]
                    else:
                        high[int(left)] = zone == 1
    pts = []
    for n in range(0, len(box), step):
        mid, half = box[n].mean(axis=1), (box[n][:, 1] - box[n][:, 0]) / 2
        for cp in (-1, 1):
            for cs in (-1, 1):
                p, sg = mid + half * np.array([cp, cs]) * (1 - 1e-9)
                if abs(sg) > 1 - 1e-7 or (high[n] and p < 1e-6):
                    continue
                nu = t.NU1 / p if high[n] else np.exp(p)
                pts.append((nu, 2.0 * np.sqrt(nu + 1.0) * sg / np.sqrt(1.0 - sg * sg)))
    return tuple(np.array(pts).T)


def _mp_ref(nu, z):
    return np.array(
        [
            float(hh_gen.log_hh(mp, mp.mpf(float(a)), mp.mpf(float(b))))
            for a, b in zip(nu, z)
        ]
    )


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    assert np.all(np.isfinite(got)), "unexpected nonfinite values"
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


@pytest.fixture(scope="module")
def refs():
    nu, z = _points()
    rng = np.random.default_rng(3)
    big = np.exp(rng.uniform(np.log(150.0), np.log(1e6), 100))
    zb = 2.0 * np.sqrt(big) * rng.uniform(-3.0, 3.0, 100)
    nu, z = np.concatenate([nu, big]), np.concatenate([z, zb])
    return nu, z, _mp_ref(nu, z)


@pytest.fixture(scope="module")
def corners():
    nu, z = _corners()
    return nu, z, _mp_ref(nu, z)


def test_log_against_mpmath(refs):
    nu, z, ln_h = refs
    assert np.max(_err(jax.jit(chebax.log_hh)(nu, z), ln_h)) <= 1e-15


def test_log_at_patch_corners(corners):
    nu, z, ln_h = corners
    assert np.max(_err(jax.jit(chebax.log_hh)(nu, z), ln_h)) <= 2e-15


def test_values_relative(refs):
    nu, z, ln_h = refs
    live = np.abs(ln_h) < 700.0
    got = np.asarray(jax.jit(chebax.hh)(nu[live], z[live]))
    rel = np.abs(got / np.exp(ln_h[live]) - 1.0)
    assert np.max(rel / np.maximum(1.0, np.abs(ln_h[live]))) <= 1.5e-15


def _x64_off(tmp_path, body, **arrays):
    """Run body in a child process with x64 off (process-global): it reads
    the arrays from d and puts its results in the dict out."""
    np.savez(tmp_path / "in.npz", **arrays)
    code = (
        textwrap.dedent(f"""
        import numpy as np, jax, jax.numpy as jnp, chebax
        assert jnp.empty(()).dtype == jnp.float32, "x64 leaked in"
        d = np.load({str(tmp_path / "in.npz")!r})
        out = {{}}
    """)
        + textwrap.dedent(body)
        + textwrap.dedent(f"""
        np.savez({str(tmp_path / "out.npz")!r}, **out)
    """)
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, run.stderr[-3000:]
    return dict(np.load(tmp_path / "out.npz"))


def test_float32(refs, corners, tmp_path):
    """x64 off: the leading K32 x K32 block of each patch, every closed form in
    float32, on float32-rounded points and corners with |z| < 1e18 (z^2/2
    overflows float32 past ~2e19)."""
    nu, z = (
        np.concatenate([u[::4], v]).astype(np.float32)
        for u, v in zip(refs[:2], corners[:2])
    )
    keep = np.abs(z) < 1e18
    nu, z = nu[keep], z[keep]
    got = _x64_off(
        tmp_path,
        """
        out["l"] = np.asarray(jax.jit(chebax.log_hh)(d["nu"], d["z"]))
        out["h"] = np.asarray(jax.jit(chebax.hh)(d["nu"], d["z"]))
        assert out["l"].dtype == np.float32
        """,
        nu=nu,
        z=z,
    )
    ln_h = _mp_ref(nu.astype(np.float64), z.astype(np.float64))
    assert np.max(_err(got["l"], ln_h)) <= 1e-6
    live = np.abs(ln_h) < 80.0
    rel = np.abs(got["h"][live] / np.exp(ln_h[live]) - 1.0)
    assert np.max(rel / np.maximum(1.0, np.abs(ln_h[live]))) <= 1e-6


def test_slopes_against_mpmath(refs):
    """d ln Hh/dz against -Hh_{nu-1}/Hh_nu (Airey; nu - 1 > -1) and d/dnu against
    mp.diff, relative to max(1, |slope|)."""
    nu, z = (v[:2000:25] for v in refs[:2])
    g = jax.jit(jax.vmap(jax.grad(chebax.log_hh, (0, 1))))
    g_nu, g_z = (np.asarray(v) for v in g(nu, z))
    worst = [0.0, 0.0]
    for k in range(nu.size):
        a, x = mp.mpf(nu[k]), mp.mpf(z[k])
        h = lambda m: mp.pcfd(-m - 1, x) * mp.exp(-x * x / 4)
        ref = (mp.diff(lambda m: mp.log(h(m)), a), -h(a - 1) / h(a))
        for m, (got, r) in enumerate(zip((g_nu[k], g_z[k]), ref)):
            worst[m] = max(worst[m], float(abs(got - r) / max(1, abs(r))))
    assert max(worst) <= 3e-14, worst


def test_both_modes_agree(refs):
    live = np.abs(refs[2]) < 700.0
    nu, z = (jnp.asarray(v[live][::50]) for v in refs[:2])
    for f in (chebax.hh, chebax.log_hh):
        rev = jax.grad(lambda a, b: f(a, b).sum(), (0, 1))(nu, z)
        for k, r in enumerate(rev):
            tan = [jnp.zeros_like(nu)] * 2
            tan[k] = jnp.ones_like(nu)
            fwd = jax.jvp(f, (nu, z), tuple(tan))[1]
            np.testing.assert_allclose(fwd, r, rtol=1e-14, atol=0)


def test_edges_and_out_of_box():
    inf, nan = np.inf, np.nan
    nu = jnp.array([2.0, 2.0, inf, 0.05, nan, 2.0])
    z = jnp.array([-inf, inf, 1.5, 1.0, 1.0, nan])
    np.testing.assert_array_equal(
        chebax.log_hh(nu, z), [inf, -inf, -inf, nan, nan, nan]
    )
    np.testing.assert_array_equal(chebax.hh(nu, z), [inf, 0.0, 0.0, nan, nan, nan])
    assert np.isnan(jax.grad(chebax.hh)(0.05, 1.0))


def test_known_values():
    """Hh_nu(0) = 2^((nu-1)/2) Gamma((nu+1)/2)/Gamma(nu+1), and Hh_1(z) =
    e^(-z^2/2) - z sqrt(pi/2) erfc(z/sqrt(2))."""
    for nu in (0.1, 0.7, 3.0, 25.0, 400.0):
        a = mp.mpf(nu)
        ref = (a - 1) / 2 * mp.log(2) + mp.loggamma((a + 1) / 2) - mp.loggamma(a + 1)
        assert abs(float(chebax.log_hh(nu, 0.0)) - float(ref)) <= 1e-15 * max(
            1, abs(float(ref))
        )
    for x in (-6.0, -1.0, 0.3, 2.0, 9.0):
        X = mp.mpf(x)
        ref = mp.exp(-X * X / 2) - X * mp.sqrt(mp.pi / 2) * mp.erfc(X / mp.sqrt(2))
        got = float(chebax.hh(1.0, x))
        assert abs(got / float(ref) - 1) <= 1e-15 * max(1, abs(float(mp.log(ref)))), x


def test_broadcast_and_jit():
    nu = jnp.array([[1.5], [70.0]])
    z = jnp.array([-3.0, 0.5, 12.0])
    got = jax.jit(chebax.log_hh)(nu, z)
    assert got.shape == (2, 3)
    ref = _mp_ref(
        *(v.ravel() for v in np.broadcast_arrays(np.asarray(nu), np.asarray(z)))
    )
    np.testing.assert_allclose(np.asarray(got).ravel(), ref, rtol=2e-15, atol=0)


def test_tree_covers_each_base_box():
    """Every base box's leaves tile it: areas add up and each leaf lies inside."""
    tree = np.asarray(t.TREE)
    box = np.asarray(t.BOX).reshape(-1, 2, 2)
    w = (np.log(t.NU1) - np.log(t.NU_LO)) / t.NP_L
    for zone, rows in enumerate(t.ROOT):
        for ip, row in enumerate(rows):
            plo, phi = (
                (0.0, 1.0)
                if zone
                else (np.log(t.NU_LO) + w * ip, np.log(t.NU_LO) + w * (ip + 1))
            )
            for isg, root in enumerate(row):
                slo, shi = t.SCUTS[isg], t.SCUTS[isg + 1]
                stack, area = [root], 0.0
                while stack:
                    d, _, left, right = tree[stack.pop()]
                    if d >= 0:
                        stack += [int(left), int(right)]
                        continue
                    (a0, a1), (b0, b1) = box[int(left)]
                    assert plo - 1e-12 <= a0 < a1 <= phi + 1e-12
                    assert slo - 1e-12 <= b0 < b1 <= shi + 1e-12
                    area += (a1 - a0) * (b1 - b0)
                full = (phi - plo) * (shi - slo)
                assert abs(area - full) <= 1e-12 * full


@pytest.mark.slow
def test_base_box_regenerates_exact():
    """One base box of zone L, rebuilt (the whole module is compared only with
    CHEBAX_FULL_REGEN=1)."""
    zone, ip, isg = "L", 3, 3
    cells, tree = hh_gen.box_cells(zone, ip, isg)
    root = t.ROOT[0][ip][isg]
    leaves, stack = [], [root]
    while stack:
        d, _, left, right = t.TREE[stack.pop()]
        if d >= 0:
            stack += [int(right), int(left)]
        else:
            leaves.append(int(left))
    assert len(leaves) == len(cells)
    k = hh_gen.BLOCK
    for n, (_, _, _, c) in zip(leaves, cells):
        assert np.array_equal(np.array(c)[:k, :k], t.TABLE[n])


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CHEBAX_FULL_REGEN"),
    reason="~15 min on six cores; set CHEBAX_FULL_REGEN=1 (pre-release)",
)
def test_table_regenerates_bit_for_bit(tmp_path):
    hh_gen.main(out_dir=tmp_path)
    here = pathlib.Path(hh_gen.__file__).parent / "hh_table.py"
    assert (tmp_path / "hh_table.py").read_bytes() == here.read_bytes()
