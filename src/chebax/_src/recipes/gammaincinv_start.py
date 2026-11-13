"""The Gamma quantile's start from gammaincinv_array_gen's table of the
quantile (zones in its docstring): one patch per element, 1e-8 in x, for
both paths' Newton steps. a >= 0.1, every zone to a = inf; the caller masks
a outside.
"""

import functools

import numpy as np

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammaincinv_array_table as _t
from chebax._src.recipes._common import cheb_basis, locate_zones, patch_sum
from chebax._src.recipes.gammaln import _gammaln_cj as _lgamma

_LO0, _LO1, _LO2, _CE1, _CE2, _UP0, _UP1, _UP2 = range(len(_t.NAMES))


@functools.lru_cache(maxsize=2)
def _tables(dtype):
    """(patch blocks, parameter cuts, argument cuts, ln lambda series) in
    the compute dtype."""
    dt = np.dtype(dtype)
    return tuple(
        np.ascontiguousarray(v, dt) for v in (_t.TABLE, _t.PCUTS, _t.YCUTS, _t.LAM)
    )


def _block_sum(tab, n, sy, sp):
    """patch_sum through one gathered block and an einsum. On the CPU the block goes
    through memory and patch_sum is faster (1.85x float32 values, 1.18x float64);
    on the GPU patch_sum writes each coefficient's index first (0.52x float32
    values, experiments/43)."""
    ky, kp = tab.shape[1:]
    return jnp.einsum(
        "...kj,...k,...j->...", tab[n], cheb_basis(sy, ky), cheb_basis(sp, kp)
    )


def _x_w(a, r):
    """x with x - (a-1) ln x = r, four Newton steps from max(r, 1.5 a + 1);
    the upper zone's leading part (q = e^-r / Gamma(a) to first order)."""
    x = jnp.maximum(r, 1.5 * a + 1.0)
    for _ in range(4):
        g = x - (a - 1.0) * jnp.log(x) - r
        x = jnp.maximum(x - g / (1.0 - (a - 1.0) / x), 0.5 * x)
    return x


def start(a, ltp, ltq, z, upper):
    """t0 = ln(x0/a) from the quantile table (zones in the generator's
    docstring); relatively accurate at large a, where t ~ z/sqrt(a)."""
    tab, _, _, lam = (jnp.asarray(v) for v in _tables(a.dtype.name))
    narg, offset = (jnp.asarray(v) for v in (_t.NARG, _t.OFFSET))
    lga = _lgamma(a)
    ln_a = jnp.log(a)

    panel = jnp.where(a < 1.0, 0, jnp.where(a < _t.ASPLIT, 1, 2))
    # lower: s = (p Gamma(a+1))^(1/a) / (a+1), as a log
    ln_y = (ltp + lga + ln_a) / a
    s = jnp.exp(ln_y - jnp.log1p(a))
    s_max = jnp.where(a < 1.0, _t.S_SMALL, _t.S_BIG)
    # centre: Temme's eta0
    eta0 = z / jnp.sqrt(a)
    # upper: the Lambert-W-type leading part
    r = -ltq - lga
    x_w = _x_w(a, r)
    tau = (a + 1.0) / x_w
    kind = jnp.where(s <= s_max, 0, jnp.where((a >= 1.0) & (eta0 <= _t.EHI), 1, 2))
    zone = jnp.select(
        [kind == 0, kind == 1],
        [_LO0 + panel, _CE1 + jnp.maximum(panel - 1, 0)],
        _UP0 + panel,
    )
    w = 1.0 / a
    par = jnp.select([kind == 1, panel == 2], [w, _t.ASPLIT / a], jnp.log1p(a))
    arg = jnp.select([kind == 0, kind == 1], [s, eta0], tau)
    i, sp = locate_zones(par, _t.PCUTS, _t.NPAR, zone)
    j, sy = locate_zones(arg, _t.YCUTS, _t.NARG, zone)
    n = offset[zone] + i * narg[zone] + j
    f = jax.lax.platform_dependent(tab, n, sy, sp, cpu=patch_sum, default=_block_sum)

    # centre: eta = eta0 + w eps, then ln lambda = eta g(eta)
    eta = jnp.clip(eta0 + w * f, _t.LAM_BAND[0], _t.LAM_BAND[1])
    te = 2.0 * (eta - _t.LAM_BAND[0]) / (_t.LAM_BAND[1] - _t.LAM_BAND[0]) - 1.0
    b1 = b2 = jnp.zeros_like(te)
    for c in lam[:0:-1]:
        b1, b2 = 2.0 * te * b1 - b2 + c, b1
    g = te * b1 - b2 + lam[0]
    return jnp.select(
        [kind == 0, kind == 1],
        [ln_y - ln_a + jnp.log1p(f), eta * g],
        jnp.log(x_w) - ln_a + jnp.log1p(f),
    )
