"""gammainc with one traceable a per call: P, Q and their logs. One lax.cond at
a = 10 picks gammainc_small or gammainc_large, each building its coefficients
once per call. Inputs arrive normalized from gammainc.py.
"""

import jax
import jax.numpy as jnp

from chebax._src.recipes import gammainc_large as _gl
from chebax._src.recipes import gammainc_small as _gs
from chebax._src.recipes import gammainc_table as _gt
from chebax._src.recipes._common import edge_slope


def _dispatch(a, x, small, kind):
    """small() or the large-a path, by one scalar cond at a = 10."""

    def large():
        tm, dl, du = _gl.series_traced(a)
        return _gl.eval_large(a, x, tm, dl, du, kind)

    return jax.lax.cond(a <= _gt.AHI, small, large)


def p(a, x):
    def small():
        lser, ltail = _gs.series_traced(a)
        return _gs.eval_gammainc(a, x, lser, ltail, lower=True)

    return _dispatch(a, x, small, "p")


def q(a, x):
    def small():
        lser, ltail = _gs.series_traced(a)
        return _gs.eval_gammainc(a, x, lser, ltail, lower=False)

    return _dispatch(a, x, small, "q")


def log_p(a, x):
    def small():
        lser, ltail = _gs.series_traced(a)
        lp_in, lq_tl = _gs.eval_logs(a, x, lser, ltail)
        core = jnp.where(x <= _gt.XS, lp_in, jnp.log1p(-jnp.exp(lq_tl)))
        out = jnp.where(x > 0.0, core, -jnp.inf)
        return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)

    out = _dispatch(a, x, small, "logp")
    # d ln P/dx = density/P ~ a/x, so the slope at x = 0 is inf for every a
    return out + edge_slope(jnp.where(x == 0.0, jnp.inf, 0.0), x)


def log_q(a, x):
    def small():
        lser, ltail = _gs.series_traced(a)
        lp_in, lq_tl = _gs.eval_logs(a, x, lser, ltail)
        core = jnp.where(x <= _gt.XS, jnp.log1p(-jnp.exp(lp_in)), lq_tl)
        out = jnp.where(x > 0.0, core, 0.0)
        return jnp.where(jnp.isnan(x) | jnp.isnan(a), jnp.nan, out)

    out = _dispatch(a, x, small, "logq")
    # Q = 1 at x = 0, so d ln Q/dx there is just -density
    return out + edge_slope(jnp.where(x == 0.0, -_gs.density0(a), 0.0), x)
