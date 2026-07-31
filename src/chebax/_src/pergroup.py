"""Per-group parameters for the traced recipes.

The traced functions (betainc_fn, besselk_fn, ...) take their parameters
uniform per call. `pergroup` relaxes that to per-GROUP: a static integer
array assigns every element of x to one of G groups, and each group gets
its own parameter set. Grouping is fixed at wrap time; parameter VALUES
stay traceable, so the wrapped function works under jit, grad and vmap
with G independent parameter sets. This is the hierarchical-model
regime: one group per chain, per mixture component, per plate level.

Cost is G table reconstructions plus one evaluation over a (G, m) padded
matrix, m the largest group. Balanced groups pay ~nothing over a single
uniform call; a group holding half the elements makes the padded matrix
half empty. Measured crossover in experiments/07_pergroup_crossover.py:
reconstruction overhead is within a few percent of the uniform floor for
n/group >= 16k and grows slowly below that.

Per-element parameters (a distinct value per point) stay out of scope;
see the scope boundary in PROJECT.md.
"""

import numpy as np

import jax
import jax.numpy as jnp


def pergroup(fn, group_idx, num_groups=None):
    """Wrap a traced recipe so each element group gets its own parameters.

    fn: a function f(param0, ..., paramK, x) whose leading arguments are
        uniform-per-call scalar parameters and whose last argument is
        the evaluation array (betainc_fn, besselk_fn, stdtr, ...).
        Keyword arguments (e.g. scaled=) are bound with functools.partial
        before wrapping.
    group_idx: STATIC integer array (numpy, not traced), same shape as
        the x the wrapped function will receive; element i belongs to
        group group_idx[i]. Rebuild the wrapper when the grouping
        changes; the index bookkeeping is numpy work done once here.
    num_groups: G, defaults to group_idx.max() + 1. Pass it explicitly
        when trailing groups may be empty.

    Returns g(param0, ..., paramK, x): each param is a length-G array
    (traceable), x has group_idx's shape, and the result is
    fn(param0[gi], ..., paramK[gi], x[i]) elementwise. Gradients flow to
    every group's parameters; an empty group contributes zero gradient.
    """
    idx = np.asarray(group_idx)
    if idx.size == 0:
        raise ValueError("group_idx is empty")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"group_idx must be integers, got {idx.dtype}")
    shape = idx.shape
    flat = idx.reshape(-1)
    g = int(flat.max()) + 1 if num_groups is None else int(num_groups)
    if flat.min() < 0 or flat.max() >= g:
        raise ValueError(f"group_idx values must lie in [0, {g}), got "
                         f"[{flat.min()}, {flat.max()}]")

    counts = np.bincount(flat, minlength=g)
    m = int(counts.max())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(flat, kind="stable")

    # gather: (g, m) element indices, rows padded with the group's first
    # element (a real in-domain x, so padding never leaves fn's domain);
    # empty groups use element 0 and their rows are never read back
    gather = np.zeros((g, m), dtype=np.int64)
    for gi in range(g):
        c = counts[gi]
        if c:
            rows = order[starts[gi]:starts[gi] + c]
            gather[gi, :c] = rows
            gather[gi, c:] = rows[0]

    # back[i] = position of element i in the flattened (g, m) result
    pos = np.empty(flat.size, dtype=np.int64)
    pos[order] = np.arange(flat.size) - np.repeat(starts, counts)
    back = flat.astype(np.int64) * m + pos

    def wrapped(*args):
        *params, x = args
        if not params:
            raise TypeError("pergroup: fn takes at least one parameter "
                            "before x")
        x = jnp.asarray(x)
        if x.shape != shape:
            raise ValueError(f"x has shape {x.shape}, group_idx has "
                             f"shape {shape}")
        ps = []
        for k, p in enumerate(params):
            p = jnp.asarray(p)
            if p.shape != (g,):
                raise ValueError(f"parameter {k} has shape {p.shape}, "
                                 f"expected ({g},) for {g} groups")
            ps.append(p)
        yg = jax.vmap(fn)(*ps, x.reshape(-1)[gather])
        return yg.reshape(-1)[back].reshape(shape)

    return wrapped
