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
    every group's parameters; an empty group contributes exactly zero
    gradient and its parameters are never passed to fn, so nan or
    out-of-domain values parked in an empty slot stay harmless.
    """
    idx = np.asarray(group_idx)
    if idx.size == 0:
        raise ValueError("group_idx is empty")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"group_idx must be integers, got {idx.dtype}")
    shape = idx.shape
    flat = idx.reshape(-1)
    if num_groups is not None and (isinstance(num_groups, bool)
                                   or int(num_groups) != num_groups):
        # 2.9 used to truncate to 2 in silence (review, 2026-08-02)
        raise ValueError(f"num_groups must be an integer, got {num_groups!r}")
    g = int(flat.max()) + 1 if num_groups is None else int(num_groups)
    if flat.min() < 0 or flat.max() >= g:
        raise ValueError(f"group_idx values must lie in [0, {g}), got "
                         f"[{flat.min()}, {flat.max()}]")

    counts = np.bincount(flat, minlength=g)
    m = int(counts.max())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(flat, kind="stable")

    empty = counts == 0
    any_empty = bool(empty.any())
    donor = int(np.flatnonzero(counts)[0])

    # gather: (g, m) element indices, rows padded with the group's first
    # element (a real in-domain x, so padding never leaves fn's domain).
    # An empty group borrows the donor's element and, below, the donor's
    # parameters, so fn only ever sees a live (param, x) pair. Passing an
    # empty group's own parameters was harmless going forward (nothing
    # reads those rows) but reverse mode turned a nan or out-of-domain
    # value there into 0 * nan, poisoning that group's gradient and, via
    # the padding index, x's.
    gather = np.empty((g, m), dtype=np.int64)
    for gi in range(g):
        c = counts[gi]
        gather[gi, :c] = order[starts[gi]:starts[gi] + c]
        gather[gi, c:] = order[starts[gi if c else donor]]

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
            ps.append(jnp.where(empty, p[donor], p) if any_empty else p)
        yg = jax.vmap(fn)(*ps, x.reshape(-1)[gather])
        return yg.reshape(-1)[back].reshape(shape)

    return wrapped
