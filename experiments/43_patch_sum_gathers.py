"""gammainc_array's patch sum: one gather per coefficient instead of a gathered block and
an einsum. Speed, compile time and memory of gammainc_array and gammaincinv_array (which
evaluates gammainc_array's _log_pq in its Newton steps and derivative), before and
after, on one CPU core and on the GPU, float64 and float32.

gammainc_array sums one 2-D Chebyshev patch per element. Before, it gathered each
element's [k, j] coefficient block and contracted it with an einsum; XLA lowers the
einsum to a batched dot, which cannot fuse with the gather, so the block (12 x 12
doubles per element) is written to memory and read back. After (_common.patch_sum),
each coefficient is gathered on its own from the flat table: elementwise work that XLA
fuses into one loop. Its custom JVP returns the value and both coordinate derivatives,
packed as one [..., 3] array behind a barrier when a block is larger than 512 bytes
(float64's 12 x 12) and as three arrays otherwise (float32's 7 x 7).

Variants on the after tree: "other packing" makes the opposite packing choice for each
table (the evidence for the threshold); "start patch_sum" also moves gammaincinv's
start table (its own block and einsum) onto patch_sum, which the change does not do
(the evidence for leaving it).

Inputs: a log-uniform in [0.1, 1000]; gammainc at x = a e^u, u uniform in
[ln 0.05, ln 20] (lower, centre and upper zones); gammaincinv at p uniform in
[1e-12, 1 - 1e-12] on a log-odds scale (1e-6 in float32). Gradients in both arguments,
as a model takes them. N = 2^18, except float32 on the GPU at 2^22 (E43_N): there a call
at 2^18 is under a millisecond, too close to launch overhead to rank variants. Per call:
median of 7 after compilation; compile time and the compiled program's temporary memory
from XLA; "block" counts results of shape [N, k, j] in the compiled HLO. One CPU core is
taskset's; float32 runs with x64 off.

Usage (one process per version and device, alternated E43_REPS times by "all"):
  [E43_DEVICES=cpu E43_DTYPES=f64,f32 E43_REPS=5 E43_CORE=6 E43_N=262144] \\
      python experiments/43_patch_sum_gathers.py all <src before> <src after>
The before/after commits are recorded by the caller in the results header.

Not established: other argument distributions; the per-family accuracy (the test suite
checks it: values move by rounding only, at most a few 1e-15 here).
"""

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# E43_N: see the docstring (float32 on the GPU is timed at 2^22)
N = int(os.environ.get("E43_N", 2**18))


def run(src, device, x64, variant):
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, src)
    import jax

    jax.config.update("jax_enable_x64", x64 == "f64")
    import jax.numpy as jnp
    import numpy as np

    import chebax

    assert pathlib.Path(chebax.__file__).resolve().is_relative_to(pathlib.Path(src).resolve())
    if variant == "other":
        _other_packing()
    elif variant == "start patch_sum":
        _start_patch_sum()
    dt = np.float64 if x64 == "f64" else np.float32
    rng = np.random.default_rng(43)
    a = np.exp(rng.uniform(np.log(0.1), np.log(1000.0), N)).astype(dt)
    x = (a * np.exp(rng.uniform(np.log(0.05), np.log(20.0), N))).astype(dt)
    lo = 1e-12 if x64 == "f64" else 1e-6
    p = (1.0 / (1.0 + np.exp(-rng.uniform(np.log(lo), -np.log(lo), N)))).astype(dt)
    a, x, p = (jax.device_put(v) for v in (a, x, p))
    cases = {
        "gammainc P": (lambda a, x: chebax.gammainc_array(a, x), (a, x)),
        "gammainc P + grad": (
            jax.grad(lambda a, x: chebax.gammainc_array(a, x).sum(), argnums=(0, 1)),
            (a, x),
        ),
        "gammaincinv x": (lambda a, p: chebax.gammaincinv_array(a, p), (a, p)),
        "gammaincinv x + grad": (
            jax.grad(lambda a, p: chebax.gammaincinv_array(a, p).sum(), argnums=(0, 1)),
            (a, p),
        ),
    }
    out = []
    for name, (fn, args) in cases.items():
        t0 = time.perf_counter()
        comp = jax.jit(fn).lower(*args).compile()
        compile_s = time.perf_counter() - t0
        hlo = comp.as_text()
        block = sum(f"[{N},{k},{k}]" in line for line in hlo.splitlines() for k in (7, 10, 12))
        jax.block_until_ready(comp(*args))
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            jax.block_until_ready(comp(*args))
            ts.append(time.perf_counter() - t0)
        out.append(dict(case=name, ns=float(np.median(ts)) / N * 1e9, compile_s=compile_s,
                        temp_mib=comp.memory_analysis().temp_size_in_bytes / 2**20, block=block))
    print(json.dumps(out))


def _other_packing():
    """patch_sum with the opposite packing choice to its default for each table (the
    ablation): unpacked where it packs, packed where it does not."""
    import jax
    import jax.numpy as jnp

    from chebax._src.recipes import _common, gammainc_array

    @jax.custom_jvp
    def other(tab, n, sy, sp):
        return _common._patch_parts(tab, n, sy, sp, False)[0]

    @other.defjvp
    def _rule(primals, tangents):
        tab, n, sy, sp = primals
        _, _, dsy, dsp = tangents
        f, fy, fp = _common._patch_parts(tab, n, sy, sp, True)
        if not tab[0].size * tab.dtype.itemsize > _common._PACK_BYTES:
            parts = jax.lax.optimization_barrier(jnp.stack((f, fy, fp), axis=-1))
            f, fy, fp = parts[..., 0], parts[..., 1], parts[..., 2]
        return f, fy * dsy + fp * dsp

    gammainc_array.patch_sum = other


def _start_patch_sum():
    """The quantile start's sum on patch_sum too (it keeps its einsum: this variant is
    the evidence), by substituting the two lines in its source."""
    import inspect
    import textwrap

    from chebax._src.recipes import _common, gammaincinv_start as gs

    src = textwrap.dedent(inspect.getsource(gs.start))
    old = src[src.index("    block = tab["):src.index("    # centre: eta = eta0")]
    assert old.count("einsum") == 1, "gammaincinv_start changed: update the substitution"
    src = src.replace(old, "    f = patch_sum(tab, offset[zone] + i * narg[zone] + j, sy, sp)\n\n")
    ns = dict(gs.__dict__, patch_sum=_common.patch_sum)
    exec(compile(src, "<start on patch_sum>", "exec"), ns)
    gs.start = ns["start"]


def main(before, after):
    # E43_DEVICES (cpu,gpu), E43_REPS (3) and E43_CORE (6): the CPU core for taskset
    # should have an idle SMT sibling
    py = sys.executable
    rows = []
    devices = os.environ.get("E43_DEVICES", "cpu,gpu").split(",")
    core = os.environ.get("E43_CORE", "6")
    jobs = [(d, x) for d in devices for x in os.environ.get("E43_DTYPES", "f64,f32").split(",")]
    for rep in range(int(os.environ.get("E43_REPS", "3"))):
        for device, x64 in jobs:
            versions = [("before", before, "plain"), ("after", after, "plain"),
                        ("other packing", after, "other"),
                        ("start patch_sum", after, "start patch_sum")]
            for label, src, variant in (versions if rep % 2 == 0 else versions[::-1]):
                cmd = [py, __file__, "run", src, device, x64, variant]
                if device == "cpu":
                    cmd = ["taskset", "-c", core, *cmd]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                line = [ln for ln in res.stdout.splitlines() if ln.startswith("[")][-1]
                for r in json.loads(line):
                    rows.append(dict(r, version=label, device=device, dtype=x64, rep=rep))
    import numpy as np

    reps = int(os.environ.get("E43_REPS", "3"))
    print(f"N = {N}; ns per element = median over {reps} alternated runs of the per-run median of 7 calls;"
          f" devices {devices}, CPU core {core}")
    print(f"{'case':22s} {'device':6s} {'dtype':5s} {'version':16s} {'ns':>8s} {'vs before':>9s}"
          f" {'compile s':>9s} {'temp MiB':>9s} {'block':>5s}")
    keys = sorted({(r["case"], r["device"], r["dtype"]) for r in rows},
                  key=lambda k: (k[1], k[2], k[0]))
    for case, device, x64 in keys:
        sel = [r for r in rows if (r["case"], r["device"], r["dtype"]) == (case, device, x64)]
        base = np.median([r["ns"] for r in sel if r["version"] == "before"])
        for label in ("before", "after", "other packing", "start patch_sum"):
            v = [r for r in sel if r["version"] == label]
            if not v:
                continue
            ns = np.median([r["ns"] for r in v])
            print(f"{case:22s} {device:6s} {x64:5s} {label:16s} {ns:8.1f} {base / ns:8.2f}x"
                  f" {np.median([r['compile_s'] for r in v]):9.1f} {np.median([r['temp_mib'] for r in v]):9.0f}"
                  f" {int(np.median([r['block'] for r in v])):5d}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:6])
    else:
        main(sys.argv[2], sys.argv[3])
