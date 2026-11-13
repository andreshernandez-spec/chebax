"""Skew-normal log CDF: where the textbook form breaks, and what chebax's
log_skewnorm_cdf costs against it.

PyMC's SkewNormal has no logcdf (main, 2026-11-11), so Censored and Truncated
reject it. The obvious one is ln(Phi(x) - 2 T(x, a)), T Owen's T, which
pytensor's owens_t can already build on every backend. This measures, against
mpmath at 40 dps (the phi-integral of tests/test_skewnorm.py):

1. Accuracy of ln F on a grid of a in +-{0.5, 1, 3, 10, 30} and x from -40 to
   40, for: the textbook form in float64 with scipy's owens_t and with jax's
   (13-point quadrature), scipy.stats.skewnorm.logcdf, and chebax. Reported:
   the worst |ln F - ref| / |ref| per method and sign of a, the number of
   nonfinite results, and per a the x nearest 0 where a method first returns a
   nonfinite value or an error above 1e-6 (its breakdown point).
2. Speed: log_skewnorm_cdf against jnp.log(ndtr(x) - 2 owens_t(x, a)), values
   and value_and_grad of the sum in (a, x), at 2^12, 2^16 and 2^20 elements,
   a uniform on [-5, 5] and x on [-4, 4] (where the textbook form is still
   finite), on one CPU core (run under taskset -c 0) or the GPU (--gpu), in
   float64, or float32 with --float32 (a separate process: x64 is global, and
   under x64 chebax computes float32 inputs in float64). Median of 20 calls
   after warm-up, two passes; a pass disagreement above 10% is printed. The
   last column times the table read alone (ln K: the coordinates, one patch
   and its closed-form factors), as a share of the value.

It does NOT establish: speed inside a PyMC model (there is no pytensor op to
lower yet), accuracy beyond this grid (tests/test_skewnorm.py covers a from
1e-8 to 1e8 and x to 1e3, and the patch seams), or float32 accuracy.

    taskset -c 0 python experiments/49_skewnorm_logcdf.py           # ~3 min
    taskset -c 0 python experiments/49_skewnorm_logcdf.py --float32
    python experiments/49_skewnorm_logcdf.py --gpu --speed-only
    python experiments/49_skewnorm_logcdf.py --gpu --float32
"""

import argparse
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--speed-only", action="store_true")
ap.add_argument("--float32", action="store_true", help="speed in float32 only")
args = ap.parse_args()
if not args.gpu:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402

if not args.float32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import mpmath as mp  # noqa: E402
from scipy import special as sc  # noqa: E402
from scipy import stats  # noqa: E402

import chebax  # noqa: E402
from chebax._src.recipes import skewnorm as S  # noqa: E402

mp.mp.dps = 40


def k_ref(s, b):
    """ln K(s, b) = ln F(-s; b), b >= 0, from the phi-integral."""
    if s == 0:
        return mp.log(mp.acot(b) / mp.pi)
    sig2 = s * s * (1 + b * b) / 2
    phi0 = mp.acot(b)
    width = 1 / (s * s * b * (1 + b * b) + s)
    pts = sorted({phi0 - width * k for k in (1, 4, 16, 64, 256) if width * k < phi0})
    g = lambda p: mp.exp(-s * s * (mp.cot(p) ** 2 - b * b) / 2)
    return mp.log(mp.quad(g, [mp.mpf(0)] + pts + [phi0]) / mp.pi) - sig2


def ref_log_cdf(a, x):
    a, x = mp.mpf(a), mp.mpf(x)
    s, b = abs(x), abs(a)
    k = mp.exp(k_ref(s, b))
    e, p = mp.erf(s / mp.sqrt(2)), mp.erfc(s / mp.sqrt(2))
    if x <= 0:
        f, sf = (k, 1 - k) if a >= 0 else (p - k, e + k)
    else:
        f, sf = (e + k, p - k) if a >= 0 else (1 - k, k)
    return float(mp.log(f) if f <= sf else mp.log1p(-sf))


def accuracy():
    xs = np.concatenate([-np.geomspace(40, 0.05, 60), [0.0], np.geomspace(0.05, 40, 60)])
    a_vals = [s * v for v in (0.5, 1.0, 3.0, 10.0, 30.0) for s in (1, -1)]
    a, x = (v.ravel() for v in np.meshgrid(a_vals, xs, indexing="ij"))
    t0 = time.perf_counter()
    ref = np.array([ref_log_cdf(ai, xi) for ai, xi in zip(a, x)])
    print(f"{len(a)} mpmath references in {time.perf_counter() - t0:.0f} s\n")
    with np.errstate(all="ignore"):
        methods = {
            "textbook, scipy owens_t": np.log(sc.ndtr(x) - 2 * sc.owens_t(x, a)),
            "textbook, jax owens_t": np.asarray(
                jnp.log(
                    jax.scipy.special.ndtr(x) - 2 * jax.scipy.special.owens_t(x, a)
                )
            ),
            "scipy.stats.skewnorm.logcdf": stats.skewnorm.logcdf(x, a),
            "chebax.log_skewnorm_cdf": np.asarray(chebax.log_skewnorm_cdf(a, x)),
        }
    print(f"{'method':30} {'a':>6} {'worst rel':>10} {'nonfinite':>9}")
    for name, got in methods.items():
        with np.errstate(all="ignore"):
            err = np.abs(got - ref) / np.abs(ref)
        bad = ~np.isfinite(got)
        for sign, m in (("a > 0", a > 0), ("a < 0", a < 0)):
            e = np.where(bad[m], np.inf, err[m])
            fin = e[np.isfinite(e)]
            worst = f"{np.max(fin):.2g}" if fin.size else "-"
            print(f"{name:30} {sign:>6} {worst:>10} {int(bad[m].sum()):>9}")
    print("\nbreakdown point: x nearest 0 with a nonfinite result or error > 1e-6")
    print(f"{'a':>6} " + "  ".join(f"{n[:22]:>22}" for n in methods))
    for av in a_vals:
        m = a == av
        row = []
        for got in methods.values():
            with np.errstate(all="ignore"):
                err = np.abs(got[m] - ref[m]) / np.abs(ref[m])
            fail = ~np.isfinite(got[m]) | (err > 1e-6)
            if not fail.any():
                row.append("none")
                continue
            i = np.flatnonzero(fail)[np.argmin(np.abs(x[m][fail]))]
            row.append(f"x={x[m][i]:.3g}")
        print(f"{av:>6} " + "  ".join(f"{r:>22}" for r in row))
    print()


def _time(f, *a):
    jax.block_until_ready(f(*a))
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def speed():
    dev = jax.devices()[0]
    print(f"speed on {dev.platform} ({dev.device_kind}), median of 20 calls, two passes")

    def textbook(a, x):
        sp = jax.scipy.special
        return jnp.log(sp.ndtr(x) - 2.0 * sp.owens_t(x, a))

    def log_k(a, x):
        return S._log_k(jnp.abs(x), jnp.abs(a))[0]

    fns = {"chebax": chebax.log_skewnorm_cdf, "textbook": textbook}
    print(f"{'dtype':8} {'n':>8} {'mode':>16} {'chebax ns/el':>13} {'textbook ns/el':>15} "
          f"{'ratio':>6} {'ln K share':>10}")
    rng = np.random.default_rng(0)
    for dt in (jnp.empty(()).dtype,):
        for logn in (12, 16, 20):
            n = 2**logn
            a = jnp.asarray(rng.uniform(-5, 5, n).astype(dt))
            x = jnp.asarray(rng.uniform(-4, 4, n).astype(dt))
            for mode in ("value", "value_and_grad"):
                t = {}
                for name, f in fns.items():
                    if mode == "value":
                        g = jax.jit(f)
                    else:
                        g = jax.jit(jax.value_and_grad(lambda a, x, f=f: jnp.sum(f(a, x)), (0, 1)))
                    passes = [_time(g, a, x) for _ in range(2)]
                    if abs(passes[0] - passes[1]) > 0.1 * min(passes):
                        print(f"   passes disagree for {name} {mode} n=2^{logn}: {passes}")
                    t[name] = min(passes) / n * 1e9
                share = ""
                if mode == "value":
                    tk = min(_time(jax.jit(log_k), a, x) for _ in range(2)) / n * 1e9
                    share = f"{tk / t['chebax']:10.2f}"
                print(f"{np.dtype(dt).name:8} {'2^' + str(logn):>8} {mode:>16} "
                      f"{t['chebax']:13.2f} {t['textbook']:15.2f} "
                      f"{t['textbook'] / t['chebax']:5.2f}x {share}")


print(f"jax {jax.__version__}, chebax {chebax.__file__}, python {sys.version.split()[0]}")
if not (args.speed_only or args.float32):
    accuracy()
speed()
