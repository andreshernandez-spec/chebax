# Example notebooks

Themed walkthroughs, not per-function references (the tests cover each
function; these show how the pieces combine). Notebook 3 needs the
`examples` extra:

```sh
pip install chebax[examples]     # adds matplotlib and numpyro
```

| Notebook | Theme |
|---|---|
| [01_bessel_family_tour](01_bessel_family_tour.ipynb) | J, Y, I, K and spherical j: values, gradients in x and in the order, ending with a Matern kernel whose smoothness nu is learned by gradient descent |
| [02_differentiable_quantiles](02_differentiable_quantiles.ipynb) | betaincinv, gammaincinv, stdtr/stdtrit (jax#2399, jax#5350, jax#20358): tail resolution, gradients in the shape parameters, reparameterized sampling, chi-square quantiles at real dof |
| [03_numpyro_truncated_and_circular](03_numpyro_truncated_and_circular.ipynb) | a truncated Student-t distribution for numpyro (numpyro#1365) fitted with NUTS, and von Mises CDF/quantile with pathwise gradients in kappa |
| [04_erfcx_lambertw_betainc](04_erfcx_lambertw_betainc.ipynb) | Gaussian tails without overflow, both real branches of Lambert W (jax#13680), and binomial reliability via betainc gradients (jax#38610) |
| [05_baking_artifacts](05_baking_artifacts.ipynb) | baking any recipe into a dependency-free pure-jax module or a standalone C++17 header |

All notebooks run on CPU in a few minutes total. They enable x64 at the
top; committed outputs were produced with fixed PRNG keys, so rerunning
reproduces them.
