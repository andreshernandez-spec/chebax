import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _bounded_compile_cache():
    # jax 0.5.3 (the CI floor) never frees compiled executables, and each
    # traced solve embeds the multi-MB tensor constants: test_quantiles
    # alone accumulated 11 GiB there (2 GiB on current jax) and the
    # 16 GB runner OOM-killed the suite. Clearing per test bounds the
    # footprint at the largest single test (measured 2.5 GiB); the
    # recompiles cost a few minutes across the suite.
    yield
    jax.clear_caches()
