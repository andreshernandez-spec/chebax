import os
import zlib

import jax
import pytest

jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(config, items):
    # CI splits the suite across runners with CHEBAX_SHARD=i/n. A hash of
    # the test id picks the shard, so every test runs in exactly one.
    shard = os.environ.get("CHEBAX_SHARD")
    if not shard:
        return
    i, n = (int(s) for s in shard.split("/"))
    keep, drop = [], []
    for item in items:
        mine = zlib.crc32(item.nodeid.encode()) % n == i
        (keep if mine else drop).append(item)
    config.hook.pytest_deselected(items=drop)
    items[:] = keep


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
