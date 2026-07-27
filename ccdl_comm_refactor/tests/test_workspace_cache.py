from ccdl_comm.config import CompressionConfig
from ccdl_comm.communication.workspace import DequantizedWorkspaceCache


class FakeTensor:
    def __init__(self, *, dtype="float32", device="cuda:0", name="tensor"):
        self.dtype = dtype
        self.device = device
        self.name = name


def test_workspace_cache_reuses_matching_bucket_workspace() -> None:
    calls = []

    def allocator(tensor, shape, config):
        calls.append((tensor, shape, config.group_size))
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"workspace-{len(calls)}")

    cache = DequantizedWorkspaceCache(allocator=allocator)
    tensor = FakeTensor()
    config = CompressionConfig(group_size=64)

    first = cache.get("bucket0", tensor, (65,), config)
    second = cache.get("bucket0", tensor, (65,), config)

    assert first is second
    assert len(calls) == 1


def test_workspace_cache_reallocates_when_shape_changes() -> None:
    calls = []

    def allocator(tensor, shape, config):
        calls.append((shape, config.group_size))
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"workspace-{len(calls)}")

    cache = DequantizedWorkspaceCache(allocator=allocator)
    tensor = FakeTensor()
    config = CompressionConfig(group_size=64)

    first = cache.get("bucket0", tensor, (64,), config)
    second = cache.get("bucket0", tensor, (65,), config)

    assert first is not second
    assert len(calls) == 2


def test_workspace_cache_reallocates_when_dtype_changes() -> None:
    calls = []

    def allocator(tensor, shape, config):
        calls.append((tensor.dtype, shape, config.group_size))
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"workspace-{len(calls)}")

    cache = DequantizedWorkspaceCache(allocator=allocator)
    config = CompressionConfig(group_size=64)

    first = cache.get("bucket0", FakeTensor(dtype="float16"), (64,), config)
    second = cache.get("bucket0", FakeTensor(dtype="float32"), (64,), config)

    assert first is not second
    assert len(calls) == 2


def test_workspace_cache_clear_drops_cached_entries() -> None:
    calls = []

    def allocator(tensor, shape, config):
        calls.append(shape)
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"workspace-{len(calls)}")

    cache = DequantizedWorkspaceCache(allocator=allocator)
    tensor = FakeTensor()
    config = CompressionConfig(group_size=64)

    first = cache.get("bucket0", tensor, (64,), config)
    cache.clear()
    second = cache.get("bucket0", tensor, (64,), config)

    assert first is not second
    assert len(calls) == 2


def test_workspace_cache_evicts_least_recently_used_entry_when_entry_limit_is_exceeded() -> None:
    calls = []

    def allocator(tensor, shape, config):
        calls.append((tensor.name, shape))
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"workspace-{len(calls)}")

    cache = DequantizedWorkspaceCache(allocator=allocator, max_entries=2)
    config = CompressionConfig(group_size=64)

    bucket0 = cache.get("bucket0", FakeTensor(name="bucket0"), (64,), config)
    bucket1 = cache.get("bucket1", FakeTensor(name="bucket1"), (64,), config)
    assert cache.get("bucket0", FakeTensor(name="bucket0"), (64,), config) is bucket0
    bucket2 = cache.get("bucket2", FakeTensor(name="bucket2"), (64,), config)
    bucket1_again = cache.get("bucket1", FakeTensor(name="bucket1"), (64,), config)

    assert bucket2.name == "workspace-3"
    assert bucket1_again.name == "workspace-4"
    assert bucket1_again is not bucket1
    assert len(calls) == 4


def test_workspace_cache_evicts_by_cached_byte_budget() -> None:
    calls = []

    def allocator(tensor, shape, config):
        calls.append(shape)
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"workspace-{len(calls)}")

    cache = DequantizedWorkspaceCache(allocator=allocator, max_cached_bytes=256)
    config = CompressionConfig(group_size=64)

    first = cache.get("bucket0", FakeTensor(dtype="float32"), (64,), config)
    second = cache.get("bucket1", FakeTensor(dtype="float32"), (64,), config)
    first_again = cache.get("bucket0", FakeTensor(dtype="float32"), (64,), config)

    assert first.name == "workspace-1"
    assert second.name == "workspace-2"
    assert first_again.name == "workspace-3"
    assert first_again is not first
    assert len(calls) == 3
