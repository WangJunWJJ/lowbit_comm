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
