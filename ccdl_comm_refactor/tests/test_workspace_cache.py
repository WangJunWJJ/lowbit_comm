from ccdl_comm.config import CompressionConfig
from ccdl_comm.communication.workspace import DequantizedWorkspaceCache, ShardCommunicationWorkspaceCache


class FakeTensor:
    def __init__(self, *, dtype="float32", device="cuda:0", name="tensor", shape=(4,)):
        self.dtype = dtype
        self.device = device
        self.name = name
        self.shape = shape

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def new_empty(self, shape, dtype=None):
        return FakeTensor(
            dtype=dtype or self.dtype,
            device=self.device,
            name=f"{self.name}-empty-{shape}",
            shape=tuple(shape),
        )


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


def test_shard_workspace_cache_reuses_all_workspace_kinds_for_matching_bucket() -> None:
    calls = []

    def quantized_allocator(tensor, config, dtype):
        calls.append(("send", tensor.name, config.bit, config.group_size, dtype))
        return FakeTensor(dtype="uint8", device=tensor.device, name=f"send-{len(calls)}", shape=(8,))

    def reduced_allocator(tensor, shape, config):
        calls.append(("reduced", tensor.name, shape, config.bit, config.group_size))
        return FakeTensor(dtype=tensor.dtype, device=tensor.device, name=f"reduced-{len(calls)}", shape=shape)

    cache = ShardCommunicationWorkspaceCache(
        quantized_allocator=quantized_allocator,
        reduced_allocator=reduced_allocator,
    )
    tensor = FakeTensor(name="bucket")
    chunk = FakeTensor(name="chunk", shape=(2,))
    config = CompressionConfig(bit=8, group_size=64)

    send0 = cache.get_quantized_chunk("bucket0", 0, chunk, config, dtype="fp32", world_size=2)
    send0_again = cache.get_quantized_chunk("bucket0", 0, chunk, config, dtype="fp32", world_size=2)
    recv0 = cache.get_received_payload("bucket0", send0, 0, world_size=2, config=config)
    recv0_again = cache.get_received_payload("bucket0", send0, 0, world_size=2, config=config)
    reduced = cache.get_reduced_shard("bucket0", tensor, (2,), config, dtype="fp32", world_size=2, rank=0)
    reduced_again = cache.get_reduced_shard("bucket0", tensor, (2,), config, dtype="fp32", world_size=2, rank=0)

    assert send0 is send0_again
    assert recv0 is recv0_again
    assert reduced is reduced_again
    assert [call[0] for call in calls] == ["send", "reduced"]


def test_shard_workspace_cache_reallocates_when_compression_config_changes() -> None:
    calls = []

    def quantized_allocator(tensor, config, dtype):
        calls.append((config.bit, config.group_size, dtype))
        return FakeTensor(dtype="uint8", device=tensor.device, name=f"send-{len(calls)}", shape=(8,))

    cache = ShardCommunicationWorkspaceCache(quantized_allocator=quantized_allocator)
    chunk = FakeTensor(name="chunk", shape=(2,))

    first = cache.get_quantized_chunk("bucket0", 0, chunk, CompressionConfig(bit=8, group_size=64), dtype="fp32", world_size=2)
    second = cache.get_quantized_chunk("bucket0", 0, chunk, CompressionConfig(bit=8, group_size=32), dtype="fp32", world_size=2)

    assert first is not second
    assert calls == [(8, 64, "fp32"), (8, 32, "fp32")]


def test_shard_workspace_cache_clear_drops_all_workspace_kinds() -> None:
    calls = []

    def quantized_allocator(tensor, config, dtype):
        calls.append("send")
        return FakeTensor(dtype="uint8", device=tensor.device, name=f"send-{len(calls)}", shape=(8,))

    cache = ShardCommunicationWorkspaceCache(quantized_allocator=quantized_allocator)
    chunk = FakeTensor(name="chunk", shape=(2,))
    config = CompressionConfig(bit=8, group_size=64)

    first = cache.get_quantized_chunk("bucket0", 0, chunk, config, dtype="fp32", world_size=2)
    cache.clear()
    second = cache.get_quantized_chunk("bucket0", 0, chunk, config, dtype="fp32", world_size=2)

    assert first is not second
    assert calls == ["send", "send"]
