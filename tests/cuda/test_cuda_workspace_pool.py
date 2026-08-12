from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread
from time import sleep

import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.workspace import (
    CudaOutputLease,
    CudaShardWorkspaceProvider,
    CudaWorkspacePool,
    WorkspaceKey,
)


KEY = WorkspaceKey(
    backend="cuda",
    collective="reduce_scatter",
    strategy="compressed",
    shape_class=(1024,),
    dtype="float16",
    world_size=4,
    bit=8,
    group_size=128,
    chunk_config=(256, 4),
    workspace_kind="send",
    device="cuda:0",
)


class FakeBuffer:
    def __init__(self, name: str, nbytes: int) -> None:
        self.name = name
        self.nbytes = nbytes
        self.recorded_streams: list[object] = []

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)

    @property
    def shape(self):
        return (self.nbytes,)

    @property
    def dtype(self):
        return "uint8"

    @property
    def device(self):
        return "cuda:0"

    def numel(self) -> int:
        return self.nbytes


class FakeEvent:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready
        self.wait_calls = 0
        self.waited_streams: list[object] = []

    def query(self) -> bool:
        return self.ready

    def wait(self) -> None:
        self.wait_calls += 1

    def wait_stream(self, stream: object) -> None:
        self.wait_calls += 1
        self.waited_streams.append(stream)


def _pool(*, max_cached_bytes: int = 4096, max_entries: int | None = None):
    calls: list[tuple[WorkspaceKey, object]] = []

    def allocator(key: WorkspaceKey, stream: object) -> FakeBuffer:
        calls.append((key, stream))
        return FakeBuffer(f"buffer-{len(calls)}", key.estimated_bytes)

    return CudaWorkspacePool(
        allocator=allocator,
        max_cached_bytes=max_cached_bytes,
        max_entries=max_entries,
    ), calls


def test_workspace_key_captures_every_allocation_dimension() -> None:
    assert replace(KEY, backend="cpu") != KEY
    assert replace(KEY, collective="all_reduce") != KEY
    assert replace(KEY, strategy="all_gather") != KEY
    assert replace(KEY, shape_class=(2048,)) != KEY
    assert replace(KEY, dtype="float32") != KEY
    assert replace(KEY, world_size=2) != KEY
    assert replace(KEY, bit=4) != KEY
    assert replace(KEY, group_size=64) != KEY
    assert replace(KEY, chunk_config=(512, 2)) != KEY
    assert replace(KEY, workspace_kind="recv") != KEY
    assert replace(KEY, device="cuda:1") != KEY


def test_workspace_pool_types_are_exported_from_cuda_package() -> None:
    from ccdl_comm.cuda import (
        CudaWorkspacePool as ExportedPool,
        CudaOutputLease as ExportedOutputLease,
        WorkspaceKey as ExportedKey,
        WorkspaceLease,
        WorkspaceStats,
    )

    assert ExportedPool is CudaWorkspacePool
    assert ExportedOutputLease is CudaOutputLease
    assert ExportedKey is WorkspaceKey
    assert WorkspaceLease.__name__ == "WorkspaceLease"
    assert WorkspaceStats.__name__ == "WorkspaceStats"


class FakeCompletionManager:
    def __init__(self) -> None:
        self.recorded: list[object] = []

    def record_for(self, value: object, *, stream: object | None = None) -> FakeEvent:
        del stream
        self.recorded.append(value)
        return FakeEvent(ready=False)


class FakeCompletedWork:
    def query(self) -> bool:
        return True


def test_output_lease_tracks_owner_completion_and_stream_safe_reuse() -> None:
    pool, calls = _pool()
    owner = object()
    lease = CudaOutputLease(
        pool.acquire(replace(KEY, workspace_kind="reduced_output"), stream="acquire"),
        owner_token=owner,
        completion_manager=FakeCompletionManager(),
        acquisition_stream="acquire",
    )

    assert lease.mark_used(owner) is lease.buffer
    lease.bind_work(owner, FakeCompletedWork())
    lease.release_after(lease.buffer)

    assert len(calls) == 1
    assert pool.stats.in_flight_bytes == KEY.estimated_bytes
    second = pool.acquire(replace(KEY, workspace_kind="reduced_output"), stream="consumer")
    assert second.buffer is lease.buffer
    assert second.buffer.recorded_streams == ["acquire", "consumer"]


def test_output_lease_rejects_foreign_double_and_invalid_release_paths() -> None:
    pool, _calls = _pool()
    owner = object()
    lease = CudaOutputLease(
        pool.acquire(KEY, stream="s0"),
        owner_token=owner,
        completion_manager=FakeCompletionManager(),
        acquisition_stream="s0",
    )

    with pytest.raises(RuntimeError, match="different executor"):
        lease.mark_used(object())
    assert lease.mark_used(owner) is lease.buffer
    with pytest.raises(RuntimeError, match="after mark_used"):
        lease.release_unused()
    with pytest.raises(RuntimeError, match="already in use"):
        lease.mark_used(owner)
    lease.bind_work(owner, FakeCompletedWork())
    lease.release_after(FakeEvent(ready=True))
    with pytest.raises(RuntimeError, match="already released"):
        lease.release_after(FakeEvent(ready=True))
    with pytest.raises(RuntimeError, match="already released"):
        lease.mark_used(owner)


def test_output_lease_releases_unused_with_acquisition_stream_ordering() -> None:
    pool, _calls = _pool()
    manager = FakeCompletionManager()
    lease = CudaOutputLease(
        pool.acquire(KEY, stream="s0"),
        owner_token=object(),
        completion_manager=manager,
        acquisition_stream="s0",
    )

    lease.release_unused()

    assert manager.recorded == [lease.buffer]
    assert pool.stats.in_flight_bytes == KEY.estimated_bytes
    with pytest.raises(RuntimeError, match="already released"):
        lease.release_unused()


def test_output_lease_records_unused_storage_on_its_acquisition_stream() -> None:
    pool, _calls = _pool()
    records = []

    class StreamAwareManager:
        def record_for(self, value, *, stream=None):
            records.append((value, stream))
            return FakeEvent(ready=True)

    lease = CudaOutputLease(
        pool.acquire(KEY, stream="acquisition"),
        owner_token=object(),
        completion_manager=StreamAwareManager(),
        acquisition_stream="acquisition",
    )

    lease.release_unused()

    assert records == [(lease.buffer, "acquisition")]


def test_output_lease_rejects_missing_owner_token() -> None:
    pool, _calls = _pool()

    with pytest.raises(TypeError, match="owner_token must not be None"):
        CudaOutputLease(
            pool.acquire(KEY, stream="s0"),
            owner_token=None,
            completion_manager=FakeCompletionManager(),
            acquisition_stream="s0",
        )


def test_output_lease_propagates_stream_aware_completion_error() -> None:
    pool, _calls = _pool()

    class RejectingManager:
        def record_for(self, value, *, stream):
            raise TypeError("stream completion failed")

    lease = CudaOutputLease(
        pool.acquire(KEY, stream="s0"),
        owner_token=object(),
        completion_manager=RejectingManager(),
        acquisition_stream="s0",
    )

    with pytest.raises(TypeError, match="stream completion failed"):
        lease.release_unused()


def test_output_lease_release_obeys_max_entries_pool_budget() -> None:
    pool, calls = _pool(max_entries=1)

    class ReadyManager:
        def record_for(self, value, *, stream):
            return FakeEvent(ready=True)

    manager = ReadyManager()
    first = CudaOutputLease(
        pool.acquire(replace(KEY, workspace_kind="reduced_output_a"), stream="s0"),
        owner_token=object(),
        completion_manager=manager,
        acquisition_stream="s0",
    )
    second = CudaOutputLease(
        pool.acquire(replace(KEY, workspace_kind="reduced_output_b"), stream="s0"),
        owner_token=object(),
        completion_manager=manager,
        acquisition_stream="s0",
    )

    first.release_unused()
    second.release_unused()
    again = pool.acquire(replace(KEY, workspace_kind="reduced_output_a"), stream="s1")

    assert again.buffer is not first.buffer
    assert len(calls) == 3
    assert pool.stats.evictions == 1


def test_output_lease_rejects_release_during_submit_to_bind_interleaving() -> None:
    pool, _calls = _pool()
    owner = object()
    entered_submission = Event()
    finish_submission = Event()
    errors = []

    class CompletedWork:
        def query(self):
            return True

    lease = CudaOutputLease(
        pool.acquire(KEY, stream="s0"),
        owner_token=owner,
        completion_manager=FakeCompletionManager(),
        acquisition_stream="s0",
    )

    def submit() -> None:
        try:
            lease.mark_used(owner)
            entered_submission.set()
            assert finish_submission.wait(timeout=2)
            lease.bind_work(owner, CompletedWork())
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=submit)
    thread.start()
    assert entered_submission.wait(timeout=2)
    with pytest.raises(RuntimeError, match="bound to work"):
        lease.release_after(lease.buffer)
    finish_submission.set()
    thread.join(timeout=2)

    assert errors == []
    lease.release_after(lease.buffer)


def test_in_flight_workspace_is_not_reused() -> None:
    pool, calls = _pool()
    first = pool.acquire(KEY, stream="s0")
    second = pool.acquire(KEY, stream="s1")

    assert second.buffer is not first.buffer
    first.release(completion=FakeEvent(ready=True))
    third = pool.acquire(KEY, stream="s2")

    assert third.buffer is first.buffer
    assert len(calls) == 2


def test_pending_workspace_can_be_handed_off_with_stream_event_ordering() -> None:
    pool, calls = _pool()
    event = FakeEvent()
    first = pool.acquire(KEY, stream="s0")
    first.release(completion=event)

    second = pool.acquire(KEY, stream="s1")

    assert second.buffer is first.buffer
    assert event.wait_calls == 1
    assert event.waited_streams == ["s1"]
    assert len(calls) == 1
    assert pool.stats.cached_bytes == 0
    assert pool.stats.in_flight_bytes == KEY.estimated_bytes


def test_pool_records_every_stream_that_uses_a_buffer() -> None:
    pool, _calls = _pool()
    first = pool.acquire(KEY, stream="s0")
    first.release(completion=FakeEvent(ready=True))
    second = pool.acquire(KEY, stream="s1")

    assert second.buffer is first.buffer
    assert second.buffer.recorded_streams == ["s0", "s1"]


def test_pool_evicts_least_recently_used_idle_workspace_over_byte_budget() -> None:
    pool, calls = _pool(max_cached_bytes=KEY.estimated_bytes * 2)
    first = pool.acquire(KEY, stream="s0")
    first.release(completion=FakeEvent(ready=True))
    second_key = replace(KEY, workspace_kind="recv")
    second = pool.acquire(second_key, stream="s0")
    second.release(completion=FakeEvent(ready=True))

    reused = pool.acquire(KEY, stream="s1")
    reused.release(completion=FakeEvent(ready=True))
    third_key = replace(KEY, workspace_kind="reduced")
    third = pool.acquire(third_key, stream="s0")
    third.release(completion=FakeEvent(ready=True))

    second_again = pool.acquire(second_key, stream="s2")

    assert second_again.buffer is not second.buffer
    assert len(calls) == 4
    assert pool.stats.evictions == 1
    assert pool.stats.cached_bytes == KEY.estimated_bytes * 2


def test_repeated_bucket_has_zero_steady_state_allocations() -> None:
    pool, calls = _pool()

    for index in range(100):
        lease = pool.acquire(KEY, stream=f"s{index % 2}")
        lease.release(completion=FakeEvent(ready=True))

    assert len(calls) == 1
    assert pool.stats.hits == 99
    assert pool.stats.misses == 1


def test_lease_cannot_be_released_twice() -> None:
    pool, _calls = _pool()
    lease = pool.acquire(KEY, stream="s0")
    lease.release(completion=FakeEvent(ready=True))

    with pytest.raises(RuntimeError, match="already released"):
        lease.release(completion=FakeEvent(ready=True))


def test_workspace_lease_exposes_public_read_only_released_state() -> None:
    pool, _calls = _pool()
    lease = pool.acquire(KEY, stream="s0")

    assert lease.released is False
    with pytest.raises(AttributeError):
        lease.released = True  # type: ignore[misc]

    lease.release(completion=FakeEvent(ready=True))

    assert lease.released is True


def test_concurrent_double_release_enters_pool_only_once() -> None:
    pool, _calls = _pool()
    lease = pool.acquire(KEY, stream="s0")
    original_release = pool._release
    entered = Event()
    continue_release = Event()
    release_calls = []
    errors = []

    def racing_release(record, completion):
        release_calls.append(record.identifier)
        entered.set()
        assert continue_release.wait(timeout=2)
        original_release(record, completion)

    pool._release = racing_release

    def release() -> None:
        try:
            lease.release(completion=FakeEvent(ready=True))
        except RuntimeError as exc:
            errors.append(exc)

    first, second = Thread(target=release), Thread(target=release)
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    sleep(0.05)
    assert len(release_calls) == 1
    continue_release.set()
    threads = [first, second]
    for thread in threads:
        thread.join(timeout=3)

    assert len(release_calls) == 1
    assert len(errors) == 1
    assert "already released" in str(errors[0])


def test_zero_budget_releases_workspace_without_caching_it() -> None:
    pool, calls = _pool(max_cached_bytes=0)
    first = pool.acquire(KEY, stream="s0")
    first.release(completion=FakeEvent(ready=True))
    second = pool.acquire(KEY, stream="s1")

    assert second.buffer is not first.buffer
    assert len(calls) == 2
    assert pool.stats.cached_bytes == 0
    assert pool.stats.evictions == 1


def test_real_cuda_workspace_waits_for_recorded_event_before_reuse() -> None:
    from ccdl_comm.communication.cuda_completion import CudaCompletion

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    key = replace(KEY, shape_class=(256,), device="cuda:0")
    pool = CudaWorkspacePool(
        allocator=lambda active_key, stream: torch.empty(
            active_key.shape_class,
            dtype=torch.float16,
            device=active_key.device,
        ),
        max_cached_bytes=4096,
    )
    stream = torch.cuda.Stream(device=key.device)
    with torch.cuda.stream(stream):
        first = pool.acquire(key, stream)
        torch.cuda._sleep(10_000_000)
        event = torch.cuda.Event()
        event.record(stream)
    first.release(completion=CudaCompletion(event))

    second = pool.acquire(key, torch.cuda.current_stream())

    assert second.buffer.data_ptr() == first.buffer.data_ptr()


def test_shard_workspace_provider_reuses_send_recv_and_reduced_buffers() -> None:
    pool, calls = _pool(max_cached_bytes=16384)
    provider = CudaShardWorkspaceProvider(
        pool,
        backend="cuda",
        collective="reduce_scatter",
        strategy="compressed",
        device="cuda:0",
    )
    config = CompressionConfig(bit=8, group_size=64)

    first = provider.begin(stream="s0")
    send = first.get_quantized_chunk(
        "bucket",
        0,
        FakeBuffer("chunk", 1024),
        config,
        dtype="fp16",
        world_size=2,
    )
    recv = first.get_received_payload(
        "bucket",
        send,
        0,
        world_size=2,
        config=config,
    )
    direct_recv = first.get_received_tensor_payload(
        "bucket",
        1,
        FakeBuffer("tensor", 2048),
        config,
        dtype="fp16",
        world_size=2,
    )
    reduced = first.get_reduced_shard(
        "bucket",
        FakeBuffer("tensor", 2048),
        (512,),
        config,
        dtype="fp16",
        world_size=2,
        rank=0,
    )
    first.release(completion=FakeEvent(ready=True))

    second = provider.begin(stream="s1")
    assert second.get_quantized_chunk(
        "bucket", 0, FakeBuffer("chunk", 1024), config, dtype="fp16", world_size=2
    ) is send
    assert second.get_received_payload(
        "bucket", send, 0, world_size=2, config=config
    ) is recv
    assert second.get_received_tensor_payload(
        "bucket",
        1,
        FakeBuffer("tensor", 2048),
        config,
        dtype="fp16",
        world_size=2,
    ) is direct_recv
    assert second.get_reduced_shard(
        "bucket",
        FakeBuffer("tensor", 2048),
        (512,),
        config,
        dtype="fp16",
        world_size=2,
        rank=0,
    ) is reduced
    assert len(calls) == 4
    assert pool.stats.hits == 4
    assert len(second.leases) == 4


def test_shard_workspace_provider_keys_and_reuses_full_output_buffer() -> None:
    pool, calls = _pool(max_cached_bytes=16384)
    provider = CudaShardWorkspaceProvider(
        pool,
        backend="cuda",
        collective="all_reduce",
        strategy="compressed_reduce_scatter",
        device="cuda:0",
    )
    config = CompressionConfig(bit=8, group_size=64)
    shard = FakeBuffer("shard", 1024)

    first = provider.begin(stream="s0")
    output = first.get_full_output(
        "bucket",
        shard,
        config,
        dtype="fp16",
        world_size=4,
    )
    first.release(completion=FakeEvent(ready=True))

    second = provider.begin(stream="s1")
    reused = second.get_full_output(
        "bucket",
        shard,
        config,
        dtype="fp16",
        world_size=4,
    )

    assert reused is output
    assert calls[0][0].workspace_kind == "full_output"
    assert calls[0][0].shape_class == (4096,)
    assert calls[0][0].chunk_config == (1024, 4096)
    assert len(calls) == 1
    assert pool.stats.hits == 1
    assert len(second.leases) == 1


def test_shard_workspace_provider_can_leave_returned_output_unpooled() -> None:
    pool, calls = _pool(max_cached_bytes=16384)
    provider = CudaShardWorkspaceProvider(
        pool,
        backend="cuda",
        collective="reduce_scatter",
        strategy="compressed",
        device="cuda:0",
        pool_reduced_output=False,
    )
    session = provider.begin(stream="s0")

    output = session.get_reduced_shard(
        "bucket",
        FakeBuffer("tensor", 2048),
        (512,),
        CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        world_size=2,
        rank=0,
    )

    assert output is None
    assert session.leases == ()
    assert calls == []
