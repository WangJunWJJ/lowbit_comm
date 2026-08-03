"""Stream-safe reusable workspaces for CUDA communication executors."""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import mul
from threading import Lock
from typing import Any
from importlib import import_module

from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.sizing import estimate_quantized_size


@dataclass(frozen=True)
class WorkspaceKey:
    """All dimensions that can change a communication workspace allocation."""

    backend: str
    collective: str
    strategy: str
    shape_class: tuple[int, ...]
    dtype: str
    world_size: int
    bit: int
    group_size: int
    chunk_config: tuple[int, ...]
    workspace_kind: str
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must not be empty")
        if not self.collective:
            raise ValueError("collective must not be empty")
        if not self.strategy:
            raise ValueError("strategy must not be empty")
        if any(dimension < 0 for dimension in self.shape_class):
            raise ValueError("shape_class dimensions must be >= 0")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.bit < 1:
            raise ValueError("bit must be >= 1")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if not self.workspace_kind:
            raise ValueError("workspace_kind must not be empty")

    @property
    def estimated_bytes(self) -> int:
        """Estimate allocation bytes when an allocator cannot report its size."""

        return reduce(mul, self.shape_class, 1) * _dtype_size_bytes(self.dtype)


@dataclass(frozen=True)
class WorkspaceStats:
    hits: int
    misses: int
    evictions: int
    cached_bytes: int
    in_flight_bytes: int


@dataclass
class _WorkspaceRecord:
    identifier: int
    key: WorkspaceKey
    buffer: Any
    size_bytes: int
    completion: Any | None = None


class WorkspaceLease:
    """Exclusive ownership of one pooled workspace until explicit release."""

    def __init__(self, pool: CudaWorkspacePool, record: _WorkspaceRecord) -> None:
        self._pool = pool
        self._record = record
        self._released = False
        self._release_lock = Lock()

    @property
    def buffer(self) -> Any:
        return self._record.buffer

    @property
    def key(self) -> WorkspaceKey:
        return self._record.key

    @property
    def size_bytes(self) -> int:
        return self._record.size_bytes

    @property
    def released(self) -> bool:
        """Whether this lease has been successfully returned to its pool."""

        with self._release_lock:
            return self._released

    def release(self, *, completion: Any) -> None:
        with self._release_lock:
            if self._released:
                raise RuntimeError("workspace lease is already released")
            self._released = True
            try:
                self._pool._release(self._record, completion)
            except BaseException:
                self._released = False
                raise


class CudaOutputLease:
    """Explicit ownership of one pooled ReducedShard output buffer.

    The transport only receives :attr:`buffer`; executor identity and release
    policy remain at the CUDA executor boundary.
    """

    _ACQUIRED = "ACQUIRED"
    _SUBMITTING = "SUBMITTING"
    _BOUND = "BOUND"
    _RELEASED = "RELEASED"

    def __init__(
        self,
        lease: WorkspaceLease,
        *,
        owner_token: object,
        completion_manager: Any,
        acquisition_stream: Any,
    ) -> None:
        if not isinstance(lease, WorkspaceLease):
            raise TypeError("lease must be a WorkspaceLease")
        if owner_token is None:
            raise TypeError("owner_token must not be None")
        if completion_manager is None:
            raise TypeError("completion_manager must not be None")
        self._lease = lease
        self._owner_token = owner_token
        self._completion_manager = completion_manager
        self._acquisition_stream = acquisition_stream
        self._state = self._ACQUIRED
        self._work: Any | None = None
        self._lock = Lock()

    @property
    def buffer(self) -> Any:
        """Return the caller-visible output storage."""

        return self._lease.buffer

    def mark_used(self, owner_token: object) -> Any:
        """Reserve this output for exactly one run by its owning executor."""

        with self._lock:
            if self._state == self._RELEASED:
                raise RuntimeError("CUDA output lease is already released")
            if owner_token is not self._owner_token:
                raise RuntimeError("CUDA output lease belongs to a different executor")
            if self._state != self._ACQUIRED:
                raise RuntimeError("CUDA output lease is already in use")
            self._state = self._SUBMITTING
            return self._lease.buffer

    def bind_work(self, owner_token: object, work: Any) -> None:
        """Attach the one Work whose completion authorizes output release."""

        with self._lock:
            if self._state == self._RELEASED:
                raise RuntimeError("CUDA output lease is already released")
            if owner_token is not self._owner_token:
                raise RuntimeError("CUDA output lease belongs to a different executor")
            if self._state != self._SUBMITTING:
                raise RuntimeError("CUDA output lease must be submitting before binding work")
            self._work = work
            self._state = self._BOUND

    def abort_use(self, owner_token: object) -> None:
        """Undo a failed executor submission before a Work has been returned."""

        with self._lock:
            if self._state == self._RELEASED:
                raise RuntimeError("CUDA output lease is already released")
            if owner_token is not self._owner_token:
                raise RuntimeError("CUDA output lease belongs to a different executor")
            if self._state != self._SUBMITTING:
                raise RuntimeError("CUDA output lease can abort only while submitting")
            self._state = self._ACQUIRED

    def release_after(self, value_or_completion: Any) -> None:
        """Return output storage after a tensor event or supplied completion."""

        with self._lock:
            if self._state == self._RELEASED:
                raise RuntimeError("CUDA output lease is already released")
            if self._state == self._ACQUIRED:
                raise RuntimeError("CUDA output lease must be marked used before release_after")
            if self._state == self._SUBMITTING or self._work is None:
                raise RuntimeError("CUDA output lease must be bound to work before release_after")
            if not _work_completed(self._work):
                raise RuntimeError("CUDA output lease cannot release until associated work completes")
            completion = _as_completion(self._completion_manager, value_or_completion)
            self._release_locked(completion)

    def release_unused(self) -> None:
        """Return untouched storage while retaining acquisition-stream ordering."""

        with self._lock:
            if self._state == self._RELEASED:
                raise RuntimeError("CUDA output lease is already released")
            if self._state != self._ACQUIRED:
                raise RuntimeError("CUDA output lease cannot release_unused after mark_used")
            completion = _record_completion(
                self._completion_manager,
                self._lease.buffer,
                stream=self._acquisition_stream,
            )
            self._release_locked(completion)

    def _release_locked(self, completion: Any) -> None:
        previous_state = self._state
        self._state = self._RELEASED
        try:
            self._lease.release(completion=completion)
            self._work = None
        except BaseException:
            self._state = previous_state
            raise


def _as_completion(completion_manager: Any, value_or_completion: Any) -> Any:
    query = getattr(value_or_completion, "query", None)
    if callable(query):
        return value_or_completion
    return _record_completion(completion_manager, value_or_completion)


def _work_completed(work: Any) -> bool:
    query = getattr(work, "query", None)
    if callable(query):
        return bool(query())
    is_completed = getattr(work, "is_completed", None)
    if callable(is_completed):
        return bool(is_completed())
    return False


def _record_completion(completion_manager: Any, value: Any, *, stream: Any = None) -> Any:
    record_for = getattr(completion_manager, "record_for", None)
    if not callable(record_for):
        raise TypeError("completion_manager must provide record_for()")
    if stream is None:
        completion = record_for(value)
    else:
        completion = record_for(value, stream=stream)
    if not callable(getattr(completion, "query", None)):
        raise TypeError("completion_manager.record_for() must return completion with query()")
    return completion


class CudaWorkspacePool:
    """LRU workspace pool with nonblocking CUDA completion ownership."""

    def __init__(
        self,
        *,
        allocator: Callable[[WorkspaceKey, Any], Any],
        max_cached_bytes: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        if not callable(allocator):
            raise TypeError("allocator must be callable")
        if max_cached_bytes is not None and max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be >= 0 or None")
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 or None")
        self._allocator = allocator
        self._max_cached_bytes = max_cached_bytes
        self._max_entries = max_entries
        self._idle: OrderedDict[int, _WorkspaceRecord] = OrderedDict()
        self._idle_by_key: dict[WorkspaceKey, deque[int]] = defaultdict(deque)
        self._in_flight: dict[int, _WorkspaceRecord] = {}
        self._next_identifier = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._cached_bytes = 0
        self._in_flight_bytes = 0
        self._lock = Lock()

    def acquire(self, key: WorkspaceKey, stream: Any) -> WorkspaceLease:
        if not isinstance(key, WorkspaceKey):
            raise TypeError("key must be a WorkspaceKey")
        with self._lock:
            self._reap_ready_locked()
            record = self._take_idle_locked(key)
            handed_off = False
            if record is None:
                record = self._take_pending_locked(key, stream)
                handed_off = record is not None
            if record is None:
                buffer = self._allocator(key, stream)
                record = _WorkspaceRecord(
                    identifier=self._next_identifier,
                    key=key,
                    buffer=buffer,
                    size_bytes=_buffer_nbytes(buffer, key.estimated_bytes),
                )
                self._next_identifier += 1
                self._misses += 1
            else:
                self._hits += 1
            _record_stream(record.buffer, stream)
            if not handed_off:
                self._in_flight[record.identifier] = record
                self._in_flight_bytes += record.size_bytes
            return WorkspaceLease(self, record)

    @property
    def stats(self) -> WorkspaceStats:
        with self._lock:
            self._reap_ready_locked()
            return WorkspaceStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                cached_bytes=self._cached_bytes,
                in_flight_bytes=self._in_flight_bytes,
            )

    def clear(self) -> None:
        """Drop idle workspaces while preserving buffers owned by active work."""

        with self._lock:
            self._idle.clear()
            self._idle_by_key.clear()
            self._cached_bytes = 0

    def _release(self, record: _WorkspaceRecord, completion: Any) -> None:
        query = getattr(completion, "query", None)
        if not callable(query):
            raise TypeError("completion must provide query()")
        with self._lock:
            if record.identifier not in self._in_flight:
                raise RuntimeError("workspace lease is not owned by this pool")
            record.completion = completion
            if bool(query()):
                self._make_idle_locked(record)

    def _reap_ready_locked(self) -> None:
        for record in tuple(self._in_flight.values()):
            completion = record.completion
            if completion is None:
                continue
            query = getattr(completion, "query", None)
            if callable(query) and bool(query()):
                self._make_idle_locked(record)

    def _make_idle_locked(self, record: _WorkspaceRecord) -> None:
        if self._in_flight.pop(record.identifier, None) is None:
            return
        self._in_flight_bytes -= record.size_bytes
        record.completion = None
        self._idle[record.identifier] = record
        self._idle_by_key[record.key].append(record.identifier)
        self._cached_bytes += record.size_bytes
        self._evict_over_budget_locked()

    def _take_idle_locked(self, key: WorkspaceKey) -> _WorkspaceRecord | None:
        identifiers = self._idle_by_key.get(key)
        if identifiers is None:
            return None
        while identifiers:
            identifier = identifiers.popleft()
            record = self._idle.pop(identifier, None)
            if record is not None:
                self._cached_bytes -= record.size_bytes
                if not identifiers:
                    self._idle_by_key.pop(key, None)
                return record
        self._idle_by_key.pop(key, None)
        return None

    def _take_pending_locked(self, key: WorkspaceKey, stream: Any) -> _WorkspaceRecord | None:
        for record in self._in_flight.values():
            if record.key != key or record.completion is None:
                continue
            wait_stream = getattr(record.completion, "wait_stream", None)
            if not callable(wait_stream):
                continue
            wait_stream(stream)
            record.completion = None
            return record
        return None

    def _evict_over_budget_locked(self) -> None:
        while self._idle and (
            (self._max_entries is not None and len(self._idle) > self._max_entries)
            or (self._max_cached_bytes is not None and self._cached_bytes > self._max_cached_bytes)
        ):
            identifier, record = self._idle.popitem(last=False)
            self._cached_bytes -= record.size_bytes
            self._evictions += 1
            identifiers = self._idle_by_key[record.key]
            try:
                identifiers.remove(identifier)
            except ValueError:
                pass
            if not identifiers:
                self._idle_by_key.pop(record.key, None)


class CudaShardWorkspaceProvider:
    """Bind shard-workspace keys to one executor-owned CUDA pool."""

    def __init__(
        self,
        pool: CudaWorkspacePool,
        *,
        backend: str,
        collective: str,
        strategy: str,
        device: str,
        pool_reduced_output: bool = True,
    ) -> None:
        if not isinstance(pool, CudaWorkspacePool):
            raise TypeError("pool must be a CudaWorkspacePool")
        self.pool = pool
        self._backend = backend
        self._collective = collective
        self._strategy = strategy
        self._device = device
        self.pool_reduced_output = bool(pool_reduced_output)

    def begin(self, *, stream: Any) -> CudaShardWorkspaceSession:
        return CudaShardWorkspaceSession(self, stream=stream)


def create_torch_workspace_pool(
    *,
    max_cached_bytes: int | None = None,
    max_entries: int | None = None,
) -> CudaWorkspacePool:
    """Create a pool whose misses allocate torch tensors lazily."""

    return CudaWorkspacePool(
        allocator=_allocate_torch_buffer,
        max_cached_bytes=max_cached_bytes,
        max_entries=max_entries,
    )


class CudaShardWorkspaceSession:
    """Exclusive send/recv/reduced leases for one shard collective call."""

    def __init__(self, provider: CudaShardWorkspaceProvider, *, stream: Any) -> None:
        self._provider = provider
        self._stream = stream
        self._leases: list[WorkspaceLease] = []

    @property
    def leases(self) -> tuple[WorkspaceLease, ...]:
        return tuple(self._leases)

    def get_quantized_chunk(
        self,
        bucket_key: Any,
        chunk_index: int,
        tensor: Any,
        config: CompressionConfig,
        *,
        dtype: str,
        world_size: int,
    ) -> Any:
        del bucket_key
        estimate = estimate_quantized_size(int(tensor.numel()), dtype=dtype, config=config)
        key = self._key(
            shape=(estimate.quantized_bytes,),
            dtype="uint8",
            world_size=world_size,
            config=config,
            chunk_config=(chunk_index, int(tensor.numel())),
            kind="send",
        )
        return self._acquire(key)

    def get_received_payload(
        self,
        bucket_key: Any,
        payload_template: Any,
        index: int,
        *,
        world_size: int,
        config: CompressionConfig,
    ) -> Any:
        del bucket_key
        key = self._key(
            shape=tuple(payload_template.shape),
            dtype=str(payload_template.dtype),
            world_size=world_size,
            config=config,
            chunk_config=(index,),
            kind="recv",
        )
        return self._acquire(key)

    def get_reduced_shard(
        self,
        bucket_key: Any,
        tensor: Any,
        shape: tuple[int, ...],
        config: CompressionConfig,
        *,
        dtype: str,
        world_size: int,
        rank: int,
    ) -> Any:
        del bucket_key, dtype
        if not self._provider.pool_reduced_output:
            return None
        padded_numel = _padded_numel(shape, config.group_size)
        key = self._key(
            shape=(padded_numel,),
            dtype=str(tensor.dtype),
            world_size=world_size,
            config=config,
            chunk_config=(rank,),
            kind="reduced",
        )
        return self._acquire(key)

    def release(self, *, completion: Any) -> None:
        while self._leases:
            lease = self._leases.pop(0)
            lease.release(completion=completion)

    def _acquire(self, key: WorkspaceKey) -> Any:
        lease = self._provider.pool.acquire(key, self._stream)
        self._leases.append(lease)
        return lease.buffer

    def _key(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str,
        world_size: int,
        config: CompressionConfig,
        chunk_config: tuple[int, ...],
        kind: str,
    ) -> WorkspaceKey:
        return WorkspaceKey(
            backend=self._provider._backend,
            collective=self._provider._collective,
            strategy=self._provider._strategy,
            shape_class=shape,
            dtype=dtype,
            world_size=world_size,
            bit=config.bit,
            group_size=config.group_size,
            chunk_config=chunk_config,
            workspace_kind=kind,
            device=self._provider._device,
        )


def _record_stream(buffer: Any, stream: Any) -> None:
    record_stream = getattr(buffer, "record_stream", None)
    if callable(record_stream) and stream is not None:
        record_stream(stream)


def _allocate_torch_buffer(key: WorkspaceKey, stream: Any) -> Any:
    del stream
    torch = import_module("torch")
    dtype_name = key.dtype.lower().removeprefix("torch.")
    dtype = {
        "fp16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float": torch.float32,
        "float32": torch.float32,
        "uint8": torch.uint8,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported workspace dtype: {key.dtype!r}")
    return torch.empty(key.shape_class, dtype=dtype, device=key.device)


def _padded_numel(shape: tuple[int, ...], group_size: int) -> int:
    numel = reduce(mul, shape, 1)
    if numel == 0:
        return 0
    return ((numel + group_size - 1) // group_size) * group_size


def _buffer_nbytes(buffer: Any, fallback: int) -> int:
    nbytes = getattr(buffer, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(buffer, "numel", None)
    element_size = getattr(buffer, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return fallback


def _dtype_size_bytes(dtype: str) -> int:
    normalized = dtype.lower().removeprefix("torch.")
    if normalized in {"float64", "double", "int64"}:
        return 8
    if normalized in {"float32", "float", "int32"}:
        return 4
    if normalized in {"float16", "half", "fp16", "bfloat16", "bf16", "int16"}:
        return 2
    return 1
