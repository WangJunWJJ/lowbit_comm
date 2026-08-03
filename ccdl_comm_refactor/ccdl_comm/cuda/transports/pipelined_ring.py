"""Immutable schedules and async submission for pipelined ring reduce-scatter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .compressed_reduce_scatter import ChunkPlan, ChunkRange


class _WorkspaceSession(Protocol):
    def release(self, *, completion: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class RingReduceScatterStep:
    """One precomputed point-to-point exchange in a ring reduce-scatter."""

    step_index: int
    send_peer: int
    recv_peer: int
    send_chunk_owner: int
    recv_chunk_owner: int
    received_contributors: tuple[int, ...]
    send_chunk: ChunkRange
    recv_chunk: ChunkRange

    def __post_init__(self) -> None:
        for name in (
            "step_index",
            "send_peer",
            "recv_peer",
            "send_chunk_owner",
            "recv_chunk_owner",
        ):
            _require_nonnegative_integer(getattr(self, name), name)
        if not isinstance(self.received_contributors, tuple):
            raise TypeError("received_contributors must be a tuple")
        if not self.received_contributors:
            raise ValueError("received_contributors must not be empty")
        for contributor in self.received_contributors:
            _require_nonnegative_integer(contributor, "received_contributors entry")
        if len(set(self.received_contributors)) != len(self.received_contributors):
            raise ValueError("received_contributors must not contain duplicates")
        if len(self.received_contributors) != self.step_index + 1:
            raise ValueError("received_contributors length must equal step_index + 1")
        for name in ("send_chunk", "recv_chunk"):
            if not isinstance(getattr(self, name), ChunkRange):
                raise TypeError(f"{name} must be a ChunkRange")


@dataclass(frozen=True, slots=True)
class PipelinedRingSchedule:
    """Rank-local ring metadata pre-bound to an immutable :class:`ChunkPlan`."""

    chunk_plan: ChunkPlan
    rank: int
    steps: tuple[RingReduceScatterStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_plan, ChunkPlan):
            raise TypeError("chunk_plan must be a ChunkPlan")
        _require_rank(self.rank, self.chunk_plan.world_size, "rank")
        expected_steps = _ring_steps(self.chunk_plan, self.rank)
        if self.steps != expected_steps:
            raise ValueError("steps must match the deterministic ring schedule")

    @property
    def local_chunk(self) -> ChunkRange:
        """The shard this rank owns after reduce-scatter completes."""

        return self.chunk_plan.chunk_for_rank(self.rank)


class PipelinedRingRuntime(Protocol):
    """Pre-bound coarse operations required by :class:`PipelinedRingExecutor`."""

    def wait_for_producer(self, tensor: Any) -> None: ...

    def quant_pack(
        self,
        tensor: Any,
        chunk: ChunkRange,
        workspace: _WorkspaceSession,
    ) -> Any: ...

    def send_recv(
        self,
        payload: Any,
        *,
        send_peer: int,
        recv_peer: int,
        recv_chunk: ChunkRange,
        workspace: _WorkspaceSession,
    ) -> tuple[Any, Any]: ...

    def fused_reduce(
        self,
        tensor: Any,
        received: Any,
        chunk: ChunkRange,
        contributors: tuple[int, ...],
        workspace: _WorkspaceSession,
    ) -> None: ...

    def record_completion(self) -> Any: ...


class _CompletionManager(Protocol):
    def create_work(
        self,
        *,
        result: Any,
        completion: Any,
        resources: tuple[Any, ...],
    ) -> Any: ...


class PipelinedRingExecutor:
    """Submit a compiled ring schedule without CPU synchronization or planning."""

    __slots__ = (
        "schedule",
        "_runtime",
        "_workspace_session_factory",
        "_completion_manager",
    )

    def __init__(
        self,
        *,
        schedule: PipelinedRingSchedule,
        runtime: PipelinedRingRuntime,
        workspace_session_factory: Callable[[Any], _WorkspaceSession],
        completion_manager: _CompletionManager,
    ) -> None:
        if not isinstance(schedule, PipelinedRingSchedule):
            raise TypeError("schedule must be a PipelinedRingSchedule")
        if not callable(workspace_session_factory):
            raise TypeError("workspace_session_factory must be callable")
        self.schedule = schedule
        self._runtime = runtime
        self._workspace_session_factory = workspace_session_factory
        self._completion_manager = completion_manager

    def run(self, tensor: Any) -> Any:
        """Enqueue the fixed schedule and return completion-backed Work."""

        workspace = self._workspace_session_factory(tensor)
        resources: list[Any] = [tensor, workspace]
        try:
            self._runtime.wait_for_producer(tensor)
            for step in self.schedule.steps:
                payload = self._runtime.quant_pack(tensor, step.send_chunk, workspace)
                received, handle = self._runtime.send_recv(
                    payload,
                    send_peer=step.send_peer,
                    recv_peer=step.recv_peer,
                    recv_chunk=step.recv_chunk,
                    workspace=workspace,
                )
                resources.extend((payload, received))
                if handle is not None:
                    resources.append(handle)
                self._runtime.fused_reduce(
                    tensor,
                    received,
                    step.recv_chunk,
                    step.received_contributors,
                    workspace,
                )
            completion = self._runtime.record_completion()
        except BaseException:
            _abort_workspace(workspace, self._runtime)
            raise

        _release_workspace(workspace, completion)
        return self._completion_manager.create_work(
            result=tensor,
            completion=completion,
            resources=tuple(resources),
        )


def compile_pipelined_ring_schedule(*, chunk_plan: ChunkPlan, rank: int) -> PipelinedRingSchedule:
    """Compile deterministic rank-local ring steps without runtime planning."""

    if not isinstance(chunk_plan, ChunkPlan):
        raise TypeError("chunk_plan must be a ChunkPlan")
    _require_rank(rank, chunk_plan.world_size, "rank")
    return PipelinedRingSchedule(
        chunk_plan=chunk_plan,
        rank=rank,
        steps=_ring_steps(chunk_plan, rank),
    )


def _ring_steps(chunk_plan: ChunkPlan, rank: int) -> tuple[RingReduceScatterStep, ...]:
    world_size = chunk_plan.world_size
    return tuple(
        RingReduceScatterStep(
            step_index=step_index,
            send_peer=(rank + 1) % world_size,
            recv_peer=(rank - 1) % world_size,
            send_chunk_owner=(rank - step_index - 1) % world_size,
            recv_chunk_owner=(rank - step_index - 2) % world_size,
            received_contributors=tuple(
                (rank - step_index - 1 + offset) % world_size
                for offset in range(step_index + 1)
            ),
            send_chunk=chunk_plan.chunk_for_rank((rank - step_index - 1) % world_size),
            recv_chunk=chunk_plan.chunk_for_rank((rank - step_index - 2) % world_size),
        )
        for step_index in range(world_size - 1)
    )


def _release_workspace(workspace: _WorkspaceSession, completion: Any) -> None:
    release = getattr(workspace, "release", None)
    if not callable(release):
        raise TypeError("workspace session must provide release(completion=...)")
    release(completion=completion)


def _abort_workspace(workspace: _WorkspaceSession, runtime: PipelinedRingRuntime) -> None:
    abort = getattr(workspace, "abort", None)
    if callable(abort):
        abort()
        return
    completion = runtime.record_completion()
    _release_workspace(workspace, completion)


def _require_rank(value: object, world_size: int, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value >= world_size:
        raise ValueError(f"{name} must be in [0, {world_size})")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
