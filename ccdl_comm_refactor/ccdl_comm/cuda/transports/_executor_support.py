"""Shared ownership support for async topology submissions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias


class WorkspaceSession(Protocol):
    def release(self, *, completion: Any) -> None: ...


class CompletionManager(Protocol):
    def create_work(
        self,
        *,
        result: Any,
        completion: Any,
        resources: tuple[Any, ...],
    ) -> Any: ...


class SubmissionContext(Protocol):
    """Nonblocking readiness for every operation submitted through this context."""

    def query(self) -> bool: ...

    def wait(self) -> None: ...

    def wait_stream(self, stream: Any) -> None: ...


class QueryableP2PDependency(Protocol):
    """P2P handle queried without blocking."""

    def query(self) -> bool: ...


class IsCompletedP2PDependency(Protocol):
    """P2P handle checked without blocking via ``is_completed``."""

    def is_completed(self) -> bool: ...


AsyncP2PDependency: TypeAlias = (
    QueryableP2PDependency | IsCompletedP2PDependency
)


@dataclass(frozen=True, slots=True)
class JoinedCompletion:
    """One completion gate joining submission-context and runtime readiness."""

    context: Any
    runtime_completion: Any

    def __post_init__(self) -> None:
        _require_completion_endpoint(self.context, "submission context")
        _require_completion_endpoint(
            self.runtime_completion, "runtime completion"
        )

    def query(self) -> bool:
        return _query_ready(self.context) and _query_ready(self.runtime_completion)

    def is_completed(self) -> bool:
        return self.query()

    def wait(self) -> None:
        self.context.wait()
        self.runtime_completion.wait()

    def wait_stream(self, stream: Any) -> None:
        self.context.wait_stream(stream)
        self.runtime_completion.wait_stream(stream)


class SubmissionRuntime(Protocol):
    def create_submission_context(self, tensor: Any) -> SubmissionContext: ...

    def record_completion(
        self, *, context: Any, dependencies: tuple[Any, ...]
    ) -> Any: ...


@dataclass(slots=True, eq=False)
class SubmissionOwner:
    tensor: Any
    workspace: WorkspaceSession
    context: Any
    resources: list[Any] = field(default_factory=list)
    dependencies: list[Any] = field(default_factory=list)
    completion: Any | None = None
    record_attempted: bool = False
    workspace_released: bool = False
    cleanup_with_abort: bool = False
    workspace_leases: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resources.extend((self.tensor, self.workspace, self.context))

    def retain(self, *values: Any) -> None:
        self.resources.extend(value for value in values if value is not None)

    def depend_on(self, value: Any) -> None:
        if value is not None:
            self.dependencies.append(value)
            self.retain(value)

    def depend_on_p2p(self, value: Any) -> None:
        self.retain(value)
        if not _has_nonblocking_query(value):
            raise TypeError(
                "P2P dependency must provide nonblocking query() or is_completed()"
            )
        self.dependencies.append(value)

    def capture_workspace_resources(self) -> None:
        leases = getattr(self.workspace, "leases", ())
        if isinstance(leases, tuple):
            known = {id(lease) for lease in self.workspace_leases}
            self.workspace_leases.extend(
                lease for lease in leases if id(lease) not in known
            )
            self.retain(*leases)
        buffers = getattr(self.workspace, "buffers", None)
        if isinstance(buffers, dict):
            self.retain(*buffers.values())


class ExecutorSupport:
    """Own resources across executor exceptions without blocking the CPU."""

    __slots__ = ("_workspace_factory", "_completion_manager", "_pending")

    def __init__(self, workspace_factory: Any, completion_manager: CompletionManager) -> None:
        self._workspace_factory = workspace_factory
        self._completion_manager = completion_manager
        self._pending: list[SubmissionOwner] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def begin(self, tensor: Any, runtime: SubmissionRuntime) -> SubmissionOwner:
        context = runtime.create_submission_context(tensor)
        # The factory retains ownership of partially acquired state until it
        # returns successfully; only the returned session transfers here.
        workspace = self._workspace_factory(tensor)
        return SubmissionOwner(tensor=tensor, workspace=workspace, context=context)

    def record(self, owner: SubmissionOwner, runtime: SubmissionRuntime) -> Any:
        owner.record_attempted = True
        runtime_completion = runtime.record_completion(
            context=owner.context,
            dependencies=tuple(owner.dependencies),
        )
        owner.retain(runtime_completion)
        completion = JoinedCompletion(owner.context, runtime_completion)
        owner.completion = completion
        owner.retain(completion)
        return completion

    def finish(self, owner: SubmissionOwner) -> Any:
        completion = owner.completion
        if completion is None:
            raise RuntimeError("submission completion was not recorded")
        owner.capture_workspace_resources()
        try:
            work = self._completion_manager.create_work(
                result=owner.tensor,
                completion=completion,
                resources=tuple(owner.resources),
            )
        except BaseException:
            self._quarantine(owner)
            raise
        owner.retain(work)
        self._quarantine(owner)
        try:
            self._release(owner, completion)
        except BaseException:
            raise
        else:
            self._pending.remove(owner)
            return work

    def fail(self, owner: SubmissionOwner, runtime: SubmissionRuntime) -> None:
        self._quarantine(owner)
        owner.cleanup_with_abort = callable(getattr(owner.workspace, "abort", None))
        if owner.record_attempted:
            return
        try:
            self.record(owner, runtime)
        except BaseException:
            return

    def reap(self) -> None:
        for owner in tuple(self._pending):
            readiness = (
                owner.completion if owner.completion is not None else owner.context
            )
            query = _nonblocking_query(readiness)
            if query is None:
                continue
            try:
                ready = bool(query())
            except BaseException:
                continue
            if not ready:
                continue
            if not owner.workspace_released:
                if owner.cleanup_with_abort:
                    abort = getattr(owner.workspace, "abort", None)
                    try:
                        if callable(abort):
                            abort()
                        else:
                            self._release(owner, readiness)
                    except BaseException:
                        continue
                    owner.workspace_released = self._captured_leases_released(owner)
                else:
                    release_succeeded = False
                    try:
                        self._release(owner, readiness)
                    except BaseException:
                        pass
                    else:
                        release_succeeded = True
                    captured_released = self._release_captured_leases(owner, readiness)
                    owner.workspace_released = captured_released or (
                        release_succeeded and not owner.workspace_leases
                    )
                if not owner.workspace_released:
                    continue
            self._pending.remove(owner)

    def _quarantine(self, owner: SubmissionOwner) -> None:
        owner.capture_workspace_resources()
        if owner not in self._pending:
            self._pending.append(owner)

    @staticmethod
    def _release(owner: SubmissionOwner, completion: Any) -> None:
        release = getattr(owner.workspace, "release", None)
        if not callable(release):
            raise TypeError("workspace session must provide release(completion=...)")
        release(completion=completion)
        owner.workspace_released = True

    @classmethod
    def _release_captured_leases(
        cls, owner: SubmissionOwner, completion: Any
    ) -> bool:
        if not owner.workspace_leases:
            return False
        for lease in owner.workspace_leases:
            if cls._lease_released(lease):
                continue
            release = getattr(lease, "release", None)
            if not callable(release):
                continue
            try:
                release(completion=completion)
            except BaseException:
                continue
        return cls._captured_leases_released(owner)

    @classmethod
    def _captured_leases_released(cls, owner: SubmissionOwner) -> bool:
        return not owner.workspace_leases or all(
            cls._lease_released(lease) for lease in owner.workspace_leases
        )

    @staticmethod
    def _lease_released(lease: Any) -> bool:
        try:
            released = lease.released
        except BaseException:
            return False
        return isinstance(released, bool) and released


def _nonblocking_query(value: Any) -> Any | None:
    query = getattr(value, "query", None)
    if callable(query):
        return query
    is_completed = getattr(value, "is_completed", None)
    if callable(is_completed):
        return is_completed
    return None


def _has_nonblocking_query(value: Any) -> bool:
    return _nonblocking_query(value) is not None


def _query_ready(value: Any) -> bool:
    query = _nonblocking_query(value)
    if query is None:
        return False
    return bool(query())


def _require_completion_endpoint(value: Any, name: str) -> None:
    if not _has_nonblocking_query(value):
        raise TypeError(
            f"{name} must provide nonblocking query() or is_completed()"
        )
    for method_name in ("wait", "wait_stream"):
        if not callable(getattr(value, method_name, None)):
            raise TypeError(f"{name} must provide {method_name}()")
