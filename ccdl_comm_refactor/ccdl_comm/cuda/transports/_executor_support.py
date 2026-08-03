"""Shared ownership support for async topology submissions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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
        if not callable(getattr(context, "query", None)):
            raise TypeError("submission context must provide nonblocking query()")
        # The factory retains ownership of partially acquired state until it
        # returns successfully; only the returned session transfers here.
        workspace = self._workspace_factory(tensor)
        return SubmissionOwner(tensor=tensor, workspace=workspace, context=context)

    def record(self, owner: SubmissionOwner, runtime: SubmissionRuntime) -> Any:
        owner.record_attempted = True
        completion = runtime.record_completion(
            context=owner.context,
            dependencies=tuple(owner.dependencies),
        )
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
            self._try_release(owner, completion)
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
        if owner.record_attempted:
            owner.cleanup_with_abort = callable(getattr(owner.workspace, "abort", None))
            self._quarantine(owner)
            return

        abort = getattr(owner.workspace, "abort", None)
        if callable(abort):
            try:
                abort()
            except BaseException:
                owner.cleanup_with_abort = True
                self._quarantine(owner)
            return

        try:
            completion = self.record(owner, runtime)
        except BaseException:
            self._quarantine(owner)
            return
        self._quarantine(owner)
        self._try_release(owner, completion)

    def reap(self) -> None:
        for owner in tuple(self._pending):
            readiness = owner.completion or owner.context
            query = getattr(readiness, "query", None)
            if not callable(query):
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
                    owner.workspace_released = True
                else:
                    try:
                        self._release(owner, readiness)
                    except BaseException:
                        if not self._release_captured_leases(owner, readiness):
                            continue
            self._pending.remove(owner)

    def _quarantine(self, owner: SubmissionOwner) -> None:
        owner.capture_workspace_resources()
        if owner not in self._pending:
            self._pending.append(owner)

    def _try_release(self, owner: SubmissionOwner, completion: Any) -> None:
        try:
            self._release(owner, completion)
        except BaseException:
            return

    @staticmethod
    def _release(owner: SubmissionOwner, completion: Any) -> None:
        release = getattr(owner.workspace, "release", None)
        if not callable(release):
            raise TypeError("workspace session must provide release(completion=...)")
        release(completion=completion)
        owner.workspace_released = True

    @staticmethod
    def _release_captured_leases(owner: SubmissionOwner, completion: Any) -> bool:
        if not owner.workspace_leases:
            return False
        all_released = True
        for lease in owner.workspace_leases:
            if bool(getattr(lease, "_released", False)):
                continue
            release = getattr(lease, "release", None)
            if not callable(release):
                all_released = False
                continue
            try:
                release(completion=completion)
            except BaseException:
                if not bool(getattr(lease, "_released", False)):
                    all_released = False
        owner.workspace_released = all_released
        return all_released
