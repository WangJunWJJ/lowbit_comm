"""CUDA-specific keys and ownership for compiled internal workspaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from lowbit_comm.core import BufferPlan, BufferSpec, WorkspaceRole
from lowbit_comm.runtime import BudgetedWorkspacePool, WorkspaceLease


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CudaWorkspaceKey:
    role: WorkspaceRole
    shape: tuple[int, ...]
    dtype: str
    device: str
    world_size: int
    bit: int
    group_size: int
    compact: bool


class CudaWorkspaceManager(Generic[T]):
    """Lease buffers described by one immutable CUDA BufferPlan."""

    def __init__(
        self,
        *,
        pool: BudgetedWorkspacePool[T],
        world_size: int,
        bit: int,
        group_size: int,
        compact: bool,
        plan: BufferPlan = BufferPlan(),
    ) -> None:
        self._pool = pool
        self._world_size = world_size
        self._bit = bit
        self._group_size = group_size
        self._compact = compact
        self._specifications: dict[WorkspaceRole, BufferSpec] = {}
        for specification in plan.buffers:
            if specification.role in self._specifications:
                raise ValueError(f"duplicate workspace role {specification.role.value}")
            self._specifications[specification.role] = specification

    def specification(self, role: WorkspaceRole) -> BufferSpec:
        try:
            return self._specifications[role]
        except KeyError as error:
            raise KeyError(f"workspace role {role.value} is not in BufferPlan") from error

    def key(self, specification: BufferSpec, device: object) -> CudaWorkspaceKey:
        return CudaWorkspaceKey(
            role=specification.role,
            shape=specification.shape,
            dtype=specification.dtype,
            device=str(device),
            world_size=self._world_size,
            bit=self._bit,
            group_size=self._group_size,
            compact=self._compact,
        )

    def acquire(
        self,
        specification: BufferSpec,
        *,
        device: object,
        allocator: Callable[[], T],
    ) -> WorkspaceLease[T]:
        return self._pool.acquire(
            self.key(specification, device),
            specification.size_bytes,
            allocator,
        )

    def acquire_role(
        self,
        role: WorkspaceRole,
        *,
        device: object,
        allocator: Callable[[], T],
    ) -> WorkspaceLease[T]:
        return self.acquire(
            self.specification(role),
            device=device,
            allocator=allocator,
        )
