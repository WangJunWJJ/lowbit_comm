"""Backend-neutral compiled executor protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .execution_info import ExecutionInfo
from .work import CollectiveWork


@runtime_checkable
class CompiledExecutor(Protocol):
    """Execute an already-resolved communication plan on the data path."""

    execution_info: ExecutionInfo

    def run(self, tensor: object) -> CollectiveWork[object]:
        """Execute communication without repeating control-plane resolution."""

        ...
