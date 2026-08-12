"""Backend-neutral consumers for rank-local reduced shards."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ccdl_comm.shard import ReducedShard


@runtime_checkable
class ReducedShardConsumer(Protocol):
    """Consume a reduced shard without prescribing a training framework."""

    def consume(self, reduced: ReducedShard) -> object:
        """Consume one rank-local reduced shard and return a completion value."""

        ...
