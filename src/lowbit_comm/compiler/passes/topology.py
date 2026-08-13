"""Pure topology parsing for compile-time communication lowering."""

from __future__ import annotations

from lowbit_comm.core.topology import (
    GroupedReductionPlan,
    Topology,
    compile_grouped_reduction,
    parse_topology_signature,
)


__all__ = [
    "GroupedReductionPlan",
    "Topology",
    "compile_grouped_reduction",
    "parse_topology_signature",
]
