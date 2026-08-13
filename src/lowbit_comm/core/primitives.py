"""Backend-neutral physical communication primitive identifiers."""

from __future__ import annotations

from enum import Enum


class PhysicalPrimitive(Enum):
    """Concrete transport primitive selected by backend lowering."""

    NCCL_ALL_REDUCE = "nccl_all_reduce"
    NCCL_ALL_GATHER_LOCAL_REDUCE = "nccl_all_gather_local_reduce"
    ALL_TO_ALL_LOCAL_REDUCE = "all_to_all_local_reduce"
    ALL_TO_ALL_QUANTIZED_ALL_GATHER = "all_to_all_quantized_all_gather"
    RING_REDUCE_SCATTER = "ring_reduce_scatter"
    TREE_REDUCE = "tree_reduce"
    HIERARCHICAL_REDUCE_SCATTER = "hierarchical_reduce_scatter"
    HIERARCHICAL_COMPRESSED_FULL_TENSOR = "hierarchical_compressed_full_tensor"
    REFERENCE_ALL_REDUCE = "reference_all_reduce"
    REFERENCE_ALL_GATHER = "reference_all_gather"
    REFERENCE_REDUCE_SCATTER = "reference_reduce_scatter"
    REFERENCE_RS_AG = "reference_rs_ag"
