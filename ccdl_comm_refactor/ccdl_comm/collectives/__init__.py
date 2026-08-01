"""Modern CCDL compressed collective APIs."""

from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.communication.gather_reduce import GatheredPayloads

from .all_gather import compressed_all_gather
from .all_reduce import compressed_all_reduce
from .dynamic_all_gather import compressed_all_gather_dynamic, qall_gather_dyn
from .hierarchical import compressed_hierarchical_all_reduce
from .reduce_scatter import ReducedShard, compressed_reduce_scatter, compressed_reduce_scatter_shard
from .work import CollectiveWork, CompletionWork, ImmediateWork

__all__ = [
    "CollectiveWork",
    "CompletionWork",
    "GatheredPayloads",
    "ImmediateWork",
    "ReducedShard",
    "UnsupportedCollective",
    "compressed_all_gather",
    "compressed_all_gather_dynamic",
    "compressed_all_reduce",
    "compressed_hierarchical_all_reduce",
    "compressed_reduce_scatter",
    "compressed_reduce_scatter_shard",
    "qall_gather_dyn",
]
