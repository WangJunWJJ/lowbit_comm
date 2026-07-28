"""Modern CCDL compressed collective APIs."""

from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.communication.gather_reduce import GatheredPayloads

from .all_gather import compressed_all_gather
from .all_reduce import compressed_all_reduce
from .reduce_scatter import compressed_reduce_scatter
from .work import CollectiveWork, ImmediateWork

__all__ = [
    "CollectiveWork",
    "GatheredPayloads",
    "ImmediateWork",
    "UnsupportedCollective",
    "compressed_all_gather",
    "compressed_all_reduce",
    "compressed_reduce_scatter",
]
