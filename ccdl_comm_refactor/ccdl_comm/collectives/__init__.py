"""Modern CCDL compressed collective APIs."""

from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.communication.gather_reduce import GatheredPayloads

from .all_reduce import compressed_all_reduce
from .work import CollectiveWork, ImmediateWork

__all__ = [
    "CollectiveWork",
    "GatheredPayloads",
    "ImmediateWork",
    "UnsupportedCollective",
    "compressed_all_reduce",
]
