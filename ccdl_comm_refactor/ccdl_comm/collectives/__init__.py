"""Modern CCDL compressed collective APIs."""

from ccdl_comm.exceptions import UnsupportedCollective

from .all_reduce import compressed_all_reduce
from .work import CollectiveWork, ImmediateWork

__all__ = [
    "CollectiveWork",
    "ImmediateWork",
    "UnsupportedCollective",
    "compressed_all_reduce",
]
