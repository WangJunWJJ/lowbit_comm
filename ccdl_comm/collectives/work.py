"""Compatibility re-exports for the backend-neutral work implementations."""

from ccdl_comm.execution_info import ExecutionCounters
from ccdl_comm.work import CollectiveWork, CompletionWork, ImmediateWork

__all__ = ["CollectiveWork", "CompletionWork", "ExecutionCounters", "ImmediateWork"]
