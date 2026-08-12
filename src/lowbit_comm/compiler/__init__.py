"""Typed Semantic IR compiler services."""

from lowbit_comm.core.errors import ProgramVerificationError

from .verifier import verify

__all__ = ["ProgramVerificationError", "verify"]
