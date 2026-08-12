"""Public Core and compiler failures."""


class LowBitCommError(RuntimeError):
    """Base lowbit_comm failure."""


class ProgramVerificationError(LowBitCommError, ValueError):
    """A Semantic IR program violates a static contract."""
