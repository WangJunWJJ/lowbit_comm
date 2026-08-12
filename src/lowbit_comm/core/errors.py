"""Public Core and compiler failures."""


class LowBitCommError(RuntimeError):
    """Base lowbit_comm failure."""


class ProgramVerificationError(LowBitCommError, ValueError):
    """A Semantic IR program violates a static contract."""


class UnsupportedProgram(LowBitCommError):
    """A valid explicit program cannot be implemented by the selected target."""
