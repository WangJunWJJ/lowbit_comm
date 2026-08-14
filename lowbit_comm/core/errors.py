"""Exception hierarchy for low-bit communication contracts."""


class LowbitCommError(Exception):
    """Base error for low-bit communication failures."""


class CompileError(LowbitCommError):
    """Raised when a communication contract cannot be compiled."""


class CapabilityError(LowbitCommError):
    """Raised when a requested capability is unavailable."""


class ExecutionError(LowbitCommError):
    """Raised when communication execution fails."""
