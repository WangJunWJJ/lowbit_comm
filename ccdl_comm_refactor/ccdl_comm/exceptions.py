from __future__ import annotations


class CCDLError(RuntimeError):
    """Base class for public CCDL communication errors."""


class CCDLUnavailableError(CCDLError):
    """Raised when CUDA-backed CCDL functionality cannot be used."""


class UnsupportedCollective(CCDLError):
    """Raised when a collective or strategy is unsupported by this runtime."""

    def __init__(self, collective: str, *, reason: str | None = None) -> None:
        message = f"unsupported CCDL collective or strategy: {collective}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)
        self.collective = collective
        self.reason = reason


class TorchDistributedUnavailableError(CCDLError):
    """Raised when torch.distributed cannot run the requested transport."""
