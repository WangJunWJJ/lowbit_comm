"""Typed low-bit communication library."""

from .compiler import BackendRegistry, BenchmarkEvidence, compile
from .core import CommunicationProgram, CompileContext, RuntimeBindings

__version__ = "0.3.0"

__all__ = [
    "BackendRegistry",
    "BenchmarkEvidence",
    "CommunicationProgram",
    "CompileContext",
    "RuntimeBindings",
    "__version__",
    "compile",
]
