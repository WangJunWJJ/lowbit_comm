from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CapabilityReport:
    """Runtime capability report consumed by ParaScale planning logic."""

    available: bool
    cuda: bool
    torch_version: str | None = None
    cuda_arch: str | None = None
    quantize: bool = False
    compressed_collectives: bool = False
    ddp_hook: bool = False
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def unavailable(cls, reason: str) -> "CapabilityReport":
        return cls(
            available=False,
            cuda=False,
            reason=reason,
            warnings=(reason,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "cuda": self.cuda,
            "torch_version": self.torch_version,
            "cuda_arch": self.cuda_arch,
            "ops": {
                "quantize": self.quantize,
                "compressed_collectives": self.compressed_collectives,
                "ddp_hook": self.ddp_hook,
            },
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def _import_torch():
    import torch

    return torch


def _import_extension():
    import ccdl_cuda_ops

    return ccdl_cuda_ops


def detect(
    *,
    import_torch: Callable[[], object] = _import_torch,
    import_extension: Callable[[], object] = _import_extension,
) -> CapabilityReport:
    """Detect whether CCDL compressed communication can be enabled safely.

    This function is deliberately import-safe for ParaScale control-plane code:
    missing Torch, missing CUDA, or missing CUDA extension all return a report
    instead of raising during configuration planning.
    """

    try:
        torch = import_torch()
    except ModuleNotFoundError:
        return CapabilityReport.unavailable("torch is not installed")

    torch_version = getattr(torch, "__version__", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return CapabilityReport(
            available=False,
            cuda=False,
            torch_version=torch_version,
            reason="CUDA is not available",
            warnings=("CUDA is not available",),
        )

    try:
        import_extension()
    except ModuleNotFoundError:
        return CapabilityReport(
            available=False,
            cuda=True,
            torch_version=torch_version,
            reason="ccdl_cuda_ops is not installed",
            warnings=("ccdl_cuda_ops is not installed",),
        )

    cuda_arch = None
    get_device_capability = getattr(cuda, "get_device_capability", None)
    if get_device_capability is not None:
        major, minor = get_device_capability(0)
        cuda_arch = f"{major}.{minor}"

    return CapabilityReport(
        available=True,
        cuda=True,
        torch_version=torch_version,
        cuda_arch=cuda_arch,
        quantize=True,
        compressed_collectives=False,
        ddp_hook=False,
        warnings=("DDP hook and compressed collectives are not implemented yet",),
    )
