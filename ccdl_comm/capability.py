from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .backend import BackendCapabilities
from .cuda.loader import CudaExtensionStatus, load_cuda_extension


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

    def to_backend_capabilities(self, *, backend: str = "cuda") -> BackendCapabilities:
        """Normalize this migration report into the backend-neutral Core model."""

        features = frozenset(
            name
            for name, enabled in (
                ("quantize", self.quantize),
                ("compressed_collectives", self.compressed_collectives),
                ("ddp_hook", self.ddp_hook),
            )
            if enabled
        )
        collectives = (
            frozenset({"all_gather", "all_reduce", "reduce_scatter"})
            if self.compressed_collectives
            else frozenset()
        )
        strategies = (
            frozenset({"all_gather", "compressed", "topology"})
            if self.compressed_collectives
            else frozenset()
        )
        details = {
            key: value
            for key, value in (
                ("cuda_arch", self.cuda_arch),
                ("torch_version", self.torch_version),
            )
            if value is not None
        }
        return BackendCapabilities(
            backend=backend,
            available=self.available,
            collectives=collectives,
            strategies=strategies,
            dtypes=frozenset({"bf16", "fp16", "fp32"}) if self.quantize else frozenset(),
            bits=frozenset({8}) if self.quantize else frozenset(),
            output_layouts=frozenset({"full", "shard"}) if self.compressed_collectives else frozenset(),
            supports_async=self.compressed_collectives,
            features=features,
            reason=self.reason,
            warnings=self.warnings,
            details=details,
        )


def _import_torch():
    import torch

    return torch


def detect(
    *,
    import_torch: Callable[[], object] = _import_torch,
    import_extension: Callable[[], object] | None = None,
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

    if import_extension is None:
        extension_status = load_cuda_extension()
    else:
        try:
            extension_status = CudaExtensionStatus(available=True, module=import_extension())
        except ModuleNotFoundError:
            extension_status = CudaExtensionStatus(
                available=False,
                module=None,
                reason="ccdl_cuda_ops is not installed",
            )
        except ImportError as exc:
            extension_status = CudaExtensionStatus(available=False, module=None, reason=str(exc))

    if not extension_status.available:
        return CapabilityReport(
            available=False,
            cuda=True,
            torch_version=torch_version,
            reason=extension_status.reason,
            warnings=(extension_status.reason or "ccdl_cuda_ops is not available",),
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
