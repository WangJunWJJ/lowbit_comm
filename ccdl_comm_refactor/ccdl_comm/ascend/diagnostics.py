from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from ccdl_comm.ascend.loader import CannExtensionStatus, load_cann_extension


@dataclass(frozen=True)
class CannCapabilityReport:
    """Runtime Ascend CANN capability report for scheduler decisions."""

    available: bool
    npu: bool
    torch_version: str | None = None
    torch_npu_version: str | None = None
    extension_available: bool = False
    quantize: bool = False
    dequantize: bool = False
    compressed_collectives: bool = False
    ddp_hook: bool = False
    quantization_path: str | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        npu: bool = False,
        torch_version: str | None = None,
        torch_npu_version: str | None = None,
    ) -> "CannCapabilityReport":
        return cls(
            available=False,
            npu=npu,
            torch_version=torch_version,
            torch_npu_version=torch_npu_version,
            reason=reason,
            warnings=(reason,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "backend": "cann",
            "npu": self.npu,
            "torch_version": self.torch_version,
            "torch_npu_version": self.torch_npu_version,
            "extension_available": self.extension_available,
            "quantization_path": self.quantization_path,
            "ops": {
                "quantize": self.quantize,
                "dequantize": self.dequantize,
                "compressed_collectives": self.compressed_collectives,
                "ddp_hook": self.ddp_hook,
            },
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def _import_torch():
    import torch

    return torch


def _import_torch_npu():
    import torch_npu

    return torch_npu


def detect_cann(
    *,
    import_torch: Callable[[], object] = _import_torch,
    import_torch_npu: Callable[[], object] = _import_torch_npu,
    load_extension: Callable[[], CannExtensionStatus] = load_cann_extension,
    environ: dict[str, str] | None = None,
) -> CannCapabilityReport:
    """Detect whether the Ascend CANN backend can run safely."""

    try:
        torch = import_torch()
    except ModuleNotFoundError:
        return CannCapabilityReport.unavailable("torch is not installed")

    torch_version = getattr(torch, "__version__", None)
    npu = getattr(torch, "npu", None)
    npu_available = bool(npu is not None and npu.is_available())

    try:
        torch_npu = import_torch_npu()
    except ModuleNotFoundError:
        return CannCapabilityReport.unavailable(
            "torch_npu is not installed",
            npu=npu_available,
            torch_version=torch_version,
        )

    torch_npu_version = getattr(torch_npu, "__version__", None)
    if not npu_available:
        return CannCapabilityReport.unavailable(
            "Ascend NPU is not available",
            npu=False,
            torch_version=torch_version,
            torch_npu_version=torch_npu_version,
        )

    extension_status = load_extension()
    if not extension_status.available:
        return CannCapabilityReport(
            available=False,
            npu=True,
            torch_version=torch_version,
            torch_npu_version=torch_npu_version,
            extension_available=False,
            reason=extension_status.reason,
            warnings=(extension_status.reason or "ccdl_cann_ops is not available",),
        )

    env = environ if environ is not None else os.environ
    experimental_aclnn = env.get("CCDL_COMM_EXPERIMENTAL_ACLNN") == "1"
    quantization_path = "experimental_aclnn" if experimental_aclnn else "aten_cann"
    warnings = ("experimental ACLNN path is enabled",) if experimental_aclnn else ()
    return CannCapabilityReport(
        available=True,
        npu=True,
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        extension_available=True,
        quantize=True,
        dequantize=True,
        compressed_collectives=True,
        ddp_hook=True,
        quantization_path=quantization_path,
        warnings=warnings,
    )
