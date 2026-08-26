"""Torch/CUDA/NCCL and extension-ABI gates for experimental RSAG/qWD."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from lowbit_comm.backends.cuda.loader import CUDA_ABI_VERSION
from lowbit_comm._version import __version__

RSAG_LOWBIT_COMM_VERSION = __version__
RSAG_CUDA_EXTENSION_ABI = CUDA_ABI_VERSION


def _require_exact_string(value: object, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact string")


def _require_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")


@dataclass(frozen=True, slots=True)
class RSAGRuntimeABI:
    """One exact binary runtime identity."""

    torch_version: str
    cuda_version: str
    nccl_version: str
    cuda_extension_abi: int

    def __post_init__(self) -> None:
        for name in ("torch_version", "cuda_version", "nccl_version"):
            _require_exact_string(getattr(self, name), name)
        _require_nonnegative_int(
            self.cuda_extension_abi,
            "cuda_extension_abi",
        )


RSAG_VERIFIED_RUNTIME_MATRIX = (
    RSAGRuntimeABI(
        torch_version="2.5.0a0+872d972e41.nv24.08",
        cuda_version="12.6",
        nccl_version="2.22.3",
        cuda_extension_abi=RSAG_CUDA_EXTENSION_ABI,
    ),
)


@dataclass(frozen=True, slots=True)
class RSAGCompatibilityReport:
    """Stable result of probing the optional CUDA runtime."""

    runtime: RSAGRuntimeABI | None
    extension_loaded: bool
    reason: str

    def __post_init__(self) -> None:
        if self.runtime is not None and type(self.runtime) is not RSAGRuntimeABI:
            raise ValueError("runtime must be an exact RSAGRuntimeABI")
        if type(self.extension_loaded) is not bool:
            raise ValueError("extension_loaded must be an exact bool")
        _require_exact_string(self.reason, "reason")

    @property
    def compatible(self) -> bool:
        """Return whether the runtime is in the verified product matrix."""
        return self.reason == "verified_runtime"


def is_verified_rsag_runtime(runtime: RSAGRuntimeABI) -> bool:
    """Require an exact runtime identity from the reviewed matrix."""
    if type(runtime) is not RSAGRuntimeABI:
        raise ValueError("runtime must be an exact RSAGRuntimeABI")
    runtime.__post_init__()
    return runtime in RSAG_VERIFIED_RUNTIME_MATRIX


def probe_rsag_compatibility() -> RSAGCompatibilityReport:
    """Probe optional dependencies without affecting stable package import."""
    try:
        torch = import_module("torch")
    except Exception:
        return RSAGCompatibilityReport(None, False, "torch_unavailable")
    if not torch.cuda.is_available():
        return RSAGCompatibilityReport(None, False, "cuda_unavailable")
    cuda_version = torch.version.cuda
    if type(cuda_version) is not str or not cuda_version:
        return RSAGCompatibilityReport(
            None,
            False,
            "cuda_version_unavailable",
        )
    try:
        nccl_version = _nccl_version_string(torch.cuda.nccl.version())
    except (TypeError, ValueError):
        return RSAGCompatibilityReport(
            None,
            False,
            "nccl_version_unavailable",
        )
    runtime_without_extension = {
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_version,
        "nccl_version": nccl_version,
    }
    try:
        loader = import_module("lowbit_comm.backends.cuda.loader")
        extension = loader.load_extension()
    except Exception:
        return RSAGCompatibilityReport(
            RSAGRuntimeABI(
                cuda_extension_abi=RSAG_CUDA_EXTENSION_ABI,
                **runtime_without_extension,
            ),
            False,
            "extension_unavailable",
        )
    extension_abi = extension.abi_version()
    if type(extension_abi) is not int or extension_abi < 0:
        return RSAGCompatibilityReport(
            None,
            True,
            "extension_abi_invalid",
        )
    runtime = RSAGRuntimeABI(
        cuda_extension_abi=extension_abi,
        **runtime_without_extension,
    )
    if extension_abi != RSAG_CUDA_EXTENSION_ABI:
        return RSAGCompatibilityReport(
            runtime,
            True,
            "extension_abi_mismatch",
        )
    reason = (
        "verified_runtime"
        if is_verified_rsag_runtime(runtime)
        else "unsupported_runtime_matrix"
    )
    return RSAGCompatibilityReport(runtime, True, reason)


def _nccl_version_string(value: object) -> str:
    if type(value) is tuple and value and all(
        type(part) is int and part >= 0 for part in value
    ):
        return ".".join(str(part) for part in value)
    if type(value) is int and value > 0:
        return str(value)
    raise ValueError("NCCL version is invalid")
