"""Torch/CUDA/NCCL and extension-ABI gates for experimental RSAG/qWD."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from lowbit_comm.backends.cuda.loader import CUDA_ABI_VERSION
from lowbit_comm.core.errors import CapabilityError
from lowbit_comm._version import __version__

RSAG_LOWBIT_COMM_VERSION = __version__
RSAG_CUDA_EXTENSION_ABI = CUDA_ABI_VERSION


def compute_rsag_build_fingerprint() -> str:
    """Hash the complete installed Python runtime and extension binary."""
    digest = sha256()
    try:
        for logical_name, content in _read_rsag_runtime_manifest():
            encoded_name = logical_name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    except Exception as error:
        raise CapabilityError("RSAG/qWD build fingerprint is unavailable.") from error
    return digest.hexdigest()


def _read_rsag_runtime_manifest() -> tuple[tuple[str, bytes], ...]:
    """Read every installed package source plus the loaded extension."""
    package_spec = find_spec("lowbit_comm")
    locations = (
        ()
        if package_spec is None
        else tuple(package_spec.submodule_search_locations or ())
    )
    if len(locations) != 1 or type(locations[0]) is not str:
        raise OSError("lowbit_comm package root is unavailable")
    package_root = Path(locations[0]).resolve(strict=True)
    if not package_root.is_dir():
        raise OSError("lowbit_comm package root is not a directory")

    paths = tuple(
        path
        for path in package_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )
    extension_spec = find_spec("lowbit_comm._C")
    if extension_spec is None or type(extension_spec.origin) is not str:
        raise OSError("lowbit_comm extension origin is unavailable")
    extension_path = Path(extension_spec.origin).resolve(strict=True)
    if not extension_path.is_file():
        raise OSError("lowbit_comm extension is not a regular file")
    paths += (extension_path,)

    manifest: list[tuple[str, bytes]] = []
    for path in paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(package_root)
        except ValueError as error:
            raise OSError("runtime file escapes lowbit_comm package") from error
        logical_name = f"lowbit_comm/{relative.as_posix()}"
        manifest.append((logical_name, resolved.read_bytes()))
    manifest.sort(key=lambda entry: entry[0])
    if len({name for name, _ in manifest}) != len(manifest):
        raise OSError("runtime manifest contains duplicate paths")
    return tuple(manifest)


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
    if (
        type(value) is tuple
        and value
        and all(type(part) is int and part >= 0 for part in value)
    ):
        return ".".join(str(part) for part in value)
    if type(value) is int and value > 0:
        return str(value)
    raise ValueError("NCCL version is invalid")
