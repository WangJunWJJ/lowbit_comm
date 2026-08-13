"""Backend extension protocol for lowering and executable binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .context import CompileContext, RuntimeBindings
from .lowered import LoweredProgram
from .program import CommunicationProgram
from .types import DataType


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """One executable combination advertised by a communication backend."""

    operation: str
    output: str
    wire: str
    algorithm: str
    dtype: DataType
    bit: int | None = None
    group_size: int | None = None
    quant_type: str | None = None
    compact: bool | None = None
    async_supported: bool = True
    physical_primitive: str = "unspecified"

    def __post_init__(self) -> None:
        if self.operation not in {"sum", "mean"}:
            raise ValueError("capability operation must be sum or mean")
        if self.output not in {"full_tensor", "reduced_shard"}:
            raise ValueError("capability output must be full_tensor or reduced_shard")
        if self.wire not in {"full_precision", "quantized"}:
            raise ValueError("capability wire must be full_precision or quantized")
        if not isinstance(self.dtype, DataType):
            raise TypeError("capability dtype must be a DataType")
        if not isinstance(self.async_supported, bool):
            raise TypeError("async_supported must be a bool")
        if not self.algorithm or not self.physical_primitive:
            raise ValueError("algorithm and physical_primitive must be non-empty")
        quantized = self.wire == "quantized"
        quantization = (self.bit, self.group_size, self.quant_type, self.compact)
        if quantized and any(value is None for value in quantization):
            raise ValueError("quantized capability requires complete wire attributes")
        if not quantized and any(value is not None for value in quantization):
            raise ValueError("full-precision capability cannot have quantized attributes")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    target: str
    specifications: tuple[CapabilitySpec, ...]
    backend_abi_version: int = 1
    extension_abi_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("backend target must be a non-empty string")
        object.__setattr__(self, "specifications", tuple(self.specifications))
        if not self.specifications:
            raise ValueError("backend must advertise at least one capability")
        if any(not isinstance(spec, CapabilitySpec) for spec in self.specifications):
            raise TypeError("specifications must contain CapabilitySpec values")
        if self.backend_abi_version <= 0:
            raise ValueError("backend_abi_version must be > 0")

    @property
    def supported_bits(self) -> frozenset[int]:
        return frozenset(
            spec.bit for spec in self.specifications if spec.bit is not None
        )

    @property
    def supported_algorithms(self) -> frozenset[str]:
        return frozenset(spec.algorithm for spec in self.specifications)

    @property
    def supports_full_tensor(self) -> bool:
        return any(spec.output == "full_tensor" for spec in self.specifications)

    @property
    def supports_reduced_shard(self) -> bool:
        return any(spec.output == "reduced_shard" for spec in self.specifications)


class CompiledExecutable(Protocol):
    def run(self, value: Any) -> Any: ...


class CommunicationBackend(Protocol):
    name: str
    abi_version: int

    def capabilities(self, context: CompileContext) -> BackendCapabilities: ...

    def lower(
        self,
        program: CommunicationProgram,
        context: CompileContext,
        bindings: RuntimeBindings,
    ) -> LoweredProgram: ...

    def compile(self, lowered: LoweredProgram) -> CompiledExecutable: ...
