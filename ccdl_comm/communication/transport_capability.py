"""Capabilities for transports that operate on encoded CCDL payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective


@dataclass(frozen=True, slots=True)
class CompressedTransportCapability:
    """Declare the encoded payload formats understood by one transport."""

    codec: str
    collectives: frozenset[str]
    bits: frozenset[int]
    group_sizes: frozenset[int]
    dtypes: frozenset[str]
    output_layouts: frozenset[str]
    supports_async: bool = False

    def __post_init__(self) -> None:
        if not self.codec.strip():
            raise ValueError("codec must be a non-empty string")
        for field_name in ("collectives", "dtypes", "output_layouts"):
            values = frozenset(getattr(self, field_name))
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            object.__setattr__(self, field_name, values)
        for field_name in ("bits", "group_sizes"):
            values = frozenset(getattr(self, field_name))
            if not values or any(isinstance(value, bool) or value <= 0 for value in values):
                raise ValueError(f"{field_name} must contain positive integers")
            object.__setattr__(self, field_name, values)

    def require_support(
        self,
        *,
        collective: str,
        bit: int,
        group_size: int,
        dtype: str | None,
        output_layout: str,
    ) -> None:
        """Raise when the requested encoded operation is not declared."""

        mismatches = []
        if collective not in self.collectives:
            mismatches.append(f"collective={collective}")
        if bit not in self.bits:
            mismatches.append(f"bit={bit}")
        if group_size not in self.group_sizes:
            mismatches.append(f"group_size={group_size}")
        if dtype is not None and _normalize_dtype(dtype) not in self.dtypes:
            mismatches.append(f"dtype={dtype}")
        if output_layout not in self.output_layouts:
            mismatches.append(f"output_layout={output_layout}")
        if mismatches:
            raise UnsupportedCollective(
                collective,
                reason="compressed payload capability mismatch: " + ", ".join(mismatches),
            )


@dataclass(frozen=True, slots=True)
class CapabilityBoundTransport:
    """Bind an encoded-payload capability to a transport callable."""

    transport: Callable[..., Any]
    ccdl_compressed_capability: CompressedTransportCapability

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transport(*args, **kwargs)


def bind_compressed_transport(
    transport: Callable[..., Any],
    capability: CompressedTransportCapability,
) -> CapabilityBoundTransport:
    """Return a callable transport carrying an immutable capability."""

    return CapabilityBoundTransport(transport=transport, ccdl_compressed_capability=capability)


def capability_for(transport: object) -> CompressedTransportCapability | None:
    """Return the declared compressed capability, when present."""

    capability = getattr(transport, "ccdl_compressed_capability", None)
    return capability if isinstance(capability, CompressedTransportCapability) else None


def require_compressed_transport(
    transport: object,
    *,
    collective: str,
    config: CompressionConfig,
    dtype: str | None,
    output_layout: str,
) -> CompressedTransportCapability:
    """Validate that a transport understands the requested CCDL payload."""

    capability = capability_for(transport)
    if capability is None:
        raise UnsupportedCollective(
            collective,
            reason="transport does not declare CCDL compressed payload capability",
        )
    capability.require_support(
        collective=collective,
        bit=config.bit,
        group_size=config.group_size,
        dtype=dtype,
        output_layout=output_layout,
    )
    return capability


def _normalize_dtype(dtype: str) -> str:
    return dtype.strip().lower().removeprefix("torch.")
