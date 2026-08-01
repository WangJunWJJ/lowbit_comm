"""Backend-neutral communication stage descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CompressionConfig


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class CommunicationStage:
    """Describe one indivisible stage of a communication plan."""

    name: str
    collective: str
    strategy: str
    backend: str = "cuda"
    compression: CompressionConfig | None = None
    process_group: object | None = None
    output_layout: str = "full"
    async_op: bool = True

    def __post_init__(self) -> None:
        for field_name in ("name", "collective", "strategy", "backend", "output_layout"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.compression is not None and not isinstance(self.compression, CompressionConfig):
            raise TypeError("compression must be a CompressionConfig or None")
