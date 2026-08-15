"""Deterministic helpers for plan and evidence signatures."""

import json
from dataclasses import fields
from enum import Enum
from typing import Any

from lowbit_comm.api.intent import (
    CommunicationIntent,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.api.policy import (
    AccumulationDType,
    CollectiveKind,
    CompressionKind,
    StrategySpec,
)
from lowbit_comm.core.errors import CompileError


_DTYPE_BIT_WIDTHS = {
    "bfloat16": 16,
    "float16": 16,
    "float32": 32,
    "float64": 64,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "uint8": 8,
}
_COLLECTIVE_SIGNATURES = {
    CollectiveKind.NATIVE: "native",
    CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE: "cag",
}
_SCALE_METADATA_BYTES = 4
StrategyKey = tuple[tuple[str, str, str], ...]
_LOCAL_INTENT_FIELDS = frozenset({"rank"})
_COLLECTIVE_INTENT_FIELDS = frozenset(
    {
        "completion",
        "output",
        "reduction",
        "shape_family",
        "tensor",
        "world_size",
    }
)


def dtype_bit_width(dtype: str) -> int:
    """Return the storage width for one supported tensor element."""
    if type(dtype) is not str:
        raise CompileError("Signature dtype must be a string.")
    try:
        return _DTYPE_BIT_WIDTHS[dtype]
    except KeyError as error:
        raise CompileError(
            f"Unsupported dtype for deterministic signature: {dtype}."
        ) from error


def compression_bit_width(strategy: StrategySpec) -> int:
    """Return the payload bit width selected by *strategy*."""
    _require_strategy(strategy)
    if strategy.compression is CompressionKind.INT8:
        return 8
    return 0


def logical_size_bytes(tensor: TensorSpec) -> int:
    """Return the uncompressed byte count for *tensor*."""
    if type(tensor) is not TensorSpec:
        raise CompileError("Signature tensor must be a TensorSpec.")
    return tensor.numel * dtype_bit_width(tensor.dtype) // 8


def intent_signature(intent: CommunicationIntent) -> str:
    """Return a stable signature for collective-shared intent semantics.

    The caller-local rank is deliberately omitted so every participant in
    one collective derives the same evidence key. All other current intent
    fields are explicitly classified, while nested tensor and shape-family
    dataclasses are reflected in full.
    """
    _require_intent(intent)
    intent_fields = {field.name for field in fields(CommunicationIntent)}
    classified_fields = _COLLECTIVE_INTENT_FIELDS | _LOCAL_INTENT_FIELDS
    if (
        classified_fields != intent_fields
        or _COLLECTIVE_INTENT_FIELDS & _LOCAL_INTENT_FIELDS
    ):
        raise CompileError(
            "Intent signature fields require an explicit shared/local "
            "classification."
        )
    components = [
        [field.name, _intent_signature_value(getattr(intent, field.name))]
        for field in fields(CommunicationIntent)
        if field.name in _COLLECTIVE_INTENT_FIELDS
    ]
    encoded = [
        "dataclass",
        _qualified_type_name(CommunicationIntent),
        components,
    ]
    return json.dumps(
        encoded,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def strategy_signature(strategy: StrategySpec) -> str:
    """Return a stable identifier for an exact strategy declaration."""
    _require_strategy(strategy)
    collective = _COLLECTIVE_SIGNATURES[strategy.collective]
    parts = [
        strategy.compression.value,
        collective,
        strategy.topology.value,
    ]
    if strategy.accumulation_dtype is not AccumulationDType.FP32:
        parts.extend(("accumulation", strategy.accumulation_dtype.value))
    if strategy.parameter_error_feedback:
        parts.append("parameter-ef")
    if strategy.workspace_budget_bytes is not None:
        parts.extend(
            ("workspace", str(strategy.workspace_budget_bytes))
        )
    return "-".join(parts)


def strategy_key(strategy: StrategySpec) -> StrategyKey:
    """Return a sortable key containing every strategy dataclass field."""
    _require_strategy(strategy)
    return tuple(
        (field.name, *_signature_value(getattr(strategy, field.name)))
        for field in fields(StrategySpec)
    )


def wire_size_bytes(
    tensor: TensorSpec,
    strategy: StrategySpec,
) -> int:
    """Return payload plus per-group scale metadata bytes."""
    if type(tensor) is not TensorSpec:
        raise CompileError("Signature tensor must be a TensorSpec.")
    _require_strategy(strategy)
    if strategy.compression is CompressionKind.NONE:
        return logical_size_bytes(tensor)
    bit_width = compression_bit_width(strategy)
    payload_bytes = (tensor.numel * bit_width + 7) // 8
    group_size = strategy.group_size
    if group_size is None:
        raise CompileError("Compressed wire signature requires a group size.")
    group_count = (tensor.numel + group_size - 1) // group_size
    return payload_bytes + group_count * _SCALE_METADATA_BYTES


def _require_strategy(strategy: StrategySpec) -> None:
    """Require an exact immutable strategy contract."""
    if type(strategy) is not StrategySpec:
        raise CompileError("Signature strategy must be a StrategySpec.")


def _require_intent(intent: CommunicationIntent) -> None:
    """Require and freshly validate the exact immutable intent graph."""
    if type(intent) is not CommunicationIntent:
        raise CompileError("Signature intent must be a CommunicationIntent.")
    if type(intent.tensor) is not TensorSpec:
        raise CompileError("Signature intent tensor must be a TensorSpec.")
    if type(intent.shape_family) is not ShapeFamily:
        raise CompileError(
            "Signature intent shape family must be a ShapeFamily."
        )
    intent.tensor.__post_init__()
    intent.shape_family.__post_init__()
    intent.__post_init__()


def _qualified_type_name(value_type: type[object]) -> str:
    """Return a stable, explicit type discriminator without ``repr``."""
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _intent_signature_value(value: object) -> Any:
    """Encode one supported intent value with stable type information."""
    if isinstance(value, Enum):
        enum_type = type(value)
        return ["enum", _qualified_type_name(enum_type), value.name]
    if type(value) in (TensorSpec, ShapeFamily):
        value.__post_init__()
        return [
            "dataclass",
            _qualified_type_name(type(value)),
            [
                [
                    field.name,
                    _intent_signature_value(getattr(value, field.name)),
                ]
                for field in fields(value)
            ],
        ]
    if type(value) is tuple:
        return ["tuple", [_intent_signature_value(item) for item in value]]
    if value is None:
        return ["none", ""]
    if type(value) is bool:
        return ["bool", "true" if value else "false"]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is str:
        return ["str", value]
    raise CompileError(
        "Intent signature contains an unsupported field type."
    )


def _signature_value(value: object) -> tuple[str, str]:
    """Normalize one immutable strategy value into sortable strings."""
    if isinstance(value, Enum):
        enum_type = type(value)
        kind = f"enum:{enum_type.__module__}.{enum_type.__qualname__}"
        return kind, value.name
    if value is None:
        return "none", ""
    if type(value) is bool:
        return "bool", "true" if value else "false"
    if type(value) is int:
        return "int", str(value)
    raise CompileError(
        "Strategy signature contains an unsupported field type."
    )
