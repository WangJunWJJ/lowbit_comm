"""Optimizer-side consumers for reduced parameter shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from ccdl_comm.shard import ReducedShard
from ccdl_comm.shard_layout import FlatShardLayout


@dataclass(frozen=True, slots=True)
class UpdatedParameterShard:
    """An updated rank-local parameter shard and its immutable layout metadata."""

    shard: Any
    shard_index: int
    shard_numel: int
    valid_numel: int
    original_numel: int
    padded_numel: int
    world_size: int
    dtype: str
    layout_version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.shard_index, "shard_index")
        _require_nonnegative_integer(self.shard_numel, "shard_numel")
        _require_nonnegative_integer(self.valid_numel, "valid_numel")
        _require_nonnegative_integer(self.original_numel, "original_numel")
        _require_nonnegative_integer(self.padded_numel, "padded_numel")
        _require_positive_integer(self.world_size, "world_size")
        _require_nonnegative_integer(self.layout_version, "layout_version")
        if self.shard_index >= self.world_size:
            raise ValueError("shard_index must be smaller than world_size")
        if self.valid_numel > self.shard_numel:
            raise ValueError("valid_numel must be <= shard_numel")
        if self.padded_numel != self.shard_numel * self.world_size:
            raise ValueError("padded_numel must equal shard_numel * world_size")
        if self.original_numel > self.padded_numel:
            raise ValueError("original_numel must be <= padded_numel")
        expected_valid = max(
            0,
            min(
                self.shard_numel,
                self.original_numel - self.shard_index * self.shard_numel,
            ),
        )
        if self.valid_numel != expected_valid:
            raise ValueError("valid_numel does not match the rank-local logical range")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise TypeError("dtype must be a non-empty string")
        if _tensor_numel(self.shard, "shard") != self.shard_numel:
            raise ValueError("shard tensor numel must equal shard_numel")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class ShardUpdateRule(Protocol):
    """Apply one optimizer update to the valid prefix of a local shard."""

    name: str

    def update(
        self,
        parameter_shard: Any,
        gradient_shard: Any,
        state: Any,
        *,
        valid_numel: int,
        step: int,
    ) -> Any: ...


class SgdShardUpdateRule:
    """Stateless SGD update that operates directly on rank-local storage."""

    name = "sgd"

    def __init__(self, learning_rate: float) -> None:
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a finite positive number")
        if not isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        self._learning_rate = float(learning_rate)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def update(
        self,
        parameter_shard: Any,
        gradient_shard: Any,
        state: Any,
        *,
        valid_numel: int,
        step: int,
    ) -> Any:
        del state, step
        parameter_shard[:valid_numel].add_(
            gradient_shard[:valid_numel],
            alpha=-self._learning_rate,
        )
        return parameter_shard


class ShardedOptimizerConsumer:
    """Validate and consume a ``ReducedShard`` without restoring full gradients."""

    def __init__(
        self,
        *,
        layout: FlatShardLayout,
        parameter_shard: Any,
        update_rule: ShardUpdateRule,
        layout_version: int = 0,
        state: Any = None,
    ) -> None:
        if not isinstance(layout, FlatShardLayout):
            raise TypeError("layout must be a FlatShardLayout")
        _require_nonnegative_integer(layout_version, "layout_version")
        if not isinstance(update_rule, ShardUpdateRule):
            raise TypeError("update_rule must implement ShardUpdateRule")
        if _tensor_numel(parameter_shard, "parameter shard") != layout.shard_numel:
            raise ValueError("parameter shard numel must equal layout shard_numel")
        _require_contiguous(parameter_shard, "parameter shard")
        self._layout = layout
        self._parameter_shard = parameter_shard
        self._update_rule = update_rule
        self._layout_version = layout_version
        self._state = state

    @property
    def parameter_shard(self) -> Any:
        return self._parameter_shard

    def consume(self, reduced: ReducedShard, *, step: int) -> UpdatedParameterShard:
        """Apply a validated reduced gradient shard and return updated storage."""

        _require_positive_integer(step, "step")
        if not isinstance(reduced, ReducedShard):
            raise TypeError("reduced must be a ReducedShard")
        try:
            self._layout.validate_reduced_shard(reduced)
        except ValueError as exc:
            raise ValueError(f"ReducedShard does not match optimizer layout: {exc}") from exc
        _require_contiguous(reduced.shard, "gradient shard")
        _require_matching_tensor_property(
            self._parameter_shard,
            reduced.shard,
            "dtype",
        )
        _require_matching_tensor_property(
            self._parameter_shard,
            reduced.shard,
            "device",
        )

        updated = self._update_rule.update(
            self._parameter_shard,
            reduced.shard,
            self._state,
            valid_numel=self._layout.valid_numel,
            step=step,
        )
        if updated is not self._parameter_shard:
            raise ValueError("update_rule must update and return parameter_shard in place")
        if self._layout.padding_numel:
            self._parameter_shard[self._layout.valid_numel :].zero_()
        return UpdatedParameterShard(
            shard=self._parameter_shard,
            shard_index=self._layout.shard_index,
            shard_numel=self._layout.shard_numel,
            valid_numel=self._layout.valid_numel,
            original_numel=self._layout.original_numel,
            padded_numel=self._layout.padded_numel,
            world_size=self._layout.world_size,
            dtype=self._layout.dtype,
            layout_version=self._layout_version,
            metadata={"update_rule": self._update_rule.name, "step": step},
        )


def _tensor_numel(tensor: Any, name: str) -> int:
    numel = getattr(tensor, "numel", None)
    if not callable(numel):
        raise TypeError(f"{name} must expose numel()")
    return int(numel())


def _require_contiguous(tensor: Any, name: str) -> None:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if callable(is_contiguous) and not bool(is_contiguous()):
        raise ValueError(f"{name} must be contiguous")


def _require_matching_tensor_property(left: Any, right: Any, name: str) -> None:
    left_value = getattr(left, name, None)
    right_value = getattr(right, name, None)
    if left_value is not None and right_value is not None and left_value != right_value:
        raise ValueError(f"gradient shard {name} must match parameter shard {name}")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "SgdShardUpdateRule",
    "ShardUpdateRule",
    "ShardedOptimizerConsumer",
    "UpdatedParameterShard",
]
