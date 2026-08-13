"""Optimizer-side consumers for reduced parameter shards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Protocol, runtime_checkable

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

    def set_learning_rate(self, learning_rate: float) -> None:
        """Update the step learning rate after validating scheduler output."""

        self._learning_rate = _finite_number(
            learning_rate,
            "learning_rate",
            positive=True,
        )

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


class AdamWShardUpdateRule:
    """Apply decoupled AdamW while owning state only for one local shard."""

    name = "adamw"

    def __init__(
        self,
        learning_rate: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1.0e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self._learning_rate = _finite_number(
            learning_rate,
            "learning_rate",
            nonnegative=True,
        )
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise TypeError("betas must be a pair of finite values in [0, 1)")
        beta1 = _finite_number(betas[0], "betas", nonnegative=True)
        beta2 = _finite_number(betas[1], "betas", nonnegative=True)
        if beta1 >= 1.0 or beta2 >= 1.0:
            raise ValueError("betas must be finite values in [0, 1)")
        self._betas = (beta1, beta2)
        self._epsilon = _finite_number(epsilon, "epsilon", positive=True)
        self._weight_decay = _finite_number(
            weight_decay,
            "weight_decay",
            nonnegative=True,
        )

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        """Update the step learning rate after validating scheduler output."""

        self._learning_rate = _finite_number(
            learning_rate,
            "learning_rate",
            nonnegative=True,
        )

    def update(
        self,
        parameter_shard: Any,
        gradient_shard: Any,
        state: Any,
        *,
        valid_numel: int,
        step: int,
    ) -> Any:
        if not isinstance(state, MutableMapping):
            raise TypeError("AdamW state must be a mutable mapping")
        previous_step = state.get("step", 0)
        if previous_step != step - 1:
            raise ValueError("AdamW step must be consecutive for the local shard")

        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None and exp_avg_sq is None:
            new_zeros = getattr(parameter_shard, "new_zeros", None)
            if not callable(new_zeros):
                raise TypeError("parameter shard must expose new_zeros()")
            exp_avg = new_zeros(parameter_shard.shape)
            exp_avg_sq = new_zeros(parameter_shard.shape)
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq
        elif exp_avg is None or exp_avg_sq is None:
            raise ValueError("AdamW state must contain both moment tensors")
        expected_numel = _tensor_numel(parameter_shard, "parameter shard")
        if (
            _tensor_numel(exp_avg, "exp_avg") != expected_numel
            or _tensor_numel(exp_avg_sq, "exp_avg_sq") != expected_numel
        ):
            raise ValueError("AdamW moment tensors must match the parameter shard")

        parameter = parameter_shard[:valid_numel]
        gradient = gradient_shard[:valid_numel]
        first_moment = exp_avg[:valid_numel]
        second_moment = exp_avg_sq[:valid_numel]
        beta1, beta2 = self._betas

        weight_decay = state.get("weight_decay")
        if weight_decay is None:
            parameter.mul_(1.0 - self._learning_rate * self._weight_decay)
        else:
            if _tensor_numel(weight_decay, "weight_decay") != expected_numel:
                raise ValueError("AdamW weight_decay tensor must match the parameter shard")
            _require_matching_tensor_property(
                parameter_shard,
                weight_decay,
                "dtype",
            )
            _require_matching_tensor_property(
                parameter_shard,
                weight_decay,
                "device",
            )
            parameter.mul_(
                1.0 - self._learning_rate * weight_decay[:valid_numel]
            )
        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        bias_correction1 = 1.0 - beta1**step
        bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
        denominator = second_moment.sqrt().div_(bias_correction2_sqrt)
        denominator.add_(self._epsilon)
        parameter.addcdiv_(
            first_moment,
            denominator,
            value=-self._learning_rate / bias_correction1,
        )
        state["step"] = step
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
        gradient_transform: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        if not isinstance(layout, FlatShardLayout):
            raise TypeError("layout must be a FlatShardLayout")
        _require_nonnegative_integer(layout_version, "layout_version")
        if not isinstance(update_rule, ShardUpdateRule):
            raise TypeError("update_rule must implement ShardUpdateRule")
        if gradient_transform is not None and not callable(gradient_transform):
            raise TypeError("gradient_transform must be callable")
        if _tensor_numel(parameter_shard, "parameter shard") != layout.shard_numel:
            raise ValueError("parameter shard numel must equal layout shard_numel")
        _require_contiguous(parameter_shard, "parameter shard")
        self._layout = layout
        self._parameter_shard = parameter_shard
        self._update_rule = update_rule
        self._layout_version = layout_version
        self._state = state
        self._gradient_transform = gradient_transform

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
        gradient_shard = reduced.shard
        if self._gradient_transform is not None:
            gradient_shard = self._gradient_transform(
                gradient_shard,
                self._parameter_shard,
            )
        if _tensor_numel(gradient_shard, "gradient shard") != self._layout.shard_numel:
            raise ValueError("transformed gradient shard must match layout shard_numel")
        _require_contiguous(gradient_shard, "gradient shard")
        _require_matching_tensor_property(
            self._parameter_shard,
            gradient_shard,
            "dtype",
        )
        _require_matching_tensor_property(
            self._parameter_shard,
            gradient_shard,
            "device",
        )

        updated = self._update_rule.update(
            self._parameter_shard,
            gradient_shard,
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


def _finite_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


__all__ = [
    "AdamWShardUpdateRule",
    "SgdShardUpdateRule",
    "ShardUpdateRule",
    "ShardedOptimizerConsumer",
    "UpdatedParameterShard",
]
