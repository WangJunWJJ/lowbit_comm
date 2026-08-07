"""Framework-neutral example adapter for ReducedShard AdamW training steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ccdl_comm.communication import ShardedStepPipeline
from ccdl_comm.optim import AdamWShardUpdateRule, ShardedOptimizerConsumer
from ccdl_comm.shard_layout import FlatShardLayout
from examples.training.compressed_sharded_optimizer import TorchFlatParameterStorage


@dataclass(frozen=True, slots=True)
class ShardedAdamWStepMetrics:
    """Numerical metadata emitted by one completed optimizer step."""

    gradient_norm: float | None
    clip_coefficient: float


class TorchShardedAdamWStep:
    """Reduce gradients, update one AdamW shard, and restore flat parameters."""

    def __init__(
        self,
        *,
        storage: TorchFlatParameterStorage,
        reduce_scatter: Callable[..., Any],
        restore: Any,
        update_rule: AdamWShardUpdateRule,
        state: dict[str, Any],
        global_l2_norm: Callable[[Any], float] | None = None,
    ) -> None:
        if not isinstance(storage, TorchFlatParameterStorage):
            raise TypeError("storage must be TorchFlatParameterStorage")
        if not callable(reduce_scatter):
            raise TypeError("reduce_scatter must be callable")
        if not callable(getattr(restore, "restore", None)):
            raise TypeError("restore must expose restore()")
        if global_l2_norm is not None and not callable(global_l2_norm):
            raise TypeError("global_l2_norm must be callable")
        self._storage = storage
        self._reduce_scatter = reduce_scatter
        self._restore = restore
        self._update_rule = update_rule
        self._state = state
        self._global_l2_norm = global_l2_norm
        self._flat_gradients = storage.padded_flat.new_zeros(
            (storage.layout.padded_numel,)
        )
        self._reduced_output = storage.local_shard.new_empty(
            (storage.layout.shard_numel,)
        )
        self._consumer = ShardedOptimizerConsumer(
            layout=storage.layout,
            parameter_shard=storage.local_shard,
            update_rule=update_rule,
            state=state,
        )
        self._pipeline = ShardedStepPipeline(
            consumer_for_bucket=lambda _bucket_id: self._consumer,
            restore_for_bucket=lambda _bucket_id: self._restore,
            max_inflight=1,
        )

    @classmethod
    def from_parameters(
        cls,
        parameters: Iterable[Any],
        *,
        rank: int,
        world_size: int,
        group_size: int,
        learning_rate: float,
        reduce_scatter: Callable[..., Any],
        restore: Any,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1.0e-8,
        weight_decay: float = 0.01,
        weight_decays: Iterable[float] | None = None,
        global_l2_norm: Callable[[Any], float] | None = None,
    ) -> "TorchShardedAdamWStep":
        """Build aligned flat storage and rank-local AdamW state."""

        active = tuple(parameters)
        if any(not bool(getattr(parameter, "requires_grad", False)) for parameter in active):
            raise ValueError("all sharded AdamW parameters must have requires_grad=True")
        decay_values = None if weight_decays is None else tuple(weight_decays)
        if decay_values is not None and len(decay_values) != len(active):
            raise ValueError("weight_decays must match the parameter count")
        if decay_values is not None:
            decay_values = tuple(
                _require_nonnegative_finite(value, "weight_decays")
                for value in decay_values
            )
        storage = TorchFlatParameterStorage.from_parameters(
            active,
            rank=rank,
            world_size=world_size,
            group_size=group_size,
        )
        state: dict[str, Any] = {}
        if decay_values is not None:
            flat_decay = storage.padded_flat.new_zeros((storage.layout.padded_numel,))
            for parameter_slice, decay in zip(
                storage.layout.parameters,
                decay_values,
                strict=True,
            ):
                flat_decay.narrow(
                    0,
                    parameter_slice.offset,
                    parameter_slice.numel,
                ).fill_(decay)
            state["weight_decay"] = flat_decay.narrow(
                0,
                storage.layout.shard_offset,
                storage.layout.shard_numel,
            ).clone()
        return cls(
            storage=storage,
            reduce_scatter=reduce_scatter,
            restore=restore,
            update_rule=AdamWShardUpdateRule(
                learning_rate,
                betas=betas,
                epsilon=epsilon,
                weight_decay=weight_decay,
            ),
            state=state,
            global_l2_norm=global_l2_norm,
        )

    @property
    def layout(self) -> FlatShardLayout:
        return self._storage.layout

    @property
    def parameters(self) -> tuple[Any, ...]:
        """Return the rebound model parameters in flat-layout order."""

        return self._storage.parameters

    @property
    def optimizer_state_numel(self) -> int:
        return sum(
            int(value.numel())
            for name, value in self._state.items()
            if name in {"exp_avg", "exp_avg_sq"}
        )

    def step(
        self,
        *,
        step: int,
        learning_rate: float | None = None,
        max_grad_norm: float | None = None,
    ) -> ShardedAdamWStepMetrics:
        """Execute one ordered ReducedShard AdamW and parameter-restore step."""

        missing = [
            index
            for index, parameter in enumerate(self._storage.parameters)
            if parameter.grad is None
        ]
        if missing:
            raise RuntimeError(f"missing gradients for parameter indices: {missing}")
        if learning_rate is not None:
            self._update_rule.set_learning_rate(learning_rate)
        if max_grad_norm is not None:
            _require_positive_finite(max_grad_norm, "max_grad_norm")

        gradients = self._storage.flatten_gradients(out=self._flat_gradients)
        reduced = self._reduce_scatter(
            gradients,
            out=self._reduced_output,
            layout=self._storage.layout,
        )
        reduced = self._storage.layout.bind_reduced_shard(reduced)
        gradient_norm, clip_coefficient = self._clip_reduced_gradient(
            reduced.shard,
            max_grad_norm=max_grad_norm,
        )
        self._pipeline.consume_bucket(
            "model",
            reduced,
            parameter_view=self._storage.padded_flat,
            step=step,
        )
        self._pipeline.finish_step()
        return ShardedAdamWStepMetrics(
            gradient_norm=gradient_norm,
            clip_coefficient=clip_coefficient,
        )

    def workspace_pointers(self) -> dict[str, int]:
        """Return stable buffer identities for allocation regression gates."""

        return {
            **self._storage.buffer_pointers(),
            "flat_gradients": int(self._flat_gradients.data_ptr()),
            "reduced_output": int(self._reduced_output.data_ptr()),
        }

    def _clip_reduced_gradient(
        self,
        gradient_shard: Any,
        *,
        max_grad_norm: float | None,
    ) -> tuple[float | None, float]:
        if max_grad_norm is None:
            return None, 1.0
        if self._global_l2_norm is None:
            if self._storage.layout.world_size != 1:
                raise RuntimeError(
                    "global_l2_norm is required to clip distributed ReducedShard gradients"
                )
            valid = gradient_shard[: self._storage.layout.valid_numel]
            norm = float(valid.float().square().sum().sqrt())
        else:
            norm = float(self._global_l2_norm(gradient_shard))
        if not isfinite(norm) or norm < 0:
            raise FloatingPointError("global gradient norm must be finite and nonnegative")
        coefficient = min(1.0, float(max_grad_norm) / (norm + 1.0e-6))
        if coefficient < 1.0:
            gradient_shard[: self._storage.layout.valid_numel].mul_(coefficient)
        return norm, coefficient


def _require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _require_nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain finite nonnegative numbers")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"{name} must contain finite nonnegative numbers")
    return result


__all__ = ["ShardedAdamWStepMetrics", "TorchShardedAdamWStep"]
