"""Rank-local authoritative optimizer state for ReducedShard consumers."""

from __future__ import annotations

from typing import Any, Mapping

from lowbit_comm.core import ReducedShardValue


class ShardedMasterState:
    """Own one FP32 master shard and its checkpoint compatibility facts."""

    def __init__(
        self,
        *,
        master: Any,
        layout_version: int,
        world_size: int,
        rank: int,
    ) -> None:
        if "float32" not in str(getattr(master, "dtype", "")):
            raise TypeError("master shard must use float32")
        if layout_version < 0:
            raise ValueError("layout_version must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be within world_size")
        self.master = master
        self.layout_version = layout_version
        self.world_size = world_size
        self.rank = rank
        self.step = 0
        self.exp_avg = master.detach().clone().zero_()
        self.exp_avg_sq = master.detach().clone().zero_()
        self.requires_fp_refresh = False

    def sgd_step(self, gradient: ReducedShardValue, *, learning_rate: float) -> None:
        if not isinstance(gradient, ReducedShardValue):
            raise TypeError("gradient must be a ReducedShardValue")
        if gradient.layout_version != self.layout_version:
            raise ValueError("gradient layout_version does not match master shard")
        if gradient.world_size != self.world_size:
            raise ValueError("gradient world_size does not match master shard")
        if gradient.shard_index != self.rank:
            raise ValueError("gradient shard_index does not match rank")
        if int(self.master.numel()) != gradient.shard_numel:
            raise ValueError("gradient shard_numel does not match master shard")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        valid = gradient.valid_numel
        self.master[:valid].add_(
            gradient.tensor[:valid].float(),
            alpha=-float(learning_rate),
        )
        self.step += 1

    def adamw_step(
        self,
        gradient: ReducedShardValue,
        *,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1.0e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self._validate_gradient(gradient)
        if learning_rate < 0 or epsilon <= 0 or weight_decay < 0:
            raise ValueError("AdamW hyperparameters are invalid")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("AdamW betas must be within [0, 1)")
        valid = gradient.valid_numel
        parameter = self.master[:valid]
        grad = gradient.tensor[:valid].float()
        first = self.exp_avg[:valid]
        second = self.exp_avg_sq[:valid]
        self.step += 1
        parameter.mul_(1.0 - learning_rate * weight_decay)
        first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        correction1 = 1.0 - beta1**self.step
        correction2 = 1.0 - beta2**self.step
        denominator = second.sqrt().div_(correction2**0.5).add_(epsilon)
        parameter.addcdiv_(first, denominator, value=-learning_rate / correction1)

    def state_dict(self) -> dict[str, object]:
        return {
            "master": self.master.detach().clone(),
            "layout_version": self.layout_version,
            "world_size": self.world_size,
            "rank": self.rank,
            "step": self.step,
            "exp_avg": self.exp_avg.detach().clone(),
            "exp_avg_sq": self.exp_avg_sq.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for name in ("layout_version", "world_size", "rank"):
            if state.get(name) != getattr(self, name):
                raise ValueError(f"checkpoint {name} does not match runtime")
        master = state.get("master")
        if int(master.numel()) != int(self.master.numel()):  # type: ignore[union-attr]
            raise ValueError("checkpoint master shard shape does not match runtime")
        self.master.copy_(master)
        self.step = int(state.get("step", 0))
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if value is not None:
                getattr(self, name).copy_(value)
        self.requires_fp_refresh = True

    def mark_fp_refreshed(self) -> None:
        self.requires_fp_refresh = False

    def _validate_gradient(self, gradient: ReducedShardValue) -> None:
        if not isinstance(gradient, ReducedShardValue):
            raise TypeError("gradient must be a ReducedShardValue")
        if gradient.layout_version != self.layout_version:
            raise ValueError("gradient layout_version does not match master shard")
        if gradient.world_size != self.world_size:
            raise ValueError("gradient world_size does not match master shard")
        if gradient.shard_index != self.rank:
            raise ValueError("gradient shard_index does not match rank")
        if int(self.master.numel()) != gradient.shard_numel:
            raise ValueError("gradient shard_numel does not match master shard")
