"""Framework-neutral example adapter for ReducedShard AdamW training steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ccdl_comm.communication import (
    ParameterCommunicationDecision,
    ParameterCommunicationPolicy,
    ParameterDeltaProvider,
    ParameterDeltaShard,
    SafeInt8QWDPolicy,
    TorchParameterDeltaProvider,
)
from ccdl_comm.optim import AdamWShardUpdateRule, ShardedOptimizerConsumer
from ccdl_comm.shard_layout import FlatShardLayout
from examples.training.compressed_sharded_optimizer import TorchFlatParameterStorage


@dataclass(frozen=True, slots=True)
class ShardedAdamWStepMetrics:
    """Numerical metadata emitted by one completed optimizer step."""

    gradient_norm: float | None
    clip_coefficient: float
    parameter_communication_mode: str
    parameter_communication_reason: str
    relative_error: float | None


@dataclass(frozen=True, slots=True)
class ShardedAdamWState:
    """Serializable rank-local AdamW moments for one flat shard."""

    step: int
    layout_version: int
    world_size: int
    shard_index: int
    shard_numel: int
    master_shard: Any
    exp_avg: Any
    exp_avg_sq: Any
    policy_state: Mapping[str, Any]


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
        global_error_ratio: Callable[[Any, Any], float] | None = None,
        policy: ParameterCommunicationPolicy | None = None,
        delta_provider: ParameterDeltaProvider | None = None,
        tensor_role: str = "weight",
        layout_version: int = 0,
    ) -> None:
        if not isinstance(storage, TorchFlatParameterStorage):
            raise TypeError("storage must be TorchFlatParameterStorage")
        if not callable(reduce_scatter):
            raise TypeError("reduce_scatter must be callable")
        for method_name in ("supports_qwd", "restore_delta", "refresh"):
            if not callable(getattr(restore, method_name, None)):
                raise TypeError(f"restore must expose {method_name}()")
        if global_l2_norm is not None and not callable(global_l2_norm):
            raise TypeError("global_l2_norm must be callable")
        if global_error_ratio is not None and not callable(global_error_ratio):
            raise TypeError("global_error_ratio must be callable")
        if policy is not None and not isinstance(policy, ParameterCommunicationPolicy):
            raise TypeError("policy must implement ParameterCommunicationPolicy")
        if delta_provider is not None and not isinstance(
            delta_provider,
            ParameterDeltaProvider,
        ):
            raise TypeError("delta_provider must implement ParameterDeltaProvider")
        if not isinstance(tensor_role, str) or not tensor_role.strip():
            raise TypeError("tensor_role must be a non-empty string")
        if isinstance(layout_version, bool) or not isinstance(layout_version, int):
            raise TypeError("layout_version must be an integer")
        if layout_version < 0:
            raise ValueError("layout_version must be nonnegative")
        self._storage = storage
        self._reduce_scatter = reduce_scatter
        self._restore = restore
        self._update_rule = update_rule
        self._state = state
        self._global_l2_norm = global_l2_norm
        self._global_error_ratio = global_error_ratio
        self._policy = policy if policy is not None else SafeInt8QWDPolicy()
        self._delta_provider = (
            delta_provider
            if delta_provider is not None
            else TorchParameterDeltaProvider()
        )
        self._tensor_role = tensor_role
        self._layout_version = layout_version
        self._master_shard = storage.local_shard.detach().float().clone()
        if storage.layout.padding_numel:
            self._master_shard.narrow(
                0,
                storage.layout.valid_numel,
                storage.layout.padding_numel,
            ).zero_()
        self._delta_workspace = self._master_shard.new_empty(
            self._master_shard.shape
        )
        self._residual_workspace = self._master_shard.new_empty(
            self._master_shard.shape
        )
        self._last_relative_error: float | None = None
        self._refresh_required = False
        self._flat_gradients = storage.padded_flat.new_zeros(
            (storage.layout.padded_numel,)
        )
        self._reduced_output = storage.local_shard.new_empty(
            (storage.layout.shard_numel,)
        )
        self._consumer = ShardedOptimizerConsumer(
            layout=storage.layout,
            parameter_shard=self._master_shard,
            update_rule=update_rule,
            state=state,
            layout_version=layout_version,
            gradient_transform=lambda gradient, master: gradient.to(master.dtype),
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
        global_error_ratio: Callable[[Any, Any], float] | None = None,
        policy: ParameterCommunicationPolicy | None = None,
        delta_provider: ParameterDeltaProvider | None = None,
        tensor_role: str = "weight",
        layout_version: int = 0,
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
            flat_decay = storage.local_shard.float().new_zeros(
                (storage.layout.padded_numel,)
            )
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
            ).clone().float()
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
            global_error_ratio=global_error_ratio,
            policy=policy,
            delta_provider=delta_provider,
            tensor_role=tensor_role,
            layout_version=layout_version,
        )

    @property
    def layout(self) -> FlatShardLayout:
        return self._storage.layout

    @property
    def parameters(self) -> tuple[Any, ...]:
        """Return the rebound model parameters in flat-layout order."""

        return self._storage.parameters

    @property
    def master_shard(self) -> Any:
        """Return the authoritative rank-local FP32 parameter shard."""

        return self._master_shard

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
        updated_master = self._consumer.consume(reduced, step=step)
        capability = bool(
            self._restore.supports_qwd(
                updated_master,
                self._storage.padded_flat,
            )
        )
        decision = self._policy.decide(
            step=step,
            tensor_role=self._tensor_role,
            numel=self._storage.layout.original_numel,
            relative_error=self._last_relative_error,
            capability=capability,
        )
        if self._refresh_required:
            decision = ParameterCommunicationDecision(
                mode="fp_refresh",
                bit=8,
                reason="checkpoint_refresh",
            )
        relative_error = self._communicate_parameters(
            updated_master,
            decision,
            step=step,
        )
        return ShardedAdamWStepMetrics(
            gradient_norm=gradient_norm,
            clip_coefficient=clip_coefficient,
            parameter_communication_mode=decision.mode,
            parameter_communication_reason=decision.reason,
            relative_error=relative_error,
        )

    def workspace_pointers(self) -> dict[str, int]:
        """Return stable buffer identities for allocation regression gates."""

        return {
            **self._storage.buffer_pointers(),
            "master_shard": int(self._master_shard.data_ptr()),
            "delta_workspace": int(self._delta_workspace.data_ptr()),
            "residual_workspace": int(self._residual_workspace.data_ptr()),
            "flat_gradients": int(self._flat_gradients.data_ptr()),
            "reduced_output": int(self._reduced_output.data_ptr()),
        }

    def export_adamw_state(self) -> ShardedAdamWState:
        """Clone optimizer moments for a distributed checkpoint boundary."""

        step = self._state.get("step")
        exp_avg = self._state.get("exp_avg")
        exp_avg_sq = self._state.get("exp_avg_sq")
        if not isinstance(step, int) or step <= 0 or exp_avg is None or exp_avg_sq is None:
            raise RuntimeError("AdamW state is unavailable before the first completed step")
        return ShardedAdamWState(
            step=step,
            layout_version=self._layout_version,
            world_size=self._storage.layout.world_size,
            shard_index=self._storage.layout.shard_index,
            shard_numel=self._storage.layout.shard_numel,
            master_shard=self._master_shard.detach().clone(),
            exp_avg=exp_avg.detach().clone(),
            exp_avg_sq=exp_avg_sq.detach().clone(),
            policy_state={
                "configuration_packet": self._policy_configuration_packet(),
                "last_relative_error": self._last_relative_error,
            },
        )

    def load_adamw_state(self, state: ShardedAdamWState) -> None:
        """Restore validated rank-local moments before the next training step."""

        if not isinstance(state, ShardedAdamWState):
            raise TypeError("state must be ShardedAdamWState")
        if isinstance(state.step, bool) or not isinstance(state.step, int) or state.step <= 0:
            raise ValueError("AdamW state step must be a positive integer")
        expected_layout = (
            self._layout_version,
            self._storage.layout.world_size,
            self._storage.layout.shard_index,
            self._storage.layout.shard_numel,
        )
        received_layout = (
            state.layout_version,
            state.world_size,
            state.shard_index,
            state.shard_numel,
        )
        if received_layout != expected_layout:
            raise ValueError("AdamW state layout does not match the local shard")
        if not isinstance(state.policy_state, Mapping):
            raise TypeError("AdamW policy_state must be a mapping")
        if state.policy_state.get(
            "configuration_packet"
        ) != self._policy_configuration_packet():
            raise ValueError("AdamW policy configuration does not match checkpoint")
        self._validate_fp32_state_tensor(state.master_shard, "master_shard")
        for name, value in (
            ("exp_avg", state.exp_avg),
            ("exp_avg_sq", state.exp_avg_sq),
        ):
            self._validate_fp32_state_tensor(value, name)
        previous_error = state.policy_state.get("last_relative_error")
        if previous_error is not None:
            previous_error = _require_nonnegative_finite(
                previous_error,
                "last_relative_error",
            )
        self._state["step"] = state.step
        self._master_shard.copy_(state.master_shard)
        self._state["exp_avg"] = state.exp_avg.detach().clone()
        self._state["exp_avg_sq"] = state.exp_avg_sq.detach().clone()
        self._last_relative_error = previous_error
        self._refresh_required = True

    def _communicate_parameters(
        self,
        updated_master: Any,
        decision: ParameterCommunicationDecision,
        *,
        step: int,
    ) -> float | None:
        if decision.mode == "fp_refresh":
            work = self._restore.refresh(
                updated_master,
                out=self._storage.padded_flat,
                async_op=True,
            )
            work.wait()
            self._last_relative_error = 0.0
            self._refresh_required = False
            return self._last_relative_error

        delta = self._prepare_delta(updated_master, out=self._delta_workspace)
        sample_error = self._should_sample_error(step)
        delta_norm_sq = (
            delta.shard.narrow(0, 0, delta.valid_numel).square().sum()
            if sample_error
            else None
        )
        work = self._restore.restore_delta(
            delta,
            out=self._storage.padded_flat,
            async_op=True,
        )
        work.wait()
        if sample_error and delta_norm_sq is not None:
            self._last_relative_error = self._measure_relative_error(delta_norm_sq)
        return self._last_relative_error

    def _prepare_delta(self, updated_master: Any, *, out: Any) -> ParameterDeltaShard:
        shard = self._delta_provider.prepare_delta(
            self._master_shard,
            self._storage.local_shard,
            out=out,
            valid_numel=self._storage.layout.valid_numel,
        )
        return ParameterDeltaShard(
            shard=shard,
            shard_index=self._storage.layout.shard_index,
            shard_numel=self._storage.layout.shard_numel,
            valid_numel=self._storage.layout.valid_numel,
            original_numel=self._storage.layout.original_numel,
            padded_numel=self._storage.layout.padded_numel,
            world_size=self._storage.layout.world_size,
            layout_version=self._layout_version,
            metadata=dict(updated_master.metadata),
        )

    def _should_sample_error(self, step: int) -> bool:
        interval = getattr(self._policy, "error_check_interval", None)
        return isinstance(interval, int) and interval > 0 and step % interval == 0

    def _measure_relative_error(self, delta_norm_sq: Any) -> float:
        self._delta_provider.prepare_delta(
            self._master_shard,
            self._storage.local_shard,
            out=self._residual_workspace,
            valid_numel=self._storage.layout.valid_numel,
        )
        residual_norm_sq = self._residual_workspace.narrow(
            0,
            0,
            self._storage.layout.valid_numel,
        ).square().sum()
        if self._global_error_ratio is not None:
            result = float(
                self._global_error_ratio(residual_norm_sq, delta_norm_sq)
            )
        else:
            if self._storage.layout.world_size != 1:
                raise RuntimeError(
                    "global_error_ratio is required for distributed error sampling"
                )
            denominator = float(delta_norm_sq)
            numerator = float(residual_norm_sq)
            result = (numerator / max(denominator, 1.0e-24)) ** 0.5
        return _require_nonnegative_finite(result, "relative_error")

    def _policy_configuration_packet(self) -> tuple[int, ...]:
        packet = getattr(self._policy, "configuration_packet", None)
        if not callable(packet):
            return ()
        values = tuple(packet())
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("policy configuration packet must contain integers")
        return values

    def _validate_fp32_state_tensor(self, value: Any, name: str) -> None:
        if int(value.numel()) != self._storage.layout.shard_numel:
            raise ValueError(f"{name} numel must equal the local shard size")
        if value.dtype != self._master_shard.dtype:
            raise ValueError(f"{name} dtype must be fp32")
        if value.device != self._master_shard.device:
            raise ValueError(f"{name} device must match the master shard")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

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


__all__ = [
    "ShardedAdamWState",
    "ShardedAdamWStepMetrics",
    "TorchShardedAdamWStep",
]
