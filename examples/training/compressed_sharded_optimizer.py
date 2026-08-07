"""Shared flat parameter storage for compressed sharded training examples."""

from __future__ import annotations

from collections.abc import Iterable
from math import lcm
from typing import Any

from ccdl_comm.shard_layout import FlatShardLayout
from examples.training.sharded_sgd import compile_torch_shard_layout


class TorchFlatParameterStorage:
    """Own padded flat storage directly viewed by all model parameters."""

    def __init__(
        self,
        *,
        parameters: tuple[Any, ...],
        padded_flat: Any,
        layout: FlatShardLayout,
    ) -> None:
        self._parameters = parameters
        self._padded_flat = padded_flat
        self._layout = layout

    @classmethod
    def from_parameters(
        cls,
        parameters: Iterable[Any],
        *,
        rank: int,
        world_size: int,
        group_size: int = 64,
    ) -> "TorchFlatParameterStorage":
        """Copy and atomically rebind homogeneous parameters to aligned storage."""

        _require_nonnegative_integer(rank, "rank")
        _require_positive_integer(world_size, "world_size")
        _require_positive_integer(group_size, "group_size")
        if rank >= world_size:
            raise ValueError("rank must be smaller than world_size")

        active = tuple(parameters)
        base_layout = compile_torch_shard_layout(
            active,
            rank=rank,
            world_size=world_size,
        )
        _validate_homogeneous_parameters(active)

        shard_alignment = lcm(group_size, 512)
        shard_numel = (
            _ceil_div(
                base_layout.original_numel,
                world_size * shard_alignment,
            )
            * shard_alignment
        )
        layout = FlatShardLayout(
            original_numel=base_layout.original_numel,
            padded_numel=shard_numel * world_size,
            shard_numel=shard_numel,
            world_size=world_size,
            shard_index=rank,
            parameters=base_layout.parameters,
        )
        padded_flat = active[0].new_zeros((layout.padded_numel,))

        for parameter, parameter_slice in zip(
            active,
            layout.parameters,
            strict=True,
        ):
            padded_flat.narrow(
                0,
                parameter_slice.offset,
                parameter_slice.numel,
            ).copy_(parameter.detach().reshape(-1))

        original_data = tuple(parameter.data for parameter in active)
        try:
            for parameter, parameter_slice in zip(
                active,
                layout.parameters,
                strict=True,
            ):
                parameter.data = padded_flat.narrow(
                    0,
                    parameter_slice.offset,
                    parameter_slice.numel,
                ).view(parameter_slice.shape)
        except Exception:
            for parameter, previous in zip(active, original_data, strict=True):
                parameter.data = previous
            raise

        return cls(
            parameters=active,
            padded_flat=padded_flat,
            layout=layout,
        )

    @property
    def layout(self) -> FlatShardLayout:
        return self._layout

    @property
    def padded_flat(self) -> Any:
        return self._padded_flat

    @property
    def original_numel(self) -> int:
        return self._layout.original_numel

    @property
    def local_shard(self) -> Any:
        return self._padded_flat.narrow(
            0,
            self._layout.shard_offset,
            self._layout.shard_numel,
        )

    @property
    def parameters(self) -> tuple[Any, ...]:
        return self._parameters


def _validate_homogeneous_parameters(parameters: tuple[Any, ...]) -> None:
    first = parameters[0]
    first_dtype = first.dtype
    first_device = first.device
    for parameter in parameters:
        if parameter.dtype != first_dtype:
            raise ValueError("all parameters must use the same dtype")
        if parameter.device != first_device:
            raise ValueError("all parameters must use the same device")
        if getattr(parameter, "layout", None) != getattr(first, "layout", None):
            raise ValueError("all parameters must use the same tensor layout")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_integer(value: object, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be > 0")


__all__ = ["TorchFlatParameterStorage"]
