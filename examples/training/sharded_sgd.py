"""Reusable Torch buffers for consuming CCDL ReducedShard gradients."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import isfinite
from typing import Any

from ccdl_comm.shard import ReducedShard
from ccdl_comm.shard_layout import FlatParameterSlice, FlatShardLayout


def compile_torch_shard_layout(
    parameters: Iterable[Any],
    *,
    rank: int,
    world_size: int,
) -> FlatShardLayout:
    """Compile a deterministic flat layout from an ordered parameter iterable."""

    active_parameters = tuple(parameters)
    if not active_parameters:
        raise ValueError("parameters must be non-empty")
    first = active_parameters[0]
    dtype = _dtype_name(first.dtype)
    device = first.device
    slices = []
    offset = 0
    for index, parameter in enumerate(active_parameters):
        if parameter.dtype != first.dtype:
            raise ValueError("all parameters must use the same dtype")
        if parameter.device != device:
            raise ValueError("all parameters must use the same device")
        numel = int(parameter.numel())
        slices.append(
            FlatParameterSlice(
                index=index,
                offset=offset,
                numel=numel,
                shape=tuple(int(value) for value in parameter.shape),
                dtype=dtype,
                requires_grad=bool(parameter.requires_grad),
            )
        )
        offset += numel
    shard_numel = (offset + world_size - 1) // world_size
    return FlatShardLayout(
        original_numel=offset,
        padded_numel=shard_numel * world_size,
        shard_numel=shard_numel,
        world_size=world_size,
        shard_index=rank,
        parameters=tuple(slices),
    )


class TorchShardedSgdConsumer:
    """Apply a reduced gradient shard and restore replicated model parameters."""

    def __init__(
        self,
        parameters: Iterable[Any],
        *,
        layout: FlatShardLayout,
        learning_rate: float,
        all_gather_into_tensor: Callable[[Any, Any], Any],
        torch: Any,
    ) -> None:
        self._parameters = tuple(parameters)
        if not self._parameters:
            raise ValueError("parameters must be non-empty")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a finite positive number")
        if not callable(all_gather_into_tensor):
            raise TypeError("all_gather_into_tensor must be callable")
        self._layout = layout
        self._learning_rate = float(learning_rate)
        self._all_gather_into_tensor = all_gather_into_tensor
        self._torch = torch
        self._validate_parameters()

        first = self._parameters[0]
        self._flat_gradients = first.new_zeros((layout.padded_numel,))
        self._reduced_gradient = first.new_empty((layout.shard_numel,))
        self._local_parameters = first.new_zeros((layout.shard_numel,))
        self._gathered_parameters = first.new_zeros((layout.padded_numel,))
        self._initialize_parameter_buffers()

    @property
    def layout(self) -> FlatShardLayout:
        return self._layout

    @property
    def local_parameters(self) -> Any:
        return self._local_parameters

    def reduced_output(self) -> Any:
        """Return the stable caller-owned output for compressed reduce-scatter."""

        return self._reduced_gradient

    def flatten_gradients(self) -> Any:
        """Copy gradients into one padded stable buffer, zeroing absent entries."""

        with self._torch.no_grad():
            self._flat_gradients.zero_()
            for parameter, parameter_slice in zip(
                self._parameters,
                self._layout.parameters,
                strict=True,
            ):
                gradient = parameter.grad
                if gradient is None:
                    continue
                target = self._flat_gradients.narrow(
                    0,
                    parameter_slice.offset,
                    parameter_slice.numel,
                )
                target.copy_(gradient.detach().reshape(-1))
        return self._flat_gradients

    def update_local(self, reduced: ReducedShard) -> Any:
        """Apply only the valid portion of a validated local gradient shard."""

        self._layout.validate_reduced_shard(reduced)
        shard = reduced.shard
        if shard.dtype != self._local_parameters.dtype:
            raise ValueError("ReducedShard tensor dtype must match parameters")
        if shard.device != self._local_parameters.device:
            raise ValueError("ReducedShard tensor device must match parameters")
        is_contiguous = getattr(shard, "is_contiguous", None)
        if not callable(is_contiguous) or not bool(is_contiguous()):
            raise ValueError("ReducedShard tensor must be contiguous")
        valid_numel = self._layout.valid_numel
        with self._torch.no_grad():
            self._local_parameters[:valid_numel].add_(
                shard[:valid_numel],
                alpha=-self._learning_rate,
            )
            if self._layout.padding_numel:
                self._local_parameters[valid_numel:].zero_()
        return self._local_parameters

    def gather_parameters(self) -> Any:
        """Gather updated fixed-size parameter shards into a contiguous buffer."""

        work = self._all_gather_into_tensor(
            self._gathered_parameters,
            self._local_parameters,
        )
        wait = getattr(work, "wait", None)
        if callable(wait):
            wait()
        return self._gathered_parameters

    def writeback_parameters(self) -> Any:
        """Copy the gathered logical prefix back to the model parameters."""

        with self._torch.no_grad():
            for parameter, parameter_slice in zip(
                self._parameters,
                self._layout.parameters,
                strict=True,
            ):
                source = self._gathered_parameters.narrow(
                    0,
                    parameter_slice.offset,
                    parameter_slice.numel,
                ).reshape(parameter_slice.shape)
                parameter.copy_(source)
        return self._gathered_parameters.narrow(0, 0, self._layout.original_numel)

    def consume(self, reduced: ReducedShard) -> Any:
        """Update, gather, and publish model parameters in stream order."""

        self.update_local(reduced)
        self.gather_parameters()
        return self.writeback_parameters()

    def buffer_pointers(self) -> dict[str, int]:
        """Expose stable storage identities for benchmark verification."""

        return {
            "flat_gradients": int(self._flat_gradients.data_ptr()),
            "reduced_gradient": int(self._reduced_gradient.data_ptr()),
            "local_parameters": int(self._local_parameters.data_ptr()),
            "gathered_parameters": int(self._gathered_parameters.data_ptr()),
        }

    def _validate_parameters(self) -> None:
        if len(self._parameters) != len(self._layout.parameters):
            raise ValueError("parameters do not match layout")
        for parameter, parameter_slice in zip(
            self._parameters,
            self._layout.parameters,
            strict=True,
        ):
            if int(parameter.numel()) != parameter_slice.numel:
                raise ValueError("parameter numel does not match layout")
            if tuple(int(value) for value in parameter.shape) != parameter_slice.shape:
                raise ValueError("parameter shape does not match layout")
            if _dtype_name(parameter.dtype) != parameter_slice.dtype:
                raise ValueError("parameter dtype does not match layout")
            if bool(parameter.requires_grad) != parameter_slice.requires_grad:
                raise ValueError("parameter requires_grad does not match layout")

    def _initialize_parameter_buffers(self) -> None:
        with self._torch.no_grad():
            for parameter, parameter_slice in zip(
                self._parameters,
                self._layout.parameters,
                strict=True,
            ):
                target = self._gathered_parameters.narrow(
                    0,
                    parameter_slice.offset,
                    parameter_slice.numel,
                )
                target.copy_(parameter.detach().reshape(-1))
            if self._layout.padded_numel > self._layout.original_numel:
                self._gathered_parameters[self._layout.original_numel :].zero_()
            valid_numel = self._layout.valid_numel
            if valid_numel:
                self._local_parameters[:valid_numel].copy_(
                    self._gathered_parameters[
                        self._layout.shard_offset : self._layout.shard_end
                    ]
                )
            if self._layout.padding_numel:
                self._local_parameters[valid_numel:].zero_()


def _dtype_name(dtype: object) -> str:
    normalized = str(dtype).lower().removeprefix("torch.")
    return {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        "float64": "fp64",
        "double": "fp64",
    }.get(normalized, normalized)
