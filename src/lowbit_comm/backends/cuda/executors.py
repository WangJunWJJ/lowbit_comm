"""Pre-bound CUDA executors with no hot-path strategy lookup."""

from __future__ import annotations

from functools import reduce
from importlib import import_module
from operator import mul
from typing import Any

from lowbit_comm.core import DataType, ReducedShard, ReducedShardValue
from lowbit_comm.core.lowered import LoweredProgram
from lowbit_comm.runtime import CompletionWork, ImmediateCompletionEvent

from .codec import payload_nbytes, quantize_into
from .loader import CudaExtensionStatus
from .transports import ShardPlan, compile_shard_plan


class _CollectiveEvent:
    def __init__(self, handle: object) -> None:
        self._handle = handle

    def query(self) -> bool:
        query = getattr(self._handle, "is_completed", None)
        return bool(query()) if callable(query) else False

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        self._handle.wait()
        return True


class CudaReducedShardExecutable:
    def __init__(
        self,
        lowered: LoweredProgram,
        extension_status: CudaExtensionStatus,
    ) -> None:
        output = lowered.program.output
        if not isinstance(output, ReducedShard):
            raise TypeError("CUDA reduced-shard executable requires ReducedShard output")
        wire = lowered.program.wire
        self.lowered = lowered
        self._status = extension_status
        self._module = _require_module(extension_status)
        self._fused = _require_callable(
            self._module,
            "inplace_dequantize_reduce_mean",
        )
        self._wire = wire
        self._output_type = output
        self.plan = compile_shard_plan(
            original_numel=reduce(mul, lowered.context.shape, 1),
            rank=lowered.context.rank,
            world_size=lowered.context.world_size,
            group_size=wire.group_size,
        )
        self.payload_numel = payload_nbytes(
            self.plan.shard_numel,
            dtype=lowered.context.dtype,
            wire=wire,
        )
        self.payload_stride = _align(self.payload_numel, 16)

    def run(self, value: Any) -> CompletionWork[ReducedShardValue]:
        torch = import_module("torch")
        dist = import_module("torch.distributed")
        flat = value.reshape(-1)
        if int(flat.numel()) != self.plan.original_numel:
            raise ValueError("input numel differs from the compiled shape")
        padded = flat.new_zeros((self.plan.padded_numel,))
        padded[: self.plan.original_numel].copy_(flat)
        send = torch.empty(
            (self.plan.world_size, self.payload_stride),
            device=flat.device,
            dtype=torch.uint8,
        )
        for destination in range(self.plan.world_size):
            source = padded.narrow(
                0,
                destination * self.plan.shard_numel,
                self.plan.shard_numel,
            )
            quantize_into(
                source,
                send[destination, : self.payload_numel],
                self._wire,
                extension_status=self._status,
            )
        received = torch.empty_like(send)
        handle = dist.all_to_all_single(
            received,
            send,
            group=self.lowered.bindings.process_group,
            async_op=True,
        )
        output = flat.new_empty((self.plan.shard_numel,))

        def complete() -> ReducedShardValue:
            payloads = [
                received[index, : self.payload_numel]
                for index in range(self.plan.world_size)
            ]
            used = self._fused(
                payloads,
                output,
                self._wire.group_size,
                0,
                self._wire.bit,
                _quant_type(self._module, self._wire.quant_type),
                self._wire.compact,
                self.plan.world_size,
            )
            if not used:
                raise RuntimeError("fused dequant-reduce-mean declined compiled payloads")
            return ReducedShardValue(
                tensor=output,
                shard_index=self.plan.rank,
                shard_numel=self.plan.shard_numel,
                original_shape=self.lowered.context.shape,
                original_numel=self.plan.original_numel,
                world_size=self.plan.world_size,
                reduction="mean",
                dtype=self.lowered.context.dtype,
                layout_version=self._output_type.layout_version,
            )

        event = _CollectiveEvent(handle) if handle is not None else ImmediateCompletionEvent()
        return CompletionWork(
            None,  # type: ignore[arg-type]
            event=event,
            complete=complete,
            resources=(padded, send, received, output),
        )


def _require_module(status: CudaExtensionStatus) -> object:
    if not status.available or status.module is None:
        raise RuntimeError(status.reason or "CUDA extension is unavailable")
    return status.module


def _require_callable(module: object, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise RuntimeError(f"CUDA extension missing required symbol: {name}")
    return value


def _quant_type(module: object, name: str) -> object:
    enum_name = {
        "linear": "Linear",
        "normal": "Normal",
        "uniform": "Uniform",
        "e3m0": "E3M0",
        "e2m1": "E2M1",
    }[name]
    return getattr(module.QuantType, enum_name)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment if value else 0
