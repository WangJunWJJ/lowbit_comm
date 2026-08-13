"""Pre-bound CUDA executors with no hot-path strategy lookup."""

from __future__ import annotations

from functools import reduce
from importlib import import_module
from operator import mul
from threading import RLock
from typing import Any

from lowbit_comm.core import (
    DataType,
    FullTensor,
    ReducedShard,
    ReducedShardValue,
    WorkspaceRole,
)
from lowbit_comm.core.lowered import LoweredProgram
from lowbit_comm.runtime import (
    BudgetedWorkspacePool,
    CompletionOutcome,
    CompletionWork,
    ImmediateCompletionEvent,
)

from .codec import dequantize_into, payload_nbytes, quantize_into
from .loader import CudaExtensionStatus
from .transports import compile_shard_plan
from .workspace import CudaWorkspaceManager


class _CollectiveEvent:
    def __init__(self, handle: object) -> None:
        self._handle = handle

    def query(self) -> bool:
        query = getattr(self._handle, "is_completed", None)
        return bool(query()) if callable(query) else False

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None:
            raise NotImplementedError("collective handle does not support timeout")
        self._handle.wait()
        return True


class _CudaRecordedEvent:
    """CompletionEvent adapter for an already recorded CUDA event."""

    def __init__(self, event: object) -> None:
        self._event = event

    def query(self) -> bool:
        return bool(self._event.query())  # type: ignore[attr-defined]

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None:
            raise NotImplementedError("CUDA event wait does not support timeout")
        self._event.synchronize()  # type: ignore[attr-defined]
        return True


class CudaNativeAllReduceExecutable:
    """Explicit native mean all-reduce compiled without CUDA extension use."""

    def __init__(self, lowered: LoweredProgram) -> None:
        self.lowered = lowered

    def run(self, value: Any) -> CompletionWork[Any]:
        dist = import_module("torch.distributed")
        handle = dist.all_reduce(
            value,
            group=self.lowered.bindings.process_group,
            async_op=True,
        )
        def complete() -> CompletionOutcome[Any]:
            if self.lowered.reduction.divisor != 1:
                value.div_(self.lowered.reduction.divisor)
            return CompletionOutcome(value, _record_current_stream(value.device))

        return CompletionWork(
            value,
            event=_CollectiveEvent(handle),
            complete=complete,
            resources=(value,),
        )


class CudaCompressedAllGatherExecutable:
    """Gather full compressed contributions and fuse local dequant-reduce-mean."""

    def __init__(self, lowered: LoweredProgram, extension_status: CudaExtensionStatus) -> None:
        self.lowered = lowered
        self._status = extension_status
        self._module = _require_module(extension_status)
        self._fused = _require_callable(self._module, "inplace_dequantize_reduce_mean")
        self._wire = lowered.program.wire
        self._workspace = _workspace_manager(lowered, self._wire)
        self.original_numel = reduce(mul, lowered.context.shape, 1)
        self.padded_numel = _align(self.original_numel, self._wire.group_size)
        self.payload_numel = payload_nbytes(
            self.original_numel,
            dtype=lowered.context.dtype,
            wire=self._wire,
        )
        self.payload_stride = _align(self.payload_numel, 16)

    def run(self, value: Any) -> CompletionWork[Any]:
        torch = import_module("torch")
        dist = import_module("torch.distributed")
        flat = value.reshape(-1)
        if int(flat.numel()) != self.original_numel:
            raise ValueError("input numel differs from the compiled shape")
        prepared_lease = self._workspace.acquire_role(
            WorkspaceRole.PADDED_INPUT,
            device=flat.device,
            allocator=lambda: flat.new_empty((self.padded_numel,)),
        )
        prepared = prepared_lease.value
        prepared.zero_()
        prepared[: self.original_numel].copy_(flat)
        send_lease = self._workspace.acquire_role(
            WorkspaceRole.SEND,
            device=flat.device,
            allocator=lambda: torch.empty(
                self.payload_stride,
                dtype=torch.uint8,
                device=flat.device,
            ),
        )
        send = send_lease.value
        send.zero_()
        quantize_into(
            prepared,
            send[: self.payload_numel],
            self._wire,
            extension_status=self._status,
        )
        gathered_lease = self._workspace.acquire_role(
            WorkspaceRole.RECEIVE,
            device=flat.device,
            allocator=lambda: torch.empty(
                self.lowered.context.world_size * self.payload_stride,
                dtype=torch.uint8,
                device=flat.device,
            ),
        )
        gathered = gathered_lease.value
        handle = dist.all_gather_into_tensor(
            gathered,
            send,
            group=self.lowered.bindings.process_group,
            async_op=True,
        )
        output = flat.new_empty((self.padded_numel,))
        return CompletionWork(
            None,
            event=_CollectiveEvent(handle),
            complete=lambda: self._finish(gathered, output),
            resources=(prepared_lease, send_lease, gathered_lease, output),
        )

    def _finish(
        self,
        gathered: Any,
        output: Any,
    ) -> CompletionOutcome[Any]:
        torch = import_module("torch")
        with torch.cuda.device(output.device):
            payloads = [
                gathered.narrow(
                    0,
                    rank * self.payload_stride,
                    self.payload_numel,
                )
                for rank in range(self.lowered.context.world_size)
            ]
            used = self._fused(
                payloads,
                output,
                self._wire.group_size,
                0,
                self._wire.bit,
                _quant_type(self._module, self._wire.quant_type),
                self._wire.compact,
                self.lowered.reduction.divisor,
            )
            if not used:
                raise RuntimeError("fused dequant-reduce-mean declined gathered payloads")
            result = output[: self.original_numel].reshape(self.lowered.context.shape)
            return CompletionOutcome(result, _record_current_stream(output.device))

    def reconstruct_local(self, value: Any) -> Any:
        """Return local quantize/dequantize reconstruction for Gradient EF."""

        torch = import_module("torch")
        flat = value.reshape(-1)
        if int(flat.numel()) != self.original_numel:
            raise ValueError("input numel differs from the compiled shape")
        prepared = flat.new_zeros((self.padded_numel,))
        prepared[: self.original_numel].copy_(flat)
        payload = torch.empty(
            self.payload_numel,
            device=flat.device,
            dtype=torch.uint8,
        )
        restored = flat.new_empty((self.padded_numel,))
        quantize_into(
            prepared,
            payload,
            self._wire,
            extension_status=self._status,
        )
        dequantize_into(
            payload,
            restored,
            self._wire,
            dtype=self.lowered.context.dtype,
            extension_status=self._status,
        )
        return restored[: self.original_numel].reshape(value.shape)


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
        self._workspace = _workspace_manager(lowered, wire)
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
        padded_lease = self._workspace.acquire_role(
            WorkspaceRole.PADDED_INPUT,
            device=flat.device,
            allocator=lambda: flat.new_empty((self.plan.padded_numel,)),
        )
        padded = padded_lease.value
        padded.zero_()
        padded[: self.plan.original_numel].copy_(flat)
        send_lease = self._workspace.acquire_role(
            WorkspaceRole.SEND,
            device=flat.device,
            allocator=lambda: torch.empty(
                (self.plan.world_size, self.payload_stride),
                device=flat.device,
                dtype=torch.uint8,
            ),
        )
        send = send_lease.value
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
        received_lease = self._workspace.acquire_role(
            WorkspaceRole.RECEIVE,
            device=flat.device,
            allocator=lambda: torch.empty_like(send),
        )
        received = received_lease.value
        handle = dist.all_to_all_single(
            received,
            send,
            group=self.lowered.bindings.process_group,
            async_op=True,
        )
        output = flat.new_empty((self.plan.shard_numel,))

        def complete() -> CompletionOutcome[ReducedShardValue]:
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
                self.lowered.reduction.divisor,
            )
            if not used:
                raise RuntimeError("fused dequant-reduce-mean declined compiled payloads")
            result = ReducedShardValue(
                tensor=output,
                shard_index=self.plan.rank,
                shard_numel=self.plan.shard_numel,
                original_shape=self.lowered.context.shape,
                original_numel=self.plan.original_numel,
                world_size=self.plan.world_size,
                reduction=self.lowered.reduction.name,
                dtype=self.lowered.context.dtype,
                layout_version=self._output_type.layout_version,
            )
            event = torch.cuda.Event(enable_timing=False)
            event.record(torch.cuda.current_stream(output.device))
            return CompletionOutcome(result, _CudaRecordedEvent(event))

        event = _CollectiveEvent(handle) if handle is not None else ImmediateCompletionEvent()
        return CompletionWork(
            None,  # type: ignore[arg-type]
            event=event,
            complete=complete,
            resources=(padded_lease, send_lease, received_lease, output),
        )


class CudaFullTensorExecutable:
    """Two-collective FullTensor path whose network wire remains quantized."""

    def __init__(
        self,
        lowered: LoweredProgram,
        extension_status: CudaExtensionStatus,
    ) -> None:
        if not isinstance(lowered.program.output, FullTensor):
            raise TypeError("CUDA full-tensor executable requires FullTensor output")
        wire = lowered.program.wire
        if wire.compact:
            raise RuntimeError(
                "fused FullTensor requantize currently requires compact=False"
            )
        if wire.bit != 8 or wire.group_size != 64 or wire.quant_type != "linear":
            raise RuntimeError("fused FullTensor requires INT8 linear group_size=64")
        self.lowered = lowered
        self._status = extension_status
        self._module = _require_module(extension_status)
        _require_callable(self._module, "inplace_quantize")
        self._requantize = _require_callable(
            self._module,
            "inplace_dequantize_reduce_mean_requantize",
        )
        self._writeback = _require_callable(
            self._module,
            "inplace_dequantize_gathered",
        )
        self._wire = wire
        self._workspace = _workspace_manager(lowered, wire)
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

    def run(self, value: Any) -> "_TwoCollectiveWork":
        torch = import_module("torch")
        dist = import_module("torch.distributed")
        flat = value.reshape(-1)
        if int(flat.numel()) != self.plan.original_numel:
            raise ValueError("input numel differs from the compiled shape")
        padded_lease = self._workspace.acquire_role(
            WorkspaceRole.PADDED_INPUT,
            device=flat.device,
            allocator=lambda: flat.new_empty((self.plan.padded_numel,)),
        )
        padded = padded_lease.value
        padded.zero_()
        padded[: self.plan.original_numel].copy_(flat)
        send_lease = self._workspace.acquire_role(
            WorkspaceRole.SEND,
            device=flat.device,
            allocator=lambda: torch.empty(
                (self.plan.world_size, self.payload_stride),
                device=flat.device,
                dtype=torch.uint8,
            ),
        )
        send = send_lease.value
        send.zero_()
        for destination in range(self.plan.world_size):
            quantize_into(
                padded.narrow(
                    0,
                    destination * self.plan.shard_numel,
                    self.plan.shard_numel,
                ),
                send[destination, : self.payload_numel],
                self._wire,
                extension_status=self._status,
            )
        received_lease = self._workspace.acquire_role(
            WorkspaceRole.RECEIVE,
            device=flat.device,
            allocator=lambda: torch.empty_like(send),
        )
        received = received_lease.value
        first_handle = dist.all_to_all_single(
            received,
            send,
            group=self.lowered.bindings.process_group,
            async_op=True,
        )
        reduced_lease = self._workspace.acquire_role(
            WorkspaceRole.REDUCED_PAYLOAD,
            device=flat.device,
            allocator=lambda: torch.empty(
                (self.payload_stride,),
                device=flat.device,
                dtype=torch.uint8,
            ),
        )
        reduced_payload = reduced_lease.value
        reduced_payload.zero_()
        gathered_lease = self._workspace.acquire_role(
            WorkspaceRole.GATHERED_PAYLOAD,
            device=flat.device,
            allocator=lambda: torch.empty(
                (self.plan.world_size * self.payload_stride,),
                device=flat.device,
                dtype=torch.uint8,
            ),
        )
        gathered = gathered_lease.value
        restored = flat.new_empty((self.plan.padded_numel,))
        resources = (
            padded_lease,
            send_lease,
            received_lease,
            reduced_lease,
            gathered_lease,
            restored,
        )
        return _TwoCollectiveWork(
            first_handle=first_handle,
            after_first=lambda: self._after_first(
                dist,
                received,
                reduced_payload,
                gathered,
            ),
            after_second=lambda: self._after_second(gathered, restored),
            resources=resources,
        )

    def reconstruct_local(self, value: Any) -> Any:
        """Return this rank's quantize/dequantize reconstruction for Gradient EF."""

        torch = import_module("torch")
        flat = value.reshape(-1)
        original_numel = int(flat.numel())
        padded_numel = (
            ((original_numel + self._wire.group_size - 1) // self._wire.group_size)
            * self._wire.group_size
        )
        prepared = flat.new_zeros((padded_numel,))
        prepared[:original_numel].copy_(flat)
        payload = torch.empty(
            payload_nbytes(
                padded_numel,
                dtype=self.lowered.context.dtype,
                wire=self._wire,
            ),
            device=flat.device,
            dtype=torch.uint8,
        )
        restored = flat.new_empty((padded_numel,))
        quantize_into(
            prepared,
            payload,
            self._wire,
            extension_status=self._status,
        )
        dequantize_into(
            payload,
            restored,
            self._wire,
            dtype=self.lowered.context.dtype,
            extension_status=self._status,
        )
        return restored[:original_numel].reshape(value.shape)

    def _after_first(
        self,
        dist: Any,
        received: Any,
        reduced_payload: Any,
        gathered: Any,
    ) -> object:
        torch = import_module("torch")
        with torch.cuda.device(received.device):
            payloads = [
                received[index, : self.payload_numel]
                for index in range(self.plan.world_size)
            ]
            used = self._requantize(
                payloads,
                reduced_payload,
                64,
                0,
                8,
                _quant_type(self._module, "linear"),
                False,
                _dtype(self._module, self.lowered.context.dtype),
                self.lowered.reduction.divisor,
            )
            if not used:
                raise RuntimeError("fused dequant-reduce-mean-requantize declined")
            return dist.all_gather_into_tensor(
                gathered,
                reduced_payload,
                group=self.lowered.bindings.process_group,
                async_op=True,
            )

    def _after_second(
        self,
        gathered: Any,
        restored: Any,
    ) -> CompletionOutcome[Any]:
        torch = import_module("torch")
        with torch.cuda.device(restored.device):
            used = self._writeback(
                gathered,
                restored,
                64,
                0,
                8,
                _quant_type(self._module, "linear"),
                False,
                _dtype(self._module, self.lowered.context.dtype),
                self.plan.world_size,
                self.payload_numel,
                self.payload_stride,
                self.plan.shard_numel,
            )
            if not used:
                raise RuntimeError("gathered-dequant-writeback declined")
            result = restored[: self.plan.original_numel].reshape(
                self.lowered.context.shape
            )
            return CompletionOutcome(result, _record_current_stream(restored.device))


class _TwoCollectiveWork:
    """Advance two ordered collectives and one output CUDA event."""

    def __init__(
        self,
        *,
        first_handle: object,
        after_first: Any,
        after_second: Any,
        resources: tuple[object, ...],
    ) -> None:
        self._first = first_handle
        self._second: object | None = None
        self._after_first = after_first
        self._after_second = after_second
        self._output: CompletionOutcome[Any] | None = None
        self._resources = resources
        self._result: Any = None
        self._error: BaseException | None = None
        self._finished = False
        self._lock = RLock()

    def query(self) -> bool:
        with self._lock:
            if self._finished:
                return True
            try:
                if self._second is None:
                    if not _handle_query(self._first):
                        return False
                    self._first.wait()  # type: ignore[attr-defined]
                    self._second = self._after_first()
                if self._output is None:
                    if not _handle_query(self._second):
                        return False
                    self._second.wait()  # type: ignore[attr-defined]
                    self._output = self._after_second()
                if not self._output.output_ready.query():
                    return False
                self._finish_locked(self._output.result)
            except BaseException as error:
                self._fail_locked(error)
            return True

    def wait(self, timeout: float | None = None) -> Any:
        if timeout is not None:
            raise NotImplementedError("two-collective work does not support timeout")
        with self._lock:
            if not self._finished:
                try:
                    if self._second is None:
                        self._first.wait()  # type: ignore[attr-defined]
                        self._second = self._after_first()
                    if self._output is None:
                        self._second.wait()  # type: ignore[attr-defined]
                        self._output = self._after_second()
                    self._output.output_ready.wait()
                    self._finish_locked(self._output.result)
                except BaseException as error:
                    self._fail_locked(error)
            if self._error is not None:
                raise self._error
            return self._result

    def _finish_locked(self, result: Any) -> None:
        self._result = result
        self._release_resources()
        self._finished = True

    def _fail_locked(self, error: BaseException) -> None:
        self._error = error
        self._release_resources()
        self._finished = True

    def _release_resources(self) -> None:
        for resource in reversed(self._resources):
            release = getattr(resource, "release", None)
            if callable(release):
                release()
        self._resources = ()


def _finish_native_reduction(value: Any, handle: object, divisor: int) -> Any:
    handle.wait()  # type: ignore[attr-defined]
    if divisor != 1:
        value.div_(divisor)
    _record_current_stream(value.device).wait()
    return value


def _record_current_stream(device: object) -> _CudaRecordedEvent:
    torch = import_module("torch")
    event = torch.cuda.Event(enable_timing=False)
    event.record(torch.cuda.current_stream(device))
    return _CudaRecordedEvent(event)


def _handle_query(handle: object) -> bool:
    query = getattr(handle, "is_completed", None)
    return bool(query()) if callable(query) else False


def _workspace_manager(lowered: LoweredProgram, wire: object) -> CudaWorkspaceManager[Any]:
    external = lowered.bindings.allocator
    pool = (
        external
        if isinstance(external, BudgetedWorkspacePool)
        else BudgetedWorkspacePool(lowered.context.workspace_budget_bytes)
    )
    return CudaWorkspaceManager(
        pool=pool,
        world_size=lowered.context.world_size,
        bit=wire.bit,
        group_size=wire.group_size,
        compact=wire.compact,
        plan=lowered.buffer_plan,
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


def _dtype(module: object, dtype: DataType) -> object:
    return getattr(
        module.DType,
        {
            DataType.FP16: "FP16",
            DataType.BF16: "BF16",
            DataType.FP32: "FP32",
        }[dtype],
    )


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment if value else 0
