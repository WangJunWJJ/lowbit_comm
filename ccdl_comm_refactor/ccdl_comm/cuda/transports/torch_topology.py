"""Torch/NCCL runtime bindings for compiled compressed topology executors."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor

from .compressed_reduce_scatter import ChunkRange
from .tree import TreeEdge


@dataclass(slots=True)
class TorchTopologyContext:
    """Per-submission stream state; no mutable state is shared across runs."""

    tensor: Any
    stream: Any
    producer_completion: Any
    quant_index: int = 0
    recv_index: int = 0
    mean_applied: bool = False

    def query(self) -> bool:
        return bool(self.producer_completion.query())

    def wait(self) -> None:
        self.producer_completion.wait()

    def wait_stream(self, stream: Any) -> None:
        self.producer_completion.wait_stream(stream)


class TorchP2PWork:
    """One nonblocking endpoint over all handles from a batched P2P call."""

    __slots__ = ("handles", "_torch")

    def __init__(self, handles: list[Any], *, torch: Any) -> None:
        self.handles = tuple(handles)
        self._torch = torch

    def query(self) -> bool:
        return all(_work_ready(handle) for handle in self.handles)

    def is_completed(self) -> bool:
        return self.query()

    def wait(self) -> None:
        for handle in self.handles:
            wait = getattr(handle, "wait", None)
            if callable(wait):
                wait()

    def wait_stream(self, stream: Any) -> None:
        guard = self._torch.cuda.stream(stream) if stream is not None else nullcontext()
        with guard:
            self.wait()


class _TorchTopologyRuntimeBase:
    def __init__(
        self,
        *,
        config: CompressionConfig,
        dtype: str,
        world_size: int,
        rank: int,
        participants: tuple[int, ...] | None = None,
        extension_status: Any,
        completion_manager: CudaCompletionManager,
        torch: Any | None = None,
        dist: Any | None = None,
        process_group: Any | None = None,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.world_size = world_size
        self.rank = rank
        self.participants = (
            tuple(range(world_size))
            if participants is None
            else tuple(participants)
        )
        if len(self.participants) != world_size:
            raise ValueError("participants length must equal world_size")
        if len(set(self.participants)) != world_size:
            raise ValueError("participants must contain unique global ranks")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be a group-local rank in [0, world_size)")
        self.extension_status = extension_status
        self.completion_manager = completion_manager
        self.torch = torch
        self.dist = dist
        self.process_group = process_group
        self.stream = _new_cuda_stream(torch) if torch is not None else None
        self._backend_validated = False

    def create_submission_context(self, tensor: Any) -> TorchTopologyContext:
        self._ensure_runtime()
        producer_stream = _current_cuda_stream(self.torch, tensor)
        producer = self.completion_manager.record_for(tensor, stream=producer_stream)
        return TorchTopologyContext(tensor=tensor, stream=self.stream, producer_completion=producer)

    def wait_for_producer(self, tensor: Any, *, context: TorchTopologyContext) -> None:
        context.wait_stream(context.stream)
        _record_stream(tensor, context.stream)

    def record_completion(
        self, *, context: TorchTopologyContext, dependencies: tuple[Any, ...]
    ) -> Any:
        for dependency in dependencies:
            wait_stream = getattr(dependency, "wait_stream", None)
            if callable(wait_stream):
                wait_stream(context.stream)
        return self.completion_manager.record_for(context.tensor, stream=context.stream)

    def _guard(self, context: TorchTopologyContext) -> Any:
        if context.stream is None:
            return nullcontext()
        stream = getattr(self.torch.cuda, "stream", None)
        return stream(context.stream) if callable(stream) else nullcontext()

    def _quantize(self, tensor: Any, workspace: Any, context: TorchTopologyContext) -> Any:
        output = workspace.get_quantized_chunk(
            id(context.tensor),
            context.quant_index,
            tensor,
            self.config,
            dtype=self.dtype,
            world_size=self.world_size,
        )
        context.quant_index += 1
        with self._guard(context):
            quantize_tensor(
                tensor,
                self.config,
                extension_status=self.extension_status,
                output=output,
            )
        _record_stream(output, context.stream)
        return output

    def _receive(self, template: Any, workspace: Any, context: TorchTopologyContext) -> Any:
        output = workspace.get_received_payload(
            id(context.tensor),
            template,
            context.recv_index,
            world_size=self.world_size,
            config=self.config,
        )
        context.recv_index += 1
        _record_stream(output, context.stream)
        return output

    def _receive_for_tensor(
        self, tensor: Any, workspace: Any, context: TorchTopologyContext
    ) -> Any:
        output = workspace.get_received_tensor_payload(
            id(context.tensor),
            context.recv_index,
            tensor,
            self.config,
            dtype=self.dtype,
            world_size=self.world_size,
        )
        context.recv_index += 1
        _record_stream(output, context.stream)
        return output

    def _send_recv(
        self,
        payload: Any,
        received: Any,
        *,
        send_peer: int,
        recv_peer: int,
        context: TorchTopologyContext,
    ) -> TorchP2PWork:
        with self._guard(context):
            ops = (
                self._p2p_op(self.dist.isend, payload, send_peer),
                self._p2p_op(self.dist.irecv, received, recv_peer),
            )
            handles = self.dist.batch_isend_irecv(list(ops))
        return TorchP2PWork(handles, torch=self.torch)

    def _p2p_op(self, operation: Any, tensor: Any, peer: int) -> Any:
        global_peer = self._global_peer(peer)
        if self.process_group is None:
            return self.dist.P2POp(operation, tensor, global_peer)
        return self.dist.P2POp(operation, tensor, global_peer, self.process_group)

    def _global_peer(self, group_peer: int) -> int:
        if group_peer < 0 or group_peer >= self.world_size:
            raise ValueError("P2P peer must be a valid group-local rank")
        return self.participants[group_peer]

    def _require_dequant_output(self, output: Any) -> None:
        numel = int(output.numel())
        if numel % self.config.group_size != 0:
            raise UnsupportedCollective(
                "topology_all_reduce",
                reason=(
                    "native inplace topology dequantization requires group-aligned "
                    f"output; numel={numel}, group_size={self.config.group_size}"
                ),
            )

    def _ensure_runtime(self) -> None:
        if self.torch is None:
            self.torch = import_module("torch")
        if self.dist is None:
            self.dist = import_module("torch.distributed")
        if not self._backend_validated:
            backend = str(self.dist.get_backend(self.process_group)).lower()
            if "nccl" not in backend:
                raise UnsupportedCollective(
                    "topology_all_reduce",
                    reason=f"async CUDA topology runtime requires NCCL; received {backend!r}",
                )
            self._backend_validated = True
        if self.stream is None:
            self.stream = _new_cuda_stream(self.torch)


class TorchPipelinedRingRuntime(_TorchTopologyRuntimeBase):
    """Allocation-free quantized ring reduce-scatter plus all-gather runtime."""

    def quant_pack(
        self, tensor: Any, chunk: ChunkRange, workspace: Any, *, context: TorchTopologyContext
    ) -> Any:
        view = _chunk_view(tensor, chunk)
        return self._quantize(view, workspace, context)

    def send_recv(
        self,
        payload: Any,
        *,
        send_peer: int,
        recv_peer: int,
        recv_chunk: ChunkRange,
        workspace: Any,
        context: TorchTopologyContext,
    ) -> tuple[Any, TorchP2PWork]:
        del recv_chunk
        received = self._receive(payload, workspace, context)
        return received, self._send_recv(
            payload,
            received,
            send_peer=send_peer,
            recv_peer=recv_peer,
            context=context,
        )

    def fused_reduce(
        self,
        tensor: Any,
        received: Any,
        chunk: ChunkRange,
        contributors: tuple[int, ...],
        workspace: Any,
        *,
        context: TorchTopologyContext,
        dependency: Any,
    ) -> Any:
        del workspace
        dependency.wait_stream(context.stream)
        output = _chunk_view(tensor, chunk)
        self._require_dequant_output(output)
        with self._guard(context):
            dequantize_tensor(
                received,
                (chunk.stop - chunk.start,),
                self.config,
                dtype=self.dtype,
                extension_status=self.extension_status,
                output=output,
                reduce_op="sum",
            )
            if len(contributors) == self.world_size - 1:
                output.div_(self.world_size)
                context.mean_applied = True
        return output

    def apply_broadcast(
        self,
        tensor: Any,
        received: Any,
        chunk: ChunkRange,
        workspace: Any,
        *,
        context: TorchTopologyContext,
        dependency: Any,
    ) -> Any:
        del workspace
        dependency.wait_stream(context.stream)
        output = _chunk_view(tensor, chunk)
        self._require_dequant_output(output)
        with self._guard(context):
            dequantize_tensor(
                received,
                (chunk.stop - chunk.start,),
                self.config,
                dtype=self.dtype,
                extension_status=self.extension_status,
                output=output,
                reduce_op="none",
            )
        return output


class TorchTreeRuntime(_TorchTopologyRuntimeBase):
    """Quantized tree reduce/broadcast runtime for arbitrary world sizes."""

    def quant_pack(
        self, tensor: Any, edge: TreeEdge, workspace: Any, *, context: TorchTopologyContext
    ) -> Any:
        if edge.parent_rank == self.rank and not context.mean_applied:
            with self._guard(context):
                tensor.div_(self.world_size)
            context.mean_applied = True
        return self._quantize(tensor, workspace, context)

    def send(
        self,
        payload: Any,
        *,
        peer: int,
        edge: TreeEdge,
        workspace: Any,
        context: TorchTopologyContext,
    ) -> TorchP2PWork:
        del edge, workspace
        with self._guard(context):
            handle = self.dist.isend(
                payload,
                dst=self._global_peer(peer),
                group=self.process_group,
            )
        return TorchP2PWork([handle], torch=self.torch)

    def receive(
        self,
        *,
        peer: int,
        edge: TreeEdge,
        workspace: Any,
        context: TorchTopologyContext,
    ) -> tuple[Any, TorchP2PWork]:
        del edge
        received = self._receive_for_tensor(context.tensor, workspace, context)
        with self._guard(context):
            handle = self.dist.irecv(
                received,
                src=self._global_peer(peer),
                group=self.process_group,
            )
        return received, TorchP2PWork([handle], torch=self.torch)

    def fused_reduce(
        self,
        tensor: Any,
        received: Any,
        edge: TreeEdge,
        workspace: Any,
        *,
        context: TorchTopologyContext,
        dependency: Any,
    ) -> Any:
        del edge, workspace
        dependency.wait_stream(context.stream)
        self._require_dequant_output(tensor)
        with self._guard(context):
            return dequantize_tensor(
                received,
                tuple(tensor.shape),
                self.config,
                dtype=self.dtype,
                extension_status=self.extension_status,
                output=tensor,
                reduce_op="sum",
            )

    def apply_broadcast(
        self,
        tensor: Any,
        received: Any,
        edge: TreeEdge,
        workspace: Any,
        *,
        context: TorchTopologyContext,
        dependency: Any,
    ) -> Any:
        del edge, workspace
        dependency.wait_stream(context.stream)
        self._require_dequant_output(tensor)
        with self._guard(context):
            result = dequantize_tensor(
                received,
                tuple(tensor.shape),
                self.config,
                dtype=self.dtype,
                extension_status=self.extension_status,
                output=tensor,
                reduce_op="none",
            )
        context.mean_applied = True
        return result


def _chunk_view(tensor: Any, chunk: ChunkRange) -> Any:
    return tensor.reshape((-1,))[chunk.start : chunk.stop]


def _record_stream(value: Any, stream: Any) -> None:
    record_stream = getattr(value, "record_stream", None)
    if callable(record_stream):
        record_stream(stream)


def _work_ready(work: Any) -> bool:
    for name in ("is_completed", "query"):
        query = getattr(work, name, None)
        if callable(query):
            return bool(query())
    return False


def _new_cuda_stream(torch: Any) -> Any | None:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    stream_type = getattr(cuda, "Stream", None)
    if cuda is None or not callable(is_available) or not is_available() or stream_type is None:
        return None
    return stream_type()


def _current_cuda_stream(torch: Any, tensor: Any) -> Any | None:
    cuda = getattr(torch, "cuda", None)
    current_stream = getattr(cuda, "current_stream", None)
    is_available = getattr(cuda, "is_available", None)
    if cuda is None or not callable(is_available) or not is_available() or not callable(current_stream):
        return None
    try:
        return current_stream(device=tensor.device)
    except (RuntimeError, TypeError):
        return current_stream()
