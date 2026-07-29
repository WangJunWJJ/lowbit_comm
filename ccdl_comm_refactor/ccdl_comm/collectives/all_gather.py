from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ccdl_comm.collectives.all_reduce import (
    _coerce_payload,
    _extension_dequantize,
    _extension_quantize,
    _make_payload_all_gather,
    _payload_buffer,
    _payloads_from_async_gather,
    _resolve_dtype,
)
from ccdl_comm.collectives.work import CollectiveWork, ImmediateWork
from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.gather_reduce import GatheredPayloads
from ccdl_comm.communication.torch_transport import make_torch_all_gather, make_torch_async_all_gather
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus


def compressed_all_gather(
    tensor: Any,
    *,
    config: CompressionConfig,
    async_op: bool = False,
    dtype: str = "auto",
    quantize: Callable[[Any, CompressionConfig], Any] | None = None,
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any] | None = None,
    all_gather: Callable[[CompressedPayload], GatheredPayloads] | None = None,
    extension_status: CudaExtensionStatus | None = None,
    completion_manager: CudaCompletionManager | Any | None = None,
) -> Sequence[Any] | CollectiveWork[Sequence[Any]]:
    """Gather compressed tensors from all ranks and return reconstructed tensors.

    The initial implementation targets same-shape tensors, which is the common
    case for DDP buckets and model-parallel control tensors. Dynamic-shape
    metadata exchange can be layered on this API without changing callers.
    """

    active_dtype = _resolve_dtype(dtype, tensor)
    shape = tuple(getattr(tensor, "shape", ()))
    active_quantize = quantize or _extension_quantize(extension_status)
    active_dequantize = dequantize or _extension_dequantize(extension_status)
    local_payload = _coerce_payload(active_quantize(tensor, config), shape=shape, dtype=active_dtype)
    if async_op and all_gather is None and not local_payload.metadata:
        gather_work = make_torch_async_all_gather()(_payload_buffer(local_payload))
        manager = completion_manager or CudaCompletionManager()

        def complete_gather() -> list[Any]:
            payloads = _payloads_from_async_gather(gather_work, local_payload)
            return [active_dequantize(payload, shape, config, active_dtype) for payload in payloads]

        return manager.create_work(
            result=[],
            handle=gather_work,
            complete=complete_gather,
            resources=(local_payload, *tuple(gather_work.payloads)),
        )

    active_all_gather = all_gather or _make_payload_all_gather(make_torch_all_gather())
    gathered = active_all_gather(local_payload)
    restored = [active_dequantize(payload, shape, config, active_dtype) for payload in gathered.payloads]
    if async_op:
        return ImmediateWork(restored)
    return restored
