"""One-shot adapters from tensor-oriented APIs to the compiled CUDA Core."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ccdl_comm.compiler import compile
from ccdl_comm.config import CompressionConfig
from ccdl_comm.executor import CompiledCommunicationPlan
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.registry import BackendRegistry

from .backend import register_cuda_backends
from .loader import CudaExtensionStatus


def compile_cuda_shortcut(
    tensor: Any,
    *,
    collective: str,
    strategy: str,
    output_layout: str,
    config: CompressionConfig,
    async_op: bool,
    dtype: str,
    extension_status: CudaExtensionStatus | None,
) -> CompiledCommunicationPlan:
    """Compile one production shortcut call without retaining global state."""

    dist = import_module("torch.distributed")
    process_group = None
    rank = int(dist.get_rank(group=process_group))
    world_size = int(dist.get_world_size(group=process_group))
    device = str(getattr(tensor, "device", "cuda"))
    active_dtype = _resolve_dtype(dtype, tensor)
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=extension_status)
    return compile(
        CommunicationPlan(
            collective=collective,
            strategy=strategy,
            backend="cuda",
            compression=config,
            output_layout=output_layout,
            async_op=async_op,
        ),
        CompileContext(
            rank=rank,
            world_size=world_size,
            device=device,
            shape=tuple(getattr(tensor, "shape", ())),
            dtype=active_dtype,
            process_group=process_group,
        ),
        registry=registry,
    )


def _resolve_dtype(dtype: str, tensor: Any) -> str:
    if dtype != "auto":
        return dtype
    value = str(getattr(tensor, "dtype", ""))
    if "bfloat16" in value:
        return "bf16"
    if "float16" in value or value.endswith("half"):
        return "fp16"
    if "float32" in value or value.endswith("float"):
        return "fp32"
    raise ValueError(f"cannot infer CCDL dtype from tensor dtype: {value!r}")
