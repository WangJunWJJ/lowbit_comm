"""Backend-neutral public shortcuts for compiled native collectives."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ccdl_comm.compiler import compile
from ccdl_comm.executor import CompiledCommunicationPlan
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.registry import BackendRegistry


def native_collectives() -> tuple[str, ...]:
    """Return native collectives in stable public protocol order."""

    from ccdl_comm.cuda.native_collectives import NATIVE_BUILDERS

    return tuple(NATIVE_BUILDERS)


def context_from_tensor(
    tensor: object | None,
    *,
    group: object | None = None,
    device: str | None = None,
    dist: Any | None = None,
    torch: Any | None = None,
) -> CompileContext:
    """Build immutable compile facts from one representative runtime tensor."""

    active_dist = dist if dist is not None else import_module("torch.distributed")
    active_torch = torch
    if tensor is None and active_torch is None:
        active_torch = import_module("torch")
    rank = int(active_dist.get_rank(group=group))
    world_size = int(active_dist.get_world_size(group=group))
    active_device = device or str(getattr(tensor, "device", ""))
    if not active_device:
        current_device = int(active_torch.cuda.current_device())
        active_device = f"cuda:{current_device}"
    shape = tuple(getattr(tensor, "shape", ()))
    dtype = _normalize_dtype(str(getattr(tensor, "dtype", "fp32")))
    return CompileContext(
        rank=rank,
        world_size=world_size,
        device=active_device,
        shape=shape,
        dtype=dtype,
        process_group=group,
        device_architecture=_device_architecture(active_device, active_torch),
    )


def compile_collective(
    sample: object | None,
    *,
    collective: str,
    group: object | None = None,
    root: int = 0,
    op: str = "sum",
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    registry: BackendRegistry | None = None,
    device: str | None = None,
) -> CompiledCommunicationPlan:
    """Compile one native collective for reuse by a steady-state caller."""

    active_plan = plan or CommunicationPlan(
        collective=collective,
        strategy="native_nccl",
        backend="cuda",
        output_layout="full",
        async_op=async_op,
        root=root,
        reduce_op=op,
    )
    if active_plan.collective != collective:
        raise ValueError(
            f"plan collective {active_plan.collective!r} does not match "
            f"requested collective {collective!r}"
        )
    context = context_from_tensor(
        sample,
        group=group,
        device=device,
    )
    active_registry = registry
    if active_registry is None:
        from ccdl_comm.cuda.backend import register_cuda_backends

        active_registry = BackendRegistry()
        register_cuda_backends(active_registry)
    return compile(
        active_plan,
        context,
        registry=active_registry,
    )


def all_reduce(
    tensor: object,
    *,
    op: str = "sum",
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Run native all-reduce and return a CCDL CollectiveWork."""

    compiled = compiled_plan or compile_collective(
        tensor,
        collective="all_reduce",
        group=group,
        op=op,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(tensor)


def all_gather(
    tensor: object,
    *,
    output_tensors: tuple[object, ...] = (),
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Run native all-gather into optional caller-owned output tensors."""

    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    compiled = compiled_plan or compile_collective(
        tensor,
        collective="all_gather",
        group=group,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(
        NativeCollectiveInput(
            tensor=tensor,
            output_tensors=output_tensors,
        )
    )


def reduce_scatter(
    output: object,
    *,
    input_tensors: tuple[object, ...],
    op: str = "sum",
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Run native list-based reduce-scatter into caller-owned output."""

    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    sample = output
    compiled = compiled_plan or compile_collective(
        sample,
        collective="reduce_scatter",
        group=group,
        op=op,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(
        NativeCollectiveInput(
            tensor=output,
            input_tensors=input_tensors,
        )
    )


def all_to_all(
    *,
    output_tensors: tuple[object, ...],
    input_tensors: tuple[object, ...],
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Run native list-based all-to-all."""

    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    sample = _first_tensor(input_tensors, output_tensors, "all_to_all")
    compiled = compiled_plan or compile_collective(
        sample,
        collective="all_to_all",
        group=group,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(
        NativeCollectiveInput(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )
    )


def broadcast(
    tensor: object,
    *,
    src: int = 0,
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Broadcast one tensor from a compile-time root."""

    compiled = compiled_plan or compile_collective(
        tensor,
        collective="broadcast",
        group=group,
        root=src,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(tensor)


def reduce(
    tensor: object,
    *,
    dst: int = 0,
    op: str = "sum",
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Reduce one tensor to a compile-time destination rank."""

    compiled = compiled_plan or compile_collective(
        tensor,
        collective="reduce",
        group=group,
        root=dst,
        op=op,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(tensor)


def gather(
    tensor: object,
    *,
    gather_list: tuple[object, ...] = (),
    dst: int = 0,
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Gather tensors to one root rank."""

    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    compiled = compiled_plan or compile_collective(
        tensor,
        collective="gather",
        group=group,
        root=dst,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(
        NativeCollectiveInput(
            tensor=tensor,
            output_tensors=gather_list,
        )
    )


def scatter(
    output: object,
    *,
    scatter_list: tuple[object, ...] = (),
    src: int = 0,
    group: object | None = None,
    async_op: bool = False,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Scatter root-owned tensors into caller-owned rank outputs."""

    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    compiled = compiled_plan or compile_collective(
        output,
        collective="scatter",
        group=group,
        root=src,
        async_op=async_op,
        plan=plan,
    )
    return compiled.run(
        NativeCollectiveInput(
            tensor=output,
            input_tensors=scatter_list,
        )
    )


def barrier(
    *,
    group: object | None = None,
    async_op: bool = False,
    device: str | None = None,
    plan: CommunicationPlan | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
):
    """Synchronize a process group through a native NCCL barrier."""

    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    compiled = compiled_plan or compile_collective(
        None,
        collective="barrier",
        group=group,
        async_op=async_op,
        plan=plan,
        device=device,
    )
    return compiled.run(NativeCollectiveInput())


def _first_tensor(
    first: tuple[object, ...],
    second: tuple[object, ...],
    collective: str,
) -> object:
    values = (*tuple(first), *tuple(second))
    if not values:
        raise ValueError(f"{collective} requires at least one tensor")
    return values[0]


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower().removeprefix("torch.")
    return {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        "float64": "fp64",
        "double": "fp64",
    }.get(normalized, normalized)


def _device_architecture(device: str, torch: Any | None) -> str:
    if torch is None:
        try:
            torch = import_module("torch")
        except (ImportError, ModuleNotFoundError):
            return "unknown"
    try:
        return str(torch.cuda.get_device_name(device))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return "unknown"


__all__ = [
    "all_gather",
    "all_reduce",
    "all_to_all",
    "barrier",
    "broadcast",
    "compile_collective",
    "context_from_tensor",
    "gather",
    "native_collectives",
    "reduce",
    "reduce_scatter",
    "scatter",
]
