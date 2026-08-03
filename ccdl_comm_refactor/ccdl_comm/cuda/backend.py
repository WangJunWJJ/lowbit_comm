"""CUDA communication backend implementing the backend-neutral Core protocol."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from ccdl_comm.backend import BackendCapabilities
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.registry import BackendKey, BackendRegistry

from .compiler import (
    OperationFactory,
    OperationKey,
    compile_cuda_plan,
    default_operation_factories,
)
from .loader import CudaExtensionStatus, load_cuda_extension
from .strategy_table import CudaStrategyTable


CUDA_BACKEND_KEYS: tuple[OperationKey, ...] = (
    ("all_reduce", "native_nccl", "full"),
    ("all_gather", "native_nccl", "full"),
    ("reduce_scatter", "native_nccl", "full"),
    ("all_to_all", "native_nccl", "full"),
    ("broadcast", "native_nccl", "full"),
    ("reduce", "native_nccl", "full"),
    ("gather", "native_nccl", "full"),
    ("scatter", "native_nccl", "full"),
    ("barrier", "native_nccl", "full"),
    ("all_reduce", "all_gather", "full"),
    ("all_reduce", "topology", "full"),
    ("all_reduce", "hierarchical", "full"),
    ("reduce_scatter", "compressed", "shard"),
)


class CudaCommunicationBackend:
    """Compile validated CUDA communication plans from existing transports."""

    name = "cuda"
    abi_version = 1

    def __init__(
        self,
        *,
        extension_status: CudaExtensionStatus | None = None,
        operation_factories: Mapping[OperationKey, OperationFactory] | None = None,
    ) -> None:
        self._extension_status = extension_status or load_cuda_extension()
        self._operation_factories = dict(
            default_operation_factories()
            if operation_factories is None
            else operation_factories
        )

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        operation_keys = tuple(
            key
            for key in self._operation_factories
            if key[1] != "hierarchical" or _hierarchical_context_available(context)
        )
        native_nccl_only = bool(self._operation_factories) and all(
            key[1] == "native_nccl" for key in self._operation_factories
        )
        available = (
            context.device.strip().lower().startswith("cuda")
            and (
                native_nccl_only
                or (
                    self._extension_status.available
                    and self._extension_status.module is not None
                )
            )
        )
        reason = None
        if not context.device.strip().lower().startswith("cuda"):
            reason = f"device {context.device!r} is not CUDA"
        elif native_nccl_only:
            reason = _native_nccl_rejection(context)
            if reason is not None:
                available = False
        elif (
            not native_nccl_only
            and (
                not self._extension_status.available
                or self._extension_status.module is None
            )
        ):
            reason = self._extension_status.reason or "CCDL CUDA extension is unavailable"
        elif context.process_group is not None and not any(
            key[1] in {"all_gather", "native_nccl"}
            for key in self._operation_factories
        ):
            available = False
            reason = "this CUDA transport does not yet support an explicit process group"
        features = {"compile_once"}
        if native_nccl_only:
            features.add("native_nccl")
        else:
            features.add("workspace_cache")
            if self._extension_status.available and self._extension_status.module is not None:
                features.add("cuda_extension")
        return BackendCapabilities(
            backend=self.name,
            available=available,
            collectives={key[0] for key in operation_keys},
            strategies={key[1] for key in operation_keys},
            dtypes={"fp16", "bf16", "fp32"},
            bits={4, 8},
            output_layouts={key[2] for key in operation_keys},
            supports_async=all(
                key[1] != "hierarchical"
                for key in operation_keys
            ),
            supports_dynamic_shape=False,
            features=features,
            reason=reason,
            details={"abi_version": self.abi_version},
        )

    def compile(self, plan: CommunicationPlan, context: CompileContext):
        native_nccl = plan.strategy == "native_nccl"
        if native_nccl:
            rejection = (
                f"device {context.device!r} is not CUDA"
                if not context.device.strip().lower().startswith("cuda")
                else _native_nccl_rejection(context)
            )
            if rejection is not None:
                raise UnsupportedCollective(
                    f"{plan.collective}:{plan.strategy}:cuda:{plan.output_layout}",
                    reason=rejection,
                )
        else:
            capabilities = self.capabilities(context)
        if not native_nccl and not capabilities.available:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}:cuda:{plan.output_layout}",
                reason=capabilities.reason,
            )
        if (
            not native_nccl
            and (
                not self._extension_status.available
                or self._extension_status.module is None
            )
        ):
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}",
                reason=self._extension_status.reason or "CCDL CUDA extension is unavailable",
            )
        if plan.compression is None and not native_nccl:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}",
                reason="CUDA compressed executor requires compression",
            )
        if context.allow_dynamic_shape:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}",
                reason="dynamic shape requires an explicit bounded shape class",
            )
        key = (plan.collective, plan.strategy, plan.output_layout)
        if key not in self._operation_factories:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}",
                reason=f"CUDA backend does not implement output layout {plan.output_layout!r}",
            )
        if context.process_group is not None and plan.strategy not in {
            "all_gather",
            "hierarchical",
            "native_nccl",
        }:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}",
                reason="this CUDA transport does not yet support an explicit process group",
            )
        if plan.async_op and plan.strategy == "hierarchical":
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}",
                reason=f"{plan.strategy} CUDA transport is synchronous",
            )
        return compile_cuda_plan(
            plan,
            context,
            self._extension_status,
            operation_factories=self._operation_factories,
        )


def register_cuda_backends(
    registry: BackendRegistry,
    *,
    extension_status: CudaExtensionStatus | None = None,
    operation_factories: Mapping[OperationKey, OperationFactory] | None = None,
) -> None:
    """Register all CUDA strategy keys using fresh backend instances."""

    if not isinstance(registry, BackendRegistry):
        raise TypeError("registry must be a BackendRegistry")
    factories = dict(
        default_operation_factories()
        if operation_factories is None
        else operation_factories
    )
    keys = tuple(key for key in CUDA_BACKEND_KEYS if key in factories)
    for collective, strategy, output_layout in keys:
        key = (collective, strategy, output_layout)
        registry.register(
            BackendKey(collective, strategy, "cuda", output_layout),
            lambda status=extension_status, active_key=key, factory=factories[key]: CudaCommunicationBackend(
                extension_status=status,
                operation_factories={active_key: factory},
            ),
        )
    registry.register_strategy_selector(
        "cuda",
        CudaStrategyTable.from_task13_a6000().as_selector(),
    )


def _native_nccl_rejection(context: CompileContext) -> str | None:
    try:
        dist = import_module("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        return f"torch.distributed is unavailable: {exc}"
    is_initialized = getattr(dist, "is_initialized", None)
    if not callable(is_initialized) or not is_initialized():
        return "torch.distributed must be initialized before compiling native NCCL"
    get_backend = getattr(dist, "get_backend", None)
    if not callable(get_backend):
        return "torch.distributed does not expose process-group backend inspection"
    try:
        backend = str(get_backend(context.process_group)).strip().lower()
    except (RuntimeError, TypeError, ValueError) as exc:
        return f"cannot inspect process-group backend: {exc}"
    if backend != "nccl":
        return f"native_nccl requires an NCCL process group; received {backend!r}"
    return None


def _hierarchical_context_available(context: CompileContext) -> bool:
    values = (
        context.local_rank,
        context.local_world_size,
        context.node_id,
        context.node_count,
    )
    if any(value is None for value in values):
        return False
    return context.world_size == context.local_world_size * context.node_count
