"""CUDA communication backend implementing the backend-neutral Core protocol."""

from __future__ import annotations

from collections.abc import Mapping

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


CUDA_BACKEND_KEYS: tuple[OperationKey, ...] = (
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
        available = (
            context.device.strip().lower().startswith("cuda")
            and self._extension_status.available
            and self._extension_status.module is not None
        )
        reason = None
        if not context.device.strip().lower().startswith("cuda"):
            reason = f"device {context.device!r} is not CUDA"
        elif not self._extension_status.available or self._extension_status.module is None:
            reason = self._extension_status.reason or "CCDL CUDA extension is unavailable"
        elif context.process_group is not None and not any(
            key[1] == "all_gather" for key in self._operation_factories
        ):
            available = False
            reason = "this CUDA transport does not yet support an explicit process group"
        return BackendCapabilities(
            backend=self.name,
            available=available,
            collectives={key[0] for key in self._operation_factories},
            strategies={key[1] for key in self._operation_factories},
            dtypes={"fp16", "bf16", "fp32"},
            bits={4, 8},
            output_layouts={key[2] for key in self._operation_factories},
            supports_async=all(
                key[1] != "hierarchical"
                for key in self._operation_factories
            ),
            supports_dynamic_shape=False,
            features={"cuda_extension", "compile_once", "workspace_cache"},
            reason=reason,
            details={"abi_version": self.abi_version},
        )

    def compile(self, plan: CommunicationPlan, context: CompileContext):
        capabilities = self.capabilities(context)
        if not capabilities.available:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.strategy}:cuda:{plan.output_layout}",
                reason=capabilities.reason,
            )
        if plan.compression is None:
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
        if context.process_group is not None and plan.strategy != "all_gather":
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
