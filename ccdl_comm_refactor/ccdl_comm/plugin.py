from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import CompressionConfig


@dataclass(frozen=True)
class PluginDecision:
    enabled: bool
    fallback: str | None
    reason: str


class CCDLCommunicationPlugin:
    """ParaScale-facing adapter for CCDL communication compression."""

    name = "ccdl"

    def explain(self, config: CompressionConfig) -> list[str]:
        return [
            "CCDL is planned as a native DDP gradient-bucket compression plugin.",
            f"It will use {config.bit}-bit {config.quant_type} quantization with group_size={config.group_size}.",
            "Training orchestration, backend selection, fallback, and benchmark decisions stay in ParaScale.",
        ]

    def plan(self, context: Mapping[str, object], config: CompressionConfig) -> PluginDecision:
        backend = context.get("training_backend")
        if backend != "native_ddp":
            return PluginDecision(
                enabled=False,
                fallback=config.fallback,
                reason="CCDL initial refactor only supports training_backend='native_ddp'",
            )

        device_type = context.get("device_type")
        if device_type != "cuda":
            return PluginDecision(
                enabled=False,
                fallback=config.fallback,
                reason="CCDL initial refactor only supports CUDA/NCCL execution",
            )

        return PluginDecision(
            enabled=True,
            fallback=None,
            reason="CCDL can be enabled for native_ddp CUDA gradient-bucket compression",
        )
