from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_BITS = {4, 8}
_SAFE_DEFAULT_BITS = {8}
_SUPPORTED_GROUP_SIZES = {16, 32, 64}
_SUPPORTED_TOPK = {0, 1, 2}
_SUPPORTED_QUANT_TYPES = {"linear", "normal", "uniform", "e3m0", "e2m1"}


@dataclass(frozen=True)
class CompressionConfig:
    """User-facing compression policy for CCDL communication.

    The default intentionally targets the safest currently validated path:
    linear 8-bit compression over native-DDP gradient buckets.
    """

    bit: int = 8
    group_size: int = 64
    topk: int = 0
    quant_type: str = "linear"
    stochastic: bool = False
    error_feedback: bool = True
    target: str = "ddp_gradient_bucket"
    warmup_steps: int = 0
    fallback: str = "bf16_compress"
    allow_experimental: bool = False
    compact: bool = False

    def __post_init__(self) -> None:
        if self.bit not in _SUPPORTED_BITS:
            raise ValueError(f"Unsupported bit={self.bit}; expected one of {sorted(_SUPPORTED_BITS)}")
        if self.bit not in _SAFE_DEFAULT_BITS and not self.allow_experimental:
            raise ValueError(f"bit={self.bit} is experimental; set allow_experimental=True to opt in")
        if self.group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsupported group_size={self.group_size}; expected one of {sorted(_SUPPORTED_GROUP_SIZES)}"
            )
        if self.topk not in _SUPPORTED_TOPK:
            raise ValueError(f"Unsupported topk={self.topk}; expected one of {sorted(_SUPPORTED_TOPK)}")
        if self.quant_type not in _SUPPORTED_QUANT_TYPES:
            raise ValueError(
                f"Unsupported quant_type={self.quant_type!r}; expected one of {sorted(_SUPPORTED_QUANT_TYPES)}"
            )
        if self.target != "ddp_gradient_bucket":
            raise ValueError("Only target='ddp_gradient_bucket' is supported in the initial refactor")
