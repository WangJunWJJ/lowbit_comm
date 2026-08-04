from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_BITS = {4, 8}
_SAFE_DEFAULT_BITS = {8}
_SUPPORTED_GROUP_SIZES = {16, 32, 64}
_SUPPORTED_TOPK = {0, 1, 2}
_SUPPORTED_QUANT_TYPES = {"linear", "normal", "uniform", "e3m0", "e2m1"}
_SUPPORTED_ERROR_FEEDBACK_POLICIES = {
    "none",
    "always",
    "large_bucket_only",
    "warmup_then_enable",
    "periodic",
}
_SUPPORTED_TARGETS = {"tensor", "ddp_gradient_bucket", "collective", "p2p"}


@dataclass(frozen=True)
class CompressionConfig:
    """User-facing compression policy for CCDL communication.

    The default intentionally targets the safest currently validated path:
    linear 8-bit compression over an independent tensor communication target.
    """

    bit: int = 8
    group_size: int = 64
    topk: int = 0
    quant_type: str = "linear"
    stochastic: bool = False
    error_feedback: bool = True
    target: str = "tensor"
    warmup_steps: int = 0
    fallback: str = "bf16_compress"
    allow_experimental: bool = False
    compact: bool = False
    error_feedback_policy: str = "always"
    error_feedback_min_numel: int = 0
    error_feedback_warmup_steps: int = 0
    error_feedback_period: int = 1

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
        if self.target not in _SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported target={self.target!r}; expected one of {sorted(_SUPPORTED_TARGETS)}")
        if self.error_feedback_policy not in _SUPPORTED_ERROR_FEEDBACK_POLICIES:
            raise ValueError(
                "Unsupported error_feedback_policy="
                f"{self.error_feedback_policy!r}; expected one of {sorted(_SUPPORTED_ERROR_FEEDBACK_POLICIES)}"
            )
        if self.error_feedback_min_numel < 0:
            raise ValueError("error_feedback_min_numel must be >= 0")
        if self.error_feedback_warmup_steps < 0:
            raise ValueError("error_feedback_warmup_steps must be >= 0")
        if self.error_feedback_period <= 0:
            raise ValueError("error_feedback_period must be > 0")

    def effective_error_feedback_policy(self) -> str:
        if not self.error_feedback:
            return "none"
        return self.error_feedback_policy
