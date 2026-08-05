"""Validated configuration for comparable DDP training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODES = ("native_ddp", "ccdl_sync", "ccdl_async")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    mode: str
    synthetic: bool = False
    data_root: Path | None = None
    steps: int = 22
    warmup_steps: int = 1
    batch_size_per_rank: int = 16
    input_dim: int = 1024
    hidden_dim: int = 4096
    depth: int = 3
    num_classes: int = 1760
    learning_rate: float = 1e-3
    seed: int = 20260805
    device: str = "auto"
    dtype: str = "fp16"
    bit: int = 8
    group_size: int = 64
    error_feedback: bool = True
    bucket_cap_mb: int = 25
    output: Path = Path("dist/ddp-training-result.json")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_root",
            None if self.data_root is None else Path(self.data_root),
        )
        object.__setattr__(self, "output", Path(self.output))
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.synthetic == (self.data_root is not None):
            raise ValueError("exactly one of synthetic=True or data_root must be provided")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps and >= 0")
        for name in (
            "batch_size_per_rank",
            "input_dim",
            "hidden_dim",
            "depth",
            "num_classes",
            "group_size",
            "bucket_cap_mb",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("dtype must be fp16, bf16, or fp32")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.bit not in {4, 8}:
            raise ValueError("bit must be 4 or 8")

    @property
    def measured_steps(self) -> int:
        return self.steps - self.warmup_steps

    def model_parameter_count(self) -> int:
        first = self.input_dim * self.hidden_dim + self.hidden_dim
        hidden = (self.depth - 1) * (
            self.hidden_dim * self.hidden_dim + self.hidden_dim
        )
        classifier = self.hidden_dim * self.num_classes + self.num_classes
        return first + hidden + classifier
