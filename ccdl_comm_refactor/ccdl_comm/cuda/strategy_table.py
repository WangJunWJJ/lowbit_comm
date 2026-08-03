"""Static CUDA strategy thresholds derived from validated benchmark evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import mul

from ccdl_comm.communication.strategy import StrategyChoice
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.plan import CommunicationPlan, CompileContext


TASK13_EVIDENCE = "tests/benchmarks/reports/task13_topology"


@dataclass(frozen=True, slots=True)
class CudaStrategyRule:
    """One immutable, benchmark-backed CUDA strategy threshold."""

    collective: str
    output_layout: str
    device_architecture: str
    world_size: int
    dtype: str
    bit: int
    group_size: int
    min_numel: int
    strategy: str
    expected_speedup: float

    def matches(
        self,
        context: CompileContext,
        compression: CompressionConfig,
        *,
        collective: str,
        output_layout: str,
    ) -> bool:
        """Return whether all validated dimensions match this rule."""

        return (
            collective == self.collective
            and output_layout == self.output_layout
            and _normalize_architecture(context.device_architecture)
            == self.device_architecture
            and context.world_size == self.world_size
            and _normalize_dtype(context.dtype) == self.dtype
            and compression.bit == self.bit
            and compression.group_size == self.group_size
            and _numel(context.shape) >= self.min_numel
        )


class CudaStrategyTable:
    """Select a CUDA strategy once from immutable, benchmarked dimensions."""

    def __init__(self, rules: tuple[CudaStrategyRule, ...]) -> None:
        self._rules = tuple(rules)

    @classmethod
    def from_task13_a6000(cls) -> CudaStrategyTable:
        """Build the checked-in Task 13 A6000 threshold table."""

        rules: list[CudaStrategyRule] = []
        speedups = {
            ("full", 2, 8_388_608): 1.63,
            ("full", 2, 33_554_432): 1.80,
            ("full", 4, 8_388_608): 1.33,
            ("full", 4, 33_554_432): 1.50,
            ("shard", 2, 8_388_608): 2.86,
            ("shard", 2, 33_554_432): 3.03,
            ("shard", 4, 8_388_608): 2.60,
            ("shard", 4, 33_554_432): 2.73,
        }
        for (layout, world_size, min_numel), speedup in speedups.items():
            rules.append(
                CudaStrategyRule(
                    collective=("all_reduce" if layout == "full" else "reduce_scatter"),
                    output_layout=layout,
                    device_architecture="nvidia_rtx_a6000",
                    world_size=world_size,
                    dtype="fp16",
                    bit=8,
                    group_size=64,
                    min_numel=min_numel,
                    strategy="topology" if layout == "full" else "compressed",
                    expected_speedup=speedup,
                )
            )
        rules.sort(key=lambda rule: rule.min_numel, reverse=True)
        return cls(tuple(rules))

    def select(
        self,
        context: CompileContext,
        compression: CompressionConfig,
        *,
        collective: str,
        output_layout: str,
    ) -> StrategyChoice:
        """Return one semantically compatible compile-time strategy choice."""

        if not isinstance(context, CompileContext):
            raise TypeError("context must be a CompileContext")
        if not isinstance(compression, CompressionConfig):
            raise TypeError("compression must be a CompressionConfig")
        semantic_key = (collective, output_layout)
        if semantic_key not in {("all_reduce", "full"), ("reduce_scatter", "shard")}:
            raise UnsupportedCollective(
                f"{collective}:{output_layout}",
                reason="CUDA strategy table has no safe semantics for this request",
            )
        for rule in self._rules:
            if rule.matches(
                context,
                compression,
                collective=collective,
                output_layout=output_layout,
            ):
                return StrategyChoice(
                    strategy=rule.strategy,
                    reason=(
                        "Task 13 A6000 benchmark matched all validated dimensions; "
                        f"expected speedup {rule.expected_speedup:.2f}x"
                    ),
                    benchmark_matched=True,
                    expected_speedup=rule.expected_speedup,
                    evidence=TASK13_EVIDENCE,
                )
        if semantic_key == ("all_reduce", "full"):
            small_bucket = _numel(context.shape) < 8_388_608
            return StrategyChoice(
                strategy="native_nccl",
                reason=(
                    "small bucket uses safe uncompressed NCCL because Task 13 measured "
                    "compressed launch overhead"
                    if small_bucket
                    else "unverified CUDA dimensions use safe uncompressed NCCL"
                ),
                benchmark_matched=False,
            )
        return StrategyChoice(
            strategy="compressed",
            reason=(
                "unverified sharded dimensions retain the only registered "
                "semantically compatible ReducedShard transport"
            ),
            benchmark_matched=False,
        )

    def as_selector(
        self,
    ) -> Callable[[CommunicationPlan, CompileContext], StrategyChoice]:
        """Adapt this table to the backend registry selector protocol."""

        def select(plan: CommunicationPlan, context: CompileContext) -> StrategyChoice:
            if plan.compression is None:
                return StrategyChoice(
                    strategy="native_nccl",
                    reason="uncompressed auto request uses native NCCL",
                    benchmark_matched=False,
                )
            return self.select(
                context,
                plan.compression,
                collective=plan.collective,
                output_layout=plan.output_layout,
            )

        return select


def _normalize_architecture(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def _normalize_dtype(value: str) -> str:
    normalized = value.strip().lower().removeprefix("torch.")
    return {"float16": "fp16", "half": "fp16", "bfloat16": "bf16"}.get(
        normalized,
        normalized,
    )


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)
