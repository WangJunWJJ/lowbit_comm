"""Static CUDA strategy thresholds derived from validated benchmark evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import mul

from ccdl_comm.backend import StrategyChoice
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.plan import CommunicationPlan, CompileContext


TASK13_EVIDENCE = "tests/benchmarks/reports/task13_topology"
TASK13_POLICY_ID = "cuda-task13-a6000-v1"
TASK13_COMPRESSION = CompressionConfig(bit=8, group_size=64)


@dataclass(frozen=True, slots=True)
class CudaStrategyRule:
    """One immutable, benchmark-backed CUDA strategy threshold."""

    collective: str
    output_layout: str
    device_architecture: str
    world_size: int
    dtype: str
    compression: CompressionConfig
    min_numel: int
    max_numel: int
    strategy: str
    speedup: float
    same_semantics_baseline: bool

    def __post_init__(self) -> None:
        if self.min_numel < 0 or self.max_numel < self.min_numel:
            raise ValueError("strategy rule requires 0 <= min_numel <= max_numel")
        if self.speedup <= 0:
            raise ValueError("strategy rule speedup must be > 0")

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
            and compression == self.compression
            and self.min_numel <= _numel(context.shape) <= self.max_numel
            and _ring_aligned(context, compression)
        )


class CudaStrategyTable:
    """Select a CUDA strategy once from immutable, benchmarked dimensions."""

    def __init__(self, rules: tuple[CudaStrategyRule, ...]) -> None:
        self._rules = tuple(rules)
        for index, rule in enumerate(self._rules):
            for other in self._rules[index + 1 :]:
                if (
                    _rule_domain(rule) == _rule_domain(other)
                    and rule.min_numel <= other.max_numel
                    and other.min_numel <= rule.max_numel
                ):
                    raise ValueError("CUDA strategy table contains overlapping rules")

    @classmethod
    def from_task13_a6000(cls) -> CudaStrategyTable:
        """Build the checked-in Task 13 A6000 threshold table."""

        rules: list[CudaStrategyRule] = []
        conservative_speedups = {
            ("full", 2): 1.6253136683796892,
            ("full", 4): 1.33102168223003,
            ("shard", 2): 2.855439327752976,
            ("shard", 4): 2.6032121073325847,
        }
        for (layout, world_size), speedup in conservative_speedups.items():
            rules.append(
                CudaStrategyRule(
                    collective=("all_reduce" if layout == "full" else "reduce_scatter"),
                    output_layout=layout,
                    device_architecture="nvidia_rtx_a6000",
                    world_size=world_size,
                    dtype="fp16",
                    compression=TASK13_COMPRESSION,
                    min_numel=8_388_608,
                    max_numel=33_554_432,
                    strategy="topology" if layout == "full" else "compressed",
                    speedup=speedup,
                    same_semantics_baseline=layout == "full",
                )
            )
        return cls(tuple(rules))

    def select(
        self,
        plan: CommunicationPlan,
        context: CompileContext,
    ) -> StrategyChoice:
        """Return one semantically compatible compile-time strategy choice."""

        if not isinstance(plan, CommunicationPlan):
            raise TypeError("plan must be a CommunicationPlan")
        if not isinstance(context, CompileContext):
            raise TypeError("context must be a CompileContext")
        compression = plan.compression
        if compression is None:
            return StrategyChoice(
                strategy="native_nccl",
                reason="uncompressed auto request uses native NCCL",
                policy_id=TASK13_POLICY_ID,
                benchmark_matched=False,
            )
        semantic_key = (plan.collective, plan.output_layout)
        if semantic_key not in {("all_reduce", "full"), ("reduce_scatter", "shard")}:
            raise UnsupportedCollective(
                f"{plan.collective}:{plan.output_layout}",
                reason="CUDA strategy table has no safe semantics for this request",
            )
        for rule in self._rules:
            if rule.matches(
                context,
                compression,
                collective=plan.collective,
                output_layout=plan.output_layout,
            ):
                return StrategyChoice(
                    strategy=rule.strategy,
                    reason=(
                        "Task 13 A6000 benchmark matched all validated dimensions; "
                        f"observed speedup {rule.speedup:.2f}x"
                    ),
                    policy_id=TASK13_POLICY_ID,
                    benchmark_matched=True,
                    expected_speedup=(rule.speedup if rule.same_semantics_baseline else None),
                    observed_speedup=(None if rule.same_semantics_baseline else rule.speedup),
                    baseline=(
                        None
                        if rule.same_semantics_baseline
                        else "native_fp16_full_output_reference"
                    ),
                    evidence=TASK13_EVIDENCE,
                )
        if semantic_key == ("all_reduce", "full"):
            mismatches = _task13_mismatches(context, compression)
            small_bucket = mismatches == ("bucket_size",) and _numel(
                context.shape
            ) < 8_388_608
            return StrategyChoice(
                strategy="native_nccl",
                reason=(
                    "small bucket uses safe uncompressed NCCL because Task 13 measured "
                    "compressed launch overhead"
                    if small_bucket
                    else (
                        "unverified CUDA dimensions use safe uncompressed NCCL: "
                        + ", ".join(mismatches)
                    )
                ),
                policy_id=TASK13_POLICY_ID,
                benchmark_matched=False,
            )
        return StrategyChoice(
            strategy="compressed",
            reason=(
                "unverified sharded dimensions retain the only registered "
                "semantically compatible ReducedShard transport"
            ),
            policy_id=TASK13_POLICY_ID,
            benchmark_matched=False,
        )

    def as_selector(
        self,
    ) -> Callable[[CommunicationPlan, CompileContext], StrategyChoice]:
        """Adapt this table to the backend registry selector protocol."""

        def select(plan: CommunicationPlan, context: CompileContext) -> StrategyChoice:
            return self.select(plan, context)

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


def _ring_aligned(
    context: CompileContext,
    compression: CompressionConfig,
) -> bool:
    numel = _numel(context.shape)
    return (
        numel % context.world_size == 0
        and (numel // context.world_size) % compression.group_size == 0
    )


def _rule_domain(rule: CudaStrategyRule) -> tuple[object, ...]:
    return (
        rule.collective,
        rule.output_layout,
        rule.device_architecture,
        rule.world_size,
        rule.dtype,
        rule.compression,
    )


def _task13_mismatches(
    context: CompileContext,
    compression: CompressionConfig,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if _normalize_architecture(context.device_architecture) != "nvidia_rtx_a6000":
        mismatches.append("device_architecture")
    if context.world_size not in {2, 4}:
        mismatches.append("world_size")
    if _normalize_dtype(context.dtype) != "fp16":
        mismatches.append("dtype")
    if compression != TASK13_COMPRESSION:
        mismatches.append("compression_profile")
    numel = _numel(context.shape)
    if not 8_388_608 <= numel <= 33_554_432:
        mismatches.append("bucket_size")
    elif not _ring_aligned(context, compression):
        mismatches.append("ring_alignment")
    return tuple(mismatches) or ("unmatched_rule",)
