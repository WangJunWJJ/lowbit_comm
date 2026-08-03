from __future__ import annotations

import pytest

from ccdl_comm import CommunicationPlan, CompileContext, CompressionConfig
from ccdl_comm.cuda.strategy_table import CudaStrategyTable
from ccdl_comm.exceptions import UnsupportedCollective


CONFIG = CompressionConfig(bit=8, group_size=64)
TABLE = CudaStrategyTable.from_task13_a6000()


def _context(
    *,
    numel: int,
    world_size: int,
    architecture: str = "NVIDIA RTX A6000",
    dtype: str = "fp16",
) -> CompileContext:
    return CompileContext(
        rank=0,
        world_size=world_size,
        device="cuda:0",
        device_architecture=architecture,
        shape=(numel,),
        dtype=dtype,
    )


def _plan(
    collective: str,
    output_layout: str,
    config: CompressionConfig = CONFIG,
) -> CommunicationPlan:
    return CommunicationPlan(
        collective,
        "auto",
        compression=config,
        output_layout=output_layout,
    )


def test_small_full_bucket_prefers_uncompressed_nccl() -> None:
    choice = TABLE.select(
        _plan("all_reduce", "full"),
        _context(numel=32_768, world_size=4),
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False
    assert "small bucket" in choice.reason


def test_four_rank_large_shard_prefers_compressed_reduce_scatter() -> None:
    choice = TABLE.select(
        _plan("reduce_scatter", "shard"),
        _context(numel=33_554_432, world_size=4),
    )

    assert choice.strategy == "compressed"
    assert choice.benchmark_matched is True
    assert choice.expected_speedup is None
    assert choice.observed_speedup == pytest.approx(2.73)
    assert choice.baseline == "native_fp16_full_output_reference"
    assert "Task 13" in choice.reason


@pytest.mark.parametrize("world_size", [2, 4])
def test_large_full_bucket_prefers_validated_pipelined_ring(world_size: int) -> None:
    choice = TABLE.select(
        _plan("all_reduce", "full"),
        _context(numel=8_388_608, world_size=world_size),
    )

    assert choice.strategy == "topology"
    assert choice.benchmark_matched is True


@pytest.mark.parametrize(
    ("architecture", "dtype", "world_size", "config"),
    [
        ("unknown", "fp16", 4, CONFIG),
        ("NVIDIA RTX A6000", "bf16", 4, CONFIG),
        ("NVIDIA RTX A6000", "fp16", 3, CONFIG),
        (
            "NVIDIA RTX A6000",
            "fp16",
            4,
            CompressionConfig(bit=4, group_size=64, allow_experimental=True),
        ),
    ],
)
def test_unverified_full_context_uses_explainable_native_fallback(
    architecture: str,
    dtype: str,
    world_size: int,
    config: CompressionConfig,
) -> None:
    choice = TABLE.select(
        _plan("all_reduce", "full", config),
        _context(
            numel=33_554_432,
            world_size=world_size,
            architecture=architecture,
            dtype=dtype,
        ),
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False
    assert "unverified" in choice.reason


def test_full_output_never_selects_reduced_shard_strategy() -> None:
    choice = TABLE.select(
        _plan("all_reduce", "full"),
        _context(numel=33_554_432, world_size=4),
    )

    assert choice.strategy != "compressed"


def test_bucket_larger_than_validated_range_uses_native_nccl() -> None:
    choice = TABLE.select(
        _plan("all_reduce", "full"),
        _context(numel=33_554_433, world_size=4),
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False
    assert "unverified" in choice.reason


def test_unvalidated_compression_profile_uses_native_nccl() -> None:
    choice = TABLE.select(
        _plan(
            "all_reduce",
            "full",
            CompressionConfig(bit=8, group_size=64, compact=True),
        ),
        _context(numel=8_388_608, world_size=4),
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False


def test_unaligned_ring_shape_uses_native_nccl() -> None:
    choice = TABLE.select(
        _plan("all_reduce", "full"),
        _context(numel=8_388_609, world_size=4),
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False


def test_table_rejects_unknown_collective_layout_semantics() -> None:
    with pytest.raises(UnsupportedCollective, match="broadcast:shard"):
        TABLE.select(
            _plan("broadcast", "shard"),
            _context(numel=33_554_432, world_size=4),
        )


def test_table_can_be_registered_as_compile_time_selector() -> None:
    selector = TABLE.as_selector()
    plan = CommunicationPlan(
        "all_reduce",
        "auto",
        compression=CONFIG,
        output_layout="full",
    )

    choice = selector(plan, _context(numel=8_388_608, world_size=4))

    assert choice.strategy == "topology"
