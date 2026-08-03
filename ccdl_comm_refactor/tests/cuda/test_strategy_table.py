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


def test_small_full_bucket_prefers_uncompressed_nccl() -> None:
    choice = TABLE.select(
        _context(numel=32_768, world_size=4),
        CONFIG,
        collective="all_reduce",
        output_layout="full",
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False
    assert "small bucket" in choice.reason


def test_four_rank_large_shard_prefers_compressed_reduce_scatter() -> None:
    choice = TABLE.select(
        _context(numel=33_554_432, world_size=4),
        CONFIG,
        collective="reduce_scatter",
        output_layout="shard",
    )

    assert choice.strategy == "compressed"
    assert choice.benchmark_matched is True
    assert choice.expected_speedup == pytest.approx(2.73)
    assert "Task 13" in choice.reason


@pytest.mark.parametrize("world_size", [2, 4])
def test_large_full_bucket_prefers_validated_pipelined_ring(world_size: int) -> None:
    choice = TABLE.select(
        _context(numel=8_388_608, world_size=world_size),
        CONFIG,
        collective="all_reduce",
        output_layout="full",
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
        _context(
            numel=33_554_432,
            world_size=world_size,
            architecture=architecture,
            dtype=dtype,
        ),
        config,
        collective="all_reduce",
        output_layout="full",
    )

    assert choice.strategy == "native_nccl"
    assert choice.benchmark_matched is False
    assert "unverified" in choice.reason


def test_full_output_never_selects_reduced_shard_strategy() -> None:
    choice = TABLE.select(
        _context(numel=33_554_432, world_size=4),
        CONFIG,
        collective="all_reduce",
        output_layout="full",
    )

    assert choice.strategy != "compressed"


def test_table_rejects_unknown_collective_layout_semantics() -> None:
    with pytest.raises(UnsupportedCollective, match="broadcast:shard"):
        TABLE.select(
            _context(numel=33_554_432, world_size=4),
            CONFIG,
            collective="broadcast",
            output_layout="shard",
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
