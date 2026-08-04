from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ccdl_comm.config import CompressionConfig


@pytest.mark.parametrize("world_size", (1, 2, 3, 4, 5, 8))
def test_chunk_plan_covers_uneven_tensor_with_equal_rank_shards(world_size: int) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan

    original_numel = world_size * 7 + 3
    plan = compile_chunk_plan(original_numel=original_numel, world_size=world_size)

    assert plan.original_numel == original_numel
    assert plan.padded_numel % world_size == 0
    assert plan.shard_numel * world_size == plan.padded_numel
    assert plan.owner_by_chunk == tuple(range(world_size))
    assert len(plan.chunks) == world_size
    assert tuple(chunk.start for chunk in plan.chunks) == tuple(
        rank * plan.shard_numel for rank in range(world_size)
    )
    assert all(chunk.numel == plan.shard_numel for chunk in plan.chunks)
    assert plan.chunks[-1].stop == plan.padded_numel


def test_chunk_plan_is_immutable_and_resolves_rank_local_range() -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan

    plan = compile_chunk_plan(original_numel=11, world_size=4)

    assert plan.chunk_for_rank(2) == plan.chunks[2]
    with pytest.raises(FrozenInstanceError):
        plan.shard_numel = 99  # type: ignore[misc]
    with pytest.raises(ValueError, match="rank"):
        plan.chunk_for_rank(4)
    with pytest.raises(TypeError, match="rank must be an integer"):
        plan.chunk_for_rank(True)


@pytest.mark.parametrize(
    ("original_numel", "world_size", "message"),
    ((-1, 2, "original_numel"), (1, 0, "world_size")),
)
def test_chunk_plan_rejects_invalid_dimensions(original_numel: int, world_size: int, message: str) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan

    with pytest.raises(ValueError, match=message):
        compile_chunk_plan(original_numel=original_numel, world_size=world_size)


def test_chunk_plan_rejects_inconsistent_manual_construction() -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import ChunkPlan, ChunkRange

    with pytest.raises(ValueError, match="padded_numel"):
        ChunkPlan(
            original_numel=3,
            world_size=2,
            padded_numel=5,
            shard_numel=2,
            chunks=(ChunkRange(0, 2), ChunkRange(2, 4)),
            owner_by_chunk=(0, 1),
        )
    with pytest.raises(ValueError, match="chunks"):
        ChunkPlan(
            original_numel=3,
            world_size=2,
            padded_numel=4,
            shard_numel=2,
            chunks=(ChunkRange(0, 2),),
            owner_by_chunk=(0, 1),
        )
    with pytest.raises(ValueError, match="owner_by_chunk"):
        ChunkPlan(
            original_numel=3,
            world_size=2,
            padded_numel=4,
            shard_numel=2,
            chunks=(ChunkRange(0, 2), ChunkRange(2, 4)),
            owner_by_chunk=(1, 0),
        )
    with pytest.raises(TypeError, match="owner_by_chunk"):
        ChunkPlan(
            original_numel=3,
            world_size=2,
            padded_numel=4,
            shard_numel=2,
            chunks=(ChunkRange(0, 2), ChunkRange(2, 4)),
            owner_by_chunk=(False, True),
        )


@pytest.mark.parametrize(
    ("original_numel", "world_size"),
    ((True, 2), (3.5, 2), (3, True), (3, 2.5)),
)
def test_chunk_plan_rejects_non_integer_dimensions(original_numel, world_size) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan

    with pytest.raises(TypeError, match="must be an integer"):
        compile_chunk_plan(original_numel=original_numel, world_size=world_size)


def test_chunk_range_rejects_non_integer_offsets() -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import ChunkRange

    with pytest.raises(TypeError, match="must be an integer"):
        ChunkRange(True, 1)
    with pytest.raises(TypeError, match="must be an integer"):
        ChunkRange(0, 1.5)


def test_empty_cuda_tensor_returns_without_launching_quantization_kernel() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    class Dist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def get_rank():
            return 1

        @staticmethod
        def all_to_all(output, input, async_op=False):
            raise AssertionError("empty tensor must bypass NCCL")

    def import_module(name):
        if name == "torch.distributed":
            return Dist
        if name == "torch":
            return torch
        raise AssertionError(name)

    result = make_torch_compressed_reduce_scatter_shard(import_module=import_module)(
        torch.empty(0, device="cuda", dtype=torch.float16),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert result.shard.is_cuda
    assert result.shard.numel() == 0
    assert result.shard_numel == 0
