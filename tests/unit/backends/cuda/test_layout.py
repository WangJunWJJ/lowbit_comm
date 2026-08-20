import pytest

from lowbit_comm.backends.cuda.layout import (
    FullTensorLayout,
    ReducedShardLayout,
    build_fulltensor_layout,
    build_reduced_shard_layout,
)
from lowbit_comm.api.policy import CompressionKind
from lowbit_comm.core.errors import CompileError


class StringSubclass(str):
    """A string subtype rejected by exact layout contracts."""


class IntSubclass(int):
    """An integer subtype rejected by exact layout contracts."""


@pytest.mark.parametrize(
    "numel,group_size,padded",
    [(1, 16, 16), (16, 16, 16), (17, 16, 32)],
)
def test_layout_pads_only_to_group_boundary(
    numel: int,
    group_size: int,
    padded: int,
) -> None:
    layout = build_fulltensor_layout(
        numel=numel,
        dtype="fp16",
        world_size=4,
        compression=CompressionKind.INT8,
        group_size=group_size,
    )

    assert layout.padded_numel == padded
    assert layout.group_count == padded // group_size
    assert layout.payload_bytes_per_rank == padded + 2 * layout.group_count


def test_layout_accounts_for_gather_output_and_workspace_bytes() -> None:
    layout = build_fulltensor_layout(
        numel=17,
        dtype="bf16",
        world_size=2,
        compression=CompressionKind.INT8,
        group_size=16,
    )

    assert layout == FullTensorLayout(
        logical_numel=17,
        padded_numel=32,
        group_size=16,
        group_count=2,
        payload_bytes_per_rank=36,
        gathered_payload_bytes=72,
        output_bytes=34,
        workspace_bytes=108,
    )


def test_layout_supports_zero_numel_without_a_phantom_group() -> None:
    layout = build_fulltensor_layout(
        numel=0,
        dtype="fp16",
        world_size=2,
        compression=CompressionKind.INT8,
        group_size=64,
    )

    assert layout.padded_numel == 0
    assert layout.group_count == 0
    assert layout.payload_bytes_per_rank == 0
    assert layout.gathered_payload_bytes == 0
    assert layout.output_bytes == 0
    assert layout.workspace_bytes == 0


def test_layout_is_immutable() -> None:
    layout = build_fulltensor_layout(
        numel=16,
        dtype="fp16",
        world_size=4,
        compression=CompressionKind.INT8,
        group_size=16,
    )

    with pytest.raises(AttributeError):
        layout.logical_numel = 32  # type: ignore[misc]


def test_layout_rejects_unregistered_domain_values() -> None:
    with pytest.raises(CompileError, match="world size"):
        build_fulltensor_layout(
            numel=16,
            dtype="fp16",
            world_size=8,
            compression=CompressionKind.INT8,
            group_size=16,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"numel": -1}, "numel"),
        ({"numel": True}, "numel"),
        ({"dtype": "fp32"}, "dtype"),
        ({"world_size": 1}, "world size"),
        ({"group_size": 8}, "group size"),
    ],
)
def test_layout_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "numel": 16,
        "dtype": "fp16",
        "world_size": 2,
        "compression": CompressionKind.INT8,
        "group_size": 16,
    }
    values.update(kwargs)

    with pytest.raises(CompileError, match=message):
        build_fulltensor_layout(**values)  # type: ignore[arg-type]


def test_layout_rejects_byte_count_overflow() -> None:
    with pytest.raises(CompileError, match="overflow"):
        build_fulltensor_layout(
            numel=1 << 63,
            dtype="fp16",
            world_size=4,
            compression=CompressionKind.INT8,
            group_size=16,
        )


def test_fp16_int8_layout_matches_compact_pack_format() -> None:
    layout = build_fulltensor_layout(
        numel=33,
        dtype="fp16",
        world_size=2,
        compression=CompressionKind.INT8,
        group_size=16,
    )

    assert layout.payload_bytes_per_rank == 3 * (16 + 2)
    assert layout.gathered_payload_bytes == 2 * 3 * (16 + 2)


def test_native_layout_has_no_quantized_workspace() -> None:
    layout = build_fulltensor_layout(
        numel=33,
        dtype="fp16",
        world_size=2,
        compression=CompressionKind.NONE,
        group_size=None,
    )

    assert layout.payload_bytes_per_rank == 0
    assert layout.gathered_payload_bytes == 0
    assert layout.workspace_bytes == 0


@pytest.mark.parametrize(
    ("numel", "world_size", "group_size"),
    [(10, 4, 16), (0, 4, 16), (1, 4, 16), (4097, 2, 64)],
)
def test_reduced_shard_layout_assigns_every_rank_one_logical_shard(
    numel: int,
    world_size: int,
    group_size: int,
) -> None:
    for rank in range(world_size):
        layout = build_reduced_shard_layout(
            numel=numel,
            dtype="fp16",
            world_size=world_size,
            compression=CompressionKind.INT8,
            group_size=group_size,
            rank=rank,
        )

        assert layout.logical_shard_length == (
            numel + world_size - 1
        ) // world_size
        assert layout.offset == min(rank * layout.logical_shard_length, numel)
        assert layout.valid_length == min(
            layout.logical_shard_length,
            numel - layout.offset,
        )
        assert layout.transport_shard_length % group_size == 0
        assert layout.output_numel == layout.logical_shard_length


def test_reduced_shard_layout_accounts_for_compact_int8_payloads() -> None:
    layout = build_reduced_shard_layout(
        numel=10,
        dtype="bf16",
        world_size=4,
        compression=CompressionKind.INT8,
        group_size=16,
        rank=3,
    )

    assert layout == ReducedShardLayout(
        global_numel=10,
        logical_shard_length=3,
        transport_shard_length=16,
        offset=9,
        valid_length=1,
        group_size=16,
        groups_per_shard=1,
        payload_bytes_per_destination=18,
        send_payload_bytes=72,
        receive_payload_bytes=72,
        output_numel=3,
        output_bytes=6,
        workspace_bytes=144,
    )


@pytest.mark.parametrize("world_size", [1, 3, 5, 7])
def test_reduced_shard_layout_accepts_arbitrary_positive_world_sizes(
    world_size: int,
) -> None:
    layout = build_reduced_shard_layout(
        numel=10,
        dtype="fp16",
        world_size=world_size,
        compression=CompressionKind.NONE,
        group_size=None,
        rank=world_size - 1,
    )

    assert layout.logical_shard_length == (10 + world_size - 1) // world_size
    assert layout.output_numel == layout.logical_shard_length


def test_reduced_shard_native_layout_reserves_only_required_input_padding(
) -> None:
    layout = build_reduced_shard_layout(
        numel=10,
        dtype="fp16",
        world_size=4,
        compression=CompressionKind.NONE,
        group_size=None,
        rank=3,
    )

    assert layout.group_size is None
    assert layout.groups_per_shard == 0
    assert layout.payload_bytes_per_destination == 0
    assert layout.send_payload_bytes == 0
    assert layout.receive_payload_bytes == 0
    assert layout.output_bytes == 6
    assert layout.workspace_bytes == 24


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"numel": -1}, "numel"),
        ({"numel": True}, "numel"),
        ({"dtype": "fp32"}, "dtype"),
        ({"dtype": StringSubclass("fp16")}, "dtype"),
        ({"world_size": 0}, "world size"),
        ({"world_size": True}, "world size"),
        ({"rank": -1}, "rank"),
        ({"rank": 2}, "rank"),
        ({"rank": True}, "rank"),
        ({"group_size": 8}, "group size"),
        ({"group_size": IntSubclass(16)}, "group size"),
        ({"compression": object()}, "compression"),
    ],
)
def test_reduced_shard_layout_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "numel": 16,
        "dtype": "fp16",
        "world_size": 2,
        "compression": CompressionKind.INT8,
        "group_size": 16,
        "rank": 1,
    }
    values.update(kwargs)

    with pytest.raises(CompileError, match=message):
        build_reduced_shard_layout(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("compression", "group_size", "message"),
    [
        (CompressionKind.NONE, 16, "native"),
        (CompressionKind.INT8, None, "group size"),
    ],
)
def test_reduced_shard_layout_requires_matching_compression_configuration(
    compression: CompressionKind,
    group_size: int | None,
    message: str,
) -> None:
    with pytest.raises(CompileError, match=message):
        build_reduced_shard_layout(
            numel=16,
            dtype="fp16",
            world_size=2,
            compression=compression,
            group_size=group_size,
            rank=0,
        )


@pytest.mark.parametrize("numel", [(1 << 63) - 1, 1 << 63])
def test_reduced_shard_layout_rejects_signed_64_bit_overflow(
    numel: int,
) -> None:
    with pytest.raises(CompileError, match="overflow"):
        build_reduced_shard_layout(
            numel=numel,
            dtype="fp16",
            world_size=2,
            compression=CompressionKind.INT8,
            group_size=16,
            rank=0,
        )
